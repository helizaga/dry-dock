"""DSP building blocks. All audio is float64, shape (frames, 2), 44.1 kHz, and may exceed 1.0 until the limiter."""

from __future__ import annotations

import math

import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, fftconvolve, hilbert, lfilter, resample_poly, sosfilt, sosfiltfilt

SR = 44100
_meter = pyln.Meter(SR)


def db(x):
    return 20 * np.log10(np.maximum(x, 1e-12))


def gain(d):
    return 10 ** (np.asarray(d, dtype=float) / 20)


def k_weight(x: np.ndarray) -> np.ndarray:
    """BS.1770 K-weighting (pre-filter shelf + RLB high-pass), same coefficients as pyloudnorm."""
    for f in _meter._filters.values():
        x = lfilter(f.b, f.a, x, axis=0)
    return x


def lufs(x: np.ndarray) -> float:
    """BS.1770-4 integrated loudness; matches pyloudnorm.Meter.integrated_loudness."""
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < int(0.4 * SR) or not np.any(x):
        return -70.0
    return lufs_weighted(k_weight(x))


def lufs_weighted(k: np.ndarray) -> float:
    """Integrated loudness of an already K-weighted signal (400 ms blocks, 75% overlap, -70 LUFS absolute and
    -10 LU relative gates). Lets several spans of one song share a single K-weighting pass."""
    if k.ndim == 1:
        k = k[:, None]
    if k.shape[0] < int(0.4 * SR) or not np.any(k):
        return -70.0
    t_g, step = 0.4, 0.25
    blocks = int(np.round((k.shape[0] / SR - t_g) / (t_g * step))) + 1
    j = np.arange(blocks)
    lo = (t_g * (j * step) * SR).astype(int)
    hi = (t_g * (j * step + 1) * SR).astype(int)
    c = np.vstack([np.zeros((1, k.shape[1])), np.cumsum(k ** 2, axis=0)])
    hi = np.minimum(hi, k.shape[0])
    z = (c[hi] - c[lo]) / (t_g * SR)  # per block, per channel mean square
    power = z.sum(axis=1)
    with np.errstate(divide="ignore"):
        loud = -0.691 + 10 * np.log10(power)
    abs_gated = loud >= -70.0
    if not abs_gated.any():
        return -70.0
    rel = -0.691 + 10 * np.log10(z[abs_gated].mean(axis=0).sum()) - 10.0
    gated = abs_gated & (loud > rel)
    if not gated.any():
        return -70.0
    return float(-0.691 + 10 * np.log10(z[gated].mean(axis=0).sum()))


def _blocks_near(x: np.ndarray, floor: float, block: int = 4096) -> list[tuple[int, int]]:
    """Sample ranges (padded for the resampler's filter) of blocks whose sample peak reaches ``floor``."""
    n = x.shape[0]
    k = int(math.ceil(n / block))
    peaks = np.pad(np.abs(x).max(axis=1), (0, k * block - n)).reshape(k, block).max(axis=1)
    hits = np.flatnonzero(peaks >= floor)
    pad = 64
    return _merge([(max(0, i * block - pad), min(n, (i + 1) * block + pad)) for i in hits])


def true_peak_db(x: np.ndarray) -> float:
    """4x-oversampled peak. Inter-sample overs are at most a few dB above the sample peak, so only blocks within
    6 dB of it are oversampled (same answer as oversampling everything, a fraction of the time)."""
    sp = float(np.abs(x).max()) if x.size else 0.0
    if sp <= 0:
        return -240.0
    best = sp
    for a, b in _blocks_near(x, sp * 10 ** (-6 / 20)):
        best = max(best, float(np.abs(resample_poly(x[a:b].astype(np.float32), 4, 1, axis=0)).max()))
    return float(db(best))


def _merge(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in ranges:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


# ---------------------------------------------------------------- EQ (RBJ biquads, causal)


def _biquad(kind: str, f0: float, g_db: float, q: float):
    a_ = 10 ** (g_db / 40)
    w = 2 * math.pi * f0 / SR
    c, s = math.cos(w), math.sin(w)
    alpha = s / (2 * q)
    if kind == "peak":
        b = [1 + alpha * a_, -2 * c, 1 - alpha * a_]
        a = [1 + alpha / a_, -2 * c, 1 - alpha / a_]
    elif kind == "highshelf":
        sq = 2 * math.sqrt(a_) * alpha
        b = [a_ * ((a_ + 1) + (a_ - 1) * c + sq), -2 * a_ * ((a_ - 1) + (a_ + 1) * c), a_ * ((a_ + 1) + (a_ - 1) * c - sq)]
        a = [(a_ + 1) - (a_ - 1) * c + sq, 2 * ((a_ - 1) - (a_ + 1) * c), (a_ + 1) - (a_ - 1) * c - sq]
    elif kind == "lowshelf":
        sq = 2 * math.sqrt(a_) * alpha
        b = [a_ * ((a_ + 1) - (a_ - 1) * c + sq), 2 * a_ * ((a_ - 1) - (a_ + 1) * c), a_ * ((a_ + 1) - (a_ - 1) * c - sq)]
        a = [(a_ + 1) + (a_ - 1) * c + sq, -2 * ((a_ - 1) + (a_ + 1) * c), (a_ + 1) + (a_ - 1) * c - sq]
    else:
        raise ValueError(f"unknown EQ band {kind!r}")
    return np.array(b) / a[0], np.array(a) / a[0]


def eq(x: np.ndarray, bands: list[tuple]) -> np.ndarray:
    """bands: ("peak"|"lowshelf"|"highshelf", hz, gain_db, q) or ("hp"|"lp", hz, order)."""
    y = x
    for band in bands:
        if band[0] in ("hp", "lp"):
            sos = butter(band[2], band[1], btype="highpass" if band[0] == "hp" else "lowpass", fs=SR, output="sos")
            y = sosfilt(sos, y, axis=0)
        elif band[2]:
            b, a = _biquad(*band)
            y = lfilter(b, a, y, axis=0)
    return y


def bandpass(x: np.ndarray, lo: float, hi: float, order: int = 2) -> np.ndarray:
    return sosfiltfilt(butter(order, [lo, hi], btype="bandpass", fs=SR, output="sos"), x, axis=0)


def split_low(x: np.ndarray, hz: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase complementary split: low + high == x exactly."""
    low = sosfiltfilt(butter(2, hz, btype="lowpass", fs=SR, output="sos"), x, axis=0)
    return low, x - low


# ---------------------------------------------------------------- dynamics


def _follow(level: np.ndarray, attack_ms: float, release_ms: float, block: int) -> np.ndarray:
    ca = math.exp(-block / (attack_ms / 1000 * SR))
    cr = math.exp(-block / (release_ms / 1000 * SR))
    out = np.empty_like(level)
    cur = level[0] if level.size else 0.0
    for i, v in enumerate(level):
        cur = v + (cur - v) * (ca if v > cur else cr)
        out[i] = cur
    return out


def _to_samples(per_block: np.ndarray, block: int, frames: int) -> np.ndarray:
    return np.interp(np.arange(frames), np.arange(per_block.size) * block + block / 2, per_block)


def compressor(x: np.ndarray, *, threshold_db: float, ratio: float, attack_ms: float, release_ms: float,
               knee_db: float = 6.0, sc_hpf: float | None = None, block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Feed-forward, linked stereo, RMS-ish detector with an optional sidechain high-pass. Returns (audio, GR dB)."""
    sc = sosfilt(butter(2, sc_hpf, btype="highpass", fs=SR, output="sos"), x, axis=0) if sc_hpf else x
    p = np.mean(sc ** 2, axis=1)
    n = int(math.ceil(p.size / block))
    lvl = 10 * np.log10(np.pad(p, (0, n * block - p.size)).reshape(n, block).mean(axis=1) + 1e-12)
    over = _follow(lvl, attack_ms, release_ms, block) - threshold_db
    slope = 1 - 1 / ratio
    gr = np.where(over <= -knee_db / 2, 0.0,
                  np.where(over >= knee_db / 2, over * slope, slope * (over + knee_db / 2) ** 2 / (2 * knee_db)))
    return x * gain(-_to_samples(gr, block, x.shape[0]))[:, None], gr


def dynamic_band(x: np.ndarray, lo: float, hi: float, *, max_cut_db: float, over_db: float = 2.0, hop: int = 512,
                 smooth: int = 7) -> tuple[np.ndarray, float]:
    """Cut lo-hi only while that band rises above its own typical share of the signal (de-esser / hat tamer).
    Returns (audio, 90th-percentile cut while active)."""
    band = bandpass(x, lo, hi)
    frames = x.shape[0] // hop
    if frames < 8:
        return x, 0.0
    full = np.sqrt(np.mean(x[: frames * hop].mean(axis=1).reshape(frames, hop) ** 2, axis=1)) + 1e-9
    part = np.sqrt(np.mean(band[: frames * hop].mean(axis=1).reshape(frames, hop) ** 2, axis=1)) + 1e-9
    share = db(part / full)
    loud = full > np.percentile(full, 30)
    typical = float(np.median(share[loud])) if loud.any() else float(np.median(share))
    excess = np.clip(share - (typical + over_db), 0, max_cut_db * 2) / 2
    w = np.hanning(smooth)
    excess = np.minimum(np.convolve(excess, w / w.sum(), mode="same"), max_cut_db)
    cut = np.interp(np.arange(x.shape[0]), np.arange(frames) * hop + hop / 2, excess)
    active = cut > 0.05
    return x - band * (1 - gain(-cut))[:, None], float(np.percentile(cut[active], 90)) if active.any() else 0.0


def soft_clip(x: np.ndarray, threshold_db: float, knee: float = 0.7) -> np.ndarray:
    """Values above knee x threshold approach the threshold asymptotically (tanh); output never exceeds it."""
    c = float(gain(threshold_db))
    k = knee * c
    a = np.abs(x)
    shaped = np.where(a > k, k + (c - k) * np.tanh((a - k) / (c - k)), a)
    return np.sign(x) * shaped


def tp_envelope(x: np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Per-sample true peak (4x oversampled). Blocks whose sample peak stays under ``floor`` keep the sample peak:
    a limiter only needs the exact value where a peak can reach its ceiling."""
    env = np.abs(x).max(axis=1)
    blocks = _blocks_near(x, floor) if floor > 0 else [(0, x.shape[0])]
    for a, b in blocks:
        over = np.abs(resample_poly(x[a:b].astype(np.float32), 4, 1, axis=0)).max(axis=1)
        m = b - a
        over = np.pad(over, (0, max(0, m * 4 - over.size)))[: m * 4].reshape(m, 4).max(axis=1)
        env[a:b] = np.maximum(env[a:b], over)
    return env


def limiter_gain(env: np.ndarray, ceiling_db: float, *, release_ms: float = 60.0, lookahead_ms: float = 4.0,
                 block: int = 32) -> np.ndarray:
    """Per-sample gain (linear) that holds ``env`` (a true-peak envelope) under the ceiling: lookahead brickwall,
    instant attack, exponential release, computed at a 32-sample control rate."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d

    need = np.minimum(1.0, float(gain(ceiling_db)) / np.maximum(env, 1e-9))
    n = int(math.ceil(need.size / block))
    need = np.pad(need, (0, n * block - need.size), constant_values=1.0).reshape(n, block).min(axis=1)
    reach = max(1, int(math.ceil(lookahead_ms / 1000 * SR / block)))
    target = uniform_filter1d(minimum_filter1d(need, size=2 * reach + 1, mode="nearest"), size=reach + 1, mode="nearest")
    if target.min() >= 1.0:
        return np.ones(env.size)
    coeff = math.exp(-block / (release_ms / 1000 * SR))
    g = np.empty(n)
    cur = 1.0
    for i, t in enumerate(target.tolist()):
        cur = t if t < cur else t + (cur - t) * coeff
        g[i] = cur
    return _to_samples(g, block, env.size)


def limiter(x: np.ndarray, ceiling_db: float, *, release_ms: float = 60.0, lookahead_ms: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Lookahead brickwall limiter on the true peak. Returns (audio, per-sample gain dB)."""
    env = tp_envelope(x, float(gain(ceiling_db - 6.0)))
    curve = limiter_gain(env, ceiling_db, release_ms=release_ms, lookahead_ms=lookahead_ms)
    return x * curve[:, None], db(curve)


def peak_tame(x: np.ndarray, over_p999_db: float = 1.5) -> tuple[np.ndarray, float]:
    """Fast limiter at (99.9th percentile + over) so only rare spikes (plosives, stray hits) get caught."""
    c = float(db(np.percentile(np.abs(x), 99.9))) + over_p999_db
    y, red = limiter(x, c, release_ms=50.0, lookahead_ms=2.0)
    return y, float(-red.min())


# ---------------------------------------------------------------- low end


def octave_up(x: np.ndarray, *, below_hz: float = 50.0, amount: float = 0.4, fundamental_db: float = -1.5) -> np.ndarray:
    """Psychoacoustic bass: add an octave-up copy of the sub band (|z| sin 2phi, which does not raise the summed peak)
    and trim the fundamental. Moves a 30-40 Hz 808 into 60-80 Hz, where phones and ears work."""
    from scipy.fft import next_fast_len

    sub = sosfiltfilt(butter(4, below_hz, btype="lowpass", fs=SR, output="sos"), x, axis=0)
    z = hilbert(sub, N=next_fast_len(sub.shape[0]), axis=0)[: sub.shape[0]]  # FFT at a fast length
    octv = np.imag(z * z) / np.maximum(np.abs(z), 1e-9)
    octv = bandpass(octv, below_hz * 0.9, below_hz * 2.4)
    return x + sub * (float(gain(fundamental_db)) - 1) + amount * octv


def upper_harmonics(x: np.ndarray, *, amount: float = 0.15, lo: float = 90.0, hi: float = 400.0) -> np.ndarray:
    """A little band-limited saturation of the bass so the line reads on laptops."""
    pk = float(np.max(np.abs(x))) or 1.0
    sat = np.tanh(2.2 * x / pk + 0.15) - np.tanh(0.15)
    return x + amount * bandpass(sat * pk, lo, hi)


def kick_duck(bass: np.ndarray, drums: np.ndarray, *, depth_db: float = 2.0, kick_hz: float = 150.0,
              release_ms: float = 90.0) -> np.ndarray:
    """Sidechain the bass to the kick (drums below kick_hz), up to depth_db, following the kick envelope."""
    k = np.abs(sosfiltfilt(butter(4, kick_hz, btype="lowpass", fs=SR, output="sos"), drums.mean(axis=1)))
    block = 32
    n = k.size // block
    if n < 2:
        return bass
    env = _follow(k[: n * block].reshape(n, block).max(axis=1), 1.0, release_ms, block)
    ref = np.percentile(env[env > 1e-6], 99) if (env > 1e-6).any() else 1.0
    g = -depth_db * np.clip(env / ref, 0, 1)
    return bass * gain(_to_samples(g, block, bass.shape[0]))[:, None]


def mono_below(x: np.ndarray, hz: float = 110.0) -> np.ndarray:
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    s = sosfiltfilt(butter(4, hz, btype="highpass", fs=SR, output="sos"), s)
    return np.stack([m + s, m - s], axis=1)


# ---------------------------------------------------------------- stereo


def decorrelator(seed: int = 7, ms_len: float = 18.0) -> np.ndarray:
    """Flat-magnitude random-phase FIR with a short decay: a diffuse copy of the input for a synthetic side."""
    n = int(ms_len / 1000 * SR)
    n += n % 2
    rng = np.random.default_rng(seed)
    spec = np.exp(1j * rng.uniform(-np.pi, np.pi, n // 2 + 1))
    spec[0] = spec[-1] = 1.0
    h = np.fft.irfft(spec, n) * np.exp(-np.linspace(0, 4, n))
    return h / np.sqrt(np.sum(h ** 2))


def widen(x: np.ndarray, *, amount: float, above_hz: float = 300.0, seed: int = 7) -> np.ndarray:
    """Add a decorrelated side signal made from the mid above ``above_hz``. The mono sum (L+R) is untouched."""
    if amount <= 0:
        return x
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    hp = sosfiltfilt(butter(2, above_hz, btype="highpass", fs=SR, output="sos"), m)
    s = s + amount * fftconvolve(hp, decorrelator(seed), mode="full")[: m.size]
    return np.stack([m + s, m - s], axis=1)


# ---------------------------------------------------------------- vocal


def presence_ride(vocal: np.ndarray, rest: np.ndarray, *, target_db: float = 6.5, max_boost_db: float = 4.5,
                  window: float = 0.4, smooth_windows: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Boost-only ride: wherever the vocal sits less than target_db over everything else in 1-4 kHz, raise it (at most
    max_boost_db), smoothed over phrases. The boost holds through gaps between words (up to hold_s) and then decays,
    so reverb tails and bleed after the last line are not lifted. Returns (vocal, gain dB per sample)."""
    vb, rb = bandpass(vocal.mean(axis=1), 1000, 4000), bandpass(rest.mean(axis=1), 1000, 4000)
    vm = vocal.mean(axis=1)
    hop, size = int(0.1 * SR), int(window * SR)
    starts = np.arange(0, max(1, vm.size - size), hop)
    lvl = window_db(vm, starts, size)
    pres = window_db(vb, starts, size) - window_db(rb, starts, size)
    active = lvl > max(-60.0, float(np.max(lvl)) - 18.0)
    hold = 5  # windows (0.5 s)
    gains, held, idle = np.zeros(lvl.size), 0.0, 0
    for i in range(lvl.size):
        if active[i]:
            held, idle = float(np.clip(target_db - pres[i], 0.0, max_boost_db)), 0
        else:
            idle += 1
            if idle > hold:
                held *= 0.7
        gains[i] = held
    k = np.hanning(smooth_windows)
    curve = np.interp(np.arange(vocal.shape[0]), starts + size / 2, np.convolve(gains, k / k.sum(), mode="same"))
    return vocal * gain(curve)[:, None], curve


def active_rms_db(x: np.ndarray, floor_db: float = 40.0) -> float:
    m = x.mean(axis=1)
    hop = 2048
    k = m.size // hop
    if k == 0:
        return -120.0
    e = np.sqrt(np.mean(m[: k * hop].reshape(k, hop) ** 2, axis=1))
    on = e > np.max(e) * 10 ** (-floor_db / 20)
    return float(db(np.sqrt(np.mean(e[on] ** 2)))) if on.any() else -120.0


def window_db(x: np.ndarray, starts: np.ndarray, size: int) -> np.ndarray:
    """Mean-square level (dB) of x[s:s+size] for every start, in one pass (cumulative sums)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(x, dtype=np.float64) ** 2)])
    ends = np.minimum(starts + size, x.size)
    return 10 * np.log10((c[ends] - c[starts]) / np.maximum(ends - starts, 1) + 1e-15)
