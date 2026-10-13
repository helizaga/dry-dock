"""Vocal production: lead/backing split, targeted note fixes, hook doubles, reverb and delay throws, and a wider,
wetter backing layer. The slow steps (split, transcription, pitch tracking) are cached per song and stem source."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from drydock import dsp, paths

Log = Callable[[str], None]


def _cache(slug: str, tag: str) -> Path:
    d = paths.work_dir(slug) / f"vocal-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- split and lyrics


def split_lead(vocals: np.ndarray, slug: str, tag: str, log: Log) -> tuple[np.ndarray, np.ndarray]:
    """Lead vs backing (harmonies, ad-lib stacks) with glass-harbor's karaoke model. Backing is everything that is
    not lead, so lead + backing is exactly the input."""
    d = _cache(slug, tag)
    if (d / "lead.wav").exists():
        lead = sf.read(d / "lead.wav", dtype="float64", always_2d=True)[0][: vocals.shape[0]]
        return lead, vocals - lead
    from glass_harbor.separate import CLEAN_MODELS

    from drydock.stems import run_separator

    log("splitting lead from backing vocals")
    pad = 6.0
    with tempfile.TemporaryDirectory(prefix="drydock-lead-") as tmp:
        src = Path(tmp) / "vocals.wav"
        sf.write(src, (vocals * dsp.gain(-pad)).astype(np.float32), dsp.SR, subtype="FLOAT")
        out = run_separator({"model": CLEAN_MODELS["lead"]}, src, Path(tmp) / "out")
        lead, _ = sf.read(out["vocals"], dtype="float64", always_2d=True)
    lead = np.pad(lead[: vocals.shape[0]], ((0, max(0, vocals.shape[0] - lead.shape[0])), (0, 0))) * dsp.gain(pad)
    sf.write(d / "lead.wav", lead.astype(np.float32), dsp.SR, subtype="FLOAT")
    return lead, vocals - lead


def transcribe(lead: np.ndarray, slug: str, tag: str, log: Log) -> dict[str, Any]:
    """Word-timed lyrics of the lead (local Whisper, the model glass-harbor uses). Stays in the gitignored work dir."""
    path = _cache(slug, tag) / "lyrics.json"
    if path.exists():
        return json.loads(path.read_text())
    import mlx_whisper
    import soxr
    from glass_harbor.lyrics import DEFAULT_MODEL, WHISPER_SR

    log("transcribing the lead vocal (local Whisper)")
    signal = soxr.resample(lead.mean(axis=1), dsp.SR, WHISPER_SR).astype(np.float32)
    result = mlx_whisper.transcribe(signal, path_or_hf_repo=DEFAULT_MODEL, word_timestamps=True, language="en",
                                    condition_on_previous_text=False, hallucination_silence_threshold=2.0, verbose=None)
    segments = []
    for seg in result.get("segments", []):
        words = [{"word": w["word"].strip(), "start": round(float(w["start"]), 3), "end": round(float(w["end"]), 3)}
                 for w in seg.get("words", []) if w["word"].strip()]
        if words:
            segments.append({"start": words[0]["start"], "end": words[-1]["end"], "text": seg["text"].strip(),
                             "words": words})
    data = {"model": DEFAULT_MODEL, "segments": segments}
    path.write_text(json.dumps(data, indent=1))
    return data


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9' ]+", "", text.lower()).strip()


def hook_spans(lyrics: dict[str, Any], min_words: int = 2) -> list[tuple[float, float]]:
    """Every occurrence of a lyric line that repeats (the hook and other refrains)."""
    lines = [(_normalize(s["text"]), s) for s in lyrics["segments"]]
    counts: dict[str, int] = {}
    for text, _ in lines:
        if len(text.split()) >= min_words:
            counts[text] = counts.get(text, 0) + 1
    return [(s["start"], s["end"]) for text, s in lines if counts.get(text, 0) >= 2]


def line_ends(lyrics: dict[str, Any], min_gap_s: float = 0.0, max_s: float = 0.8) -> list[tuple[float, float]]:
    """The last word of each line that is followed by at least ``min_gap_s`` of space (room for the echo to be
    heard): where a delay throw goes."""
    segs = lyrics["segments"]
    out = []
    for i, s in enumerate(segs):
        w = s["words"][-1]
        gap = segs[i + 1]["start"] - w["end"] if i + 1 < len(segs) else float("inf")
        if gap >= min_gap_s:
            out.append((w["start"], min(w["end"], w["start"] + max_s)))
    return out


# ---------------------------------------------------------------- breaths

# Calibrated on a separated lead (reverb and beat bleed included) against spectrogram-checked breaths: breaths are
# noise-like at 1-5 kHz, flat at 150-1500 Hz, and their strongest spectral peaks stand at most ~14 dB over the
# surrounding spectrum (breaths 11-14 dB; held-note and backing tails 17-22 dB; sung words 15 dB and up).
BREATH_HI_FLATNESS = 0.18
BREATH_LOW_FLATNESS = 0.14
BREATH_MAX_PEAK_DB = 14.5


def find_breaths(v: np.ndarray, words: list[tuple[float, float]], *, min_s: float = 0.12, max_s: float = 0.9,
                 hop: int = 441) -> list[tuple[float, float]]:
    """Breaths: gaps between words (>= 40 ms from any word) that sit 6-40 dB under the singing level, last
    min_s-max_s, and are noise-like rather than pitched."""
    import librosa

    m = v.mean(axis=1).astype(np.float32)
    p = np.abs(librosa.stft(m, n_fft=4096, hop_length=hop)) ** 2 + 1e-12
    f = librosa.fft_frequencies(sr=dsp.SR, n_fft=4096)

    def flatness(lo: float, hi: float) -> np.ndarray:
        band = (f >= lo) & (f <= hi)
        return np.exp(np.mean(np.log(p[band]), 0)) / np.mean(p[band], 0)

    from scipy.ndimage import median_filter

    hi_flat, lo_flat = flatness(1000, 5000), flatness(150, 1500)
    band_db = 10 * np.log10(p[(f >= 150) & (f <= 2000)])
    # how far the 5 strongest peaks stand over a ~320 Hz median of the spectrum: harmonics of a note poke out
    peaks = np.sort(band_db - median_filter(band_db, size=(31, 1)), axis=0)[-5:].mean(axis=0)
    level = 10 * np.log10(np.mean(p, 0))
    t = np.arange(p.shape[1]) * hop / dsp.SR
    ref = np.percentile(level[level > level.max() - 40], 90)
    gap = (level > ref - 40) & (level < ref - 6)
    for a, b in words:
        gap[(t >= a - 0.04) & (t <= b + 0.04)] = False
    spans, i = [], 0
    while i < t.size:
        if not gap[i]:
            i += 1
            continue
        j = i
        while j + 1 < t.size and (gap[j + 1] or (j + 2 < t.size and gap[j + 2])):  # bridge one-frame dips
            j += 1
        dur = (j - i + 1) * hop / dsp.SR
        if (min_s <= dur <= max_s and np.median(hi_flat[i:j + 1]) >= BREATH_HI_FLATNESS
                and np.median(lo_flat[i:j + 1]) >= BREATH_LOW_FLATNESS
                and np.median(peaks[i:j + 1]) <= BREATH_MAX_PEAK_DB):
            spans.append((round(float(t[i]), 3), round(float(t[j]), 3)))
        i = j + 1
    return spans


def reduce_breaths(v: np.ndarray, spans: list[tuple[float, float]], depth_db: float, fade_s: float = 0.03) -> np.ndarray:
    """Turn each breath down by depth_db with short fades inside the span (the words around it are untouched)."""
    if not spans or depth_db <= 0:
        return v
    g = np.ones(v.shape[0])
    floor = float(dsp.gain(-depth_db))
    f = max(1, int(fade_s * dsp.SR))
    for s0, s1 in spans:
        a, b = int(s0 * dsp.SR), min(v.shape[0], int(s1 * dsp.SR))
        if b - a < 2 * f:
            continue
        seg = np.full(b - a, floor)
        seg[:f] = np.linspace(1, floor, f)
        seg[-f:] = np.linspace(floor, 1, f)
        g[a:b] = np.minimum(g[a:b], seg)
    return v * g[:, None]


# ---------------------------------------------------------------- pitch


def _key_pitch_classes(key_name: str | None) -> set[int] | None:
    if not key_name:
        return None
    m = re.match(r"([A-G][#♯b♭]?)\s*(major|minor)", key_name.strip(), re.I)
    if not m:
        return None
    import librosa

    tonic = int(librosa.note_to_midi(m.group(1).replace("♯", "#").replace("♭", "b") + "4")) % 12
    steps = (0, 2, 3, 5, 7, 8, 10) if m.group(2).lower() == "minor" else (0, 2, 4, 5, 7, 9, 11)
    return {(tonic + s) % 12 for s in steps}


def pitch_track(x: np.ndarray, hop_s: float = 0.01) -> np.ndarray:
    """F0 in Hz every ``hop_s`` (0 = unvoiced). FCPE, a neural tracker built for singing (about as accurate as
    RMVPE, which scores ~97% raw pitch accuracy where pYIN scores ~75% on MIR-1K, and ~10x faster than pYIN here);
    pYIN when torchfcpe is missing."""
    import librosa

    y = librosa.resample(x.mean(axis=1).astype(np.float32), orig_sr=dsp.SR, target_sr=16000)
    n = max(1, int(len(y) / 16000 / hop_s))
    try:
        import torch
        from torchfcpe import spawn_bundled_infer_model
    except ImportError:
        f0, voiced, prob = librosa.pyin(y, fmin=70, fmax=1000, sr=16000, frame_length=1024, hop_length=int(16000 * hop_s))
        return np.where(voiced & (prob >= 0.5), np.nan_to_num(f0), 0.0)[:n]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):  # the model prints its device on load
        model = spawn_bundled_infer_model(device="cpu")
    f0 = model.infer(torch.from_numpy(y)[None, :, None], sr=16000, decoder_mode="local_argmax", threshold=0.006,
                     f0_min=70, f0_max=1000, interp_uv=False, output_interp_target_length=n)
    return f0.squeeze().numpy().astype(np.float64)


def held_notes(x: np.ndarray, hop_s: float = 0.01) -> list[dict[str, Any]]:
    """Steady notes (>= 150 ms within +-40 cents of their own median)."""
    f0 = pitch_track(x, hop_s)
    midi = np.where(f0 > 0, 69 + 12 * np.log2(np.maximum(f0, 1e-6) / 440.0), np.nan)
    notes, i, n = [], 0, len(midi)
    while i < n:
        if not np.isfinite(midi[i]):
            i += 1
            continue
        j = i
        while j + 1 < n and np.isfinite(midi[j + 1]) and abs(midi[j + 1] - np.nanmedian(midi[i:j + 1])) < 0.4:
            j += 1
        if (j - i + 1) * hop_s >= 0.15:
            seg = midi[i:j + 1]
            core = seg[len(seg) // 5: len(seg) - len(seg) // 5] if len(seg) > 10 else seg
            notes.append({"t0": round(i * hop_s, 3), "t1": round((j + 1) * hop_s, 3), "midi": round(float(np.median(core)), 3)})
        i = j + 1
    return notes


def off_notes(lead: np.ndarray, slug: str, tag: str, key_name: str | None, *, min_cents: float = 25.0,
              max_cents: float = 60.0, min_s: float = 0.25) -> list[dict[str, Any]]:
    """Held notes (>= min_s) that land min_cents-max_cents from the nearest semitone, when that semitone is in the
    song's key. Out-of-key targets and bigger misses are left alone (likely slides, blue notes, or tracking errors)."""
    path = _cache(slug, tag) / "held-notes-fcpe.json"
    if path.exists():
        notes = json.loads(path.read_text())
    else:
        notes = held_notes(lead)
        path.write_text(json.dumps(notes))
    scale = _key_pitch_classes(key_name)
    out = []
    for nte in notes:
        target = round(nte["midi"])
        cents = (nte["midi"] - target) * 100
        if nte["t1"] - nte["t0"] < min_s or not min_cents <= abs(cents) <= max_cents:
            continue
        if scale is not None and target % 12 not in scale:
            continue
        out.append({**nte, "target": target, "cents": round(cents, 1)})
    return out


def fix_notes(lead: np.ndarray, notes: list[dict[str, Any]], ramp_s: float = 0.025, pad_s: float = 0.12) -> np.ndarray:
    """Shift only the flagged notes onto pitch. Each note is re-pitched in a short window (formants kept) and
    crossfaded back, so the rest of the vocal is untouched."""
    import pedalboard

    out = lead.copy()
    for nte in notes:
        a = max(0, int((nte["t0"] - pad_s) * dsp.SR))
        b = min(lead.shape[0], int((nte["t1"] + pad_s) * dsp.SR))
        seg = lead[a:b]
        t = np.arange(b - a) / dsp.SR + a / dsp.SR
        shift = -nte["cents"] / 100.0
        ramp_in = np.clip((t - nte["t0"]) / ramp_s + 1, 0, 1)
        ramp_out = np.clip((nte["t1"] - t) / ramp_s + 1, 0, 1)
        curve = shift * np.minimum(ramp_in, ramp_out)
        moved = pedalboard.time_stretch(np.ascontiguousarray(seg.T, dtype=np.float32), dsp.SR, stretch_factor=1.0,
                                        pitch_shift_in_semitones=curve.astype(np.float64), high_quality=True,
                                        preserve_formants=True).T[: seg.shape[0]].astype(np.float64)
        moved = np.pad(moved, ((0, seg.shape[0] - moved.shape[0]), (0, 0)))
        # crossfade into the processed window over the padding, so the edges are the original audio
        fade = np.clip(np.minimum(t - (nte["t0"] - pad_s), (nte["t1"] + pad_s) - t) / (pad_s * 0.8), 0, 1)[:, None]
        out[a:b] = seg * (1 - fade) + moved * fade
    return out


# ---------------------------------------------------------------- treatment and effects


def _span_mask(n: int, spans: list[tuple[float, float]], fade_s: float = 0.05) -> np.ndarray:
    m = np.zeros(n)
    f = max(1, int(fade_s * dsp.SR))
    for s0, s1 in spans:
        a, b = max(0, int(s0 * dsp.SR) - f), min(n, int(s1 * dsp.SR) + f)
        if b <= a:
            continue
        seg = np.ones(b - a)
        k = min(f, (b - a) // 2)
        if k:
            seg[:k] = np.linspace(0, 1, k)
            seg[-k:] = np.linspace(1, 0, k)
        m[a:b] = np.maximum(m[a:b], seg)
    return m


def treat_backing(backing: np.ndarray, *, width: float, level_db: float) -> np.ndarray:
    """Backing and ad-libs: thinner, de-essed, held steady, wider than the lead, so the lead owns the centre."""
    x = dsp.eq(backing, [("hp", 160, 2), ("peak", 400, -2.0, 1.0)])
    x, _ = dsp.dynamic_band(x, 5500, 10000, max_cut_db=6.0, over_db=4.0, hop=256, smooth=5)
    x, _ = dsp.compressor(x, threshold_db=dsp.active_rms_db(x), ratio=3.0, attack_ms=8, release_ms=120)
    x = dsp.widen(x, amount=width, above_hz=300, seed=23)
    return x * dsp.gain(level_db)


def hook_doubles(lead: np.ndarray, hooks: list[tuple[float, float]]) -> np.ndarray:
    """Two detuned, slightly late, panned doubles (glass-harbor's add_layers) on the hook lines only."""
    if not hooks:
        return lead
    from glass_harbor.vocals import add_layers

    layered = add_layers(lead.astype(np.float32), doubles=True, octave=False).astype(np.float64)[: lead.shape[0]]
    m = _span_mask(lead.shape[0], hooks)[:, None]
    return lead + m * (layered - lead)


def sends(lead: np.ndarray, backing: np.ndarray, throws: list[tuple[float, float]], beat_s: float, *,
          reverb_mix: float, throw_mix: float) -> np.ndarray:
    """Plate reverb (lead light, backing heavier) and a dotted-eighth delay on line endings. Returns the wet signal."""
    import pedalboard

    def plate(x: np.ndarray) -> np.ndarray:
        board = pedalboard.Pedalboard([pedalboard.HighpassFilter(cutoff_frequency_hz=250.0),
                                       pedalboard.LowpassFilter(cutoff_frequency_hz=9000.0),
                                       pedalboard.Delay(delay_seconds=0.025, feedback=0.0, mix=1.0),
                                       pedalboard.Reverb(room_size=0.35, damping=0.6, wet_level=1.0, dry_level=0.0,
                                                         width=1.0)])
        return board(np.ascontiguousarray(x.T, dtype=np.float32), dsp.SR).T.astype(np.float64)[: x.shape[0]]

    wet = reverb_mix * plate(lead) + 2 * reverb_mix * plate(backing)
    if throws and throw_mix:
        m = _span_mask(lead.shape[0], throws, fade_s=0.02)[:, None]
        echo = pedalboard.Pedalboard([pedalboard.HighpassFilter(cutoff_frequency_hz=300.0),
                                      pedalboard.LowpassFilter(cutoff_frequency_hz=6000.0),
                                      pedalboard.Delay(delay_seconds=0.75 * beat_s, feedback=0.38, mix=1.0)])
        wet = wet + throw_mix * echo(np.ascontiguousarray((lead * m).T, dtype=np.float32), dsp.SR).T.astype(np.float64)
    return wet
