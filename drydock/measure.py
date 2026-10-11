"""Measurements shared by the decisions and the QA report."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import welch

from drydock import dsp

TONE_BANDS = {"sub <45": (20, 45), "bass 45-90": (45, 90), "low-mid 150-400": (150, 400),
              "mid 400-1.5k": (400, 1500), "presence 1.5-4k": (1500, 4000), "6-9k": (6000, 9000),
              "air 10-16k": (10000, 16000)}


def tone(x: np.ndarray) -> dict[str, float]:
    """Share of the spectrum in each band, dB relative to the 20 Hz+ total."""
    f, p = welch(x.mean(axis=1), dsp.SR, nperseg=16384)
    total = p[f >= 20].sum() + 1e-30
    return {k: round(float(10 * np.log10(p[(f >= lo) & (f < hi)].sum() / total + 1e-30)), 1)
            for k, (lo, hi) in TONE_BANDS.items()}


def third_octaves(x: np.ndarray, *, loud_only: bool = True) -> dict[int, float]:
    """1/3-octave levels relative to 1 kHz, over the loud sections (short-term loudness within 6 LU of the max)."""
    sig = x
    if loud_only and x.shape[0] > 4 * dsp.SR:
        win = 3 * dsp.SR
        segs = [(s, dsp.lufs(x[s:s + win])) for s in range(0, x.shape[0] - win, dsp.SR)]
        top = max(v for _, v in segs)
        keep = [x[s:s + dsp.SR] for s, v in segs if v > top - 6]
        sig = np.concatenate(keep) if keep else x
    f, p = welch(sig.mean(axis=1), dsp.SR, nperseg=16384)
    out = {}
    for c in 1000 * 2 ** (np.arange(-16, 13) / 3):
        m = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        out[int(round(c))] = 10 * np.log10(p[m].sum() + 1e-30)
    ref = out[1000]
    return {k: round(float(v - ref), 1) for k, v in out.items()}


def stereo(x: np.ndarray) -> dict[str, float]:
    m, s = (x[:, 0] + x[:, 1]) / 2, (x[:, 0] - x[:, 1]) / 2
    hm, hs = dsp.bandpass(m, 300, 16000), dsp.bandpass(s, 300, 16000)
    corr = float(np.corrcoef(x[:, 0], x[:, 1])[0, 1]) if np.std(x[:, 0]) and np.std(x[:, 1]) else 1.0
    return {"correlation": round(corr, 3),
            "side_vs_mid_above_300_db": round(float(10 * np.log10(np.mean(hs ** 2) / (np.mean(hm ** 2) + 1e-30) + 1e-30)), 1)}


def levels(x: np.ndarray) -> dict[str, Any]:
    lu = dsp.lufs(x)
    tp = dsp.true_peak_db(x)
    st = np.array([dsp.lufs(x[s:s + 3 * dsp.SR]) for s in range(0, max(1, x.shape[0] - 3 * dsp.SR), dsp.SR)])
    gated = st[st > lu - 20] if (st > lu - 20).any() else st
    return {"lufs": round(lu, 2), "true_peak_dbtp": round(tp, 2), "plr_db": round(tp - lu, 2),
            "lra_lu": round(float(np.percentile(gated, 95) - np.percentile(gated, 10)), 2),
            "short_term_max": round(float(st.max()), 2),
            "samples_over_full_scale": int((np.abs(x) > 1.0).sum())}


def content_cutoff_hz(x: np.ndarray) -> int:
    """Highest frequency with real content (a lossy encoder's low-pass shows up as a cliff)."""
    f, p = welch(x.mean(axis=1), dsp.SR, nperseg=8192)
    pdb = 10 * np.log10(p + 1e-30)
    ref = np.median(pdb[(f > 2000) & (f < 8000)])
    above = f[(pdb > ref - 35) & (f > 8000)]
    return int(above.max()) if above.size else 8000


def bass_fundamental_hz(bass: np.ndarray) -> float:
    f, p = welch(bass.mean(axis=1), dsp.SR, nperseg=32768)
    sel = (f >= 20) & (f <= 120)
    if not sel.any() or p[sel].sum() <= 0:
        return 0.0
    # energy-weighted median frequency of the bass line
    c = np.cumsum(p[sel])
    return float(f[sel][np.searchsorted(c, c[-1] / 2)])


# ---------------------------------------------------------------- song structure


def bar_bounds(grid, bar: int, frames: int) -> tuple[int, int]:
    a = max(0, int(grid.bar_time(bar) * dsp.SR))
    b = min(frames, int(grid.bar_time(bar + 1) * dsp.SR))
    return a, b


def section_spans(grid, sections: list[dict], frames: int) -> list[tuple[int, int, str]]:
    spans = []
    for i, s in enumerate(sections):
        a = 0 if i == 0 else int(grid.bar_time(s["start_bar"]) * dsp.SR)
        b = frames if i == len(sections) - 1 else int(grid.bar_time(s["start_bar"] + s["bars"]) * dsp.SR)
        spans.append((max(0, a), min(frames, b), f"bars {s['start_bar']}-{s['start_bar'] + s['bars'] - 1}"))
    return spans


def section_loudness(x: np.ndarray, spans: list[tuple[int, int, str]]) -> list[float]:
    return [round(dsp.lufs(x[a:b]), 2) for a, b, _ in spans]


def vocal_audibility(vocal: np.ndarray, rest: np.ndarray, grid, *, window: float = 0.4) -> dict[int, float]:
    """Per bar: median of (vocal over everything else, 1-4 kHz) across the windows where the vocal is singing."""
    vb, rb = dsp.bandpass(vocal.mean(axis=1), 1000, 4000), dsp.bandpass(rest.mean(axis=1), 1000, 4000)
    vm = vocal.mean(axis=1)
    size, hop = int(window * dsp.SR), int(0.1 * dsp.SR)

    def e(sig, a, b):
        return 10 * np.log10(np.mean(sig[a:b] ** 2) + 1e-15)

    starts = range(0, max(1, vm.size - size), hop * 5)
    gate = max(-60.0, max(e(vm, s, s + size) for s in starts) - 18)
    out = {}
    for bar in range(1, grid.bar_count + 1):
        a, b = bar_bounds(grid, bar, vm.size)
        act = [s for s in range(a, b - size, hop) if e(vm, s, s + size) > gate]
        if act:
            out[bar] = round(float(np.median([e(vb, s, s + size) - e(rb, s, s + size) for s in act])), 1)
    return out
