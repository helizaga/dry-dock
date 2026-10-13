"""Reference matching: measure a released track the listener wants to sound like (tonal balance, vocal over the
beat, loudness, width) and steer the master toward it. Tone is matched with a smooth, bounded EQ; the vocal level
is set to land near the reference's balance. Loudness stays with the limiter budget (louder is not better)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from drydock import dsp, measure, paths

Log = Callable[[str], None]
MAX_EQ_DB = 3.0  # the most the match EQ moves any band


class _Windows:
    """A stand-in beat grid of fixed 2 s windows (vocal_audibility only needs bar_count and bar_time)."""

    def __init__(self, seconds: float, size: float = 2.0):
        self.size = size
        self.bar_count = int(seconds // size)

    def bar_time(self, bar: float) -> float:
        return (bar - 1) * self.size


GENERIC_NAMES = {"audio", "mix", "master", "track", "song", "final", "bounce", "output"}


def display_name(path: Path) -> str:
    """The reference's name for labels: the file name, or its folder's when the file name says nothing."""
    return path.parent.name if path.stem.lower() in GENERIC_NAMES else path.stem


def analyze(path: Path, log: Log) -> dict[str, Any]:
    """Cached per file hash under work/_references/."""
    from drydock import stems

    digest = stems.sha256(path)
    cache = paths.root() / "work" / "_references" / digest[:16]
    if (cache / "reference.json").exists():
        info = json.loads((cache / "reference.json").read_text())
        info["curve"] = {int(k): v for k, v in info["curve"].items()}  # JSON keys come back as text
        info["file"] = display_name(path)
        return info
    cache.mkdir(parents=True, exist_ok=True)
    log(f"measuring the reference {path.name}")
    x = stems.decode(path)
    info: dict[str, Any] = {"file": display_name(path), "lufs": round(dsp.lufs(x), 2), "curve": measure.third_octaves(x),
                            "stereo": measure.stereo(x), "tone": measure.tone(x), "vocal_median_db": None}
    vocals = _reference_vocals(x, cache, log)
    if vocals is not None:
        a = measure.vocal_audibility(vocals, x - vocals, _Windows(x.shape[0] / dsp.SR))
        if a:
            info["vocal_median_db"] = round(float(np.median(list(a.values()))), 1)
    (cache / "reference.json").write_text(json.dumps(info, indent=1))
    return info


def _reference_vocals(x: np.ndarray, cache: Path, log: Log) -> np.ndarray | None:
    """The reference's vocal (one fast RoFormer pass), or None for an instrumental."""
    import tempfile

    import soundfile as sf
    from glass_harbor.separate import PROFILES

    from drydock import engineer, stems

    log("separating the reference's vocal")
    with tempfile.TemporaryDirectory(prefix="drydock-ref-") as tmp:
        src = Path(tmp) / "ref.wav"
        sf.write(src, (x * dsp.gain(-6)).astype(np.float32), dsp.SR, subtype="FLOAT")
        out = stems.run_separator(PROFILES["balanced"]["vocals"], src, Path(tmp) / "out")
        v, _ = sf.read(out["vocals"], dtype="float64", always_2d=True)
    v = np.pad(v[: x.shape[0]], ((0, max(0, x.shape[0] - v.shape[0])), (0, 0))) * dsp.gain(6)
    return v if engineer.has_vocal(v, x) else None


def match_curve(own: dict[int, float], ref: dict[int, float], *, cutoff_hz: float) -> dict[int, float]:
    """Per 1/3 octave: how much to move ``own`` toward ``ref`` (both relative to 1 kHz), smoothed over about an
    octave and bounded to +-MAX_EQ_DB. No boosts above the source's content cutoff (nothing there to raise but
    noise) or below 30 Hz (sub boosts only eat headroom)."""
    bands = sorted(set(own) & set(ref))
    raw = np.array([ref[b] - own[b] for b in bands])
    # compare shape, not level: both curves are relative to one 1/3-octave band (1 kHz), so a bump in that one
    # band would otherwise read as "everything else is off". Remove the broadband offset first.
    mid = np.array([100 <= b <= 10000 for b in bands])
    raw = raw - np.median(raw[mid])
    k = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    smooth = np.convolve(np.pad(raw, 2, mode="edge"), k / k.sum(), mode="valid")
    out = {}
    for b, d in zip(bands, smooth):
        d = float(np.clip(d, -MAX_EQ_DB, MAX_EQ_DB))
        if d > 0 and (b > cutoff_hz * 0.9 or b < 30):
            d = 0.0
        out[b] = round(d, 2)
    return out


def apply_curve(x: np.ndarray, curve: dict[int, float], numtaps: int = 8191) -> np.ndarray:
    """Linear-phase FIR through the curve's band gains (no phase shift, delay compensated)."""
    from scipy.signal import fftconvolve, firwin2

    bands = sorted(curve)
    freqs = [0.0] + [float(b) for b in bands] + [dsp.SR / 2]
    gains = [dsp.gain(curve[bands[0]])] + [float(dsp.gain(curve[b])) for b in bands] + [float(dsp.gain(curve[bands[-1]]))]
    h = firwin2(numtaps, freqs, gains, fs=dsp.SR)
    delay = (numtaps - 1) // 2
    return np.stack([fftconvolve(x[:, c], h)[delay:delay + x.shape[0]] for c in range(x.shape[1])], axis=1)


def tone_distance_db(a: dict[int, float], b: dict[int, float], lo: int = 40, hi: int = 12000) -> float:
    """RMS difference (dB) between two 1/3-octave curves over lo-hi, after removing their median offset (shape)."""
    bands = [k for k in a if k in b and lo <= k <= hi]
    d = np.array([a[k] - b[k] for k in bands])
    return round(float(np.sqrt(np.mean((d - np.median(d)) ** 2))), 2)
