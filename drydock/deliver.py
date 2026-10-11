"""Deliverables: masters for distribution, files that play on phones, and loudness-matched comparisons."""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from drydock import dsp

PHONE_LIMIT_BYTES = 28 * 2 ** 20  # chat apps and the Claude mobile app cap attachments around 30 MiB


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise RuntimeError("ffmpeg is required (brew install ffmpeg)")
    return exe


def _encode(src: Path, dst: Path, args: list[str]) -> Path:
    subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-y", "-i", str(src), *args, str(dst)], check=True)
    return dst


def aac_encoder() -> str:
    """Apple's AudioToolbox AAC when ffmpeg has it (cleaner, fewer overs than ffmpeg's native encoder)."""
    out = subprocess.run([_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    return "aac_at" if re.search(r"\baac_at\b", out) else "aac"


def peak_safe(make, dst: Path, max_tp_db: float = -1.0) -> tuple[Path, float]:
    """Encode with ``make(volume_db)``; if the decoded file's true peak exceeds ``max_tp_db`` (lossy encoders overshoot
    on dense, bright masters), re-encode turned down by the overshoot. Returns (path, gain applied in dB)."""
    vol = 0.0
    for _ in range(3):
        make(vol)
        tp = dsp.true_peak_db(decode_file(dst))
        if tp <= max_tp_db:
            break
        vol -= tp - max_tp_db + 0.1
    return dst, round(vol, 2)


def aac_true_peak(x: np.ndarray) -> float:
    """True peak after a 256 kbps AAC round trip: a stand-in for what streaming transcodes do to the master."""
    with tempfile.TemporaryDirectory(prefix="drydock-aac-") as tmp:
        wav, m4a = Path(tmp) / "x.wav", Path(tmp) / "x.m4a"
        sf.write(wav, np.clip(x, -1.0, 1.0 - 2 ** -23), dsp.SR, subtype="PCM_24")
        _encode(wav, m4a, ["-c:a", aac_encoder(), "-b:a", "256k"])
        return dsp.true_peak_db(decode_file(m4a))


def cover(path: Path, title: str, subtitle: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5.4, 5.4), dpi=100, facecolor="#0b0b0b")
    fig.text(0.5, 0.84, title, ha="center", va="center", color="#f2f2f2", fontsize=24, weight="bold",
             family=["Helvetica", "DejaVu Sans"], wrap=True)
    if subtitle:
        fig.text(0.5, 0.76, subtitle, ha="center", va="center", color="#c3c2b7", fontsize=14, family=["Helvetica", "DejaVu Sans"])
    fig.savefig(path, facecolor="#0b0b0b")
    plt.close(fig)
    return path


def video(wav: Path, dst: Path, title: str, subtitle: str = "", volume_db: float = 0.0) -> Path:
    """Square video with a title card and a moving waveform: plays inline in Messages, WhatsApp, Instagram DMs."""
    with tempfile.TemporaryDirectory(prefix="drydock-vid-") as tmp:
        card = cover(Path(tmp) / "cover.png", title, subtitle)
        for crf in (33, 38, 44):
            subprocess.run([_ffmpeg(), "-v", "error", "-nostdin", "-y", "-loop", "1", "-framerate", "24", "-i", str(card),
                            "-i", str(wav), "-filter_complex",
                            "[1:a]showwaves=s=540x220:mode=cline:rate=24:colors=0xF2F2F2:scale=sqrt,colorkey=black:0.1:0.0[w];"
                            "[0:v][w]overlay=0:195:shortest=1,format=yuv420p[v];"
                            f"[1:a]volume={volume_db}dB[a]",
                            "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                            "-c:a", aac_encoder(), "-b:a", "256k", "-movflags", "+faststart", "-shortest", str(dst)], check=True)
            if dst.stat().st_size <= PHONE_LIMIT_BYTES:
                break
    return dst


def at_lufs(x: np.ndarray, target: float = -14.0) -> np.ndarray:
    return np.clip(x * dsp.gain(target - dsp.lufs(x)), -1.0, 1.0 - 2 ** -23)


def write_all(out: Path, title: str, source: np.ndarray, master: np.ndarray) -> dict[str, str]:
    base = f"{title} - master"
    files: dict[str, Path] = {}
    files["wav"] = out / f"{base}.wav"
    sf.write(files["wav"], np.clip(master, -1.0, 1.0 - 2 ** -23), dsp.SR, subtype="PCM_24")
    files["flac"] = out / f"{base}.flac"
    sf.write(files["flac"], np.clip(master, -1.0, 1.0 - 2 ** -23), dsp.SR, format="FLAC", subtype="PCM_24")
    wav, aac = files["wav"], aac_encoder()
    trims: dict[str, float] = {}
    files["mp3"], trims["mp3"] = peak_safe(lambda v: _encode(wav, out / f"{base}.mp3", ["-af", f"volume={v}dB", "-c:a",
                                           "libmp3lame", "-b:a", "320k"]), out / f"{base}.mp3")
    files["m4a"], trims["m4a"] = peak_safe(lambda v: _encode(wav, out / f"{base}.m4a", ["-af", f"volume={v}dB", "-c:a", aac,
                                           "-b:a", "256k", "-movflags", "+faststart"]), out / f"{base}.m4a")
    files["mp4"], trims["mp4"] = peak_safe(lambda v: video(wav, out / f"{base}.mp4", title, "master", v), out / f"{base}.mp4")
    cmp_dir = out / "compare-at-14-lufs"
    cmp_dir.mkdir(exist_ok=True)
    files["compare_source"] = cmp_dir / f"1 {title} - original (-14 LUFS).flac"
    sf.write(files["compare_source"], at_lufs(source), dsp.SR, format="FLAC", subtype="PCM_24")
    files["compare_master"] = cmp_dir / f"2 {title} - master (-14 LUFS).flac"
    sf.write(files["compare_master"], at_lufs(master), dsp.SR, format="FLAC", subtype="PCM_24")
    out_files = {k: str(v) for k, v in files.items()}
    out_files["_share_trim_db"] = trims  # type: ignore[assignment]
    return out_files


def decode_file(path: Path) -> np.ndarray:
    from drydock.stems import decode

    return decode(path)
