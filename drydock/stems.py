"""Decode, analyze (glass-harbor's beat grid and sections), and split a song into stems. Everything is cached
under work/<slug>/ keyed by the source file's hash."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

from drydock import dsp, paths

Log = Callable[[str], None]
STEMS = ("vocals", "drums", "bass", "other")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def decode(path: Path) -> np.ndarray:
    """Float decode at 44.1 kHz stereo that keeps samples above full scale (bounces often peak over 0 dBFS)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required (brew install ffmpeg)")
    probe = subprocess.run([ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=sample_rate", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    sr_in = int(probe.stdout.strip() or dsp.SR)
    raw = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
                          "-ac", "2", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).astype(np.float64)
    if sr_in != dsp.SR:
        import soxr

        x = soxr.resample(x, sr_in, dsp.SR, quality="VHQ")
    return x


def load_source(src: Path, slug: str, log: Log) -> tuple[np.ndarray, dict[str, Any]]:
    work = paths.work_dir(slug)
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "source.json"
    digest = sha256(src)
    cached = work / "source.wav"
    if meta_path.exists() and cached.exists() and json.loads(meta_path.read_text()).get("sha256") == digest:
        x, _ = sf.read(cached, dtype="float64", always_2d=True)
        return x, json.loads(meta_path.read_text())
    for stale in ("analysis.json", "stems"):
        p = work / stale
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    log(f"decoding {src.name}")
    x = decode(src)
    sf.write(cached, x.astype(np.float32), dsp.SR, subtype="FLOAT")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_name,sample_rate,bit_rate", "-of", "json", str(src)], capture_output=True, text=True)
    stream = (json.loads(probe.stdout or "{}").get("streams") or [{}])[0]
    meta = {"file": src.name, "sha256": digest, "codec": stream.get("codec_name"),
            "sample_rate": int(stream.get("sample_rate", 0) or 0),
            "bit_rate": int(stream.get("bit_rate", 0) or 0) or None, "seconds": round(x.shape[0] / dsp.SR, 2)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return x, meta


def analyze(x: np.ndarray, slug: str, log: Log) -> dict[str, Any]:
    path = paths.work_dir(slug) / "analysis.json"
    if path.exists():
        return json.loads(path.read_text())
    from glass_harbor.analyze import analyze_track

    log("analyzing beat grid, key, sections (glass-harbor)")
    result = analyze_track(np.clip(x, -1, 1).astype(np.float32))
    path.write_text(json.dumps(result))
    return result


def separate(x: np.ndarray, slug: str, log: Log) -> dict[str, np.ndarray]:
    """BS-RoFormer vocals, then HT-Demucs drums/bass/other from the instrumental (glass-harbor's balanced profile).
    Runs on a -6 dB copy so peaks above full scale survive the models' integer output."""
    out_dir = paths.work_dir(slug) / "stems"
    if all((out_dir / f"{n}.wav").exists() for n in STEMS):
        return {n: sf.read(out_dir / f"{n}.wav", dtype="float64", always_2d=True)[0] for n in STEMS}
    from glass_harbor.paths import model_dir
    from glass_harbor.separate import PROFILES

    import os

    os.environ.setdefault("TQDM_DISABLE", "1")  # the models' progress bars flood agent logs
    profile = PROFILES["balanced"]
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = 6.0
    with tempfile.TemporaryDirectory(prefix="drydock-sep-") as tmp, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from audio_separator.separator import Separator

        tmp_dir = Path(tmp)
        src = tmp_dir / "mix.wav"
        sf.write(src, (x * dsp.gain(-pad)).astype(np.float32), dsp.SR, subtype="FLOAT")

        def run(model: str, inp: Path, name: str) -> dict[str, Path]:
            sep = Separator(log_level=30, model_file_dir=str(model_dir()), output_dir=str(tmp_dir / name),
                            output_format="WAV")
            sep.load_model(model_filename=model)
            found = {}
            for o in sep.separate(str(inp)):
                p = Path(o) if Path(o).is_absolute() else tmp_dir / name / o
                labels = re.findall(r"\(([^)]+)\)", p.name)  # the last label is this pass's; earlier ones are the input's
                if labels:
                    found.setdefault(labels[-1].lower(), p)
            return found

        log(f"separating vocals ({profile['vocals']['model']})")
        two = run(profile["vocals"]["model"], src, "vocals")
        inst = two.get("instrumental") or two["other"]
        log(f"separating drums/bass/other ({profile['four_stem']['model']})")
        four = run(profile["four_stem"]["model"], inst, "four")

        def read(p: Path) -> np.ndarray:
            y, _ = sf.read(p, dtype="float64", always_2d=True)
            y = y[: x.shape[0]]
            return np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0))) * dsp.gain(pad)

        stems = {"vocals": read(two["vocals"]) + (read(four["vocals"]) if "vocals" in four else 0.0),
                 "drums": read(four["drums"]), "bass": read(four["bass"]), "other": read(four["other"])}
    for n, y in stems.items():
        sf.write(out_dir / f"{n}.wav", y.astype(np.float32), dsp.SR, subtype="FLOAT")
    return stems
