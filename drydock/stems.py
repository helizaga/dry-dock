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
    """Float decode at 44.1 kHz stereo that keeps samples above full scale (bounces often peak over 0 dBFS).
    Mono files are duplicated to both channels at full level (ffmpeg's own upmix would apply a -3 dB pan law)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required (brew install ffmpeg)")
    probe = subprocess.run([ffmpeg.replace("ffmpeg", "ffprobe"), "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=sample_rate,channels", "-of", "json", str(path)], capture_output=True, text=True)
    stream = (json.loads(probe.stdout or "{}").get("streams") or [{}])[0]
    sr_in = int(stream.get("sample_rate") or dsp.SR)
    mono = int(stream.get("channels") or 2) == 1
    raw = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(path), "-f", "f32le", "-acodec", "pcm_f32le",
                          "-ac", "1" if mono else "2", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    x = np.repeat(x[:, None], 2, axis=1) if mono else x.reshape(-1, 2)
    if sr_in != dsp.SR:
        import soxr

        x = soxr.resample(x, sr_in, dsp.SR, quality="VHQ")
    return x


def clear_work(work: Path) -> None:
    """A new source invalidates everything cached for the song (analysis, stems, vocal split, lyrics, notes)."""
    for p in work.iterdir():
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def load_cached(slug: str) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray] | None]:
    """The song as decoded on an earlier run (the original file may be gone: chat attachments get cleared).
    Returns (source, meta, session stems or None)."""
    work = paths.work_dir(slug)
    if not ((work / "source.wav").exists() and (work / "source.json").exists()):
        raise FileNotFoundError(f"nothing cached for {slug}; run drydock master with the file once")
    meta = json.loads((work / "source.json").read_text())
    source = sf.read(work / "source.wav", dtype="float64", always_2d=True)[0]
    user = None
    if meta.get("user_stems"):
        user = {r: sf.read(work / "stems" / f"{r}.wav", dtype="float64", always_2d=True)[0] for r in ROLES}
    return source, meta, user


def load_source(src: Path, slug: str, log: Log) -> tuple[np.ndarray, dict[str, Any]]:
    work = paths.work_dir(slug)
    work.mkdir(parents=True, exist_ok=True)
    meta_path = work / "source.json"
    digest = sha256(src)
    cached = work / "source.wav"
    if meta_path.exists() and cached.exists() and json.loads(meta_path.read_text()).get("sha256") == digest:
        x, _ = sf.read(cached, dtype="float64", always_2d=True)
        return x, json.loads(meta_path.read_text())
    clear_work(work)
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


def run_separator(spec: dict[str, str], inp: Path, out: Path) -> dict[str, Path]:
    """One audio-separator pass (a single model or a glass-harbor ensemble preset). Outputs keyed by the last
    label in each file name: later passes' names also carry the labels of their input."""
    import os

    os.environ.setdefault("TQDM_DISABLE", "1")  # the models' progress bars flood agent logs
    from glass_harbor.paths import model_dir

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from audio_separator.separator import Separator

        preset = spec.get("ensemble_preset")
        sep = Separator(log_level=30, model_file_dir=str(model_dir()), output_dir=str(out), output_format="WAV",
                        **({"ensemble_preset": preset} if preset else {}))
        sep.load_model() if preset else sep.load_model(model_filename=spec["model"])
        found: dict[str, Path] = {}
        for o in sep.separate(str(inp)):
            p = Path(o) if Path(o).is_absolute() else out / o
            labels = re.findall(r"\(([^)]+)\)", p.name)
            if labels:
                found.setdefault(labels[-1].lower(), p)
        return found


def separate(x: np.ndarray, slug: str, log: Log, quality: str = "best") -> dict[str, np.ndarray]:
    """Vocals (BS-RoFormer, or glass-harbor's "best" vocal ensemble), then HT-Demucs drums/bass/other from the
    instrumental. Runs on a -6 dB copy so peaks above full scale survive the models' integer output."""
    out_dir = paths.work_dir(slug) / ("stems" if quality == "balanced" else f"stems-{quality}")
    if all((out_dir / f"{n}.wav").exists() for n in STEMS):
        return {n: sf.read(out_dir / f"{n}.wav", dtype="float64", always_2d=True)[0] for n in STEMS}
    from glass_harbor.separate import PROFILES

    profile = PROFILES[quality]
    out_dir.mkdir(parents=True, exist_ok=True)
    pad = 6.0
    with tempfile.TemporaryDirectory(prefix="drydock-sep-") as tmp:
        tmp_dir = Path(tmp)
        src = tmp_dir / "mix.wav"
        sf.write(src, (x * dsp.gain(-pad)).astype(np.float32), dsp.SR, subtype="FLOAT")
        log(f"separating vocals ({quality}: {profile['vocals']})")
        two = run_separator(profile["vocals"], src, tmp_dir / "vocals")
        inst = two.get("instrumental") or two.get("other") or next(p for k, p in two.items() if k != "vocals")
        log(f"separating drums/bass/other ({profile['four_stem']['model']})")
        four = run_separator(profile["four_stem"], inst, tmp_dir / "four")

        def read(p: Path) -> np.ndarray:
            y, _ = sf.read(p, dtype="float64", always_2d=True)
            y = y[: x.shape[0]]
            return np.pad(y, ((0, x.shape[0] - y.shape[0]), (0, 0))) * dsp.gain(pad)

        stems = {"vocals": read(two["vocals"]) + (read(four["vocals"]) if "vocals" in four else 0.0),
                 "drums": read(four["drums"]), "bass": read(four["bass"]), "other": read(four["other"])}
    for n, y in stems.items():
        sf.write(out_dir / f"{n}.wav", y.astype(np.float32), dsp.SR, subtype="FLOAT")
    return stems


# ---------------------------------------------------------------- user-provided stems

AUDIO_EXTS = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".m4a", ".caf", ".ogg"}
ROLE_WORDS = (  # checked in order; the first role with a matching word wins. "x*" matches a prefix, otherwise
    # whole words (plural "s" allowed). "lead" and "break" are left out: "Lead Synth" and "Break FX" are not vocals/drums.
    ("drums", ("bass drum", "bassdrum", "kick drum")),
    ("vocals", ("vox", "vocal", "voc", "adlib", "ad lib", "bgv", "backing", "harmon*", "double", "dbl", "hook", "verse",
                "rap", "singer", "acap*", "acapella")),
    ("bass", ("808", "bass", "sub", "subbass")),
    ("drums", ("drum", "kick", "snare", "clap", "hat", "hihat", "hh", "perc*", "cymbal", "crash", "ride", "tom", "rim",
               "shaker", "snap", "loop drum")),
)
ROLES = ("vocals", "drums", "bass", "other")


def guess_role(name: str) -> str:
    """Role from a stem's file name: "808 Final.wav" -> bass, "Adlibs L.wav" -> vocals, anything unknown -> other."""
    low = re.sub(r"[_\-.]+", " ", Path(name).stem.lower())
    for role, words in ROLE_WORDS:
        for w in words:
            pattern = rf"(?<![a-z]){re.escape(w[:-1])}" if w.endswith("*") else rf"(?<![a-z]){re.escape(w)}s?(?![a-z])"
            if re.search(pattern, low):
                return role
    return "other"


def stem_files(folder: Path) -> list[Path]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))
    if not files:
        raise ValueError(f"no audio files in {folder} (looked for {', '.join(sorted(AUDIO_EXTS))})")
    return files


def load_user_stems(folder: Path, slug: str, log: Log, roles: dict[str, str] | None = None
                    ) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Stems exported from the session (no separation). Each file is assigned a role by name (``roles`` overrides
    by file name), files sharing a role are summed, and the song is their sum. Cached like a decoded bounce."""
    roles = roles or {}
    files = stem_files(folder)
    unknown = set(roles) - {f.name for f in files}
    if unknown:
        raise ValueError(f"--role names files that are not in {folder}: {', '.join(sorted(unknown))}")
    bad = {r for r in roles.values() if r not in ROLES}
    if bad:
        raise ValueError(f"roles must be one of {', '.join(ROLES)}; got {', '.join(sorted(bad))}")
    assignment = {f.name: roles.get(f.name, guess_role(f.name)) for f in files}
    digest = hashlib.sha256(json.dumps([[f.name, sha256(f), assignment[f.name]] for f in files]).encode()).hexdigest()
    work = paths.work_dir(slug)
    work.mkdir(parents=True, exist_ok=True)
    meta_path, stem_dir = work / "source.json", work / "stems"
    if (meta_path.exists() and json.loads(meta_path.read_text()).get("sha256") == digest
            and all((stem_dir / f"{r}.wav").exists() for r in ROLES)):
        source = sf.read(work / "source.wav", dtype="float64", always_2d=True)[0]
        stems = {r: sf.read(stem_dir / f"{r}.wav", dtype="float64", always_2d=True)[0] for r in ROLES}
        return source, json.loads(meta_path.read_text()), stems
    clear_work(work)
    decoded, info = {}, []
    for f in files:
        log(f"  {f.name} -> {assignment[f.name]}")
        decoded[f.name] = decode(f)
        stream = _probe(f)
        info.append({"file": f.name, "role": assignment[f.name], "codec": stream.get("codec_name"),
                     "sample_rate": int(stream.get("sample_rate", 0) or 0),
                     "seconds": round(decoded[f.name].shape[0] / dsp.SR, 2)})
    n = max(y.shape[0] for y in decoded.values())
    if n - min(y.shape[0] for y in decoded.values()) > dsp.SR:
        log("  warning: stem lengths differ by more than a second; export every stem from the session start")
    stems = {r: np.zeros((n, 2)) for r in ROLES}
    for name, y in decoded.items():
        stems[assignment[name]][: y.shape[0]] += y
    source = sum(stems.values())
    stem_dir.mkdir(parents=True, exist_ok=True)
    for r, y in stems.items():
        sf.write(stem_dir / f"{r}.wav", y.astype(np.float32), dsp.SR, subtype="FLOAT")
    sf.write(work / "source.wav", source.astype(np.float32), dsp.SR, subtype="FLOAT")
    meta = {"file": folder.name, "sha256": digest, "codec": "stems", "user_stems": True, "stems": info,
            "sample_rate": max(i["sample_rate"] for i in info), "bit_rate": None, "seconds": round(n / dsp.SR, 2)}
    meta_path.write_text(json.dumps(meta, indent=2))
    return source, meta, stems


def _probe(path: Path) -> dict[str, Any]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                            "stream=codec_name,sample_rate,bit_rate", "-of", "json", str(path)], capture_output=True, text=True)
    return (json.loads(probe.stdout or "{}").get("streams") or [{}])[0]
