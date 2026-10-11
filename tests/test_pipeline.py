"""End-to-end runs on a synthetic song (no ML models: the beat analysis and separation are stubbed)."""

import json
import shutil

import numpy as np
import pytest
import soundfile as sf

from drydock import deliver, dsp, engineer, master, stems

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
SR = dsp.SR
BPM, BARS = 120, 10  # 2 s bars, 20 s song


def _song(seed=0, vocal=True):
    rng = np.random.default_rng(seed)
    n = BARS * 2 * SR
    t = np.arange(n) / SR
    kick = np.zeros(n)
    for beat in np.arange(0, 20, 0.5):
        i = int(beat * SR)
        k = np.arange(int(0.15 * SR))
        kick[i:i + k.size] += 0.8 * np.sin(2 * np.pi * 55 * k / SR) * np.exp(-k / (0.04 * SR))
    sub = 0.5 * np.sin(2 * np.pi * 36 * t)
    hats = dsp.bandpass(rng.standard_normal(n) * 0.05, 6000, 12000)
    keys = 0.08 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 660 * t)
    voc = (0.12 * np.sin(2 * np.pi * 220 * t) * (1 + np.sin(2 * np.pi * 1.5 * t)) / 2
           + dsp.bandpass(rng.standard_normal(n) * 0.04, 1000, 4000)) if vocal else np.zeros(n)
    intro = np.where(t < 4.0, 0.25, 1.0)  # bars 1-2 are a quiet intro
    parts = {"drums": (kick + hats) * intro, "bass": sub * intro, "other": keys, "vocals": voc * (t >= 4.0)}
    st = {k: np.stack([v, v], axis=1) for k, v in parts.items()}
    st["other"][:, 1] *= 0.9
    return sum(st.values()), st


def _analysis():
    from glass_harbor.grid import Grid

    grid = Grid(beats=list(np.arange(0, 20.01, 0.5)), beats_per_bar=4, first_downbeat=0, mode="fixed", bpm=BPM,
                duration=20.0)
    return {"bpm": BPM, "key": {"name": "A minor"}, "grid": grid.to_dict(),
            "sections": [{"start_bar": 1, "bars": 2, "energy": "low"}, {"start_bar": 3, "bars": 8, "energy": "high"}]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DRYDOCK_HOME", str(tmp_path))
    monkeypatch.setattr(stems, "analyze", lambda x, slug, log: _analysis())
    return tmp_path


def _run(home, monkeypatch, mode, vocal=True):
    import soxr

    mix, st = _song(vocal=vocal)
    src = home / "song.wav"
    # 48 kHz / 24-bit input exercises the resampling decode
    sf.write(src, np.clip(soxr.resample(mix * 0.9, SR, 48000), -1, 1), 48000, subtype="PCM_24")
    if mode == "stems":
        monkeypatch.setattr(stems, "separate", lambda x, slug, log: {k: v[: x.shape[0]] * 0.9 for k, v in st.items()})
    out = master.run(src, "synth", engineer.Settings(mode=mode), log=lambda m: None)
    return out, json.loads((out / "qa.json").read_text())


@pytest.mark.parametrize("mode", ["2track", "stems"])
def test_pipeline_writes_everything_and_passes_hard_checks(home, mode, monkeypatch):
    out, qa = _run(home, monkeypatch, mode)
    names = qa["files"]
    for key in ("wav", "flac", "mp3", "m4a", "mp4", "compare_source", "compare_master"):
        assert (out / names[key]).exists() or (out / "compare-at-14-lufs" / names[key]).exists(), key
    status = {c["check"]: c["status"] for c in qa["checks"]}
    assert status["true peak"] == "pass" and status["clipping"] == "pass"
    assert status["peaks after MP3/AAC encoding"] == "pass"
    assert status["phone share file size"] == "pass"
    assert (out / names["mp4"]).stat().st_size <= deliver.PHONE_LIMIT_BYTES
    assert qa["source"]["sample_rate"] == 48000
    assert abs(qa["sections"]["balance_shift_lu"][0]) <= 1.5  # intro contrast kept
    if mode == "stems":
        assert qa["processing"]["stems"]["has_vocal"] is True
        assert qa["vocal"] is not None


def test_instrumental_skips_vocal_moves(home, monkeypatch):
    _, qa = _run(home, monkeypatch, "stems", vocal=False)
    assert qa["processing"]["stems"]["has_vocal"] is False
    assert qa["vocal"] is None
    assert not any(k.startswith("vocal") or k.startswith("duck") for k in qa["decisions"]["why"])


def test_short_and_silent_inputs_are_rejected(home):
    short = home / "short.wav"
    sf.write(short, np.zeros((3 * SR, 2)) + 0.1, SR)
    with pytest.raises(ValueError, match="at least"):
        master.run(short, "short", engineer.Settings(mode="2track"), log=lambda m: None)
    silent = home / "silent.wav"
    sf.write(silent, np.zeros((15 * SR, 2)), SR)
    with pytest.raises(ValueError, match="silent"):
        master.run(silent, "silent", engineer.Settings(mode="2track"), log=lambda m: None)


def test_settings_validation():
    with pytest.raises(ValueError):
        engineer.Settings(mode="loud").validate()
    with pytest.raises(ValueError):
        engineer.Settings(target_lufs=-3).validate()
    engineer.Settings(target_lufs=-9, width=0.3).validate()
