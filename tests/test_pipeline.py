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


def _fake_lyrics():
    w = lambda word, a, b: {"word": word, "start": a, "end": b}
    line = lambda a: {"start": a, "end": a + 1.0, "text": "need you now",
                      "words": [w("need", a, a + 0.3), w("you", a + 0.3, a + 0.6), w("now", a + 0.6, a + 1.0)]}
    return {"segments": [line(5.0), line(9.0), line(13.0)]}


@pytest.fixture
def home(tmp_path, monkeypatch):
    from drydock import vocal

    monkeypatch.setenv("DRYDOCK_HOME", str(tmp_path))
    monkeypatch.setattr(stems, "analyze", lambda x, slug, log: _analysis())
    # no ML in tests: a fixed lead/backing split, canned lyrics, one flagged note
    monkeypatch.setattr(vocal, "split_lead", lambda v, slug, tag, log: (v * 0.8, v * 0.2))
    monkeypatch.setattr(vocal, "transcribe", lambda lead, slug, tag, log: _fake_lyrics())
    monkeypatch.setattr(vocal, "off_notes", lambda lead, slug, tag, key, **k: [
        {"t0": 6.0, "t1": 6.5, "midi": 57.3, "target": 57, "cents": 30.0}])
    return tmp_path


def _run(home, monkeypatch, mode, vocal=True):
    import soxr

    mix, st = _song(vocal=vocal)
    src = home / "song.wav"
    # 48 kHz / 24-bit input exercises the resampling decode
    sf.write(src, np.clip(soxr.resample(mix * 0.9, SR, 48000), -1, 1), 48000, subtype="PCM_24")
    if mode == "stems":
        monkeypatch.setattr(stems, "separate", lambda x, slug, log, **k: {n: v[: x.shape[0]] * 0.9 for n, v in st.items()})
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


# ---------------------------------------------------------------- session stems


def _stem_folder(home, vocal=True):
    _, st = _song(vocal=vocal)
    folder = home / "Song Stems"
    folder.mkdir()
    sf.write(folder / "Lead Vox.wav", st["vocals"][:, 0] * 0.7, SR, subtype="PCM_24")  # mono file
    sf.write(folder / "Adlibs.wav", st["vocals"] * 0.3, SR, subtype="PCM_24")
    sf.write(folder / "808.wav", st["bass"], SR, subtype="PCM_24")
    sf.write(folder / "Kick and Hats.wav", st["drums"], SR, subtype="PCM_24")
    sf.write(folder / "Keys.wav", st["other"], SR, subtype="PCM_24")
    sf.write(folder / "notes.txt", np.zeros((10, 1)), SR, format="WAV")  # not audio by extension: ignored
    return folder, st


def test_guess_role_reads_session_file_names():
    expected = {"808 Final.wav": "bass", "Adlibs L.wav": "vocals", "Lead Vox.wav": "vocals", "Bass Drum.aif": "drums",
                "Hi-Hats.wav": "drums", "Keys.wav": "other", "Lead Synth.wav": "other", "Tombstone pad.wav": "other",
                "Synth Bass.wav": "bass", "Harmonies.wav": "vocals", "Percussion.wav": "drums", "Subtle FX.wav": "other"}
    assert {n: stems.guess_role(n) for n in expected} == expected


def test_user_stems_are_summed_by_role(home):
    folder, st = _stem_folder(home)
    source, meta, user = stems.load_user_stems(folder, "s", lambda m: None)
    assert meta["user_stems"] and len(meta["stems"]) == 5
    assert {s["file"]: s["role"] for s in meta["stems"]}["Kick and Hats.wav"] == "drums"
    assert np.allclose(source, sum(user.values()))
    assert np.allclose(user["bass"], st["bass"], atol=1e-6)
    assert np.allclose(user["vocals"][:, 1], st["vocals"][:, 1] * 0.7 + st["vocals"][:, 1] * 0.3, atol=1e-5)
    _, meta2, user2 = stems.load_user_stems(folder, "s2", lambda m: None, {"Keys.wav": "vocals"})
    assert np.allclose(user2["other"], 0) and {s["file"]: s["role"] for s in meta2["stems"]}["Keys.wav"] == "vocals"
    with pytest.raises(ValueError):
        stems.load_user_stems(folder, "s3", lambda m: None, {"Nope.wav": "bass"})
    with pytest.raises(ValueError):
        stems.load_user_stems(folder, "s4", lambda m: None, {"Keys.wav": "piano"})


def test_master_from_session_stems(home, monkeypatch):
    def no_separation(*a, **k):
        raise AssertionError("session stems must not be separated")

    monkeypatch.setattr(stems, "separate", no_separation)
    folder, _ = _stem_folder(home)
    out = master.run(folder, "stemsong", engineer.Settings(), log=lambda m: None)
    qa = json.loads((out / "qa.json").read_text())
    assert qa["processing"]["stems"]["has_vocal"] is True
    assert any(n.startswith("Session stems") for n in qa["source_notes"])
    assert not any("Separation artifacts" in e for e in qa["needs_ears"])
    assert "separation_residual_db" not in qa["processing"]
    assert {c["check"]: c["status"] for c in qa["checks"]}["true peak"] == "pass"


# ---------------------------------------------------------------- listening loop


def test_review_server_and_feedback_loop(home, monkeypatch, capsys):
    import threading
    import urllib.request

    from drydock import cli, review

    out, _ = _run(home, monkeypatch, "2track")
    run = out.name
    server = review.serve("synth", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        def get(path, headers=None):
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {})) as r:
                return r.status, r.headers, r.read()

        status, _, body = get("/api/session")
        session = json.loads(body)
        assert session["runs"] == [review.ORIGINAL, run] and "808" in session["tags"]
        assert "808" in session["placeholder"] and "Janice" not in session["placeholder"]
        data = json.loads(get(f"/api/run/{run}")[2])
        assert data["timeline"]["bars"][0]["bar"] == 1 and len(data["timeline"]["clips"]) == 2
        assert len(data["peaks"]) == 2400
        status, headers, chunk = get("/audio/original", {"Range": "bytes=0-99"})
        assert status == 206 and len(chunk) == 100 and headers["Content-Type"] == "audio/flac"
        req = urllib.request.Request(base + f"/api/run/{run}/comments", method="POST",
                                     data=json.dumps({"time_s": 5.0, "end_s": 6.0, "tag": "vocal",
                                                      "text": "vocal too quiet"}).encode(),
                                     headers={"Content-Type": "application/json"})
        comment = json.loads(urllib.request.urlopen(req).read())
        assert comment["bar"] == pytest.approx(3.5, abs=0.01) and comment["clips"] == ["bars 3-10"]
        bad = urllib.request.Request(base + f"/api/run/{run}/comments", method="POST", data=b"{}",
                                     headers={"Content-Type": "text/plain"})  # simple POST from another site
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(bad)
        assert err.value.code == 403
    finally:
        server.shutdown()

    assert cli.main(["comment", "synth", "808 too loud", "--at", "0:12", "--tag", "808"]) == 0
    assert cli.main(["feedback", "synth"]) == 0
    printed = capsys.readouterr().out
    assert "2 open comment(s)" in printed and "vocal too quiet" in printed and "808 too loud" in printed
    assert cli.main(["resolve", "synth", comment["id"], "--note", "ride target 8"]) == 0
    _, still_open, _ = review.open_comments("synth")
    assert [c["text"] for c in still_open] == ["808 too loud"]
    assert cli.main(["feedback", "nope"]) == 2


def test_vocal_production_clean_and_produced(home, monkeypatch):
    _, clean = _run(home, monkeypatch, "stems")
    vp = clean["processing"]["vocal_production"]
    assert vp["production"] == "clean" and len(vp["notes_fixed"]) == 1 and vp["notes_fixed"][0]["bar"] == pytest.approx(4.0, abs=0.01)
    assert any("Re-pitched note at 0:06.0" in e for e in clean["needs_ears"])
    from drydock import engineer

    mix, st = _song()
    src = home / "song.wav"
    monkeypatch.setattr(stems, "separate", lambda x, slug, log, **k: {n: v[: x.shape[0]] * 0.9 for n, v in st.items()})
    out = master.run(src, "synth", engineer.Settings(production="produced"), log=lambda m: None)
    produced = json.loads((out / "qa.json").read_text())
    vp = produced["processing"]["vocal_production"]
    assert vp["production"] == "produced" and len(vp["hook_lines"]) == 3 and vp["throws"] == 3
    assert "## Vocal production" in (out / "qa.md").read_text()
    assert {c["check"]: c["status"] for c in produced["checks"]}["true peak"] == "pass"
