import numpy as np
import pytest

from drydock import dsp, engineer, master, measure

SR = dsp.SR


def tone(hz, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * hz * t)
    return np.stack([x, x], axis=1)


def noise(seconds=2.0, amp=0.2, seed=0):
    return amp * np.random.default_rng(seed).standard_normal((int(seconds * SR), 2))


def band_db(x, lo, hi):
    from scipy.signal import welch

    f, p = welch(x.mean(axis=1), SR, nperseg=8192)
    return 10 * np.log10(p[(f >= lo) & (f < hi)].sum())


def test_limiter_holds_true_peak_ceiling():
    x = noise(amp=0.6) + tone(60, amp=0.8)
    y, red = dsp.limiter(x, -1.15)
    assert dsp.true_peak_db(y) <= -1.0
    assert red.max() <= 1e-9


def test_soft_clip_never_exceeds_threshold():
    x = noise(amp=2.0)
    assert np.abs(dsp.soft_clip(x, -3.0)).max() <= dsp.gain(-3.0) + 1e-12


def test_split_low_reconstructs_exactly():
    x = noise()
    low, high = dsp.split_low(x, 120)
    assert np.allclose(low + high, x)


def test_widen_keeps_mono_sum_and_adds_side():
    x = noise()
    x[:, 1] = x[:, 0]
    y = dsp.widen(x, amount=0.5)
    assert np.allclose(y.sum(axis=1), x.sum(axis=1))
    assert np.abs(y[:, 0] - y[:, 1]).max() > 0.01


def test_octave_up_moves_energy_up_an_octave():
    x = tone(35, seconds=3)
    y = dsp.octave_up(x, below_hz=50, amount=0.4, fundamental_db=-1.5)
    assert band_db(y, 65, 75) - band_db(x, 65, 75) > 20
    assert band_db(y, 30, 40) < band_db(x, 30, 40)
    assert np.abs(y).max() <= np.abs(x).max() * 1.15


def test_presence_ride_only_boosts_and_respects_cap():
    vocal = noise(seconds=4, amp=0.05, seed=1)
    rest = noise(seconds=4, amp=0.2, seed=2)
    y, curve = dsp.presence_ride(vocal, rest, target_db=6.5, max_boost_db=3.0)
    assert curve.min() >= 0
    assert curve.max() <= 3.0 + 1e-9
    assert dsp.active_rms_db(y) > dsp.active_rms_db(vocal)


def test_kick_duck_only_dips_the_bass():
    bass = tone(40, seconds=2, amp=0.5)
    drums = np.zeros_like(bass)
    drums[SR // 2: SR // 2 + 2000] = 0.9  # one kick
    y = dsp.kick_duck(bass, drums, depth_db=3.0)
    ratio = np.abs(y).max(axis=1) / np.maximum(np.abs(bass).max(axis=1), 1e-9)
    assert ratio.max() <= 1 + 1e-9
    assert dsp.db(ratio[SR // 2 + 1000]) == pytest.approx(-3.0, abs=1.0)


def test_section_rides_restore_contrast_with_cuts_only():
    quiet, loud = noise(4, 0.05, 1), noise(4, 0.4, 2)
    source = np.concatenate([quiet, loud])
    flattened = np.concatenate([quiet * 4, loud])  # a master that squashed the contrast
    spans = [(0, quiet.shape[0], "intro"), (quiet.shape[0], source.shape[0], "drop")]
    out, info = master.section_rides(source, flattened, spans, beat_s=0.5)
    assert all(c <= 0 for c in info["cut_db"])
    contrast_src = info["source_lufs"][1] - info["source_lufs"][0]
    contrast_out = info["after_lufs"][1] - info["after_lufs"][0]
    assert abs(contrast_out - contrast_src) < 1.0
    assert np.abs(out).max() <= np.abs(flattened).max() + 1e-9


def test_finalize_hits_target_under_ceiling():
    x = noise(6, 0.1) + tone(50, 6, 0.3)
    mask = np.ones(x.shape[0], bool)
    y, info = master.finalize(x, -1.0, mask, target_lufs=-12.0)
    assert info["lufs"] == pytest.approx(-12.0, abs=0.3)
    assert dsp.true_peak_db(y) <= -1.0


def test_choose_loudness_respects_budget():
    x = noise(6, 0.1) + tone(35, 6, 0.6)  # sub-heavy: the limiter would ride every cycle if pushed
    mask = np.ones(x.shape[0], bool)
    gain_db, tried = master.choose_loudness(x, mask, -1.0)
    chosen = tried[-1]
    assert chosen["gain_db"] == round(gain_db, 2)
    within = chosen["median_db"] <= master.GR_MEDIAN_MAX_DB and chosen["p995_db"] <= master.GR_P995_MAX_DB
    assert within or len(tried) == len(master.LOUDNESS_CANDIDATES)
    # the final pass at that gain carries exactly the limiter load the search measured
    _, info = master.finalize(x, -1.0, mask, gain_db=gain_db)
    assert info["limiter_beat_sections"]["median_db"] == chosen["median_db"]


def test_settings_override_parses_types():
    s = engineer.Settings().override(["ride_target_db=7", "octave_up=false", "width=auto", "mode=2track"])
    assert s.ride_target_db == 7.0 and s.octave_up is False and s.width is None and s.mode == "2track"
    with pytest.raises(ValueError):
        engineer.Settings().override(["nope=1"])


def test_decide_flags_mono_and_sub_808():
    x = tone(36, 4, 0.5) + noise(4, 0.02)
    x[:, 1] = x[:, 0]
    s, why = engineer.decide(engineer.Settings(mode="2track"), x, None)
    assert s.octave_up is True and s.width == 0.55
    assert "octave_up" in why and "width" in why


def test_vocal_audibility_per_bar():
    class G:
        bar_count = 2

        def bar_time(self, bar):
            return (bar - 1) * 2.0

    vocal = np.concatenate([noise(2, 0.3, 1), noise(2, 0.08, 2)])  # bar 2 quieter but above the singing gate
    rest = noise(4, 0.1, 3)
    a = measure.vocal_audibility(vocal, rest, G())
    assert a[1] > a[2]


def test_presence_ride_releases_after_the_last_line():
    vocal = np.concatenate([noise(3, 0.05, 1), noise(3, 0.0005, 2)])  # singing, then only faint bleed
    rest = noise(6, 0.2, 3)
    _, curve = dsp.presence_ride(vocal, rest, target_db=6.5, max_boost_db=4.5)
    assert curve[int(1.5 * SR)] > 2.0
    assert curve[int(5.5 * SR)] < 0.2


def test_vocal_moves_back_off_when_the_vocal_is_already_forward():
    x = noise(4, 0.1)
    buried, why = engineer.decide(engineer.Settings(mode="2track"), x, None, vocal_median_db=3.0)
    forward, _ = engineer.decide(engineer.Settings(mode="2track"), x, None, vocal_median_db=16.0)
    assert buried.vocal_presence_db > 0 and buried.duck_other_db > 0 and "vocal_lift_db" in why
    assert forward.vocal_presence_db == 0 and forward.duck_other_db == 0 and forward.vocal_lift_db == 0
