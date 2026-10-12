import numpy as np
import pytest

from drydock import dsp, vocal

SR = dsp.SR


def _sine(freq, seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * freq * t)
    return np.stack([x, x], axis=1)


def _pitch_cents(x, ref_hz):
    import librosa

    y = librosa.resample(x.mean(axis=1), orig_sr=SR, target_sr=22050)
    f0, voiced, _ = librosa.pyin(y, fmin=200, fmax=900, sr=22050, frame_length=2048, hop_length=256)
    return float(np.nanmedian(1200 * np.log2(f0[voiced] / ref_hz)))


def test_fix_notes_repitches_only_the_flagged_note():
    sharp = 440 * 2 ** (35 / 1200)
    x = np.concatenate([_sine(330, 1.0), _sine(sharp, 1.0), _sine(330, 1.0)])
    out = vocal.fix_notes(x, [{"t0": 1.0, "t1": 2.0, "cents": 35.0}])
    assert abs(_pitch_cents(out[int(1.2 * SR):int(1.8 * SR)], 440)) < 8  # was +35
    assert np.array_equal(out[: int(0.8 * SR)], x[: int(0.8 * SR)])  # untouched outside the window
    assert np.array_equal(out[int(2.2 * SR):], x[int(2.2 * SR):])


def test_key_filter_and_spans():
    assert vocal._key_pitch_classes("A minor") == {9, 11, 0, 2, 4, 5, 7}
    assert vocal._key_pitch_classes("C major") == {0, 2, 4, 5, 7, 9, 11}
    assert vocal._key_pitch_classes(None) is None
    w = lambda word, a, b: {"word": word, "start": a, "end": b}
    lyrics = {"segments": [
        {"start": 1.0, "end": 2.0, "text": "I need somebody", "words": [w("I", 1.0, 1.2), w("need", 1.2, 1.5), w("somebody", 1.5, 2.0)]},
        {"start": 3.0, "end": 4.0, "text": "something else here", "words": [w("something", 3.0, 3.4), w("else", 3.4, 3.7), w("here", 3.7, 4.0)]},
        {"start": 5.0, "end": 6.0, "text": "I need somebody!", "words": [w("I", 5.0, 5.2), w("need", 5.2, 5.5), w("somebody", 5.5, 6.0)]},
    ]}
    assert vocal.hook_spans(lyrics) == [(1.0, 2.0), (5.0, 6.0)]
    assert vocal.line_ends(lyrics) == [(1.5, 2.0), (3.7, 4.0), (5.5, 6.0)]
    assert vocal.line_ends(lyrics, min_gap_s=1.5) == [(5.5, 6.0)]  # only where the echo has room


def test_hook_doubles_touch_only_hook_lines():
    rng = np.random.default_rng(0)
    lead = dsp.bandpass(rng.standard_normal((6 * SR, 2)) * 0.1, 200, 3000)
    out = vocal.hook_doubles(lead, [(2.0, 3.0)])
    assert np.array_equal(out[: int(1.5 * SR)], lead[: int(1.5 * SR)])
    assert np.array_equal(out[int(3.5 * SR):], lead[int(3.5 * SR):])
    inside = slice(int(2.2 * SR), int(2.8 * SR))
    side = lambda y: np.mean(((y[inside, 0] - y[inside, 1]) / 2) ** 2)
    assert side(out) > side(lead) + 1e-6  # doubles are panned: the hook gets width


def test_sends_add_reverb_and_throws():
    lead = _sine(300, 4.0, 0.2) * (np.arange(4 * SR) < 2 * SR)[:, None]
    backing = np.zeros_like(lead)
    beat = 0.6
    dry = vocal.sends(lead, backing, [], beat, reverb_mix=0.0, throw_mix=0.0)
    assert not np.any(dry)
    wet = vocal.sends(lead, backing, [(1.5, 2.0)], beat, reverb_mix=0.08, throw_mix=0.3)
    tail = wet[int(2.3 * SR):int(3.5 * SR)]
    assert np.sqrt(np.mean(tail ** 2)) > 1e-3  # reverb + echo ring after the line ends


def test_treat_backing_widens_and_keeps_level_setting():
    rng = np.random.default_rng(1)
    b = dsp.bandpass(rng.standard_normal((4 * SR, 2)) * 0.05, 300, 4000)
    b[:, 1] = b[:, 0]
    out = vocal.treat_backing(b, width=0.4, level_db=-3.0)
    assert np.mean((out[:, 0] - out[:, 1]) ** 2) > 0
    assert dsp.active_rms_db(out) < dsp.active_rms_db(b)
