"""Mix repair: decide per-song moves from measurements, then process the stems (or the stereo file) into a bus."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from drydock import dsp, measure


@dataclass
class Settings:
    """Every knob. ``None`` means "decide from the song"; any value can be overridden with --set name=value."""

    mode: str = "stems"  # "stems" (separate and fix each element) or "2track" (stereo file only)
    # vocal (auto values scale with how far the vocal already sits over the beat)
    vocal_lift_db: float | None = None
    vocal_presence_db: float | None = None  # bell at 3 kHz on the vocal stem
    deess_max_db: float = 5.0
    ride_target_db: float | None = None  # vocal over everything else in 1-4 kHz while singing; 0 turns the ride off
    ride_max_db: float = 4.5
    duck_other_db: float | None = None  # make room: melodic stem dips in the vocal's bins while it sings
    duck_drums_db: float | None = None
    # drums
    drum_clip_db: float = 2.5
    hat_cut_db: float | None = None
    # bass
    octave_up: bool | None = None
    bass_db: float | None = None
    kick_duck_db: float | None = None
    sub_trim_db: float | None = None
    # melodic layer / stereo
    width: float | None = None
    # bus and master
    bus_presence_db: float | None = None
    target_lufs: float | None = None  # None: loudest level that keeps the limiter within its budget
    ceiling_db: float | None = None  # None: -1 dBTP, or -2 if an AAC transcode of the master would clip
    section_rides: bool = True

    def override(self, pairs: list[str]) -> "Settings":
        kinds = {f.name: f.type for f in fields(self)}
        for pair in pairs:
            name, _, raw = pair.partition("=")
            if name not in kinds:
                raise ValueError(f"unknown setting {name!r}; one of: {', '.join(kinds)}")
            kind = str(kinds[name])
            if raw.lower() in ("none", "auto"):
                value: Any = None
            elif "bool" in kind:
                value = raw.lower() in ("1", "true", "yes", "on")
            elif "float" in kind:
                value = float(raw)
            else:
                value = raw
            setattr(self, name, value)
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> "Settings":
        if self.mode not in ("stems", "2track"):
            raise ValueError(f"mode must be 'stems' or '2track', not {self.mode!r}")
        if self.target_lufs is not None and not -24 <= self.target_lufs <= -6:
            raise ValueError("target_lufs must be between -24 and -6")
        if self.ceiling_db is not None and not -6 <= self.ceiling_db <= 0:
            raise ValueError("ceiling_db must be between -6 and 0")
        if self.width is not None and not 0 <= self.width <= 1:
            raise ValueError("width must be between 0 and 1")
        for name in ("ride_max_db", "deess_max_db", "drum_clip_db", "hat_cut_db", "kick_duck_db", "duck_other_db",
                     "duck_drums_db"):
            value = getattr(self, name)
            if value is not None and not 0 <= value <= 12:
                raise ValueError(f"{name} must be between 0 and 12 dB")
        return self


def has_vocal(vocal: np.ndarray, source: np.ndarray) -> bool:
    """A lead vocal is present when the vocal stem's active level is within 25 dB of the song's (not just bleed)."""
    return dsp.active_rms_db(vocal) > dsp.active_rms_db(source) - 25


def decide(s: Settings, source: np.ndarray, stems: dict[str, np.ndarray] | None,
           vocal_median_db: float | None = None) -> tuple[Settings, dict[str, str]]:
    """Fill every ``None`` from measurements; returns the settings and why each auto value was chosen.
    ``vocal_median_db``: the source vocal's median audibility (vocal over the rest, 1-4 kHz, while singing)."""
    why: dict[str, str] = {}
    t = measure.tone(source)
    st = measure.stereo(source)
    if vocal_median_db is not None:
        v = vocal_median_db
        level = "buried" if v < 6 else "in the beat" if v < 10 else "already forward"
        auto = {"vocal_presence_db": 3.0 if v < 6 else 1.5 if v < 10 else 0.0,
                "vocal_lift_db": 0.5 if v < 8 else 0.0,
                # an already-forward vocal's quieter lines are its dynamics, not burial: riding them up flattens it
                "ride_target_db": 6.5 if v < 10 else 0.0,
                "duck_other_db": 3.0 if v < 10 else 0.0,
                "duck_drums_db": 1.5 if v < 10 else 0.0}
        for k, val in auto.items():
            if getattr(s, k) is None:
                setattr(s, k, val)
                why[k] = f"vocal sits {v:+.1f} dB over the beat in 1-4 kHz while singing ({level}; buried is under +6)"
    for k in ("vocal_presence_db", "vocal_lift_db", "duck_other_db", "duck_drums_db"):
        if getattr(s, k) is None:
            setattr(s, k, 0.0)
    if s.ride_target_db is None:
        s.ride_target_db = 6.5
    if s.hat_cut_db is None:
        excess = t["6-9k"] - t["presence 1.5-4k"] + 3.0
        s.hat_cut_db = round(float(np.clip(excess, 0.0, 2.5)), 1)
        why["hat_cut_db"] = (f"6-9 kHz sits {t['6-9k'] - t['presence 1.5-4k']:+.1f} dB vs 1.5-4 kHz "
                             f"(released hip-hop masters run about -5); cut {s.hat_cut_db} dB")
    bass = stems["bass"] if stems else dsp.split_low(source, 120.0)[0]
    fund = measure.bass_fundamental_hz(bass)
    if s.octave_up is None:
        s.octave_up = bool(0 < fund < 45)
        why["octave_up"] = f"bass line centres on {fund:.0f} Hz ({'below' if s.octave_up else 'above'} 45 Hz, where phones stop)"
    if s.sub_trim_db is None:
        heavy = t["sub <45"] - t["bass 45-90"]
        s.sub_trim_db = -1.0 if heavy > 2 else 0.0
        why["sub_trim_db"] = f"below 45 Hz is {heavy:+.1f} dB vs 45-90 Hz"
    if s.bass_db is None:
        s.bass_db = 1.0 if s.octave_up else 0.0
        why["bass_db"] = "makes up the trimmed fundamental and the limiter's toll on the 808" if s.octave_up else "no change"
    if s.kick_duck_db is None:
        s.kick_duck_db = 2.0 if 0 < fund < 60 else 0.0
        why["kick_duck_db"] = "808-style sub: duck it under the kick so both punch" if s.kick_duck_db else "bass is not a sub 808"
    if s.width is None:
        c = st["correlation"]
        s.width = 0.55 if c > 0.95 else 0.3 if c > 0.9 else 0.0
        why["width"] = f"L/R correlation {c:.3f} ({'near mono' if c > 0.95 else 'narrow' if c > 0.9 else 'already wide'})"
    if s.bus_presence_db is None:
        p = t["presence 1.5-4k"]
        s.bus_presence_db = (0.5 if s.mode == "stems" else 1.5) if p < -16.5 else 0.0
        why["bus_presence_db"] = f"1.5-4 kHz holds {p:.1f} dB of the spectrum (references about -16)"
    return s, why


# ---------------------------------------------------------------- stems


def process_stems(stems: dict[str, np.ndarray], source: np.ndarray, s: Settings) -> tuple[dict[str, np.ndarray], dict]:
    """Per-element fixes. The residual (source minus stems) is passed through untouched by the caller."""
    from glass_harbor.spectral import spectral_duck

    info: dict[str, Any] = {}
    vocal_present = has_vocal(stems["vocals"], source)

    def vocal_chain(v: np.ndarray) -> np.ndarray:
        if not vocal_present:
            return v
        pre = dsp.active_rms_db(v)
        v = dsp.eq(v, [("hp", 90, 2), ("peak", 500, -1.5, 1.4), ("peak", 3000, s.vocal_presence_db, 0.9),
                       ("highshelf", 11000, 1.0, 0.7)])
        v, ess = dsp.dynamic_band(v, 5500, 10000, max_cut_db=s.deess_max_db, over_db=4.0, hop=256, smooth=5)
        # threshold at the vocal's own active level: about 1-2 dB typical, more on the loudest syllables
        v, vgr = dsp.compressor(v, threshold_db=dsp.active_rms_db(v), ratio=3.0, attack_ms=5, release_ms=90)
        v = v * dsp.gain(pre + s.vocal_lift_db - dsp.active_rms_db(v))
        v, spike = dsp.peak_tame(v, 1.5)
        on = vgr[vgr > 0.05]
        info["vocal"] = {"deess_p90_db": round(ess, 1), "comp_gr_p50_db": round(float(np.median(on)), 1) if on.size else 0.0,
                         "spike_cut_max_db": round(spike, 1), "lift_db": s.vocal_lift_db}
        return v

    def drum_chain(d: np.ndarray) -> np.ndarray:
        d = dsp.eq(d, [("peak", 7200, -s.hat_cut_db, 1.4)])
        d, hat = dsp.dynamic_band(d, 6000, 9500, max_cut_db=2.0 if s.hat_cut_db else 1.0)
        d = dsp.widen(d, amount=0.2 if s.width else 0.0, above_hz=5000, seed=11)
        if s.drum_clip_db:
            d = dsp.soft_clip(d, float(dsp.db(np.percentile(np.abs(d), 99.995))) - s.drum_clip_db, knee=0.8)
        info["drums"] = {"hat_cut_db": s.hat_cut_db, "hat_dynamic_p90_db": round(hat, 1), "transient_clip_db": s.drum_clip_db}
        return d

    def bass_chain(b: np.ndarray) -> np.ndarray:
        b = dsp.eq(b, [("hp", 20, 4), ("peak", 30, s.sub_trim_db, 1.5)])
        if s.octave_up:
            b = dsp.octave_up(b, below_hz=50, amount=0.4, fundamental_db=-1.5)
        b = dsp.upper_harmonics(b, amount=0.15) * dsp.gain(s.bass_db)
        if s.kick_duck_db:
            b = dsp.kick_duck(b, stems["drums"], depth_db=s.kick_duck_db)
        info["bass"] = {"octave_up": s.octave_up, "level_db": s.bass_db, "kick_duck_db": s.kick_duck_db,
                        "sub_trim_db": s.sub_trim_db}
        return b

    def other_chain(o: np.ndarray) -> np.ndarray:
        info["other"] = {"width": s.width}
        return dsp.widen(o, amount=s.width, above_hz=300, seed=7)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:  # the four chains are independent (scipy releases the GIL)
        jobs = [pool.submit(f, stems[k]) for f, k in ((vocal_chain, "vocals"), (drum_chain, "drums"),
                                                       (bass_chain, "bass"), (other_chain, "other"))]
        v, d, b, o = (j.result() for j in jobs)

    if vocal_present:
        if s.duck_other_db:
            o = spectral_duck(o.astype(np.float32), v.astype(np.float32), max_db=s.duck_other_db, level_db=0.0).astype(np.float64)
        if s.duck_drums_db:
            d = spectral_duck(d.astype(np.float32), v.astype(np.float32), max_db=s.duck_drums_db, level_db=0.0,
                              high_hz=5000.0).astype(np.float64)
        residual = source - sum(stems.values())
        if s.ride_target_db:
            v, curve = dsp.presence_ride(v, d + b + o + residual, target_db=s.ride_target_db, max_boost_db=s.ride_max_db)
            on = curve > 0.1
            info["vocal_ride"] = {"share_of_song_boosted": round(float(on.mean()), 2),
                                  "boost_p50_db": round(float(np.median(curve[on])), 1) if on.any() else 0.0,
                                  "boost_max_db": round(float(curve.max()), 1)}
    info["has_vocal"] = bool(vocal_present)
    return {"vocals": v, "drums": d, "bass": b, "other": o}, info


def bus_from_stems(processed: dict[str, np.ndarray], residual: np.ndarray, s: Settings) -> tuple[np.ndarray, dict]:
    bus = sum(processed.values()) + residual
    return _bus(bus, [("peak", 62, 1.0 if s.octave_up else 0.0, 1.0), ("peak", 2800, s.bus_presence_db, 0.8)])


def two_track(source: np.ndarray, s: Settings) -> tuple[np.ndarray, dict]:
    x = dsp.eq(source, [("hp", 22, 2), ("peak", 30, s.sub_trim_db, 1.2), ("peak", 2800, s.bus_presence_db, 0.8),
                        ("peak", 7200, -min(1.5, s.hat_cut_db), 1.2)])
    x, harsh = dsp.dynamic_band(x, 6000, 9500, max_cut_db=2.5 if s.hat_cut_db else 1.0)
    if s.octave_up:
        x = dsp.octave_up(x, below_hz=48, amount=0.35, fundamental_db=-0.5)
    x = dsp.upper_harmonics(x, amount=0.08, lo=110, hi=550)
    bus, info = _bus(x, [])
    info["harsh_dynamic_p90_db"] = round(harsh, 1)
    return bus, info


def _bus(bus: np.ndarray, bands: list[tuple]) -> tuple[np.ndarray, dict]:
    bus = bus * dsp.gain(-14.0 - dsp.lufs(bus))  # working level for the glue compressor
    bus = dsp.eq(bus, bands)
    bus, gr = dsp.compressor(bus, threshold_db=-20.0, ratio=2.0, attack_ms=30, release_ms=160, sc_hpf=150)
    bus = dsp.mono_below(bus, 110)
    on = gr[gr > 0.05]
    return bus, {"glue_gr_p50_db": round(float(np.median(on)), 1) if on.size else 0.0,
                 "glue_gr_p95_db": round(float(np.percentile(on, 95)), 1) if on.size else 0.0}
