"""Loudness stage, section rides, and the end-to-end pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from drydock import dsp, engineer, measure, paths, stems as stem_io

Log = Callable[[str], None]

# Limiter budget for the automatic loudness search, measured inside the beat sections. Past this the limiter starts
# riding every hit (a sub-heavy 808 pins it) and the master gets smaller, not louder.
GR_MEDIAN_MAX_DB = 1.0
GR_P995_MAX_DB = 5.0
LOUDNESS_CANDIDATES = [-9.0, -9.5, -10.0, -10.5, -11.0, -11.5, -12.0, -12.5]
MIN_SECONDS = 10.0


class FinalStage:
    """Low-band soft clip (808/kick peaks set the master's peak level; clipping the full mix distorts the vocal),
    then a true-peak limiter. The clip threshold follows the low band's own level, so the clipped signal scales
    linearly with gain: clip and oversample once, then any gain is just a limiter pass on a scaled envelope."""

    def __init__(self, bus: np.ndarray, low_clip_db: float = 2.0):
        low, high = dsp.split_low(bus, 120.0)
        lp = np.percentile(np.abs(low), 99.97)
        clipped = dsp.soft_clip(low, float(dsp.db(lp)) - low_clip_db, knee=0.8)
        self.shaped = clipped + high
        self.env = dsp.tp_envelope(self.shaped)
        diff = dsp.bandpass((clipped - low).mean(axis=1), 300, 5000)
        sig = dsp.bandpass(self.shaped.mean(axis=1), 300, 5000)
        self.products_db = float(10 * np.log10(np.mean(diff ** 2) / (np.mean(sig ** 2) + 1e-30) + 1e-30))
        self.base_lufs = dsp.lufs(bus)

    def render(self, gain_db: float, ceiling_db: float) -> tuple[np.ndarray, np.ndarray]:
        """(audio, limiter gain dB per sample) at this gain."""
        g = float(dsp.gain(gain_db))
        curve = dsp.limiter_gain(self.env * g, ceiling_db - 0.15)
        return self.shaped * (g * curve)[:, None], dsp.db(curve)


def _gr_stats(red: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gr = -red[mask] if mask.any() else -red
    return {"median_db": round(float(np.median(gr)), 2), "p995_db": round(float(np.percentile(gr, 99.5)), 2),
            "share_over_1db": round(float((gr > 1).mean()), 2)}


def finalize(stage: FinalStage, ceiling_db: float, beat_mask: np.ndarray, *, target_lufs: float | None = None,
             gain_db: float | None = None) -> tuple[np.ndarray, dict]:
    """Either hit ``target_lufs`` (gain iterated: limiting eats loudness) or apply a fixed ``gain_db`` (the gain the
    loudness search measured, so the limiter load is exactly what it checked)."""
    if gain_db is None:
        if target_lufs is None:
            raise ValueError("finalize needs target_lufs or gain_db")
        gain_db = target_lufs - stage.base_lufs
        for _ in range(6):
            y, _ = stage.render(gain_db, ceiling_db)
            got = dsp.lufs(y)
            if abs(got - target_lufs) < 0.1:
                break
            gain_db += target_lufs - got
    y, red = stage.render(gain_db, ceiling_db)
    tp = dsp.true_peak_db(y)
    if tp > ceiling_db:
        y = y * dsp.gain(ceiling_db - tp - 0.02)
    return y, {"target_lufs": target_lufs, "gain_db": round(gain_db, 2), "lufs": round(dsp.lufs(y), 2),
               "true_peak_dbtp": round(dsp.true_peak_db(y), 2), "limiter_beat_sections": _gr_stats(red, beat_mask),
               "low_clip_products_db": round(stage.products_db, 1)}


def choose_loudness(stage: FinalStage, beat_mask: np.ndarray, ceiling_db: float) -> tuple[float, list[dict]]:
    """Gain for the loudest candidate whose limiter stays within budget in the beat sections. Each candidate is
    judged at the level it actually reaches after limiting, not the gain's nominal target."""
    tried = []
    for t in LOUDNESS_CANDIDATES:
        g = t - stage.base_lufs
        y, red = stage.render(g, ceiling_db)
        st = _gr_stats(red, beat_mask)
        tried.append({"gain_db": round(g, 2), "reaches_lufs": round(dsp.lufs(y), 2), **st})
        if st["median_db"] <= GR_MEDIAN_MAX_DB and st["p995_db"] <= GR_P995_MAX_DB:
            return g, tried
    return LOUDNESS_CANDIDATES[-1] - stage.base_lufs, tried


def section_rides(source: np.ndarray, master: np.ndarray, spans: list[tuple[int, int, str]], beat_s: float) -> tuple[np.ndarray, dict]:
    """Per-section cuts (never boosts) so each section sits as far under the loud ones as it did in the source.
    Limiting turns loud sections down and leaves quiet ones alone; this puts the arrangement's contrast back."""
    n = min(source.shape[0], master.shape[0])
    lo = measure.section_loudness(source[:n], spans)
    lm = measure.section_loudness(master[:n], spans)
    loud = [i for i, v in enumerate(lo) if v > max(lo) - 1.0]
    offset = float(np.median([lo[i] - lm[i] for i in loud]))
    cuts = [min(0.0, (lo[i] - offset) - lm[i]) for i in range(len(spans))]
    cuts = [c if c < -0.5 else 0.0 for c in cuts]
    g = np.zeros(master.shape[0])
    for (a, b, _), c in zip(spans, cuts):
        g[a:b] = c
    for k in range(1, len(spans)):
        t, prev, cur = spans[k][0], cuts[k - 1], cuts[k]
        if cur > prev:  # into a louder section: short ramp that lands on the downbeat
            w = min(int(0.03 * dsp.SR), t)
            g[t - w:t] = np.linspace(prev, cur, w)
        elif cur < prev:  # into a quieter one: one-beat ramp from the downbeat so the previous tail is not chopped
            w = int(beat_s * dsp.SR)
            g[t:t + w] = np.linspace(prev, cur, len(g[t:t + w]))
    out = master * dsp.gain(g)[:, None]
    return out, {"sections": [s for _, _, s in spans], "source_lufs": lo, "before_lufs": lm,
                 "cut_db": [round(c, 2) for c in cuts], "after_lufs": measure.section_loudness(out[:n], spans)}


def vocal_prep(vocals: np.ndarray, slug: str, tag: str, analysis: dict, grid, settings: engineer.Settings,
               log: Log) -> tuple[dict[str, Any], dict[str, Any]]:
    """Lead/backing split, note fixes, and (for "produced") the hook and line-end spans from the lyrics."""
    from drydock import vocal

    lead, backing = vocal.split_lead(vocals, slug, tag, log)
    notes = vocal.off_notes(lead, slug, tag, (analysis.get("key") or {}).get("name")) if settings.note_fix else []
    if notes:
        log(f"re-pitching {len(notes)} held note(s)")
        lead = vocal.fix_notes(lead, notes)
    hooks: list[tuple[float, float]] = []
    throws: list[tuple[float, float]] = []
    breaths: list[tuple[float, float]] = []
    if settings.production == "produced" or settings.breath_cut_db:
        lyrics = vocal.transcribe(lead, slug, tag, log)
        if settings.production == "produced":
            hooks, throws = vocal.hook_spans(lyrics), vocal.line_ends(lyrics, min_gap_s=grid.period)
        if settings.breath_cut_db:
            words = [(w["start"], w["end"]) for s in lyrics["segments"] for w in s["words"]]
            breaths = vocal.find_breaths(lead, words)
            log(f"turning {len(breaths)} breath(s) down {settings.breath_cut_db:g} dB")
            lead = vocal.reduce_breaths(lead, breaths, settings.breath_cut_db)
    info = {"lead_split": True, "backing_level_vs_lead_db": round(dsp.active_rms_db(backing) - dsp.active_rms_db(lead), 1),
            "notes_fixed": [{"time_s": n["t0"], "bar": round(grid.bar_at_time(n["t0"]), 1), "cents": n["cents"],
                             "seconds": round(n["t1"] - n["t0"], 2)} for n in notes],
            "production": settings.production,
            "hook_lines": [{"bar": round(grid.bar_at_time(a), 1), "seconds": round(b - a, 1)} for a, b in hooks],
            "throws": len(throws),
            "breaths": {"cut_db": settings.breath_cut_db, "count": len(breaths),
                        "times_s": [round(a, 2) for a, _ in breaths]}}
    return {"lead": lead, "backing": backing, "hooks": hooks, "throws": throws, "beat_s": grid.period}, info


def beat_mask(frames: int, spans: list[tuple[int, int, str]], loudness: list[float]) -> np.ndarray:
    mask = np.zeros(frames, bool)
    for (a, b, _), v in zip(spans, loudness):
        if v > max(loudness) - 3.0:
            mask[a:b] = True
    return mask


def run(src: Path, slug: str, settings: engineer.Settings, log: Log = print, *,
        roles: dict[str, str] | None = None, tag: str | None = None) -> Path:
    """Master ``src``: a bounce (separated into stems) or a folder of session stems (used as they are)."""
    from glass_harbor.grid import Grid

    from drydock import deliver, qa

    started = time.time()
    settings.validate()
    tag = engineer.variant_tag(settings) if tag is None else tag  # before the auto values are filled in
    user_stems = None
    if src.is_dir():
        log(f"reading stems from {src.name}")
        source, meta, user_stems = stem_io.load_user_stems(src, slug, log, roles)
    else:
        source, meta = stem_io.load_source(src, slug, log)
    if source.shape[0] < MIN_SECONDS * dsp.SR:
        raise ValueError(f"{src.name} is {source.shape[0] / dsp.SR:.1f} s; dry-dock needs at least {MIN_SECONDS} s")
    if dsp.lufs(source) <= -60:
        raise ValueError(f"{src.name} is silent or nearly silent ({dsp.lufs(source):.0f} LUFS)")
    analysis = stem_io.analyze(source, slug, log)
    grid = Grid.from_dict(analysis["grid"])
    spans = measure.section_spans(grid, analysis["sections"], source.shape[0])
    src_sections = measure.section_loudness(source, spans)
    mask = beat_mask(source.shape[0], spans, src_sections)

    if settings.mode != "stems":
        stems = None
    else:
        stems = user_stems if user_stems is not None else stem_io.separate(source, slug, log, quality=settings.separation)
    has_vocal = stems is not None and engineer.has_vocal(stems["vocals"], source)
    if stems is not None and not has_vocal:
        log("no lead vocal found: treating the song as an instrumental")
    vocal_before = measure.vocal_audibility(stems["vocals"], source - stems["vocals"], grid) if has_vocal else {}
    vocal_median = float(np.median(list(vocal_before.values()))) if vocal_before else None
    settings, why = engineer.decide(settings, source, stems, vocal_median)
    for k, v in why.items():
        log(f"  {k} = {getattr(settings, k)}: {v}")
    info: dict[str, Any] = {}
    if stems is not None:
        residual = source - sum(stems.values())
        if user_stems is None:
            info["separation_residual_db"] = round(dsp.active_rms_db(residual) - dsp.active_rms_db(source), 1)
        prep = None
        if has_vocal and settings.lead_split:
            prep, info["vocal_production"] = vocal_prep(stems["vocals"], slug,
                                                        "session" if user_stems is not None else settings.separation,
                                                        analysis, grid, settings, log)
        log("fixing stems")
        processed, info["stems"] = engineer.process_stems(stems, source, settings, prep)
        bus, info["bus"] = engineer.bus_from_stems(processed, residual, settings)
    else:
        processed = None
        log("processing the stereo file")
        bus, info["bus"] = engineer.two_track(source, settings)

    from drydock.deliver import aac_true_peak

    stage = FinalStage(bus)

    def master_at(ceiling: float) -> tuple[np.ndarray, dict]:
        meta_: dict[str, Any] = {}
        if settings.target_lufs is None:
            log(f"searching for the loudest level within the limiter budget ({ceiling} dBTP)")
            gain_db, meta_["loudness_search"] = choose_loudness(stage, mask, ceiling)
            y, meta_["final"] = finalize(stage, ceiling, mask, gain_db=gain_db)
        else:
            y, meta_["final"] = finalize(stage, ceiling, mask, target_lufs=settings.target_lufs)
        if settings.section_rides:
            y, meta_["rides"] = section_rides(source, y, spans, grid.period)
        return y, meta_

    ceiling = settings.ceiling_db if settings.ceiling_db is not None else -1.0
    master, stage_info = master_at(ceiling)
    overshoot = 0.0
    if settings.ceiling_db is None:
        tp_aac = aac_true_peak(master)
        overshoot = max(0.0, tp_aac - dsp.true_peak_db(master))
        why["ceiling_db"] = f"AAC transcode of a -1 dBTP master peaks at {tp_aac:+.2f} dBTP"
        if tp_aac > -0.3:
            log(f"an AAC transcode would peak at {tp_aac:+.2f} dBTP; re-mastering at -2 dBTP")
            ceiling = -2.0
            master, stage_info = master_at(ceiling)
            why["ceiling_db"] += " (would clip on streaming transcodes; Spotify asks for -2 dBTP above -14 LUFS)"
        settings.ceiling_db = ceiling
    info.update(stage_info)
    log(f"mastered to {info['final']['lufs']} LUFS at {ceiling} dBTP before section rides")
    fade = int(0.005 * dsp.SR)
    master[:fade] *= np.linspace(0, 1, fade)[:, None]

    out = paths.runs_dir(slug) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    title = Path(meta["file"]).stem
    from concurrent.futures import ThreadPoolExecutor

    log("writing deliverables and running QA")
    with ThreadPoolExecutor(max_workers=1) as pool:  # encodes run in ffmpeg processes while QA measures in memory
        pending = pool.submit(deliver.write_all, out, title, source, master, encode_overshoot_db=overshoot, tag=tag)
        report = qa.report(source=source, master=master, meta=meta, analysis=analysis, spans=spans, stems=stems,
                           processed=processed, settings=settings, why=why, info=info, grid=grid,
                           vocal_before=vocal_before)
        qa.chart(out / "before-after.png", source, master, spans, report)
        files = pending.result()
    report = qa.add_file_checks(report, files)
    report["variant"] = tag
    (out / "qa.json").write_text(json.dumps(report, indent=1, default=float))
    (out / "qa.md").write_text(qa.markdown(report, title))
    (out / "settings.json").write_text(json.dumps(settings.to_dict(), indent=1))
    (out / "analysis.json").write_text(json.dumps({k: analysis[k] for k in ("bpm", "key", "grid", "sections") if k in analysis}))
    log(f"done in {time.time() - started:.0f}s: {out}")
    log(f"listen and comment: drydock review {slug}")
    return out
