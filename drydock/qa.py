"""QA: deterministic checks on the master, source notes, and what only ears can judge."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from drydock import deliver, dsp, measure

LOSSY = {"mp3", "aac", "vorbis", "opus", "mp2", "wmav2"}
# Vocal over the beat (1-4 kHz, median while singing) measured on released rap/R&B masters with the same method:
# about 0 dB (Lil Bean "yeah bop") to +7 dB (Drake "Janice STFU"). Past +12 the vocal is far in front of the beat.
REFERENCE_VOCAL_RANGE = "about 0 to +7 dB"
VOCAL_FORWARD_DB = 12.0


def _check(name: str, ok: bool, detail: str, *, fail: bool = False) -> dict[str, str]:
    return {"check": name, "status": "pass" if ok else ("fail" if fail else "warn"), "detail": detail}


def _rel(values: list[float]) -> list[float]:
    top = max(values)
    return [v - top for v in values]


def report(*, source, master, meta, analysis, spans, stems, processed, settings, why, info, grid,
           vocal_before=None) -> dict[str, Any]:
    """Everything measurable on the master in memory (runs while the share files encode). ``add_file_checks``
    finishes the report once the files exist."""
    written = np.round(np.clip(master, -1.0, 1.0 - 2 ** -23) * 2 ** 23) / 2 ** 23  # exactly what the 24-bit WAV holds
    src_lv, mst_lv = measure.levels(source), measure.levels(written)
    src_st, mst_st = measure.stereo(source), measure.stereo(written)
    sec_src, sec_mst = measure.section_loudness(source, spans), measure.section_loudness(written, spans)
    drift = [round(m - s, 2) for s, m in zip(_rel(sec_src), _rel(sec_mst))]
    checks = [
        _check("true peak", mst_lv["true_peak_dbtp"] <= settings.ceiling_db + 0.05,
               f"{mst_lv['true_peak_dbtp']} dBTP (ceiling {settings.ceiling_db})", fail=True),
        _check("clipping", int((np.abs(written) >= 0.9999).sum()) == 0,
               f"{int((np.abs(written) >= 0.9999).sum())} samples at full scale", fail=True),
    ]
    checks.append(_check("section contrast kept", max(abs(d) for d in drift) <= 1.5,
                         f"largest shift vs the source's section balance {max(drift, key=abs):+.1f} LU"))
    lim = info["final"]["limiter_beat_sections"]
    checks.append(_check("limiter load (beat sections)", lim["median_db"] <= 1.0 and lim["p995_db"] <= 5.5,
                         f"median {lim['median_db']} dB, 99.5th pct {lim['p995_db']} dB"))
    checks.append(_check("low-band clip distortion", info["final"]["low_clip_products_db"] <= -35,
                         f"{info['final']['low_clip_products_db']} dB re signal in 300 Hz-5 kHz"))
    mono = np.repeat(written.mean(axis=1, keepdims=True), 2, axis=1)
    mono_change = round(dsp.lufs(mono) - mst_lv["lufs"], 2)
    checks.append(_check("mono compatibility", mono_change >= -1.0, f"mono sum {mono_change:+.2f} LU"))

    vocal = None
    if stems is not None and info.get("stems", {}).get("has_vocal"):
        residual = source - sum(stems.values())
        before = vocal_before or measure.vocal_audibility(stems["vocals"], source - stems["vocals"], grid)
        rest = processed["drums"] + processed["bass"] + processed["other"] + residual
        # score only the bars sung in the source: processing can lift bleed and tails over the singing gate
        after = {b: v for b, v in measure.vocal_audibility(processed["vocals"], rest, grid).items() if b in before}
        weak_b = sorted(b for b, v in before.items() if v < 4)
        weak_a = sorted(b for b, v in after.items() if v < 4)
        vocal = {"before": before, "after": after, "median_before": round(float(np.median(list(before.values()))), 1),
                 "median_after": round(float(np.median(list(after.values()))), 1), "weak_bars_before": weak_b,
                 "weak_bars_after": weak_a}
        forward = vocal["median_after"]
        ref_vocal = (info.get("reference") or {}).get("vocal_median_db")
        if ref_vocal is not None:
            checks.append(_check("vocal balance vs reference", abs(forward - ref_vocal) <= 3.0,
                                 f"vocal sits {forward:+.1f} dB over the beat; the reference {ref_vocal:+.1f} dB"))
        else:
            advice = ("in range" if forward <= VOCAL_FORWARD_DB else "very forward: the beat may feel small under it; "
                      f"A/B --set vocal_lift_db={(settings.vocal_lift_db or 0) - 3:g}")
            checks.append(_check("vocal balance", forward <= VOCAL_FORWARD_DB,
                                 f"vocal sits {forward:+.1f} dB over the beat in 1-4 kHz (released references measured "
                                 f"{REFERENCE_VOCAL_RANGE}); {advice}"))
        checks.append(_check("vocal audibility", len(weak_a) <= 2,
                             f"bars under +4 dB while singing: {len(weak_b)} -> {len(weak_a)}"
                             + (f" ({', '.join(map(str, weak_a))})" if weak_a else "")))

    ref = info.get("reference")
    if ref:
        from drydock.reference import tone_distance_db

        curve = {int(k): v for k, v in ref["curve"].items()}
        ref["tone_distance_db"]["master"] = tone_distance_db(measure.third_octaves(written), curve)
        d = ref["tone_distance_db"]
        checks.append(_check("tone vs reference", d["master"] <= 2.5 or d["master"] <= d["source"] - 1.0,
                             f"1/3-octave distance to {ref['file']}: source {d['source']} dB -> master {d['master']} dB"))
    notes = []
    if meta.get("user_stems"):
        by_role: dict[str, int] = {}
        for s_ in meta["stems"]:
            by_role[s_["role"]] = by_role.get(s_["role"], 0) + 1
        notes.append("Session stems, no separation: " + ", ".join(f"{r} {n}" for r, n in sorted(by_role.items()))
                     + " file(s). 'Original' in comparisons is the unmastered sum of the stems.")
        lossy = [s_["file"] for s_ in meta["stems"] if (s_.get("codec") or "") in LOSSY]
        if lossy:
            notes.append(f"Lossy stems ({', '.join(lossy)}); export WAV from the session for full quality.")
    elif (meta.get("codec") or "") in LOSSY:
        kbps = f" at {meta['bit_rate'] // 1000} kbps" if meta.get("bit_rate") else ""
        notes.append(f"Source is {meta['codec'].upper()}{kbps}; a 24-bit WAV bounce (or stems) would master cleaner.")
    if src_lv["samples_over_full_scale"]:
        notes.append(f"The bounce peaks at {src_lv['true_peak_dbtp']:+.1f} dBTP: it clips on playback.")
    cutoff = measure.content_cutoff_hz(source)
    if cutoff < 17000:
        notes.append(f"Little above ~{cutoff / 1000:.1f} kHz (encoder low-pass); the top octave cannot be restored.")
    if src_st["correlation"] > 0.95:
        notes.append(f"The mix is near mono (L/R correlation {src_st['correlation']}).")

    ears = ["Does the drop still hit as hard as in the bounce?"]
    if stems is not None and not meta.get("user_stems"):
        ears.append("Separation artifacts: watery vocal edges, swirly hats, a ghost vocal in the sides.")
    if settings.width:
        ears.append("Width on keys/pads: natural or phasey on headphones?")
    if settings.octave_up:
        ears.append("808 on a big system vs a phone: fundamentals are slightly lower, the octave is louder.")
    if vocal and settings.ride_target_db:
        ears.append("Vocal ride: any phrase that now jumps out, or a breath lifted? Vocal level is a taste call.")
    vp = info.get("vocal_production")
    if vp:
        for n in vp["notes_fixed"]:
            ears.append(f"Re-pitched note at {int(n['time_s'] // 60)}:{n['time_s'] % 60:04.1f} (bar {n['bar']}, "
                        f"{n['cents']:+.0f} cents): natural, or a pitch-shift artifact?")
        ears.append(f"Backing/ad-libs sit {vp['backing_level_vs_lead_db']:+.1f} dB vs the lead, wider: do they still read?")
        br = vp.get("breaths") or {}
        if br.get("count"):
            first = ", ".join(f"{int(s // 60)}:{s % 60:04.1f}" for s in br["times_s"][:4])
            ears.append(f"{br['count']} breath(s) turned down {br['cut_db']:g} dB (e.g. {first}): any word start or end "
                        "clipped, or a reverb tail that now stops short?")
        if vp["production"] == "produced":
            ears.append(f"Hook doubles on {len(vp['hook_lines'])} line(s) and delay throws on {vp['throws']} line ending(s): "
                        "too much, too little?")
    return {"source": {**meta, **src_lv, "tone": measure.tone(source), "stereo": src_st, "content_cutoff_hz": cutoff},
            "master": {**mst_lv, "tone": measure.tone(written), "stereo": mst_st},
            "sections": {"names": [s for _, _, s in spans], "source_lufs": sec_src, "master_lufs": sec_mst,
                         "balance_shift_lu": drift},
            "song": {"bpm": analysis.get("bpm"), "key": (analysis.get("key") or {}).get("name"),
                     "bars": grid.bar_count},
            "vocal": vocal, "checks": checks, "source_notes": notes, "needs_ears": ears,
            "decisions": {"settings": settings.to_dict(), "why": why}, "processing": info}


def add_file_checks(r: dict[str, Any], files: dict[str, Any]) -> dict[str, Any]:
    """Checks that need the written files: peaks after lossy encoding and the phone file size."""
    encoded = {}
    for kind in ("mp3", "m4a"):
        y = deliver.decode_file(Path(files[kind]))
        encoded[kind] = {"lufs": round(dsp.lufs(y), 2), "true_peak_dbtp": round(dsp.true_peak_db(y), 2)}
    worst = max(e["true_peak_dbtp"] for e in encoded.values())
    trims = {k: v for k, v in (files.get("_share_trim_db") or {}).items() if v}
    r["checks"].insert(2, _check("peaks after MP3/AAC encoding", worst <= -0.3, f"worst {worst} dBTP"
                                 + (f" (share files turned down {', '.join(f'{k} {v} dB' for k, v in trims.items())})"
                                    if trims else "")))
    size = Path(files["mp4"]).stat().st_size
    r["checks"].append(_check("phone share file size", size <= deliver.PHONE_LIMIT_BYTES, f"mp4 {size / 2 ** 20:.1f} MiB"))
    r["master"]["encoded"] = encoded
    r["files"] = {k: Path(v).name for k, v in files.items() if not k.startswith("_")}
    r["share_trim_db"] = files.get("_share_trim_db", {})
    return r


def markdown(r: dict[str, Any], title: str) -> str:
    s, m = r["source"], r["master"]
    sec = r["sections"]
    lines = [f"# {title}: master report" + (f" ({r['variant']})" if r.get("variant") else ""), "",
             f"{r['song']['bpm']} BPM, {r['song']['key']}, {r['song']['bars']} bars. Source: `{s['file']}` "
             + (f"({len(s['stems'])} session stems)." if s.get("user_stems") else f"({s.get('codec')}, {s.get('sample_rate')} Hz).")
             , "",
             "| | source | master |", "|---|---|---|",
             f"| Integrated loudness | {s['lufs']} LUFS | {m['lufs']} LUFS |",
             f"| True peak | {s['true_peak_dbtp']:+.2f} dBTP | {m['true_peak_dbtp']:+.2f} dBTP |",
             f"| Peak-to-loudness | {s['plr_db']} dB | {m['plr_db']} dB |",
             f"| Loudness range | {s['lra_lu']} LU | {m['lra_lu']} LU |",
             f"| Stereo (side vs mid above 300 Hz) | {s['stereo']['side_vs_mid_above_300_db']} dB | {m['stereo']['side_vs_mid_above_300_db']} dB |"]
    if r["vocal"]:
        v = r["vocal"]
        lines.append(f"| Vocal over the beat, 1-4 kHz (median while singing) | {v['median_before']:+.1f} dB | {v['median_after']:+.1f} dB |")
    lines += ["", "## Checks", ""]
    lines += [f"- **{c['status'].upper()}** {c['check']}: {c['detail']}" for c in r["checks"]]
    lines += ["", "## Sections (LUFS)", "", "| section | source | master | balance shift |", "|---|---|---|---|"]
    lines += [f"| {n} | {a} | {b} | {d:+.1f} |" for n, a, b, d in zip(sec["names"], sec["source_lufs"], sec["master_lufs"], sec["balance_shift_lu"])]
    lines += ["", "## Decisions", ""]
    lines += [f"- `{k}` = {r['decisions']['settings'][k]}: {why}" for k, why in r["decisions"]["why"].items()]
    ref = r["processing"].get("reference")
    if ref:
        moved = [f"{b} Hz {g:+.1f}" for b, g in ref["match_eq_db"].items() if abs(g) >= 0.5]
        lines += ["", "## Reference", "",
                  f"- `{ref['file']}`: {ref['lufs']} LUFS"
                  + (f", vocal {ref['vocal_median_db']:+.1f} dB over its beat" if ref["vocal_median_db"] is not None else ""),
                  f"- Tone distance (1/3 octave, 40 Hz-12 kHz): source {ref['tone_distance_db']['source']} dB -> "
                  f"master {ref['tone_distance_db'].get('master')} dB",
                  "- Match EQ (bounded to 3 dB, no boosts above the source's content or below 30 Hz): "
                  + (", ".join(moved) if moved else "under 0.5 dB everywhere")]
    vp = r["processing"].get("vocal_production")
    if vp:
        lines += ["", "## Vocal production", "",
                  f"- Lead and backing treated separately (backing {vp['backing_level_vs_lead_db']:+.1f} dB vs lead, widened).",
                  f"- Notes re-pitched: {len(vp['notes_fixed'])}"
                  + "".join(f"\n  - {int(n['time_s'] // 60)}:{n['time_s'] % 60:04.1f} bar {n['bar']}: {n['cents']:+.0f} cents, "
                            f"{n['seconds']} s" for n in vp["notes_fixed"]),
                  (f"- Breaths turned down {vp['breaths']['cut_db']:g} dB: {vp['breaths']['count']}"
                   if (vp.get("breaths") or {}).get("count") else "- Breaths: kept"),
                  f"- Production: {vp['production']}"
                  + (f" (hook doubles on {len(vp['hook_lines'])} line(s), delay throws on {vp['throws']} line ending(s), "
                     "plate reverb)" if vp["production"] == "produced" else " (no added doubles or effects)")]
    if r["source_notes"]:
        lines += ["", "## Source notes", ""] + [f"- {n}" for n in r["source_notes"]]
    lines += ["", "## Needs ears", ""] + [f"- {e}" for e in r["needs_ears"]]
    lines += ["", "## Files", ""] + [f"- {k}: `{v}`" for k, v in r["files"].items()]
    return "\n".join(lines) + "\n"


def chart(path: Path, source: np.ndarray, master: np.ndarray, spans, r: dict[str, Any]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surf, t2, grid_c = "#fcfcfb", "#52514e", "#e4e3df"
    c_src, c_mst = "#eb6834", "#2a78d6"
    plt.rcParams.update({"font.family": ["Helvetica", "DejaVu Sans"], "font.size": 9, "axes.edgecolor": grid_c,
                         "axes.labelcolor": t2, "xtick.color": t2, "ytick.color": t2})
    panels = 3 if r["vocal"] else 2
    fig, ax = plt.subplots(1, panels, figsize=(5.2 * panels, 4.2), facecolor=surf)
    for a in ax:
        a.set_facecolor(surf)
        a.grid(True, color=grid_c, lw=0.6)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    for x, col, lab in ((source, c_src, "Source"), (master, c_mst, "Master")):
        curve = measure.third_octaves(x)
        ax[0].plot(list(curve), list(curve.values()), color=col, lw=2, label=lab)
    ax[0].set_xscale("log")
    ax[0].set_xlim(25, 16000)
    ax[0].set_xticks([30, 60, 120, 250, 500, 1000, 2000, 4000, 8000, 16000])
    ax[0].set_xticklabels(["30", "60", "120", "250", "500", "1k", "2k", "4k", "8k", "16k"])
    ax[0].set_ylabel("dB re 1 kHz (1/3 octave, loud sections)")
    ax[0].set_title("Tone", loc="left")
    ax[0].legend(frameon=False)
    sec = r["sections"]
    y = np.arange(len(sec["names"]))[::-1]
    for yi, a_, b_ in zip(y, sec["source_lufs"], sec["master_lufs"]):
        ax[1].plot([a_, b_], [yi, yi], color=grid_c, lw=3, zorder=1)
    ax[1].scatter(sec["source_lufs"], y, s=50, color=c_src, edgecolor=surf, lw=1.5, zorder=3, label="Source")
    ax[1].scatter(sec["master_lufs"], y, s=50, color=c_mst, edgecolor=surf, lw=1.5, zorder=3, label="Master")
    ax[1].set_yticks(y)
    ax[1].set_yticklabels(sec["names"])
    ax[1].set_xlabel("section loudness (LUFS)")
    ax[1].set_title("Arrangement dynamics", loc="left")
    ax[1].legend(frameon=False, loc="lower left")
    if r["vocal"]:
        v = r["vocal"]
        ax[2].axhspan(-30, 4, color="#f3f2ee", zorder=0)
        for d, col, lab in ((v["before"], c_src, "Source"), (v["after"], c_mst, "Master")):
            xs = sorted(d)
            ax[2].plot(xs, [d[k] for k in xs], color=col, lw=1.6, marker="o", ms=3, label=lab, zorder=3)
        allv = list(v["before"].values()) + list(v["after"].values())
        ax[2].set_ylim(min(allv) - 1, max(allv) + 2)
        ax[2].set_xlabel("bar")
        ax[2].set_ylabel("vocal over the rest, 1-4 kHz (dB)")
        ax[2].set_title("Vocal audibility (shaded: easy to lose)", loc="left")
        ax[2].legend(frameon=False)
    plt.tight_layout()
    fig.savefig(path, dpi=110, facecolor=surf)
    plt.close(fig)
    return path
