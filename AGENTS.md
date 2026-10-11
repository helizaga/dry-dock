# dry-dock Agent Guide

## Purpose
- Mix repair and mastering for finished original songs (the user's or a collaborator's bounce).
- The agent runs the pipeline, reads QA, and hands off; the `drydock` CLI does every audio step deterministically.
- Mashups and DJ transitions belong in the sibling repo glass-harbor, which dry-dock depends on for its engine
  (beat grid, sections, stem models, spectral ducking). Fix shared DSP upstream there, not by copying it here.
- Workflow and craft rules: `skills/mastering/SKILL.md`.

## Ground rules
- Work only from audio the user provides. Never download music.
- Never commit audio, stems, analysis, or reports: `work/` and `runs/` are gitignored. This repo is public; keep
  song names, artists, and personal paths out of committed files.
- Everything runs locally and free. Do not add paid services.
- You cannot listen. Report what QA verified, what it cannot (groove, tone taste, separation artifacts), and ask
  for ear feedback.
- Ask for a lossless bounce (24-bit WAV) or session stems when the source is an MP3.

## Commands
- `drydock master <audio> [--slug s] [--mode stems|2track] [--set name=value ...]`
- `drydock settings` (every knob; `None` means decided from the song)

## Working style
- If a judgment can be measured, encode it as a decision rule (`engineer.decide`) or a QA check (`qa.py`), with
  the reason recorded, not as prose.
- New knobs go in `engineer.Settings` with a default or an auto rule.
- Run `.venv/bin/python -m pytest` after code changes.
