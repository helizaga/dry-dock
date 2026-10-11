---
name: mastering
description: Engineer and master a finished song from a bounce with the drydock CLI, read its QA, and iterate on the listener's feedback. Use when the user asks to master, mix, engineer, clean up, or make release-ready a song they (or a collaborator) made.
---

# Mastering a finished song

The CLI makes every audio decision from measurements and records why. Your job: run it, read `qa.md`, explain the
result in plain terms, and turn listening notes into `--set` overrides.

## 1. Source
- Best: stems exported from the session (every track or bus from the session start, WAV). Then a 24-bit WAV
  bounce. An MP3 works, but say what it costs (no top octave, lossy grit, separation artifacts).
- Never commit or upload the audio. Outputs stay in gitignored `runs/`.

## 2. Run
```sh
.venv/bin/drydock master "<bounce>" --slug <name>
.venv/bin/drydock master --stems "<folder>" --slug <name>     # session stems: check the role each file got
```
With `--stems`, read the `file -> role` lines in the output; fix a wrong guess with `--role "File.wav=bass"`.
Stems are cached in `work/<slug>/`; re-runs with new `--set` values take about a minute.
Use `--mode 2track` when the user hears separation artifacts (watery vocal edges, swirly hats).

## 3. Read qa.md
- Every check should pass. A `warn` needs a reason or a fix before hand-off; a `fail` blocks it.
- Decisions list each auto move and why (vocal forwardness, 808 fundamental, width, hat hump, ceiling).
- Source notes: lossy source, bounce peaks over 0 dBFS, encoder low-pass, near-mono mix. Pass these on.

## 4. Hand off
- Send the `.mp3` or `.mp4` for phones (the `.mp4` plays inline anywhere), the `.wav`/`.flac` for distribution, and
  the `compare-at-14-lufs/` pair for a fair A/B.
- One line on what changed, one on what QA found, one on what needs ears (from "Needs ears").
- Loudness: the master is as loud as the song allows before the limiter rides every hit. Streaming plays
  everything at about -14 LUFS, so louder files only lose punch there. A louder version means clipping the 808 or
  the whole mix; offer it as a separate choice, never silently.

## 5. Iterate on ears
The listener runs `drydock review <slug>` (or says it in chat; record it with `drydock comment <slug> "..." --at m:ss
--tag <tag>`). Then:
1. `drydock feedback <slug>`: open comments with bar, section, and the settings that run used.
2. Re-run `drydock master` on the same source and slug with `--set` changes from the table below.
3. `drydock resolve <slug> <ids> --run <old run> --note "what changed, new run id"`. The new run appears in the
   review page next to the old one for an A/B.

| The listener says | Try |
|---|---|
| vocal too loud / too quiet | `--set vocal_lift_db=-1` / `--set ride_target_db=8` |
| vocal harsh / sibilant | `--set vocal_presence_db=1 --set deess_max_db=7` |
| keys or snare disappear under the vocal | `--set duck_other_db=1.5 --set duck_drums_db=0` |
| 808 thin on big speakers | `--set octave_up=false` or `--set sub_trim_db=0` |
| 808 missing on phones | `--set bass_db=2` |
| too wide / phasey | `--set width=0` |
| hats harsh | `--set hat_cut_db=2.5` |
| needs to be louder | `--set target_lufs=-9` (QA will warn about limiter load; say so) |
| drop does not hit | check the sections table; `--set section_rides=true` is the default |

Pitch problems (flat held notes) and arrangement changes belong in the session (Logic Flex Pitch, Melodyne), not here.
