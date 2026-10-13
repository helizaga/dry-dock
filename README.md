# dry-dock

Mix repair and mastering for finished songs, built for an AI agent to run and a person to judge by ear.

Give it a bounce. It splits the song into stems, decides per-element fixes from measurements, masters to the
loudest level the song can take without the limiter riding every hit, keeps the arrangement's dynamics, writes
files for distribution and for phones, and reports what it checked and what still needs ears.

It shares its engine with [glass-harbor](https://github.com/helizaga/glass-harbor) (beat grid, sections, stem
models, spectral ducking). glass-harbor makes mashups and DJ transitions; dry-dock finishes original songs.

## Setup

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
brew install ffmpeg
```

This installs glass-harbor from GitHub at the commit pinned in `pyproject.toml`, so work in progress in a
glass-harbor checkout never changes dry-dock's behavior. To develop both together, swap in the checkout with
`.venv/bin/pip install --no-deps -e ../glass-harbor` (and move back with `pip install -e .`). First use downloads
the stem models (a few GB) into `~/.cache/glass-harbor/models`.

## Use

```sh
.venv/bin/drydock master "path/to/bounce.wav"                 # split into stems, fix each element, master
.venv/bin/drydock master --stems "path/to/Stems Folder"       # session stems: no separation, best quality
.venv/bin/drydock master "path/to/bounce.wav" --mode 2track   # stereo file only (no separation artifacts)
.venv/bin/drydock master song.wav --set ride_target_db=7 --set width=0
.venv/bin/drydock master song.wav --reference "a released track.flac"   # match its tone and vocal balance
.venv/bin/drydock master --slug <slug> --set ...              # re-run from the cache (the original file can be gone)
.venv/bin/drydock settings                                    # every knob and its default
```

`--reference` measures a track you want to sound like: its tonal balance is matched with a smooth EQ (at most
3 dB per band, shape only, no boosts above the source's content or below 30 Hz), and the vocal level moves toward
its vocal-over-beat balance, but never so far that more than two extra bars of singing get buried. Loudness still
follows the limiter budget. The reference's vocal is separated once and cached.

With `--stems`, each file gets a role from its name (`808.wav` -> bass, `Adlibs.wav` -> vocals, `Hats.wav` ->
drums, anything else -> other); files sharing a role are summed. Fix a guess with `--role "Keys 2.wav=vocals"`.
Export every stem from the session start so they line up.

## Listen and comment

```sh
.venv/bin/drydock review <slug>                               # local page: A/B masters, comments pinned to time
.venv/bin/drydock comment <slug> "vocal too quiet" --at 1:12 --tag vocal   # or leave it from chat
.venv/bin/drydock feedback <slug>                             # open comments with bar, section, settings used
.venv/bin/drydock resolve <slug> <ids> --note "ride target 8"
```

The page (glass-harbor's review page) plays every master and the original at -14 LUFS, so the A/B is fair. A new
`drydock master` run shows up in the page next to the previous one.

Each run writes `runs/<slug>/<stamp>/`:

| file | for |
|---|---|
| `<title> - master.wav` / `.flac` | distribution (44.1 kHz / 24-bit) |
| `<title> - master.mp3` / `.m4a` | listening and sharing |
| `<title> - master.mp4` | a waveform video that plays inline in Messages, WhatsApp, Instagram DMs |
| `compare-at-14-lufs/` | the original and the master at streaming loudness, for a fair A/B |
| `qa.md`, `qa.json`, `before-after.png` | checks, decisions and why, what needs ears |

## What it does

- **Stems** (glass-harbor's "best" vocal ensemble, or `--set separation=balanced` for one faster model; HT-Demucs
  drums/bass/other). The residual (mix minus stems) is added back untouched.
- **Vocal**: de-box, presence, de-ess, compression at its own level, spike limiter, and a boost-only ride wherever
  it sits under the beat in 1-4 kHz. If the vocal is buried, the melodic layer and drums dip in the vocal's
  frequency bins while it sings. A vocal that is already forward gets no extra lift; an instrumental (no lead
  vocal in the stems) skips the vocal moves.
- **Vocal production** (stems mode): the lead and the backing/ad-libs are split and treated apart (backing thinner,
  wider, steadier); held notes 25-60 cents off are re-pitched one by one (in key only, formants kept, everything
  else untouched; pitch from FCPE, a neural tracker for singing, with pYIN as the fallback). `--set breath_cut_db=15` turns breaths between words down (found from Whisper's word timings
  and checked to be noise, not a note or a tail). `--set production=produced` adds detuned doubles on the repeated lyric lines (hooks, from local
  Whisper), a light plate reverb, and dotted-eighth delay throws on line endings.
- **Drums**: hat/ess hump cut when 6-9 kHz runs hot, transient soft clip.
- **808**: when the bass line sits below 45 Hz (phones stop there), an octave-up copy, a trimmed fundamental, a
  little upper harmonics, and ducking under the kick.
- **Width**: decorrelated side on the melodic layer only when the mix is near mono (the mono sum is unchanged).
- **Master**: glue compression, mono lows, soft clip on the low band only (clipping the full mix distorts the
  vocal), a 4x-oversampled true-peak limiter at -1 dBTP (-2 if an AAC transcode would clip), and the loudest level
  whose limiter stays within budget in the beat sections. Then per-section cuts so the intro, breaks and outro sit
  where they did in the bounce.

## Speed

A 3-minute song takes about 15 minutes the first time (best-quality separation, the lead/backing split, and
Whisper on Apple Silicon; about 4 with `separation=balanced`) and about 30 seconds per
re-run after that: stems and analysis are cached, the four stem chains and the three lossy encodes run in
parallel, and QA measures the master while it encodes.

## Limits

It cannot listen. QA measures loudness, peaks, contrast, limiter load, distortion, mono, and vocal audibility; it
cannot judge groove, tone taste, or separation artifacts. A lossless bounce or session stems beat an MP3. Inputs
under 10 seconds or silent files are rejected with a message.

Development: `.venv/bin/python -m pytest`.
