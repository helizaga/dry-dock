# dry-dock

Mix repair and mastering for finished songs, built for an AI agent to run and a person to judge by ear.

Give it a bounce. It splits the song into stems, decides per-element fixes from measurements, masters to the
loudest level the song can take without the limiter riding every hit, keeps the arrangement's dynamics, writes
files for distribution and for phones, and reports what it checked and what still needs ears.

It shares its engine with [glass-harbor](https://github.com/helizaga/glass-harbor) (beat grid, sections, stem
models, spectral ducking). glass-harbor makes mashups and DJ transitions; dry-dock finishes original songs.

## Setup

```sh
git clone https://github.com/helizaga/glass-harbor ../glass-harbor   # sibling checkout
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/pip install -e '../glass-harbor[ml]'
.venv/bin/pip install --no-deps -e . && .venv/bin/pip install 'matplotlib>=3.8' 'pytest>=8'
brew install ffmpeg
```

A plain `pip install -e .` also works: it pulls glass-harbor from GitHub at the pinned commit. First use downloads
the stem models (a few GB) into `~/.cache/glass-harbor/models`.

## Use

```sh
.venv/bin/drydock master "path/to/bounce.wav"                 # stems mode: fix each element, then master
.venv/bin/drydock master "path/to/bounce.wav" --mode 2track   # stereo file only (no separation artifacts)
.venv/bin/drydock master song.wav --set ride_target_db=7 --set width=0
.venv/bin/drydock settings                                    # every knob and its default
```

Each run writes `runs/<slug>/<stamp>/`:

| file | for |
|---|---|
| `<title> - master.wav` / `.flac` | distribution (44.1 kHz / 24-bit) |
| `<title> - master.mp3` / `.m4a` | listening and sharing |
| `<title> - master.mp4` | a waveform video that plays inline in Messages, WhatsApp, Instagram DMs |
| `compare-at-14-lufs/` | the original and the master at streaming loudness, for a fair A/B |
| `qa.md`, `qa.json`, `before-after.png` | checks, decisions and why, what needs ears |

## What it does

- **Stems** (BS-RoFormer vocals, HT-Demucs drums/bass/other). The residual (mix minus stems) is added back untouched.
- **Vocal**: de-box, presence, de-ess, compression at its own level, spike limiter, and a boost-only ride wherever
  it sits under the beat in 1-4 kHz. If the vocal is buried, the melodic layer and drums dip in the vocal's
  frequency bins while it sings. A vocal that is already forward gets no extra lift; an instrumental (no lead
  vocal in the stems) skips the vocal moves.
- **Drums**: hat/ess hump cut when 6-9 kHz runs hot, transient soft clip.
- **808**: when the bass line sits below 45 Hz (phones stop there), an octave-up copy, a trimmed fundamental, a
  little upper harmonics, and ducking under the kick.
- **Width**: decorrelated side on the melodic layer only when the mix is near mono (the mono sum is unchanged).
- **Master**: glue compression, mono lows, soft clip on the low band only (clipping the full mix distorts the
  vocal), a 4x-oversampled true-peak limiter at -1 dBTP (-2 if an AAC transcode would clip), and the loudest level
  whose limiter stays within budget in the beat sections. Then per-section cuts so the intro, breaks and outro sit
  where they did in the bounce.

## Speed

A 3-minute song takes about 4 minutes the first time (stem separation on Apple Silicon) and about 30 seconds per
re-run after that: stems and analysis are cached, the four stem chains and the three lossy encodes run in
parallel, and QA measures the master while it encodes.

## Limits

Inputs under 10 seconds or silent files are rejected with a message, not a traceback. It cannot listen. QA measures loudness, peaks, contrast, limiter load, distortion, mono, and vocal audibility; it
cannot judge groove, tone taste, or separation artifacts. A lossless bounce or session stems beat an MP3.

Development: `.venv/bin/python -m pytest`.
