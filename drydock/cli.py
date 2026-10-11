"""drydock: mix repair and mastering for a finished song."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drydock import engineer, paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="drydock", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("master", help="separate, fix, master, and QA a song")
    m.add_argument("audio", type=Path, help="the bounce (WAV/AIFF/FLAC preferred; MP3/M4A work)")
    m.add_argument("--slug", help="name for work/ and runs/ (default: from the file name)")
    m.add_argument("--mode", choices=["stems", "2track"], default="stems",
                   help="stems: fix each element (default); 2track: stereo file only, no separation artifacts")
    m.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a setting (see `drydock settings`); repeatable")
    sub.add_parser("settings", help="list every setting and its default ('None' = decided from the song)")
    args = p.parse_args(argv)

    if args.cmd == "settings":
        print(json.dumps(engineer.Settings().to_dict(), indent=1))
        return 0
    if not args.audio.exists():
        print(f"no such file: {args.audio}", file=sys.stderr)
        return 2
    settings = engineer.Settings(mode=args.mode).override(args.overrides)
    slug = args.slug or paths.slugify(args.audio.stem)
    from drydock.master import run

    out = run(args.audio, slug, settings)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
