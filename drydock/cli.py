"""drydock: mix repair and mastering for a finished song."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from drydock import engineer, paths, review


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="drydock", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("master", help="separate, fix, master, and QA a song")
    m.add_argument("audio", type=Path, nargs="?", help="the bounce (WAV/AIFF/FLAC preferred; MP3/M4A work)")
    m.add_argument("--stems", type=Path, metavar="FOLDER",
                   help="a folder of stems exported from the session: used as they are, no separation")
    m.add_argument("--role", dest="roles", action="append", default=[], metavar="FILE=ROLE",
                   help="with --stems: override the role guessed from a file name (vocals, drums, bass, other)")
    m.add_argument("--slug", help="name for work/ and runs/ (default: from the file name)")
    m.add_argument("--mode", choices=["stems", "2track"], default="stems",
                   help="stems: fix each element (default); 2track: stereo file only, no separation artifacts")
    m.add_argument("--tag", help="label for this version in file names and the video (default: built from --set)")
    m.add_argument("--set", dest="overrides", action="append", default=[], metavar="NAME=VALUE",
                   help="override a setting (see `drydock settings`); repeatable")
    sub.add_parser("settings", help="list every setting and its default ('None' = decided from the song)")
    rv = sub.add_parser("review", help="listening page: A/B masters at matched loudness, comments pinned to time")
    rv.add_argument("slug")
    rv.add_argument("--port", type=int, default=8766)
    rv.add_argument("--no-open", action="store_true", help="do not open the browser")
    cm = sub.add_parser("comment", help="add a comment from chat, e.g. drydock comment <slug> 'vocal too loud' --at 1:12")
    cm.add_argument("slug")
    cm.add_argument("text")
    cm.add_argument("--at", required=True, help="m:ss or seconds")
    cm.add_argument("--to", help="end of the range (m:ss or seconds)")
    cm.add_argument("--tag", default="other", choices=review.TAGS)
    cm.add_argument("--run", help="run id (default: latest; 'original' for the source)")
    fb = sub.add_parser("feedback", help="open comments on the latest master, with bar, section, and its settings")
    fb.add_argument("slug")
    fb.add_argument("--run")
    fb.add_argument("--json", action="store_true")
    rs = sub.add_parser("resolve", help="close comments: drydock resolve <slug> <ids...> --note 'what changed'")
    rs.add_argument("slug")
    rs.add_argument("ids", nargs="+")
    rs.add_argument("--note", default="")
    rs.add_argument("--wontfix", action="store_true")
    rs.add_argument("--run")
    args = p.parse_args(argv)

    if args.cmd in ("review", "comment", "feedback", "resolve"):
        try:
            return _listening(args)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            print(f"drydock: {exc}", file=sys.stderr)
            return 2

    if args.cmd == "settings":
        print(json.dumps(engineer.Settings().to_dict(), indent=1))
        return 0
    if (args.audio is None) == (args.stems is None):
        print("drydock: give a bounce or --stems FOLDER (one of them)", file=sys.stderr)
        return 2
    src = args.stems or args.audio
    if not src.exists() or (args.stems is not None and not src.is_dir()):
        print(f"drydock: no such {'folder' if args.stems else 'file'}: {src}", file=sys.stderr)
        return 2
    slug = args.slug or paths.slugify(src.stem if src.is_file() else src.name)
    from drydock.master import run

    try:
        roles = dict(r.split("=", 1) for r in args.roles if "=" in r)
        if len(roles) != len(args.roles):
            raise ValueError("--role takes FILE=ROLE, for example --role 'Keys 2.wav=other'")
        settings = engineer.Settings(mode=args.mode).override(args.overrides).validate()
        out = run(src, slug, settings, roles=roles, tag=args.tag)
    except ValueError as exc:
        print(f"drydock: {exc}", file=sys.stderr)
        return 2
    print(out)
    return 0


def _seconds(text: str) -> float:
    parts = [float(p) for p in str(text).split(":")]
    total = 0.0
    for p in parts:
        total = total * 60 + p
    return total


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


def _listening(args: argparse.Namespace) -> int:
    if args.cmd == "review":
        review.latest(args.slug)  # fail early when there is nothing to review
        server = review.serve(args.slug, args.port)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"review page for {args.slug}: {url}  (Ctrl-C to stop)", flush=True)
        if not args.no_open:
            import webbrowser

            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return 0
    if args.cmd == "comment":
        name = args.run or review.latest(args.slug).name
        c = review.add_comment(args.slug, name, time_s=_seconds(args.at), end_s=_seconds(args.to) if args.to else None,
                               tag=args.tag, text=args.text)
        print(f"[{c['id']}] {name} {_clock(c['time_s'])} bar {c['bar']} ({', '.join(c['clips'])}) {c['tag']}: {c['text']}")
        return 0
    if args.cmd == "feedback":
        name, comments, verdict = review.open_comments(args.slug, args.run)
        settings = {}
        if name != review.ORIGINAL:
            settings = json.loads((review.run_dir(args.slug, name) / "settings.json").read_text())
        if args.json:
            print(json.dumps({"run": name, "comments": comments, "verdict": verdict, "settings": settings}, indent=1))
            return 0
        print(f"run {name}: {len(comments)} open comment(s)" + (f"; verdict {verdict['verdict']}" if verdict else ""))
        for c in comments:
            span = _clock(c["time_s"]) + (f"-{_clock(c['end_s'])}" if c["end_s"] > c["time_s"] else "")
            print(f"  [{c['id']}] {span}  bar {c['bar']}  {', '.join(c['clips'])}  {c['tag']}: {c['text']}")
        if comments and settings:
            print("settings used: " + ", ".join(f"{k}={v}" for k, v in settings.items()))
            print("revise with: drydock master <same source> --slug " + args.slug + " --set name=value ...")
        return 0
    name = args.run or review.latest(args.slug).name
    for cid in args.ids:
        c = review.set_status(args.slug, name, cid, "wontfix" if args.wontfix else "resolved", args.note)
        print(f"[{c['id']}] {c['status']}: {c['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
