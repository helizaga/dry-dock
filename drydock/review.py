"""Listening loop: a local page (glass-harbor's review page) that A/Bs masters at matched loudness and takes
comments pinned to time ranges; the agent reads them with ``drydock feedback`` and closes them with ``resolve``.

Every run in the page plays its -14 LUFS copy, and "original" plays the source at the same level, so louder never
wins the comparison. Comments are stored next to each run (``feedback.json``) with the bar and section they hit.
"""

from __future__ import annotations

import json
import re
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import numpy as np
import soundfile as sf

from drydock import paths

TAGS = ("vocal", "808", "drums", "level", "tone", "width", "dynamics", "artifact", "keep", "other")
PLACEHOLDER = ("What should change here? e.g. vocal too quiet in the hook, 808 too loud on phones, hats harsh, "
               "drop hits softer than the bounce")
ORIGINAL = "original"


def list_runs(slug: str) -> list[Path]:
    root = paths.runs_dir(slug)
    return sorted(p for p in root.iterdir() if (p / "qa.json").exists()) if root.exists() else []


def run_dir(slug: str, name: str) -> Path:
    """A run folder, or the song's work folder for "original" (its comments live there)."""
    if name == ORIGINAL:
        return paths.work_dir(slug)
    if not re.fullmatch(r"[0-9A-Za-z_\-]+", name):
        raise ValueError("bad run id")
    for run in list_runs(slug):
        if run.name == name:
            return run
    raise FileNotFoundError(f"no run {name} for {slug}")


def latest(slug: str) -> Path:
    runs = list_runs(slug)
    if not runs:
        raise FileNotFoundError(f"no runs for {slug}; run drydock master first")
    return runs[-1]


def audio_path(slug: str, name: str) -> Path:
    run = latest(slug) if name == ORIGINAL else run_dir(slug, name)
    files = json.loads((run / "qa.json").read_text())["files"]
    return run / "compare-at-14-lufs" / files["compare_source" if name == ORIGINAL else "compare_master"]


def timeline(slug: str, name: str) -> dict[str, Any]:
    """Bar start times and section spans (shown as the page's lanes) for a run's audio."""
    from glass_harbor.grid import Grid

    run = latest(slug) if name == ORIGINAL else run_dir(slug, name)
    analysis = json.loads((run / "analysis.json").read_text())  # each run keeps its own copy
    grid = Grid.from_dict(analysis["grid"])
    duration = sf.info(str(audio_path(slug, name))).duration
    bars = [{"bar": b, "t": round(grid.bar_time(b), 4)} for b in range(1, grid.bar_count + 2)
            if 0 <= grid.bar_time(b) <= duration]
    sections = analysis.get("sections") or []
    lanes = []
    for i, s in enumerate(sections):
        start = 0.0 if i == 0 else grid.bar_time(s["start_bar"])
        end = duration if i == len(sections) - 1 else grid.bar_time(s["start_bar"] + s["bars"])
        lanes.append({"id": f"bars {s['start_bar']}-{s['start_bar'] + s['bars'] - 1}", "track": s.get("energy", ""),
                      "stem": "", "start": round(start, 3), "end": round(min(end, duration), 3)})
    qa = json.loads((run / "qa.json").read_text())
    variant = "original" if name == ORIGINAL else qa.get("variant") or "master"
    title = f"{Path(qa['source']['file']).stem} ({variant})"
    key = qa["song"].get("key") or ""
    return {"duration": duration, "bars": bars, "clips": lanes, "title": title,
            "tempo": f"{qa['song']['bpm']:.0f} BPM {key}".strip()}


def peaks(slug: str, name: str, buckets: int = 2400) -> list[float]:
    folder = run_dir(slug, name)
    cache = folder / ("peaks-original.json" if name == ORIGINAL else "peaks.json")
    if cache.exists():
        return json.loads(cache.read_text())
    y, _ = sf.read(audio_path(slug, name), dtype="float32", always_2d=True)
    m = np.abs(y).max(axis=1)
    edges = np.linspace(0, m.size, buckets + 1).astype(int)
    data = [round(float(m[a:b].max()) if b > a else 0.0, 4) for a, b in zip(edges[:-1], edges[1:])]
    cache.write_text(json.dumps(data))
    return data


# ---------------------------------------------------------------- comments (glass-harbor's storage format)


def load(slug: str, name: str) -> dict[str, Any]:
    from glass_harbor import feedback

    return feedback.load_feedback(run_dir(slug, name))


def add_comment(slug: str, name: str, *, time_s: float, text: str, tag: str = "other", end_s: float | None = None) -> dict:
    import datetime as dt
    import uuid

    from glass_harbor import feedback

    if tag not in TAGS:
        raise ValueError(f"tag must be one of {', '.join(TAGS)}")
    if not text.strip():
        raise ValueError("comment text is empty")
    tl = timeline(slug, name)
    end = end_s if end_s is not None and end_s > time_s else time_s
    comment = {"id": uuid.uuid4().hex[:8], "time_s": round(time_s, 3), "end_s": round(end, 3),
               "bar": round(feedback.bar_at(tl, time_s), 2), "end_bar": round(feedback.bar_at(tl, end), 2),
               "clips": [c["id"] for c in tl["clips"] if c["start"] <= end and c["end"] >= time_s],
               "tag": tag, "text": text.strip(), "status": "open",
               "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    folder = run_dir(slug, name)
    data = feedback.load_feedback(folder)
    data["comments"].append(comment)
    feedback.save_feedback(folder, data)
    return comment


def set_status(slug: str, name: str, comment_id: str, status: str, note: str = "") -> dict:
    from glass_harbor import feedback

    return feedback.set_comment_status(run_dir(slug, name), comment_id, status, note)


def delete_comment(slug: str, name: str, comment_id: str) -> None:
    from glass_harbor import feedback

    feedback.delete_comment(run_dir(slug, name), comment_id)


def set_verdict(slug: str, name: str, verdict: str, note: str = "") -> dict:
    import datetime as dt

    from glass_harbor import feedback

    if verdict not in ("approve", "reject"):
        raise ValueError("verdict must be approve or reject")
    folder = run_dir(slug, name)
    data = feedback.load_feedback(folder)
    data["verdict"] = {"verdict": verdict, "note": note,
                       "at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    feedback.save_feedback(folder, data)
    return data["verdict"]


def open_comments(slug: str, name: str | None = None) -> tuple[str, list[dict[str, Any]], dict[str, Any] | None]:
    """(run name, open comments, verdict) for a run (default: the latest)."""
    name = name or latest(slug).name
    data = load(slug, name)
    return name, [c for c in data["comments"] if c["status"] == "open"], data.get("verdict")


# ---------------------------------------------------------------- server


def make_handler(slug: str) -> type[BaseHTTPRequestHandler]:
    page = resources.files("glass_harbor").joinpath("review.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # quiet
            return

        def _json(self, payload: Any, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, error: Exception) -> None:
            status = (HTTPStatus.FORBIDDEN if isinstance(error, PermissionError) else
                      HTTPStatus.NOT_FOUND if isinstance(error, (FileNotFoundError, KeyError)) else HTTPStatus.BAD_REQUEST)
            self._json({"error": str(error)}, status)

        def _check_host(self) -> None:
            """Only this local server may be addressed (blocks DNS-rebinding pages in the browser)."""
            port = self.server.server_address[1]
            if self.headers.get("Host", "") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                raise PermissionError("unexpected Host header")

        def _body(self) -> dict[str, Any]:
            # JSON only: a JSON content type forces a CORS preflight this server never answers, so other sites
            # open in the browser cannot post here.
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise PermissionError("requests must be application/json")
            return json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0") or 0)) or b"{}")

        def _audio(self, path: Path) -> None:
            size = path.stat().st_size
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                self.send_response(HTTPStatus.PARTIAL_CONTENT)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "audio/flac")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with path.open("rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    left -= len(chunk)

        def do_GET(self) -> None:  # noqa: N802
            path = unquote(self.path.split("?", 1)[0])
            try:
                self._check_host()
                if path in ("/", "/index.html"):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(page)))
                    self.end_headers()
                    self.wfile.write(page)
                elif path == "/api/session":
                    runs = [r.name for r in list_runs(slug)]
                    if not runs:
                        raise FileNotFoundError(f"no runs for {slug}; run drydock master first")
                    self._json({"slug": slug, "runs": [ORIGINAL, *runs], "tags": list(TAGS), "placeholder": PLACEHOLDER})
                elif m := re.fullmatch(r"/api/run/([^/]+)", path):
                    self._json({"timeline": timeline(slug, m.group(1)), "peaks": peaks(slug, m.group(1)),
                                "feedback": load(slug, m.group(1))})
                elif m := re.fullmatch(r"/audio/([^/]+)", path):
                    self._audio(audio_path(slug, m.group(1)))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:  # surface errors to the page
                self._error(error)

        def do_POST(self) -> None:  # noqa: N802
            path = unquote(self.path)
            try:
                self._check_host()
                body = self._body()
                if m := re.fullmatch(r"/api/run/([^/]+)/comments", path):
                    self._json(add_comment(slug, m.group(1), time_s=float(body["time_s"]), end_s=body.get("end_s"),
                                           tag=body.get("tag", "other"), text=str(body.get("text", ""))))
                elif m := re.fullmatch(r"/api/run/([^/]+)/comments/([0-9a-f]+)", path):
                    self._json(set_status(slug, m.group(1), m.group(2), body.get("status", "resolved"), body.get("note", "")))
                elif m := re.fullmatch(r"/api/run/([^/]+)/verdict", path):
                    self._json(set_verdict(slug, m.group(1), body.get("verdict", ""), body.get("note", "")))
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._error(error)

        def do_DELETE(self) -> None:  # noqa: N802
            try:
                self._check_host()
                if m := re.fullmatch(r"/api/run/([^/]+)/comments/([0-9a-f]+)", unquote(self.path)):
                    delete_comment(slug, m.group(1), m.group(2))
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            except Exception as error:
                self._error(error)

    return Handler


def serve(slug: str, port: int = 8766) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(slug))
