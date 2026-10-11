"""Workspace layout (all gitignored: songs are private audio).

- ``work/<slug>/``: decoded source, analysis, stems (cached by the source file's hash)
- ``runs/<slug>/<stamp>/``: masters, share files, QA report

The root is the repo by default, or ``DRYDOCK_HOME``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def root() -> Path:
    explicit = os.environ.get("DRYDOCK_HOME", "").strip()
    return Path(explicit).expanduser().resolve() if explicit else REPO_ROOT


def work_dir(slug: str) -> Path:
    return root() / "work" / slug


def runs_dir(slug: str) -> Path:
    return root() / "runs" / slug


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)[:64].rstrip("-") or "song"
