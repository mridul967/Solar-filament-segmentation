"""Run-directory helpers; no model or Colab dependencies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def create_run_directory(runs_root: str | Path, name: str = "vanilla") -> Path:
    """Create a unique timestamped experiment directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(runs_root) / f"{timestamp}-{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "splits").mkdir()
    return run_dir


def write_json(path: str | Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
