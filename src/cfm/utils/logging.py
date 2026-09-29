"""Per-round adaptation logging -> CSV, plus a final JSON summary. Kept
dependency-light (just `csv`/`json`) so it's trivial to unit test."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class RoundLogger:
    def __init__(self):
        self.rows: list[dict[str, Any]] = []

    def log(self, **fields: Any) -> None:
        self.rows.append(fields)

    def to_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            path.write_text("")
            return
        fieldnames = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

    def write_summary(self, path: str | Path, summary: dict[str, Any]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2))
