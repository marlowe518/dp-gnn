"""Minimal logger for DP-GNN runs.

- Respects debug flag for verbosity.
- Can dump JSONL lines for metrics (stub; no real logging yet).
Target: match reference repo logging / metrics format where needed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any


def setup_logging(debug: bool = False, log_file: Path | None = None) -> logging.Logger:
    """Configure root logger; debug=True sets level to DEBUG."""
    log = logging.getLogger("dpgnn_repro")
    log.setLevel(logging.DEBUG if debug else logging.INFO)
    if not log.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.DEBUG if debug else logging.INFO)
        log.addHandler(h)
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG if debug else logging.INFO)
            log.addHandler(fh)
    return log


def log_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one JSON object as a line to path. Stub for metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
