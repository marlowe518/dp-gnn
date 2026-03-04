"""Placeholder test: neighbor sampling (Alg. 1–3). Asserts imports and CLI flags; no logic."""

import sys
from pathlib import Path

# Ensure src is on path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.sampling import neighbor_sample  # noqa: E402


def test_neighbor_sampling_stub():
    """Import and stub exist; real implementation TODO."""
    # Would call neighbor_sample(...) once implemented
    assert callable(neighbor_sample)
    print("NOT IMPLEMENTED YET")


if __name__ == "__main__":
    test_neighbor_sampling_stub()
    sys.exit(0)
