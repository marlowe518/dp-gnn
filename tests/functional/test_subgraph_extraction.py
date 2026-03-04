"""Placeholder test: subgraph extraction. Asserts imports and CLI flags; no logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.subgraph import extract_subgraph  # noqa: E402


def test_subgraph_extraction_stub():
    """Import and stub exist; real implementation TODO."""
    assert callable(extract_subgraph)
    print("NOT IMPLEMENTED YET")


if __name__ == "__main__":
    test_subgraph_extraction_stub()
    sys.exit(0)
