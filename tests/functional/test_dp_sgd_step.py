"""Placeholder test: DP-SGD step. Asserts imports and CLI flags; no logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.dp import dp_step  # noqa: E402


def test_dp_sgd_step_stub():
    """Import and stub exist; real implementation TODO."""
    assert callable(dp_step)
    print("NOT IMPLEMENTED YET")


if __name__ == "__main__":
    test_dp_sgd_step_stub()
    sys.exit(0)
