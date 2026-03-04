"""Placeholder test: privacy accounting. Asserts imports and CLI flags; no logic."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.accounting import get_epsilon  # noqa: E402


def test_privacy_accounting_stub():
    """Import and stub exist; real implementation TODO."""
    assert callable(get_epsilon)
    print("NOT IMPLEMENTED YET")


if __name__ == "__main__":
    test_privacy_accounting_stub()
    sys.exit(0)
