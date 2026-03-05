"""Privacy accounting: get_epsilon and DP accountant integration."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

pytest.importorskip("dp_accounting")
import numpy as np

from dpgnn_repro.accounting import get_epsilon  # noqa: E402


def test_get_epsilon_callable():
    """get_epsilon is callable (API exists)."""
    assert callable(get_epsilon)


def test_get_epsilon_returns_finite_epsilon():
    """get_epsilon(steps, noise, batch_size, delta, num_samples) returns finite float >= 0."""
    eps = get_epsilon(
        steps=10,
        noise_multiplier=1.0,
        batch_size=50,
        target_delta=1e-5,
        num_samples=500,
    )
    assert isinstance(eps, (float, np.floating))
    assert eps >= 0.0
    assert np.isfinite(eps)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
