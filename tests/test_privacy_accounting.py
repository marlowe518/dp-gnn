"""Tests for FU9: Privacy accounting (dp.privacy_accounting).

Verifies: dpsgd_privacy_accountant, multiterm_dpsgd_privacy_accountant,
get_training_privacy_accountant. No dataset downloads; synthetic inputs only.

Requires: dp-accounting (pip install dp-accounting) and scipy.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

# Skip entire module if dp_accounting not installed (e.g. minimal env)
pytest.importorskip("dp_accounting")
import numpy as np

# Ensure src is on path for imports (pytest pythonpath should do this; fallback for direct run)
if "src" not in sys.path:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from dpgnn_repro.dp.privacy_accounting import (
    dpsgd_privacy_accountant,
    get_training_privacy_accountant,
    multiterm_dpsgd_privacy_accountant,
)


# ---- Happy path ----

def test_dpsgd_privacy_accountant_happy_path():
    """Happy path: single-term (MLP-style) accountant returns finite epsilon."""
    eps = dpsgd_privacy_accountant(
        num_training_steps=10,
        noise_multiplier=1.0,
        target_delta=1e-5,
        sampling_probability=0.01,
    )
    assert isinstance(eps, (float, np.floating))
    assert np.isfinite(eps)
    assert eps >= 0.0
    # More steps -> larger epsilon (monotonic)
    eps_more = dpsgd_privacy_accountant(
        num_training_steps=100,
        noise_multiplier=1.0,
        target_delta=1e-5,
        sampling_probability=0.01,
    )
    assert eps_more >= eps


def test_multiterm_dpsgd_privacy_accountant_happy_path():
    """Happy path: multi-term (GCN-style) accountant returns finite epsilon."""
    eps = multiterm_dpsgd_privacy_accountant(
        num_training_steps=10,
        noise_multiplier=1.0,
        target_delta=1e-5,
        num_samples=1000,
        batch_size=100,
        max_terms_per_node=5,
    )
    assert isinstance(eps, (float, np.floating))
    assert np.isfinite(eps)
    assert eps >= 0.0


def test_get_training_privacy_accountant_mlp_returns_callable():
    """Happy path: get_training_privacy_accountant(mlp) returns callable that returns float."""
    config = SimpleNamespace(
        model="mlp",
        differentially_private_training=True,
        training_noise_multiplier=1.0,
        batch_size=50,
    )
    accountant = get_training_privacy_accountant(
        config, num_training_nodes=500, max_terms_per_node=1
    )
    assert callable(accountant)
    eps = accountant(10)
    assert isinstance(eps, (float, np.floating))
    assert np.isfinite(eps)
    assert eps >= 0.0


def test_get_training_privacy_accountant_gcn_returns_callable():
    """Happy path: get_training_privacy_accountant(gcn) returns callable."""
    config = SimpleNamespace(
        model="gcn",
        differentially_private_training=True,
        training_noise_multiplier=1.0,
        batch_size=100,
    )
    accountant = get_training_privacy_accountant(
        config, num_training_nodes=1000, max_terms_per_node=6
    )
    assert callable(accountant)
    eps = accountant(5)
    assert isinstance(eps, (float, np.floating))
    assert np.isfinite(eps)
    assert eps >= 0.0


def test_get_training_privacy_accountant_non_dp_returns_zero():
    """When DP is disabled, accountant always returns 0."""
    config = SimpleNamespace(
        model="mlp",
        differentially_private_training=False,
        training_noise_multiplier=1.0,
        batch_size=50,
    )
    accountant = get_training_privacy_accountant(
        config, num_training_nodes=500, max_terms_per_node=1
    )
    assert accountant(0) == 0.0
    assert accountant(100) == 0.0


# ---- Edge cases ----

def test_dpsgd_low_noise_multiplier_returns_inf():
    """Edge case: very small noise_multiplier returns np.inf (reference behavior)."""
    eps = dpsgd_privacy_accountant(
        num_training_steps=10,
        noise_multiplier=0.0,
        target_delta=1e-5,
        sampling_probability=0.1,
    )
    assert eps == np.inf

    eps_tiny = dpsgd_privacy_accountant(
        num_training_steps=10,
        noise_multiplier=1e-25,
        target_delta=1e-5,
        sampling_probability=0.1,
    )
    assert eps_tiny == np.inf


def test_multiterm_low_noise_multiplier_returns_inf():
    """Edge case: very small noise_multiplier returns np.inf."""
    eps = multiterm_dpsgd_privacy_accountant(
        num_training_steps=10,
        noise_multiplier=0.0,
        target_delta=1e-5,
        num_samples=1000,
        batch_size=100,
        max_terms_per_node=5,
    )
    assert eps == np.inf


def test_multiterm_vs_standard_ordering():
    """Edge case: for same params with max_terms_per_node=1, multiterm >= standard (reference test)."""
    num_training_steps = 10
    noise_multiplier = 1.0
    target_delta = 1e-5
    num_samples = 1000
    batch_size = 100
    sampling_prob = batch_size / num_samples

    standard_eps = dpsgd_privacy_accountant(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sampling_probability=sampling_prob,
    )
    multiterm_eps = multiterm_dpsgd_privacy_accountant(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        num_samples=num_samples,
        batch_size=batch_size,
        max_terms_per_node=1,
    )
    print(standard_eps, multiterm_eps)
    assert standard_eps <= multiterm_eps + 1e-6  # allow small numerical difference


def test_zero_steps():
    """Edge case: zero steps gives zero or very small epsilon."""
    eps = dpsgd_privacy_accountant(
        num_training_steps=0,
        noise_multiplier=1.0,
        target_delta=1e-5,
        sampling_probability=0.1,
    )
    assert eps >= 0.0
    assert np.isfinite(eps)


# ---- Failure cases ----

def test_dpsgd_negative_steps_raises():
    """Failure: negative num_training_steps raises ValueError."""
    with pytest.raises(ValueError, match="num_training_steps must be non-negative"):
        dpsgd_privacy_accountant(
            num_training_steps=-1,
            noise_multiplier=1.0,
            target_delta=1e-5,
            sampling_probability=0.1,
        )


def test_dpsgd_invalid_sampling_probability_raises():
    """Failure: sampling_probability outside (0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="sampling_probability"):
        dpsgd_privacy_accountant(
            num_training_steps=10,
            noise_multiplier=1.0,
            target_delta=1e-5,
            sampling_probability=1.5,
        )
    with pytest.raises(ValueError, match="sampling_probability"):
        dpsgd_privacy_accountant(
            num_training_steps=10,
            noise_multiplier=1.0,
            target_delta=1e-5,
            sampling_probability=0.0,
        )


def test_multiterm_invalid_max_terms_raises():
    """Failure: max_terms_per_node < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_terms_per_node"):
        multiterm_dpsgd_privacy_accountant(
            num_training_steps=10,
            noise_multiplier=1.0,
            target_delta=1e-5,
            num_samples=1000,
            batch_size=100,
            max_terms_per_node=0,
        )


def test_get_training_privacy_accountant_unknown_model_raises():
    """Failure: config.model not 'mlp' or 'gcn' raises ValueError."""
    config = SimpleNamespace(
        model="unknown",
        differentially_private_training=True,
        training_noise_multiplier=1.0,
        batch_size=50,
    )
    with pytest.raises(ValueError, match="Could not create privacy accountant for model"):
        get_training_privacy_accountant(
            config, num_training_nodes=500, max_terms_per_node=1
        )


def test_get_training_privacy_accountant_missing_attrs_raises():
    """Failure: config missing required attributes raises ValueError."""
    config = SimpleNamespace(model="mlp", differentially_private_training=True)
    # Missing training_noise_multiplier and batch_size
    with pytest.raises(ValueError, match="training_noise_multiplier|batch_size"):
        get_training_privacy_accountant(
            config, num_training_nodes=500, max_terms_per_node=1
        )


def test_get_training_privacy_accountant_zero_training_nodes_raises():
    """Failure: num_training_nodes <= 0 raises ValueError."""
    config = SimpleNamespace(
        model="mlp",
        differentially_private_training=True,
        training_noise_multiplier=1.0,
        batch_size=50,
    )
    with pytest.raises(ValueError, match="num_training_nodes must be positive"):
        get_training_privacy_accountant(
            config, num_training_nodes=0, max_terms_per_node=1
        )


# ---- Reference-reproduced tests (ref: privacy_accountants_test.py) ----


def _get_privacy_accountant(training_type: str):
    """Return the accountant function for the given training type (reference API)."""
    if training_type in ("sgd", "adam"):
        return dpsgd_privacy_accountant
    if training_type in ("multiterm-sgd", "multiterm-adam"):
        return multiterm_dpsgd_privacy_accountant
    raise ValueError("Unsupported training_type: %s" % training_type)


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("num_training_steps", [1, 10])
@pytest.mark.parametrize("noise_multiplier", [1])
@pytest.mark.parametrize("target_delta", [1e-5])
@pytest.mark.parametrize("batch_size", [10, 20])
@pytest.mark.parametrize("num_samples", [1000, 2000])
@pytest.mark.parametrize("max_terms_per_node", [1])
def test_ref_multiterm_vs_standard_dpsgd(
    optimizer,
    num_training_steps,
    noise_multiplier,
    target_delta,
    batch_size,
    num_samples,
    max_terms_per_node,
):
    """Reference: standard epsilon <= multiterm epsilon (same params, max_terms=1)."""
    multiterm_fn = _get_privacy_accountant("multiterm-" + optimizer)
    multiterm_epsilon = multiterm_fn(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        num_samples=num_samples,
        batch_size=batch_size,
        max_terms_per_node=max_terms_per_node,
    )
    standard_fn = _get_privacy_accountant(optimizer)
    standard_epsilon = standard_fn(
        num_training_steps=num_training_steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sampling_probability=batch_size / num_samples,
    )
    assert standard_epsilon <= multiterm_epsilon


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("noise_multiplier", [-1, 0])
def test_ref_low_noise_multiplier_multiterm_dpsgd(optimizer, noise_multiplier):
    """Reference: multiterm with low/zero noise returns np.inf."""
    privacy_accountant = _get_privacy_accountant("multiterm-" + optimizer)
    eps = privacy_accountant(
        num_training_steps=10,
        noise_multiplier=noise_multiplier,
        target_delta=1e-5,
        num_samples=1000,
        batch_size=10,
        max_terms_per_node=1,
    )
    assert eps == np.inf


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
@pytest.mark.parametrize("noise_multiplier", [-1, 0])
def test_ref_low_noise_multiplier_dpsgd(optimizer, noise_multiplier):
    """Reference: standard (single-term) with low/zero noise returns np.inf."""
    privacy_accountant = _get_privacy_accountant(optimizer)
    eps = privacy_accountant(
        num_training_steps=10,
        noise_multiplier=noise_multiplier,
        target_delta=1e-5,
        sampling_probability=0.1,
    )
    assert eps == np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-q"])
