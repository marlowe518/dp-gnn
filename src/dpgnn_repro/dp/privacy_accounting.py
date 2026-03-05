"""Privacy accountants for DP-SGD/DP-Adam (node-level DP-GNN).

Matches reference: reference_repo/differentially_private_gnns/privacy_accountants.py
Paper: Node-Level Differentially Private Graph Neural Networks (arXiv:2111.15521).
"""

from __future__ import annotations

import functools
from typing import Any, Callable

import dp_accounting
import numpy as np
import scipy.special
import scipy.stats


def multiterm_dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    batch_size: int,
    max_terms_per_node: int,
) -> float:
    """Compute epsilon after a given number of training steps with DP-SGD/Adam.

    Accounts for the exact distribution of terms in a minibatch, assuming
    sampling without replacement (hypergeometric). Used for GCN where one node
    can affect multiple gradient terms.

    Returns np.inf if the noise multiplier is too small (< 1e-20).

    Args:
        num_training_steps: Number of training steps.
        noise_multiplier: Noise multiplier that scales the sensitivity.
        target_delta: Privacy parameter delta to choose epsilon for.
        num_samples: Total number of samples (training nodes) in the dataset.
        batch_size: Size of every batch.
        max_terms_per_node: Maximum number of gradient terms affected by the
            removal of one node.

    Returns:
        Privacy parameter epsilon (float).

    Raises:
        ValueError: If the RDP lower bound is violated (internal sanity check).
    """
    if noise_multiplier < 1e-20:
        return np.inf

    if num_training_steps < 0:
        raise ValueError(
            "num_training_steps must be non-negative, got %s" % num_training_steps
        )
    if num_samples <= 0 or batch_size <= 0:
        raise ValueError(
            "num_samples and batch_size must be positive, got num_samples=%s batch_size=%s"
            % (num_samples, batch_size)
        )
    if max_terms_per_node < 1:
        raise ValueError(
            "max_terms_per_node must be at least 1, got %s" % max_terms_per_node
        )
    if not 0 < target_delta < 1:
        raise ValueError("target_delta must be in (0, 1), got %s" % target_delta)

    # Hypergeometric: population num_samples, max_terms_per_node "successes" per node,
    # batch_size draws; count how many of the node's terms are in the batch.
    terms_rv = scipy.stats.hypergeom(
        num_samples, max_terms_per_node, batch_size
    )
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    # Unamplified RDP (sampling probability = 1).
    orders = np.arange(1, 10, 0.1)[1:]
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
    unamplified_rdps = accountant._rdp  # noqa: SLF001

    # Amplified RDP for each (order, unamplified RDP) pair.
    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node)
        )
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
            order - 1
        )
        amplified_rdps.append(amplified_rdp)

    amplified_rdps = np.asarray(amplified_rdps)
    lower_bound = unamplified_rdps * (batch_size / num_samples) ** 2
    if not np.all(lower_bound <= amplified_rdps + 1e-6):
        raise ValueError(
            "The lower bound has been violated. Something is wrong."
        )

    amplified_rdps_total = amplified_rdps * num_training_steps
    return dp_accounting.rdp.compute_epsilon(
        orders, amplified_rdps_total, target_delta
    )[0]


def dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    sampling_probability: float,
) -> float:
    """Compute epsilon after a given number of training steps with DP-SGD/Adam.

    Assumes a single affected term per node (e.g. MLP). Uses Poisson sampling
    for subsampling amplification.

    Returns np.inf if the noise multiplier is too small (< 1e-20).

    Args:
        num_training_steps: Number of training steps.
        noise_multiplier: Noise multiplier that scales the sensitivity.
        target_delta: Privacy parameter delta to choose epsilon for.
        sampling_probability: Probability that a single sample is in a batch.
            For uniform sampling without replacement: batch_size / num_samples.

    Returns:
        Privacy parameter epsilon (float).

    Raises:
        ValueError: If sampling_probability or target_delta are invalid.
    """
    if noise_multiplier < 1e-20:
        return np.inf

    if num_training_steps < 0:
        raise ValueError(
            "num_training_steps must be non-negative, got %s" % num_training_steps
        )
    if num_training_steps == 0:
        return 0.0
    if not 0 < sampling_probability <= 1:
        raise ValueError(
            "sampling_probability must be in (0, 1], got %s"
            % sampling_probability
        )
    if not 0 < target_delta < 1:
        raise ValueError("target_delta must be in (0, 1), got %s" % target_delta)

    orders = np.arange(1, 200, 0.1)[1:]
    event = dp_accounting.PoissonSampledDpEvent(
        sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(event, num_training_steps)
    return accountant.get_epsilon(target_delta)


def get_training_privacy_accountant(
    config: Any,
    num_training_nodes: int,
    max_terms_per_node: int,
) -> Callable[[int], float]:
    """Return a callable that computes DP epsilon for a given number of steps.

    The returned function takes one argument (num_training_steps: int) and
    returns the current epsilon (float). For non-DP training it always returns 0.

    Config must have (duck-typed):
        - model: str, one of "mlp", "gcn"
        - differentially_private_training: bool
        - training_noise_multiplier: float
        - batch_size: int

    Args:
        config: Object with model, differentially_private_training,
            training_noise_multiplier, batch_size.
        num_training_nodes: Total number of training nodes.
        max_terms_per_node: Max gradient terms per node (for GCN; ignored for MLP).

    Returns:
        Callable (num_training_steps: int) -> float.

    Raises:
        ValueError: If config.model is not "mlp" or "gcn", or required
            config attributes are missing.
    """
    try:
        dp_enabled = getattr(config, "differentially_private_training", None)
        model = getattr(config, "model", None)
    except (TypeError, AttributeError):
        raise ValueError(
            "config must support attribute access (e.g. config.model, "
            "config.differentially_private_training)"
        ) from None

    if dp_enabled is None or model is None:
        raise ValueError(
            "config must have 'differentially_private_training' and 'model'"
        )

    if not dp_enabled:
        return lambda num_training_steps: 0.0

    if num_training_nodes <= 0:
        raise ValueError(
            "num_training_nodes must be positive, got %s" % num_training_nodes
        )

    noise_multiplier = getattr(config, "training_noise_multiplier", None)
    batch_size = getattr(config, "batch_size", None)
    if noise_multiplier is None or batch_size is None:
        raise ValueError(
            "config must have 'training_noise_multiplier' and 'batch_size' "
            "when differentially_private_training is True"
        )

    target_delta = 1.0 / (10 * num_training_nodes)

    if model == "mlp":
        return functools.partial(
            dpsgd_privacy_accountant,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            sampling_probability=batch_size / num_training_nodes,
        )
    if model == "gcn":
        return functools.partial(
            multiterm_dpsgd_privacy_accountant,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            num_samples=num_training_nodes,
            batch_size=batch_size,
            max_terms_per_node=max_terms_per_node,
        )

    raise ValueError(
        "Could not create privacy accountant for model: %s." % model
    )
