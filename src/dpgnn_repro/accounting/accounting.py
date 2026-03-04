"""Privacy accounting: (ε, δ) from steps, noise multiplier, batch size, delta.

Implements RDP-based accounting matching the reference. Multi-term uses
numerically stable RDP orders (orders >= 1.5) to avoid epsilon blow-up.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.special
import scipy.stats

try:
    import dp_accounting
except ImportError:
    dp_accounting = None  # type: ignore[assignment]


def _multiterm_dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    num_samples: int,
    batch_size: int,
    max_terms_per_node: int,
) -> float:
    """Multi-term DP-SGD accountant (GCN). Stable RDP orders from 1.5 to avoid blow-up."""
    if dp_accounting is None:
        raise ImportError("pip install dp-accounting")
    if noise_multiplier < 1e-20:
        return np.inf

    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)]

    orders = np.arange(1.5, 10.0, 0.1)

    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier))
    unamplified_rdps = accountant._rdp  # pylint: disable=protected-access

    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node)
        )
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (order - 1)
        amplified_rdps.append(amplified_rdp)

    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(
        unamplified_rdps * (batch_size / num_samples) ** 2 <= amplified_rdps + 1e-6
    ):
        raise ValueError("Lower bound violated in multi-term accountant.")

    amplified_rdps_total = amplified_rdps * num_training_steps
    result = dp_accounting.rdp.compute_epsilon(
        orders, amplified_rdps_total, target_delta
    )
    return result[0] if isinstance(result, (list, tuple)) else float(result)


def _dpsgd_privacy_accountant(
    num_training_steps: int,
    noise_multiplier: float,
    target_delta: float,
    sampling_probability: float,
) -> float:
    """Standard DP-SGD accountant (single term, Poisson sampling)."""
    if dp_accounting is None:
        raise ImportError("pip install dp-accounting")
    if noise_multiplier < 1e-20:
        return np.inf

    orders = np.arange(1, 200, 0.1)[1:]
    event = dp_accounting.PoissonSampledDpEvent(
        sampling_probability, dp_accounting.GaussianDpEvent(noise_multiplier)
    )
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(event, num_training_steps)
    return accountant.get_epsilon(target_delta)


def get_epsilon(
    steps: int,
    noise_multiplier: float,
    batch_size: int,
    target_delta: float,
    *,
    num_samples: int | None = None,
    max_terms_per_node: int = 1,
    use_multiterm: bool = True,
    **_: Any,
) -> float:
    """Return ε for (steps, noise_multiplier, batch_size, target_delta).

    For GCN: use_multiterm=True, pass num_samples and max_terms_per_node.
    For MLP: use_multiterm=False, pass num_samples for sampling_probability.
    """
    if use_multiterm and num_samples is not None:
        return _multiterm_dpsgd_privacy_accountant(
            num_training_steps=steps,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            num_samples=num_samples,
            batch_size=batch_size,
            max_terms_per_node=max_terms_per_node,
        )
    if num_samples is not None:
        return _dpsgd_privacy_accountant(
            num_training_steps=steps,
            noise_multiplier=noise_multiplier,
            target_delta=target_delta,
            sampling_probability=batch_size / num_samples,
        )
    if dp_accounting is None:
        raise ImportError("pip install dp-accounting")
    orders = np.arange(1.5, 200, 0.1)
    accountant = dp_accounting.rdp.RdpAccountant(orders)
    accountant.compose(dp_accounting.GaussianDpEvent(noise_multiplier), steps)
    return accountant.get_epsilon(target_delta)
