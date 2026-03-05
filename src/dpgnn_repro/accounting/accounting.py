"""Privacy accounting interface.

Delegates to dp.privacy_accounting (RDP) for (ε, δ) from steps, noise, batch.
Target: reference repo accounting. Paper: Theorem 1, RDP composition.
"""

from __future__ import annotations

from dpgnn_repro.dp.privacy_accounting import dpsgd_privacy_accountant


def get_epsilon(
    steps: int,
    noise_multiplier: float,
    batch_size: int,
    target_delta: float,
    num_samples: int | None = None,
    **kwargs: float,
) -> float:
    """Return ε for given steps, noise, batch_size, delta (single-term / MLP style).

    Uses RDP accountant with Poisson subsampling. Requires num_samples so that
    sampling_probability = batch_size / num_samples can be computed.

    Args:
        steps: Number of training steps.
        noise_multiplier: Noise multiplier.
        batch_size: Batch size.
        target_delta: Target delta.
        num_samples: Total number of training samples (required).
        **kwargs: Ignored (for API compatibility).

    Returns:
        Privacy parameter epsilon.

    Raises:
        ValueError: If num_samples is None or not positive.
    """
    if num_samples is None:
        raise ValueError(
            "get_epsilon requires num_samples (total training samples) "
            "for sampling_probability = batch_size / num_samples"
        )
    if num_samples <= 0:
        raise ValueError("num_samples must be positive, got %s" % num_samples)
    sampling_probability = batch_size / num_samples
    return dpsgd_privacy_accountant(
        num_training_steps=steps,
        noise_multiplier=noise_multiplier,
        target_delta=target_delta,
        sampling_probability=sampling_probability,
    )
