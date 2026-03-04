"""DP optimizer step interface. Stub only.

Will clip per-sample gradients and add Gaussian noise (DP-SGD / DP-Adam).
Target: reference repo DP optimizer. Paper: Alg. 4, gradient clipping, noise scale.
"""

from __future__ import annotations

from typing import Any


def dp_step(
    gradients: Any,
    clip_norm: float,
    noise_multiplier: float,
    batch_size: int,
    **kwargs: Any,
) -> Any:
    """One DP gradient step (clip + noise). TODO: implement; return noisy gradients.

    Target: reference repo DP step. Paper: Alg. 4 (clipping + noise), Theorem 1 (noise scale).
    """
    # TODO: implement clip and add noise; no implementation in skeleton
    raise NotImplementedError("DP step not implemented yet (skeleton only)")
