"""Privacy accounting interface. Stub only.

Will compute (ε, δ) from steps, noise multiplier, batch size, delta.
Target: reference repo accounting. Paper: Theorem 1, RDP or moments accountant.
"""

from __future__ import annotations


def get_epsilon(
    steps: int,
    noise_multiplier: float,
    batch_size: int,
    target_delta: float,
    **kwargs: float,
) -> float:
    """Return ε for given (steps, noise, batch_size, delta). TODO: implement RDP/moments.

    Target: reference repo accounting. Paper: Theorem 1 (privacy guarantee).
    """
    # TODO: implement accounting; no implementation in skeleton
    raise NotImplementedError("Privacy accounting not implemented yet (skeleton only)")
