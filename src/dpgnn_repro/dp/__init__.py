"""DP-SGD / DP-Adam placeholders. No implementation.

Target: reference repo DP optimizer. Paper: Alg. 4 (DP-SGD), clipping and noise.
"""

from .optimizer import dp_step

__all__ = ["dp_step"]
