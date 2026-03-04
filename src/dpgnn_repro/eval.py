"""Evaluation stub. No algorithm logic.

Will compute accuracy / metrics on val/test after training.
Target: reference repo eval. Paper: same metrics as reported.
"""

from __future__ import annotations

from .config import Config


def run(cfg: Config) -> None:
    """Stub: evaluate model. TODO: load checkpoint, run inference, report metrics."""
    # TODO: load best checkpoint from cfg.workdir
    # TODO: run model on val/test (with same sampling/subgraph as reference)
    # TODO: compute and log accuracy
    pass
