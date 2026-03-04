"""Training loop stub. No algorithm logic.

Will run: load data -> sampling (Alg.1–3) -> subgraph -> forward -> DP step -> accounting.
Target: reference repo training script. Paper: full pipeline Alg. 1–5, Theorem 1.
"""

from __future__ import annotations

from .config import Config


def run(cfg: Config) -> None:
    """Stub: one training run. TODO: implement loop, dataloader, model, DP optimizer."""
    # TODO: load dataset via data.load_dataset(cfg)
    # TODO: for each batch: sampling.sample(...) -> subgraph.extract(...) -> model forward
    # TODO: dp.dp_step(...) with clipping and noise
    # TODO: accounting.update(...)
    pass
