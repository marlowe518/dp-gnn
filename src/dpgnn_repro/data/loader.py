"""Dataset loader interface. Stub only; no real dataset download.

Will load graph dataset (e.g. ogbn-arxiv) and return PyG Data or equivalent.
Target: reference repo dataset loading. Paper: dataset and split description.
"""

from __future__ import annotations

from typing import Any

from ..config import Config


def load_dataset(cfg: Config) -> Any:
    """Load dataset for cfg.dataset. TODO: implement; return graph + train/val/test masks.

    Target: reference repo data module. Paper: same splits and features.
    """
    # TODO: implement ogbn-arxiv (and others) load; no download in skeleton
    raise NotImplementedError("Dataset loader not implemented yet (skeleton only)")
