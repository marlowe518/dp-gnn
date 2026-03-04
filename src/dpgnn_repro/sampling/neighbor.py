"""Neighbor sampling interface. Stub for Alg. 1–3.

Will perform k-hop neighbor sampling for minibatches.
Target: reference repo neighbor sampler. Paper: Alg. 1 (and related Alg. 2, 3).
"""

from __future__ import annotations

from typing import Any


def neighbor_sample(
    node_ids: Any,
    num_hops: int,
    num_neighbors: int,
    edge_index: Any,
    **kwargs: Any,
) -> Any:
    """Sample neighbors for given nodes. TODO: implement; return (subset of) edge_index / adj.

    Target: reference repo sampling. Paper: Alg. 1 (neighbor sampling), Alg. 2–3 as needed.
    """
    # TODO: implement sampling logic; no implementation in skeleton
    raise NotImplementedError("Neighbor sampling not implemented yet (skeleton only)")
