"""Subgraph-related utilities.

This module provides:
- A placeholder `extract_subgraph` interface (for future batch-level subgraphs).
- A PyTorch/PyG-compatible implementation of `get_subgraphs` from the reference
  training pipeline, which builds padded neighbor index arrays per node.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

_SUBGRAPH_PADDING_VALUE = -1


def extract_subgraph(sampled_nodes: Any, edge_index: Any, **kwargs: Any) -> Any:
    """Extract subgraph for batch. TODO: implement; return subgraph (e.g. PyG Data).

    Target: reference repo subgraph module. Paper: same subgraph as in Alg.
    """
    # TODO: implement extraction; no implementation in skeleton
    raise NotImplementedError("Subgraph extraction not implemented yet (skeleton only)")


def get_subgraphs(data: Data, pad_to: int) -> torch.Tensor:
    """PyG equivalent of train.get_subgraphs: padded 1-hop subgraphs per node.

    For each node u:
      - Collect [u] + outgoing neighbors (excluding self-loops).
      - Deduplicate while preserving first-occurrence order (as in the reference:
        np.unique(..., return_index=True) + index sort trick).
      - Truncate to length `pad_to` if necessary.
      - Pad the remainder with _SUBGRAPH_PADDING_VALUE (-1).

    Args:
      data: PyG `Data` with at least `edge_index` (and optionally `num_nodes`).
      pad_to: Maximum subgraph length per node.

    Returns:
      A LongTensor of shape [num_nodes, pad_to] with node indices or -1 padding.
    """
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    src, dst = data.edge_index

    # Build outgoing adjacency, ignoring self-loops.
    outgoing = {u: [] for u in range(num_nodes)}
    for u, v in zip(src.tolist(), dst.tolist()):
        if u != v:
            outgoing[u].append(v)

    subgraphs = torch.full(
        (num_nodes, pad_to),
        _SUBGRAPH_PADDING_VALUE,
        dtype=torch.long,
        device=data.edge_index.device,
    )

    for node in range(num_nodes):
        neighbors = outgoing[node]
        sub_idx = [node] + neighbors  # root first
        arr = np.asarray(sub_idx, dtype=np.int64)

        # Deduplicate while preserving first occurrence order.
        _, first_idx = np.unique(arr, return_index=True)
        arr = arr[np.sort(first_idx)]

        # Truncate and pad.
        arr = arr[:pad_to]
        k = arr.shape[0]
        if k > 0:
            subgraphs[node, :k] = torch.as_tensor(arr, dtype=torch.long, device=subgraphs.device)

    return subgraphs

