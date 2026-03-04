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

from ..adjacency import compute_edge_weight

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


def make_subgraph_from_indices(
    data: Data,
    subgraph_indices: torch.Tensor,
    add_reverse_edges: bool,
    adjacency_normalization: str | None,
) -> Data:
    """PyG equivalent of train.make_subgraph_from_indices.

    Args:
      data: Full PyG graph with node features `x`.
      subgraph_indices: 1D LongTensor of node indices (in original graph),
        padded with _SUBGRAPH_PADDING_VALUE (-1) where absent.
      add_reverse_edges: Whether to add reverse edges (ignoring self-loops),
        as in the reference implementation.
      adjacency_normalization: One of {None, "none", "inverse-degree",
        "inverse-sqrt-degree"}.

    Returns:
      A PyG `Data` representing the subgraph/star around the root node, with:
        - x: [K+1, F] where K = len(subgraph_indices), last node is dummy.
        - edge_index: edges from root to valid indices, plus padding edges to
          dummy node (and optional reverse edges).
        - edge_weight: normalized edge weights according to adjacency_normalization.
    """
    if not torch.is_tensor(subgraph_indices):
        subgraph_indices = torch.as_tensor(subgraph_indices, dtype=torch.long)
    subgraph_indices = subgraph_indices.to(dtype=torch.long)

    device = data.x.device
    subgraph_indices = subgraph_indices.to(device)

    # Extract valid positions.
    valid_mask = subgraph_indices != _SUBGRAPH_PADDING_VALUE

    # Node features: gather, then zero out padding positions.
    gathered = data.x[subgraph_indices.clamp(min=0)]
    gathered = torch.where(
        valid_mask.unsqueeze(-1),
        gathered,
        torch.zeros_like(gathered),
    )

    # Add dummy padding node as last entry (index K).
    padding_node = subgraph_indices.numel()
    dummy = torch.zeros((1, gathered.size(1)), dtype=gathered.dtype, device=device)
    subgraph_nodes = torch.cat([gathered, dummy], dim=0)

    # Replace invalid indices with padding_node.
    idx_clipped = torch.where(
        valid_mask,
        subgraph_indices,
        torch.full_like(subgraph_indices, padding_node),
    )

    # Remap indices to within the subgraph: star graph from root (0).
    length = idx_clipped.numel()
    senders = torch.zeros(length, dtype=torch.int32, device=device)
    receivers = torch.arange(length, dtype=torch.int32, device=device)

    # Handle padding: edges for invalid positions point to/from dummy.
    senders = torch.where(
        valid_mask.to(torch.bool),
        senders,
        torch.full_like(senders, padding_node, dtype=torch.int32),
    )
    receivers = torch.where(
        valid_mask.to(torch.bool),
        receivers,
        torch.full_like(receivers, padding_node, dtype=torch.int32),
    )

    # Add reverse edges, ignoring self-loops (skip index 0).
    if add_reverse_edges:
        rev_senders = receivers[1:]
        rev_receivers = senders[1:]
        senders = torch.cat([senders, rev_senders], dim=0)
        receivers = torch.cat([receivers, rev_receivers], dim=0)

    edge_index = torch.stack([senders, receivers], dim=0)
    num_nodes_sub = subgraph_nodes.size(0)

    edge_weight = compute_edge_weight(
        edge_index=edge_index,
        num_nodes=num_nodes_sub,
        mode=adjacency_normalization,
    )

    sub_data = Data(x=subgraph_nodes, edge_index=edge_index)
    sub_data.edge_weight = edge_weight

    return sub_data


