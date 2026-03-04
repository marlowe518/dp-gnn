"""Degree-bounded subsampling utilities (PyG).

This module implements a PyTorch/PyG-compatible equivalent of the reference
`differentially_private_gnns.input_pipeline.subsample_graph` behavior:
it applies an in-degree constraint from training nodes while leaving
non-training nodes' adjacency unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
from torch_geometric.data import Data


AdjacencyDict = Dict[int, List[int]]


def get_adjacency_lists_pyg(data: Data) -> AdjacencyDict:
    """Build adjacency lists from a PyG `Data` object (like sampler.get_adjacency_lists)."""
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    edges: AdjacencyDict = {u: [] for u in range(num_nodes)}
    src, dst = data.edge_index
    for u, v in zip(src.tolist(), dst.tolist()):
        edges[u].append(v)
    return edges


def _reverse_edges(edges: AdjacencyDict) -> AdjacencyDict:
    """Reverse adjacency: v -> [u ...] where u->v exists."""
    reversed_edges: AdjacencyDict = {u: [] for u in edges}
    for u, nbrs in edges.items():
        for v in nbrs:
            reversed_edges[v].append(u)
    return reversed_edges


def sample_adjacency_lists_pyg(
    edges: AdjacencyDict,
    train_nodes: Sequence[int],
    max_degree: int,
    base_seed: int,
) -> AdjacencyDict:
    """Sample adjacency lists with in-degree constraints (NumPy version of sampler.sample_adjacency_lists).

    Args:
      edges: dict mapping node -> list of outgoing neighbors.
      train_nodes: sequence of training node ids.
      max_degree: in-degree bound from training nodes.
      base_seed: global integer seed; per-node seeds are derived from it.
    """
    train_nodes_set = set(int(t) for t in train_nodes)
    all_nodes = list(edges.keys())

    reversed_edges = _reverse_edges(edges)
    sampled_reversed_edges: AdjacencyDict = {u: [] for u in all_nodes}

    dropped_count = 0

    for u in all_nodes:
        incoming_edges = reversed_edges[u]
        incoming_train_edges = [v for v in incoming_edges if v in train_nodes_set]
        if not incoming_train_edges:
            continue

        in_degree = len(incoming_train_edges)
        # Emulate jax.random.fold_in(rng, u) by deriving a per-node seed.
        node_seed = (base_seed + int(u)) & 0x7FFFFFFF
        rng = np.random.default_rng(node_seed)

        sampling_prob = max_degree / (2.0 * in_degree)
        mask = rng.uniform(size=in_degree) <= sampling_prob
        incoming_sel = np.asarray(incoming_train_edges, dtype=np.int64)[mask]
        unique_incoming = np.unique(incoming_sel)

        # Enforce in-degree bound; otherwise drop this node.
        if len(unique_incoming) <= max_degree:
            sampled_reversed_edges[u] = unique_incoming.tolist()
        else:
            dropped_count += 1

    print("dropped count", dropped_count)
    sampled_edges: AdjacencyDict = _reverse_edges(sampled_reversed_edges) # Convert sampled incoming-edge view back into a standard outgoing adjacency list:

    # For non-train nodes, keep full adjacency. Transductive setup.
    for u in all_nodes:
        if u not in train_nodes_set:
            sampled_edges[u] = edges[u]

    return sampled_edges


def subsample_graph_pyg(
    data: Data,
    max_degree: int,
    train_nodes: torch.Tensor,
    base_seed: int,
) -> Data:
    """Apply degree-bounded subsampling to a PyG graph (equivalent to input_pipeline.subsample_graph).

    Args:
      data: Full `Data` graph with `edge_index` and node features.
      max_degree: in-degree bound from training nodes.
      train_nodes: 1D tensor of training node indices.
      base_seed: global integer seed used to derive per-node RNGs.

    Returns:
      A new `Data` with:
        - same node set and x
        - `edge_index` replaced by the sampled edge list.
    """
    edges = get_adjacency_lists_pyg(data)
    sampled_edges = sample_adjacency_lists_pyg(
        edges, train_nodes.tolist(), max_degree, base_seed
    )

    senders: List[int] = []
    receivers: List[int] = []
    for u, nbrs in sampled_edges.items():
        for v in nbrs:
            senders.append(u)
            receivers.append(v)

    edge_index_sub = torch.tensor(
        [senders, receivers],
        dtype=torch.long,
        device=data.edge_index.device,
    )

    new_data = data.clone()
    new_data.edge_index = edge_index_sub
    return new_data

