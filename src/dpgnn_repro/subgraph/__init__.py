"""Subgraph utilities.

Includes:
- `extract_subgraph`: batch-level subgraph extraction (placeholder).
- `get_subgraphs`: padded per-node subgraph indices (reference train.get_subgraphs).
- `make_subgraph_from_indices`: PyG equivalent of train.make_subgraph_from_indices.
"""

from .extract import extract_subgraph, get_subgraphs, make_subgraph_from_indices

__all__ = ["extract_subgraph", "get_subgraphs", "make_subgraph_from_indices"]
