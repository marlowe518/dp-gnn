"""Subgraph utilities.

Includes:
- `extract_subgraph`: batch-level subgraph extraction (placeholder).
- `get_subgraphs`: padded per-node subgraph indices (reference train.get_subgraphs).
"""

from .extract import extract_subgraph, get_subgraphs

__all__ = ["extract_subgraph", "get_subgraphs"]
