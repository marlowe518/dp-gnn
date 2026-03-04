"""Neighbor / subgraph sampling (Alg. 1–3).

This package contains PyTorch/PyG-compatible sampling utilities that mirror the
reference repository's behavior where feasible.
"""

from .neighbor import neighbor_sample
from .subsample import subsample_graph_pyg

__all__ = ["neighbor_sample", "subsample_graph_pyg"]
