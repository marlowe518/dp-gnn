"""Input pipeline for DP-GNN training (PyTorch/PyG version).

Reference: `refrence_repo/differentially_private_gnns/input_pipeline.py`.
This module mirrors the structure of the reference input pipeline but is adapted
to the PyTorch / PyG stack used in this reproduction project.

Scope (current phase):
- Load graph data via `dpgnn_repro.data.load_dataset`.
- Add reverse edges (as in the reference).
- Add self-loops (as in the reference).
- Expose train/validation/test masks and labels in a convenient form.

Out of scope (handled elsewhere / future work):
- Sampling / subsampling (Alg. 1–3) — uses `sampling` module stubs for now.
- DP-related logic (DP-SGD / DP-Adam, privacy accounting).
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch_geometric.data import Data

from .config import Config
from .data import load_dataset
from .sampling import subsample_graph_pyg
from .adjacency import compute_edge_weight


def add_reverse_edges(data: Data) -> Data:
    """Add reverse edges to the graph (undirected view), in-place.

    Reference behavior:
    - Concatenate (senders, receivers) with their reversed counterparts.
    - Duplicates are allowed; no deduplication is performed.
    """
    edge_index = data.edge_index
    assert edge_index.dim() == 2 and edge_index.size(0) == 2
    senders, receivers = edge_index
    rev = torch.stack((receivers, senders), dim=0)
    data.edge_index = torch.cat((edge_index, rev), dim=1)
    return data


def add_self_loops(data: Data) -> Data:
    """Add self-loops to all nodes, in-place.

    Reference behavior:
    - For each node i, add an edge i->i.
    - Duplicates with existing self-loops are allowed.
    """
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    self_nodes = torch.arange(num_nodes, dtype=torch.long)
    self_edges = torch.stack((self_nodes, self_nodes), dim=0)
    data.edge_index = torch.cat((self_edges, data.edge_index), dim=1)
    return data


def compute_masks_for_splits(data: Data) -> Dict[str, torch.Tensor]:
    """Compute boolean masks for the train, validation and test splits.

    Reference behavior:
    - In the original code, masks are constructed from index arrays
      (train_nodes, validation_nodes, test_nodes).
    - Here we reuse the boolean masks stored on the PyG `Data` object.
    """
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()
    test_mask = data.test_mask.bool()
    return {"train": train_mask, "validation": val_mask, "test": test_mask}


def get_dataset(cfg: Config, rng: torch.Generator | None = None) -> Tuple[Data, torch.Tensor, Dict[str, torch.Tensor]]:
    """Load graph dataset and apply basic preprocessing.

    Returns:
      - data: PyG `Data` object with:
          - x: node features
          - edge_index: edges (possibly with reverse edges and self-loops)
          - y: integer node labels
          - train_mask / val_mask / test_mask
      - labels: same as `data.y`
      - masks: dict with boolean masks for 'train', 'validation', 'test'

    Notes:
      - Sampling / subsampling (Alg. 1–3) is *not* implemented here yet,
        to keep algorithm logic separate and match the project’s phase.
      - Adjacency normalization hooks are kept in config, but actual
        normalization is expected to be handled by the model / layers.
    """
    # Load base graph + splits.
    data = load_dataset(cfg, ensure_download=True)

    # Add reverse edges to mimic reference behavior.
    data = add_reverse_edges(data)

    # Optional degree-bounded subsampling (Alg. 1–3) via sampling module.
    max_degree = getattr(cfg, "max_degree", None)
    if max_degree is not None and max_degree > 0 and hasattr(data, "train_nodes"):
        data = subsample_graph_pyg(
            data,
            max_degree=max_degree,
            train_nodes=data.train_nodes,
            base_seed=getattr(cfg, "seed", 0),
        )

    # Build masks dict.
    masks = compute_masks_for_splits(data)

    # Add self-loops after masks are defined (matches reference ordering).
    data = add_self_loops(data)

    # Compute edge weights analogous to reference normalization step.
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    data.edge_weight = compute_edge_weight(
        data.edge_index,
        num_nodes=num_nodes,
        mode=getattr(cfg, "adjacency_normalization", None),
    )

    labels = data.y
    return data, labels, masks

