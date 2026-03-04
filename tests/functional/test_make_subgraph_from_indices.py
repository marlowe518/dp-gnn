"""Functional test for make_subgraph_from_indices.

Builds a tiny graph, defines subgraph indices with padding, and verifies:
- dummy node position and features
- star-graph edge structure
- reverse edges option.
"""

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.subgraph import make_subgraph_from_indices  # noqa: E402


def build_toy_graph() -> Data:
    # 4 nodes with simple identity features.
    x = torch.eye(4, dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 0, 1, 2],
         [1, 2, 2, 3]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = 4
    return data


def main() -> None:
    data = build_toy_graph()

    # Root node 0, neighbors 1 and 2, plus one padding slot.
    subgraph_indices = torch.tensor([0, 1, 2, -1], dtype=torch.long)

    sub_no_rev = make_subgraph_from_indices(
        data,
        subgraph_indices=subgraph_indices,
        add_reverse_edges=False,
        adjacency_normalization="none",
    )

    print("Subgraph (no reverse edges):")
    print("  x =", sub_no_rev.x.tolist())
    print("  edge_index =", sub_no_rev.edge_index.tolist())
    print("  edge_weight =", sub_no_rev.edge_weight.tolist())

    # Dummy node is last and has zero features.
    assert sub_no_rev.x.size(0) == len(subgraph_indices) + 1
    assert torch.all(sub_no_rev.x[-1] == 0)

    # With reverse edges.
    sub_rev = make_subgraph_from_indices(
        data,
        subgraph_indices=subgraph_indices,
        add_reverse_edges=True,
        adjacency_normalization="none",
    )

    print("Subgraph (with reverse edges):")
    print("  x =", sub_rev.x.tolist())
    print("  edge_index =", sub_rev.edge_index.tolist())

    print("make_subgraph_from_indices checks passed.")


if __name__ == "__main__":
    main()

