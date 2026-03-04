"""Functional test for get_subgraphs (reference train.get_subgraphs behavior).

Builds a small synthetic graph, computes per-node padded subgraph indices, and
validates:
- shape and padding
- root node at position 0
- indices correspond to outgoing neighbors (no self-loops).
"""

import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.subgraph import get_subgraphs  # noqa: E402


def build_toy_graph() -> Data:
    # 4 nodes, directed edges:
    # 0 -> 1, 0 -> 2, 1 -> 2, 2 -> 2 (self-loop), 2 -> 3
    x = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 2],
         [1, 2, 2, 2, 3]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = 4
    return data


def main() -> None:
    data = build_toy_graph()
    pad_to = 4

    subgraphs = get_subgraphs(data, pad_to=pad_to)

    print("Subgraphs indices (rows = nodes):")
    print(subgraphs.tolist())

    num_nodes = data.num_nodes
    assert subgraphs.shape == (num_nodes, pad_to)

    src, dst = data.edge_index
    edges = list(zip(src.tolist(), dst.tolist()))

    for u in range(num_nodes):
        row = subgraphs[u].tolist()
        # Root node must appear in position 0.
        assert row[0] == u

        for v in row[1:]:
            if v == -1:
                continue  # padding
            # No self-loops should have been included.
            assert not (u == v and (u, v) in edges)
            # Either v is a neighbor of u, or equal to u (already checked).
            assert (u, v) in edges or v == u

    print("get_subgraphs checks passed.")


if __name__ == "__main__":
    main()

