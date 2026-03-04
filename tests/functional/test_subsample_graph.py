"""Functional test for subsample_graph_pyg (degree-bounded subsampling).

Builds a small graph, marks train nodes, applies subsampling, and verifies:
- in-degree constraints from training nodes
- overall structure and node set unchanged.
"""

import argparse
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.sampling import subsample_graph_pyg  # noqa: E402


def build_toy_graph() -> Data:
    # 6 nodes, directed edges:
    # 0,1,2 are train nodes; 3,4,5 are non-train nodes.
    #
    # Edges:
    # 0 -> 3, 1 -> 3, 2 -> 3,
    # 0 -> 4, 1 -> 4, 2 -> 4,
    # 3 -> 5
    x = torch.arange(6, dtype=torch.float32).unsqueeze(-1)
    edge_index = torch.tensor(
        [[0, 1, 2, 0, 1, 2, 3],
         [3, 3, 3, 4, 4, 4, 5]],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = 6
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--max_degree", type=int, default=1, help="In-degree bound from train nodes.")
    p.add_argument("--seed", type=int, default=0, help="Base seed for sampling.")
    args = p.parse_args()

    data = build_toy_graph()
    train_nodes = torch.tensor([0, 1, 2], dtype=torch.long)

    print("Original graph:")
    print(f"  num_nodes={data.x.size(0)}, num_edges={data.edge_index.size(1)}")
    print("  edge_index =", data.edge_index.tolist())

    sub_data = subsample_graph_pyg(
        data, max_degree=args.max_degree, train_nodes=train_nodes, base_seed=args.seed
    )

    print("Subsampled graph:")
    print(f"  num_nodes={sub_data.x.size(0)}, num_edges={sub_data.edge_index.size(1)}")
    print("  edge_index =", sub_data.edge_index.tolist())

    # Check in-degree constraints from training nodes.
    src, dst = sub_data.edge_index
    num_nodes = sub_data.x.size(0)
    train_set = set(train_nodes.tolist())
    indeg_from_train = [0] * num_nodes
    for u, v in zip(src.tolist(), dst.tolist()):
        if u in train_set:
            indeg_from_train[v] += 1

    for u in range(num_nodes):
        assert indeg_from_train[u] <= args.max_degree, (
            f"Node {u} has in-degree {indeg_from_train[u]} from train nodes, "
            f"exceeding max_degree={args.max_degree}"
        )

    print("In-degree constraints satisfied for all nodes.")


if __name__ == "__main__":
    main()

