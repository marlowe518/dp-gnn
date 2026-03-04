"""Functional test: ensure ogbn-arxiv is available (download if missing), then print stats."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.config import Config  # noqa: E402
from dpgnn_repro.data import ensure_ogbn_arxiv, load_dataset  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data", help="Data root (dataset will be data_dir/ogbn_arxiv)")
    args = p.parse_args()

    # 1) Check / trigger download
    path = ensure_ogbn_arxiv(args.data_dir)
    print(f"Dataset at: {path}")

    # 2) Load and print statistics
    cfg = Config(
        dataset="ogbn-arxiv",
        data_dir=args.data_dir,
        split_mode="transductive",
        feature_norm="none",
        adjacency_normalization="inverse-degree",
    )
    data = load_dataset(cfg, ensure_download=False)

    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1

    print("Dataset statistics:")
    print(f"  number of nodes: {num_nodes}")
    print(f"  number of edges: {num_edges}")
    print(f"  feature dimension: {feat_dim}")
    print(f"  number of classes: {num_classes}")


if __name__ == "__main__":
    main()
