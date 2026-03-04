"""Functional test for input_pipeline.get_dataset equivalence.

Default: uses toy dataset (no downloads, fast).
Optional: --use_real_dataset to exercise OGBN-Arxiv path.
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from dpgnn_repro.config import Config  # noqa: E402
from dpgnn_repro.input_pipeline import get_dataset  # noqa: E402


def summarize(data, labels, masks) -> None:
    num_nodes = data.x.size(0)
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1)
    num_classes = int(labels.max().item()) + 1

    train_mask = masks["train"]
    val_mask = masks["validation"]
    test_mask = masks["test"]

    print("get_dataset summary:")
    print(f"  num_nodes={num_nodes}, num_edges={num_edges}")
    print(f"  feature_dim={feat_dim}, num_classes={num_classes}")
    print(
        "  split sizes: "
        f"train={int(train_mask.sum().item())}, "
        f"val={int(val_mask.sum().item())}, "
        f"test={int(test_mask.sum().item())}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--use_real_dataset",
        action="store_true",
        help="If set, use ogbn-arxiv (will download if missing).",
    )
    p.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Data root; ogbn-arxiv will live under data_dir/ogbn_arxiv.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for toy dataset (determinism check).",
    )
    args = p.parse_args()

    dataset_name = "ogbn-arxiv" if args.use_real_dataset else "toy"
    cfg = Config(
        dataset=dataset_name,
        data_dir=args.data_dir,
        seed=args.seed,
        split_mode="transductive",
        feature_norm="none",
        adjacency_normalization="inverse-degree",
    )

    # Call twice with same config to verify determinism (at least for toy).
    data1, labels1, masks1 = get_dataset(cfg)
    data2, labels2, masks2 = get_dataset(cfg)

    summarize(data1, labels1, masks1)

    if not args.use_real_dataset:
        # Determinism: same masks, labels, and structure.
        assert data1.edge_index.equal(data2.edge_index)
        assert data1.x.equal(data2.x)
        assert labels1.equal(labels2)
        for key in ("train", "validation", "test"):
            assert masks1[key].equal(masks2[key])
        print("Determinism check passed for toy dataset.")
    else:
        print("Real dataset path exercised (ogbn-arxiv). Determinism not enforced (splits from disk).")


if __name__ == "__main__":
    main()

