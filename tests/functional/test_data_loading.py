"""Functional test for data loading. Uses toy dataset by default; optional ogbn-arxiv with --use_real_dataset."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import dpgnn_repro  # noqa: E402
from dpgnn_repro.config import Config  # noqa: E402
from dpgnn_repro.data import load_dataset, log_data_debug  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="Random seed for deterministic toy load")
    p.add_argument("--use_real_dataset", action="store_true", help="If set, try loading ogbn-arxiv from data_dir")
    p.add_argument("--data_dir", type=str, default="data", help="Data root for ogbn-arxiv")
    args = p.parse_args()

    cfg = Config(
        dataset="toy" if not args.use_real_dataset else "ogbn-arxiv",
        data_dir=args.data_dir,
        seed=args.seed,
        split_mode="transductive",
        feature_norm="none",
        adjacency_normalization="inverse-degree",
    )

    # Load twice with same seed and verify deterministic
    data1 = load_dataset(cfg)
    data2 = load_dataset(cfg)

    # Debug summary (required)
    log_data_debug(data1, cfg)

    if not args.use_real_dataset:
        # Determinism check: same splits and features
        assert data1.train_mask.equal(data2.train_mask)
        assert data1.val_mask.equal(data2.val_mask)
        assert data1.test_mask.equal(data2.test_mask)
        assert data1.x.equal(data2.x)
        assert data1.y.equal(data2.y)
        assert data1.edge_index.equal(data2.edge_index)
        print("Determinism check passed: repeated load with same seed produced identical data.")
    else:
        print("Real dataset loaded; determinism not checked (splits from disk).")


if __name__ == "__main__":
    main()
