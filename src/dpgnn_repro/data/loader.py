"""Dataset loader and preprocessing. Matches reference repo behavior.

Reference: refrence_repo/differentially_private_gnns/dataset_readers.py,
input_pipeline.py (load + split + feature norm only; no sampling/normalization).
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from ..config import Config


# Toy graph sizes matching reference DummyDataset (dataset_readers.py)
TOY_NUM_TRAIN = 3
TOY_NUM_VAL = 3
TOY_NUM_TEST = 3
TOY_NUM_FEATURES = 5
TOY_NUM_CLASSES = 3


def _toy_graph(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build in-memory toy graph. Deterministic for given seed. Matches reference DummyDataset."""
    rng = np.random.default_rng(seed)
    n = TOY_NUM_TRAIN + TOY_NUM_VAL + TOY_NUM_TEST
    # Senders / receivers: ring graph like reference (senders 0..n-1, receivers rolled by 1)
    senders = np.arange(n, dtype=np.int64)
    receivers = np.roll(np.arange(n, dtype=np.int64), -1)
    # Node features: deterministic from seed (reference uses repeat/reshape; we use rng for variety)
    node_features = rng.standard_normal((n, TOY_NUM_FEATURES)).astype(np.float32)
    # Labels: reference uses zeros; we use 0..num_classes-1 cyclically for simple tests
    node_labels = np.arange(n, dtype=np.int64) % TOY_NUM_CLASSES
    train_nodes = np.arange(TOY_NUM_TRAIN)
    val_nodes = np.arange(TOY_NUM_TRAIN, TOY_NUM_TRAIN + TOY_NUM_VAL)
    test_nodes = np.arange(TOY_NUM_TRAIN + TOY_NUM_VAL, n)
    return senders, receivers, node_features, node_labels, train_nodes, val_nodes, test_nodes


def _load_ogbn_arxiv(data_dir: str, ensure_download: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load ogbn-arxiv from disk. If missing and ensure_download, download first. Layout matches reference OGBTransductiveDataset."""
    from .download_dataset import ensure_ogbn_arxiv

    if ensure_download:
        ensure_ogbn_arxiv(data_dir)
    base = Path(data_dir) / "ogbn_arxiv"
    split_dir = base / "split" / "time"
    raw_dir = base / "raw"
    if not base.exists():
        raise FileNotFoundError(
            f"OGB directory not found: {base}. "
            "Run with ensure_download=True or download via ensure_ogbn_arxiv(data_dir)."
        )

    def _read_csv_gz(path: Path) -> np.ndarray:
        with gzip.open(path, "rt") as f:
            return pd.read_csv(f, header=None).values

    node_features = _read_csv_gz(raw_dir / "node-feat.csv.gz").astype(np.float32)
    node_labels = _read_csv_gz(raw_dir / "node-label.csv.gz").astype(np.int64).squeeze()
    edge_df = _read_csv_gz(raw_dir / "edge.csv.gz")
    senders = edge_df[:, 0].astype(np.int64)
    receivers = edge_df[:, 1].astype(np.int64)
    train_nodes = _read_csv_gz(split_dir / "train.csv.gz").squeeze().astype(np.int64)
    validation_nodes = _read_csv_gz(split_dir / "valid.csv.gz").squeeze().astype(np.int64)
    test_nodes = _read_csv_gz(split_dir / "test.csv.gz").squeeze().astype(np.int64)
    return senders, receivers, node_features, node_labels, train_nodes, validation_nodes, test_nodes


def _to_pyg(
    senders: np.ndarray,
    receivers: np.ndarray,
    node_features: np.ndarray,
    node_labels: np.ndarray,
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    test_nodes: np.ndarray,
) -> Data:
    """Build PyG Data with masks and split index arrays."""
    num_nodes = len(node_labels)
    edge_index = np.stack([senders, receivers], axis=0)
    train_mask = np.zeros(num_nodes, dtype=bool)
    val_mask = np.zeros(num_nodes, dtype=bool)
    test_mask = np.zeros(num_nodes, dtype=bool)
    train_mask[train_nodes] = True
    val_mask[val_nodes] = True
    test_mask[test_nodes] = True

    data = Data(
        x=torch.from_numpy(node_features),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(node_labels),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )
    data.train_nodes = torch.from_numpy(train_nodes)
    data.val_nodes = torch.from_numpy(val_nodes)
    data.test_nodes = torch.from_numpy(test_nodes)
    data.num_nodes = num_nodes
    return data


def _apply_feature_norm(data: Data, method: str) -> None:
    """Apply feature normalization in-place. Train-only fit for 'standard' (reference GraphSAINT)."""
    if method == "none":
        return
    if method == "standard":
        train_x = data.x[data.train_mask].numpy()
        scaler = StandardScaler()
        scaler.fit(train_x)
        data.x = torch.from_numpy(scaler.transform(data.x.numpy()).astype(np.float32))
        return
    raise ValueError(f"Unknown feature_norm: {method}")


def _filter_disjoint_edges(
    data: Data,
    train_nodes: np.ndarray,
    val_nodes: np.ndarray,
    test_nodes: np.ndarray,
) -> None:
    """Keep only edges whose endpoints are in the same split. Reference: OGBDisjointDataset."""
    train_set = set(train_nodes.tolist())
    val_set = set(val_nodes.tolist())
    test_set = set(test_nodes.tolist())

    def same_split(u: int, v: int) -> bool:
        for s in (train_set, val_set, test_set):
            if u in s and v in s:
                return True
        return False

    ei = data.edge_index.numpy()
    keep = np.array([same_split(int(ei[0, i]), int(ei[1, i])) for i in range(ei.shape[1])])
    data.edge_index = torch.from_numpy(ei[:, keep])


def load_dataset(cfg: Config, ensure_download: bool = True) -> Data:
    """Load dataset and apply preprocessing (feature norm, disjoint split). Returns PyG Data. If ogbn-arxiv is missing, download when ensure_download=True."""
    dataset_name = (cfg.dataset or "").strip().lower().replace("-", "_")
    disjoint = cfg.split_mode == "disjoint" or "disjoint" in (cfg.dataset or "")
    if dataset_name == "toy":
        senders, receivers, node_features, node_labels, train_nodes, val_nodes, test_nodes = _toy_graph(cfg.seed)
    elif dataset_name in ("ogbn_arxiv", "ogbn_arxiv_disjoint"):
        senders, receivers, node_features, node_labels, train_nodes, val_nodes, test_nodes = _load_ogbn_arxiv(cfg.data_dir, ensure_download=ensure_download)
        if "disjoint" in dataset_name:
            disjoint = True
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}. Use 'toy' or 'ogbn-arxiv'.")

    data = _to_pyg(senders, receivers, node_features, node_labels, train_nodes, val_nodes, test_nodes)

    if disjoint:
        _filter_disjoint_edges(data, train_nodes, val_nodes, test_nodes)

    _apply_feature_norm(data, cfg.feature_norm or "none")
    return data


def log_data_debug(data: Data, cfg: Config) -> None:
    """Log dataset stats when --debug is set. Reference: inputs expected by training pipeline."""
    num_nodes = data.num_nodes if hasattr(data, "num_nodes") else data.x.size(0)
    num_edges = data.edge_index.size(1)
    feat_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    train_size = data.train_mask.sum().item()
    val_size = data.val_mask.sum().item()
    test_size = data.test_mask.sum().item()
    label_counts: Any = {}
    for i in range(num_classes):
        label_counts[i] = (data.y == i).sum().item()

    lines = [
        "Data loading debug:",
        f"  num_nodes={num_nodes}, num_edges={num_edges}",
        f"  feature_dim={feat_dim}, num_classes={num_classes}",
        f"  split sizes: train={train_size}, val={val_size}, test={test_size}",
        f"  label distribution: {label_counts}",
        f"  feature_norm={getattr(cfg, 'feature_norm', 'none')}, adjacency_normalization={getattr(cfg, 'adjacency_normalization', 'inverse-degree')}",
    ]
    import logging
    log = logging.getLogger("dpgnn_repro")
    for line in lines:
        log.info(line)
