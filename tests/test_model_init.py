"""Tests for FU5: Model init (create_model, GraphMLP, GCN).

Reference: refrence_repo/differentially_private_gnns/train.create_model, models.py.
Uses synthetic graphs; no dataset download.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data

# Ensure src on path for direct run
if "src" not in sys.path:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "src"))

from dpgnn_repro.models import create_model, GraphMultiLayerPerceptron, GraphConvolutionalNetwork


def _make_graph(
    num_nodes: int = 4,
    in_features: int = 3,
    num_edges: int = 4,
) -> Data:
    """Minimal PyG Data for init."""
    x = torch.randn(num_nodes, in_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    return Data(x=x, edge_index=edge_index)


# ---- Happy path ----

def test_create_model_mlp_happy_path():
    """Happy path: create_model with mlp config returns model and params; forward shape correct."""
    config = SimpleNamespace(
        model="mlp",
        latent_size=8,
        num_classes=5,
        activation_fn="tanh",
        num_layers=2,
    )
    graph = _make_graph(num_nodes=6, in_features=3)
    rng = 42
    model, params = create_model(config, graph, rng)
    assert model is not None
    assert isinstance(params, dict)
    assert len(params) > 0
    assert all(isinstance(v, torch.Tensor) for v in params.values())
    # Forward
    out = model(graph.x)
    assert out.shape == (6, 5)
    assert torch.isfinite(out).all()


def test_create_model_gcn_happy_path():
    """Happy path: create_model with gcn config returns model and params; forward shape correct."""
    config = SimpleNamespace(
        model="gcn",
        latent_size=8,
        num_classes=5,
        activation_fn="relu",
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_message_passing_steps=1,
    )
    graph = _make_graph(num_nodes=6, in_features=3, num_edges=10)
    graph.edge_weight = torch.ones(graph.edge_index.size(1))
    rng = 42
    model, params = create_model(config, graph, rng)
    assert model is not None
    assert isinstance(params, dict)
    assert len(params) > 0
    out = model(graph.x, graph.edge_index, graph.edge_weight)
    assert out.shape == (6, 5)
    assert torch.isfinite(out).all()


def test_create_model_accepts_tuple_graph():
    """create_model accepts (x, edge_index) tuple for GCN."""
    config = SimpleNamespace(
        model="gcn",
        latent_size=4,
        num_classes=2,
        activation_fn="tanh",
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_message_passing_steps=1,
    )
    x = torch.randn(3, 2)
    edge_index = torch.tensor([[0, 1], [1, 2]])
    model, params = create_model(config, (x, edge_index), 0)
    out = model(x, edge_index, None)
    assert out.shape == (3, 2)


# ---- Edge cases ----

def test_create_model_mlp_single_node():
    """Edge case: single node graph."""
    config = SimpleNamespace(
        model="mlp",
        latent_size=4,
        num_classes=2,
        activation_fn="tanh",
        num_layers=1,
    )
    graph = _make_graph(num_nodes=1, in_features=2)
    model, params = create_model(config, graph, 0)
    out = model(graph.x)
    assert out.shape == (1, 2)


def test_create_model_deterministic_with_seed():
    """Same seed yields same params."""
    config = SimpleNamespace(
        model="mlp",
        latent_size=4,
        num_classes=2,
        activation_fn="tanh",
        num_layers=1,
    )
    graph = _make_graph(num_nodes=2, in_features=2)
    _, p1 = create_model(config, graph, 123)
    _, p2 = create_model(config, graph, 123)
    for k in p1:
        assert k in p2
        assert torch.allclose(p1[k], p2[k]), "Same seed should give same params"


# ---- Failure cases ----

def test_create_model_unsupported_model_raises():
    """Unsupported config.model raises ValueError."""
    config = SimpleNamespace(
        model="unknown",
        latent_size=4,
        num_classes=2,
        activation_fn="tanh",
        num_layers=1,
    )
    graph = _make_graph(num_nodes=2, in_features=2)
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model(config, graph, 0)


def test_create_model_missing_config_raises():
    """Missing config attributes raise ValueError."""
    config = SimpleNamespace(model="mlp")  # missing latent_size, num_classes, num_layers
    graph = _make_graph(num_nodes=2, in_features=2)
    with pytest.raises(ValueError, match="latent_size|num_classes"):
        create_model(config, graph, 0)


def test_create_model_unsupported_activation_raises():
    """Unsupported activation_fn raises ValueError."""
    config = SimpleNamespace(
        model="mlp",
        latent_size=4,
        num_classes=2,
        activation_fn="silu",
        num_layers=1,
    )
    graph = _make_graph(num_nodes=2, in_features=2)
    with pytest.raises(ValueError, match="activation_fn|Unsupported"):
        create_model(config, graph, 0)


def test_create_model_gcn_missing_edge_index_raises():
    """GCN with tuple (x,) only raises ValueError."""
    config = SimpleNamespace(
        model="gcn",
        latent_size=4,
        num_classes=2,
        activation_fn="tanh",
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_message_passing_steps=1,
    )
    x = torch.randn(2, 2)
    with pytest.raises(ValueError, match="edge_index|length"):
        create_model(config, (x,), 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
