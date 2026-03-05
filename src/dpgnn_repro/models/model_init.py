"""Model creation from config and representative graph (FU5: Model init).

Reference: refrence_repo/differentially_private_gnns/train.py create_model.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple, Union

import torch
from torch import nn
from torch_geometric.data import Data

from .gcn import GraphConvolutionalNetwork
from .mlp import GraphMultiLayerPerceptron


def _get_activation(activation_fn: str) -> nn.Module:
    """Resolve activation name to nn.Module (reference: getattr(nn, config.activation_fn))."""
    name = str(activation_fn).strip().lower()
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    raise ValueError("Unsupported activation_fn: %s. Use 'tanh' or 'relu'." % activation_fn)


def create_model(
    config: Any,
    graph: Union[Data, Tuple[torch.Tensor, torch.Tensor]],
    rng: Union[int, torch.Generator],
) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
    """Create the model and initial parameters.

    Matches reference create_model(config, graph, rng) -> (model, params).
    Config is duck-typed; graph is PyG Data or (x, edge_index) for init.

    Args:
        config: Object with at least:
            - model: str, 'mlp' or 'gcn'
            - latent_size: int
            - num_classes: int
            - activation_fn: str, 'tanh' or 'relu'
            - For mlp: num_layers: int
            - For gcn: num_encoder_layers, num_decoder_layers, num_message_passing_steps: int
        graph: PyG Data (x, edge_index [, edge_weight]) or (x, edge_index) tuple.
        rng: Random seed (int) or torch.Generator for reproducibility.

    Returns:
        (model, params) where params is model.state_dict().

    Raises:
        ValueError: If config.model is not 'mlp' or 'gcn', or required config missing.
    """
    if isinstance(rng, int):
        torch.manual_seed(rng)
    # rng as Generator could be used for dropout etc. in future

    model_name = getattr(config, "model", None)
    if model_name is None:
        raise ValueError("config must have attribute 'model' ('mlp' or 'gcn').")
    model_name = str(model_name).strip().lower()

    latent_size = getattr(config, "latent_size", None)
    num_classes = getattr(config, "num_classes", None)
    activation_fn = getattr(config, "activation_fn", "tanh")
    if latent_size is None or num_classes is None:
        raise ValueError(
            "config must have 'latent_size' and 'num_classes'. "
            "Got latent_size=%s, num_classes=%s." % (latent_size, num_classes)
        )
    activation = _get_activation(activation_fn)

    if isinstance(graph, Data):
        x = graph.x
        edge_index = graph.edge_index
        edge_weight = getattr(graph, "edge_weight", None)
    else:
        if len(graph) < 2 and model_name == "gcn":
            raise ValueError(
                "For model='gcn', graph must be Data or (x, edge_index [, edge_weight]). "
                "Got tuple of length %d." % len(graph)
            )
        x, edge_index = graph[0], graph[1]
        edge_weight = graph[2] if len(graph) > 2 else None

    if x is None:
        raise ValueError("graph must provide x (node features).")
    if model_name == "gcn" and edge_index is None:
        raise ValueError("For model='gcn', graph must provide edge_index.")
    if x.dim() != 2:
        raise ValueError("graph.x must be 2D (N, in_features), got shape %s." % (tuple(x.shape),))

    in_features = x.size(-1)
    device = x.device
    dtype = x.dtype

    if model_name == "mlp":
        num_layers = getattr(config, "num_layers", 1)
        if num_layers is None or num_layers < 1:
            raise ValueError("config.num_layers must be >= 1 for model='mlp', got %s." % num_layers)
        dimensions = [in_features] + [latent_size] * num_layers + [num_classes]
        model = GraphMultiLayerPerceptron(dimensions=dimensions, activation=activation)
        model = model.to(device=device, dtype=dtype)
        with torch.no_grad():
            _ = model(x)
        params = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return model, params

    if model_name == "gcn":
        num_encoder_layers = getattr(config, "num_encoder_layers", 1)
        num_decoder_layers = getattr(config, "num_decoder_layers", 1)
        num_message_passing_steps = getattr(config, "num_message_passing_steps", 1)
        if num_encoder_layers is None or num_decoder_layers is None or num_message_passing_steps is None:
            raise ValueError(
                "config must have num_encoder_layers, num_decoder_layers, num_message_passing_steps for model='gcn'."
            )
        model = GraphConvolutionalNetwork(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            num_message_passing_steps=num_message_passing_steps,
            latent_size=latent_size,
            num_classes=num_classes,
            activation=activation,
        )
        model = model.to(device=device, dtype=dtype)
        with torch.no_grad():
            _ = model(x, edge_index, edge_weight)
        params = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        return model, params

    raise ValueError("Unsupported model: %s. Use 'mlp' or 'gcn'." % config.model)
