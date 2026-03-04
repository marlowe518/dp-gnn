"""PyTorch models for the DP-GNN reproduction.

Matches reference GraphMultiLayerPerceptron and GraphConvolutionalNetwork
(encoder -> message passing -> decoder with configurable activation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def _get_activation(name: str):
    if (name or "tanh").lower() == "tanh":
        return torch.tanh
    return F.relu


def _activation_module(name: str) -> nn.Module:
    if (name or "tanh").lower() == "tanh":
        return nn.Tanh()
    return nn.ReLU()


class MLP(nn.Module):
    """Feed-forward network for encoder/decoder."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int):
        super().__init__()
        layers: list[nn.Module] = []
        if num_layers <= 1:
            layers.append(nn.Linear(in_dim, out_dim))
        else:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMLP(nn.Module):
    """MLP applied to node features (non-graph baseline)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        self.mlp = MLP(in_dim, hidden_dim, num_layers, num_classes)

    def forward(self, data: Data) -> torch.Tensor:
        return self.mlp(data.x)


class GCN(nn.Module):
    """Kipf-style GCN with optional encoder/decoder MLPs (reference match)."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_classes: int,
        *,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        activation_fn: str = "tanh",
    ):
        super().__init__()
        act = _activation_module(activation_fn)

        enc_layers: list[nn.Module] = []
        if num_encoder_layers >= 1:
            enc_layers.append(nn.Linear(in_dim, hidden_dim))
            for _ in range(num_encoder_layers - 1):
                enc_layers.append(nn.Linear(hidden_dim, hidden_dim))
                enc_layers.append(_activation_module(activation_fn))
            enc_layers.append(_activation_module(activation_fn))
        self.encoder = nn.Sequential(*enc_layers) if enc_layers else None
        enc_out = hidden_dim if self.encoder is not None else in_dim

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(enc_out, hidden_dim, add_self_loops=False, normalize=False))

        dec_layers: list[nn.Module] = []
        if num_decoder_layers <= 1:
            dec_layers.append(nn.Linear(hidden_dim, num_classes))
        else:
            dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
            dec_layers.append(_activation_module(activation_fn))
            for _ in range(num_decoder_layers - 2):
                dec_layers.append(nn.Linear(hidden_dim, hidden_dim))
                dec_layers.append(_activation_module(activation_fn))
            dec_layers.append(nn.Linear(hidden_dim, num_classes))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        if self.encoder is not None:
            x = self.encoder(x)
        edge_index = data.edge_index
        edge_weight = getattr(data, "edge_weight", None)
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            x = torch.tanh(x)
        return self.decoder(x)


@dataclass
class ModelConfig:
    """Model build config (from global Config or data)."""
    kind: Literal["mlp", "gcn"] = "gcn"
    in_dim: int = 0
    hidden_dim: int = 100
    num_layers: int = 1
    num_classes: int = 40
    num_encoder_layers: int = 1
    num_decoder_layers: int = 1
    activation_fn: str = "tanh"


def build_model(cfg: ModelConfig) -> nn.Module:
    """Build model from config. Uses getattr for optional fields."""
    if cfg.kind == "mlp":
        return GraphMLP(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_classes=cfg.num_classes,
        )
    if cfg.kind == "gcn":
        gcn_kw: dict = {
            "in_dim": cfg.in_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_layers": cfg.num_layers,
            "num_classes": cfg.num_classes,
        }
        for key in ("num_encoder_layers", "num_decoder_layers", "activation_fn"):
            if hasattr(cfg, key):
                gcn_kw[key] = getattr(cfg, key)
        try:
            return GCN(**gcn_kw)
        except TypeError:
            return GCN(
                in_dim=cfg.in_dim,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                num_classes=cfg.num_classes,
            )
    raise ValueError(f"Unknown model kind: {cfg.kind}")
