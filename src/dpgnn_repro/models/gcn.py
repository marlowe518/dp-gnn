"""GCN: encoder -> message passing (weighted) -> decoder.

Reference: refrence_repo/differentially_private_gnns/models.py GraphConvolutionalNetwork,
OneHopGraphConvolution (message = edge_weight * x[sender], aggregate by receiver, then update_fn).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import Data


def _one_hop_weighted(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    """Aggregate: for each node, sum (edge_weight * x[sender]) over incoming edges.

    edge_index: (2, E) with (source, target); message = edge_weight * x[source], aggregate to target.
    """
    source, target = edge_index[0], edge_index[1]
    # (E, F) * (E,) or (E,1) -> (E, F)
    if edge_weight.dim() == 1:
        edge_weight = edge_weight.unsqueeze(-1)
    messages = edge_weight * x[source]
    return scatter_add(messages, target, dim=0, dim_size=x.size(0))


class GraphConvolutionalNetwork(nn.Module):
    """GCN: encoder MLP -> K hops of weighted conv + MLP (with skip) -> decoder MLP.

    Matches reference: encoder (latent_size layers, activate_final=True), then
    num_message_passing_steps x (weighted aggregation + MLP with skip), then
    decoder (latent_size*(decoder_layers-1) + num_classes, no activation on final).
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_message_passing_steps: int,
        latent_size: int,
        num_classes: int,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_message_passing_steps = num_message_passing_steps
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.activation = activation

        # Encoder: in_features -> latent_size, then (num_encoder_layers-1) x latent_size -> latent_size, activate_final
        # Built in first forward when in_features is known.
        self._encoder: Optional[nn.Module] = None

        # Core: per-hop update (MLP with skip): one Linear(latent_size, latent_size) + activation
        self.core_update = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_size, latent_size),
                activation if activation is not None else nn.Identity(),
            )
            for _ in range(num_message_passing_steps)
        ])

        # Decoder: [latent_size]*(num_decoder_layers-1) + [num_classes], no activation on final
        decoder_layers = []
        for i in range(num_decoder_layers - 1):
            decoder_layers.append(nn.Linear(latent_size, latent_size))
            decoder_layers.append(activation if activation is not None else nn.Identity())
        decoder_layers.append(nn.Linear(latent_size, num_classes))
        self.decoder = nn.Sequential(*decoder_layers)

        self._in_features: Optional[int] = None

    def _build_encoder(self, in_features: int) -> None:
        if self._in_features == in_features:
            return
        self._in_features = in_features
        layers = []
        for i in range(self.num_encoder_layers):
            layers.append(nn.Linear(in_features if i == 0 else self.latent_size, self.latent_size))
            if self.activation is not None:
                layers.append(self.activation)
        self._encoder = nn.Sequential(*layers).to(next(self.core_update.parameters()).device)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward: x (N, F), edge_index (2, E), optional edge_weight (E,) or (E,1). Output (N, num_classes)."""
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device, dtype=x.dtype)
        self._build_encoder(x.size(-1))
        h = self._encoder(x)
        # Reference: message-passing "against the direction" (senders=receivers, receivers=senders)
        # So aggregate from edge_index[1] to edge_index[0].
        edge_index_rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
        for hop in range(self.num_message_passing_steps):
            agg = _one_hop_weighted(h, edge_index_rev, edge_weight)
            h = self.core_update[hop](agg) + h  # skip connection
        return self.decoder(h)
