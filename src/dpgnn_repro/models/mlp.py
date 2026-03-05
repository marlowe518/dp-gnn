"""Graph MLP: MLP applied to node features (no message passing).

Reference: refrence_repo/differentially_private_gnns/models.py GraphMultiLayerPerceptron.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class GraphMultiLayerPerceptron(nn.Module):
    """MLP applied to each node's feature vector; output shape (N, num_classes).

    Matches reference GraphMultiLayerPerceptron: dimensions = [latent_size]*num_layers + [num_classes],
    no skip connections, no activation on final layer.
    """

    def __init__(
        self,
        dimensions: List[int],
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dimensions = dimensions
        self.activation = activation
        layers = []
        for i in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2 and activation is not None:
                layers.append(activation)
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x (N, in_dim) -> (N, num_classes)."""
        return self.mlp(x)
