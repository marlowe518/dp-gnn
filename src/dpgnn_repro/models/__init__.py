"""Model definitions and creation for DP-GNN.

Reference: refrence_repo/differentially_private_gnns/models.py and train.create_model.
"""

from .model_init import create_model
from .gcn import GraphConvolutionalNetwork
from .mlp import GraphMultiLayerPerceptron

__all__ = [
    "create_model",
    "GraphConvolutionalNetwork",
    "GraphMultiLayerPerceptron",
]
