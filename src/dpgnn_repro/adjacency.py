"""Adjacency / edge-weight utilities shared across modules.

Implements PyTorch equivalents of the reference `normalizations.normalize_edges_with_mask`
for the common case where the mask is None (all edges valid).
"""

from __future__ import annotations

import torch


def compute_edge_weight(
    edge_index: torch.Tensor,
    num_nodes: int,
    mode: str | None,
) -> torch.Tensor:
    """Compute edge weights according to adjacency normalization mode.

    Args:
      edge_index: LongTensor of shape [2, E] with (senders, receivers).
      num_nodes: Total number of nodes in the graph.
      mode: One of {None, "none", "inverse-degree", "inverse-sqrt-degree"}.

    Returns:
      FloatTensor of shape [E] with edge weights.
    """
    senders = edge_index[0]
    receivers = edge_index[1]
    num_edges = senders.size(0)

    if mode is None or mode == "none":
        return torch.ones(num_edges, dtype=torch.float32, device=edge_index.device)

    out_deg = torch.bincount(senders, minlength=num_nodes).to(torch.float32)
    in_deg = torch.bincount(receivers, minlength=num_nodes).to(torch.float32)

    if mode == "inverse-degree":
        out_deg = torch.clamp(out_deg, min=1.0)
        coeff = 1.0 / out_deg
        return coeff[senders]

    if mode == "inverse-sqrt-degree":
        out_deg = torch.clamp(out_deg, min=1.0)
        in_deg = torch.clamp(in_deg, min=1.0)
        coeff_out = 1.0 / torch.sqrt(out_deg)
        coeff_in = 1.0 / torch.sqrt(in_deg)
        return coeff_out[senders] * coeff_in[receivers]

    raise ValueError(f"Unknown adjacency_normalization: {mode}")

