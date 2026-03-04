"""DP-SGD step: per-example clip by L2, sum, add Gaussian noise.

Matches reference optimizers.dp_aggregate (clip_by_norm + sum + noise).
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch


def _tree_flatten(tree: Any) -> Tuple[List[torch.Tensor], Any]:
    """Flatten gradient tree to list of tensors."""
    if isinstance(tree, dict):
        out = []
        for k in sorted(tree.keys()):
            o, _ = _tree_flatten(tree[k])
            out.extend(o)
        return out, type(tree)
    if isinstance(tree, (list, tuple)):
        out = []
        for x in tree:
            o, _ = _tree_flatten(x)
            out.extend(o)
        return out, type(tree)
    if isinstance(tree, torch.Tensor):
        return [tree], None
    return [], None


def _tree_unflatten(flat: List[torch.Tensor], treedef: Any, structure: Any) -> Any:
    """Reconstruct tree from flat list using structure as template."""
    if structure is None and treedef is None:
        return flat[0] if len(flat) == 1 else flat
    if isinstance(structure, dict):
        res = {}
        idx = 0
        for k in sorted(structure.keys()):
            sub_struct = structure[k]
            sub_flat_len = len(_tree_flatten(sub_struct)[0])
            res[k] = _tree_unflatten(flat[idx : idx + sub_flat_len], None, sub_struct)
            idx += sub_flat_len
        return res
    if isinstance(structure, (list, tuple)):
        res = []
        for sub_struct in structure:
            sub_flat_len = len(_tree_flatten(sub_struct)[0])
            res.append(_tree_unflatten(flat[: sub_flat_len], None, sub_struct))
            flat = flat[sub_flat_len:]
        return type(structure)(res)
    if isinstance(structure, torch.Tensor):
        return flat[0]
    return flat[0] if flat else None


def _per_example_l2_norms(grad: torch.Tensor, batch_dim: int = 0) -> torch.Tensor:
    """L2 norm of each example along batch_dim. grad: [B, ...]."""
    flat = grad.reshape(grad.size(batch_dim), -1)
    return torch.norm(flat, p=2, dim=1)


def _clip_per_example(grad: torch.Tensor, clip_norm: float, batch_dim: int = 0) -> torch.Tensor:
    """Scale each example so L2 norm <= clip_norm."""
    norms = _per_example_l2_norms(grad, batch_dim)
    scale = clip_norm / (norms + 1e-12)
    scale = torch.clamp(scale, max=1.0)
    for _ in range(batch_dim):
        scale = scale.unsqueeze(-1)
    for _ in range(grad.dim() - 1 - batch_dim):
        scale = scale.unsqueeze(-1)
    return grad * scale


def _tree_map(fn, tree: Any, batch_dim: int = 0) -> Any:
    """Map fn over each tensor in tree (fn receives tensor)."""
    if isinstance(tree, dict):
        return {k: _tree_map(fn, tree[k], batch_dim) for k in tree}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_tree_map(fn, x, batch_dim) for x in tree)
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    return tree


def dp_step(
    gradients: Any,
    clip_norm: float,
    noise_multiplier: float,
    batch_size: int,
    *,
    base_sensitivity: float = 1.0,
    generator: Optional[torch.Generator] = None,
    **_: Any,
) -> Any:
    """Clip per-example gradients by L2, sum over batch, add Gaussian noise.

    gradients: tree of tensors each of shape (batch_size, *param_shape).
    Returns: tree of tensors (*param_shape) (no batch dim).
    """
    def clip_and_sum(g: torch.Tensor) -> torch.Tensor:
        clipped = _clip_per_example(g, clip_norm, batch_dim=0)
        summed = clipped.sum(dim=0)
        return summed

    summed = _tree_map(clip_and_sum, gradients)
    noise_std = clip_norm * base_sensitivity * noise_multiplier

    def add_noise(t: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(t, generator=generator, device=t.device, dtype=t.dtype)
        return t + noise * noise_std

    return _tree_map(add_noise, summed)
