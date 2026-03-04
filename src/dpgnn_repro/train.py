"""Training loop: load data, subgraphs, per-example gradients, DP step, accounting."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from .accounting import get_epsilon
from .config import Config
from .dp.optimizer import dp_step
from .input_pipeline import get_dataset
from .models import ModelConfig, build_model
from .subgraph.extract import get_subgraphs, make_subgraph_from_indices

log = logging.getLogger(__name__)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_config(cfg: Config, in_dim: int, num_classes: int) -> ModelConfig:
    return ModelConfig(
        kind=cfg.model,
        in_dim=in_dim,
        hidden_dim=cfg.latent_size,
        num_layers=cfg.num_message_passing_steps,
        num_classes=num_classes,
        num_encoder_layers=getattr(cfg, "num_encoder_layers", 1),
        num_decoder_layers=getattr(cfg, "num_decoder_layers", 1),
        activation_fn=getattr(cfg, "activation_fn", "tanh"),
    )


def _compute_metrics(
    model: torch.nn.Module,
    data: Data,
    labels: torch.Tensor,
    masks: dict,
    device: torch.device,
) -> dict:
    model.eval()
    with torch.no_grad():
        data_dev = data.to(device)
        logits = model(data_dev)
        pred = logits.argmax(dim=-1)
    res = {}
    for name, mask in masks.items():
        m = mask.to(device)
        acc = (pred[m] == labels.to(device)[m]).float().mean().item()
        res[f"{name}_acc"] = acc
    return res


def compute_max_terms_per_node(cfg: Config) -> int:
    if cfg.model == "mlp":
        return 1
    k = cfg.num_message_passing_steps
    d = cfg.max_degree
    if k == 1:
        return d + 1
    if k == 2:
        return d * d + d + 1
    raise ValueError("Only 0/1/2 message passing steps supported.")


def compute_base_sensitivity(cfg: Config) -> float:
    if cfg.model == "mlp":
        return 1.0
    k = cfg.num_message_passing_steps
    d = cfg.max_degree
    if k == 1:
        return float(2 * (d + 1))
    if k == 2:
        return float(2 * (d * d + d + 1))
    raise ValueError("Only 0/1/2 message passing steps supported.")


def _estimate_clip_norm(
    model: torch.nn.Module,
    data: Data,
    subgraphs: torch.Tensor,
    train_indices: torch.Tensor,
    cfg: Config,
    device: torch.device,
    num_samples: int,
) -> float:
    """Estimate global clip norm from percentile of per-example gradient norms."""
    model.train()
    norms = []
    pad_to = getattr(cfg, "pad_subgraphs_to", 100)
    adj_norm = getattr(cfg, "adjacency_normalization", "inverse-degree")
    n_est = min(num_samples, train_indices.size(0))
    indices = train_indices[torch.randperm(train_indices.size(0), device=train_indices.device)[:n_est]]
    for idx in indices:
        idx = idx.item()
        sub_idx = subgraphs[idx]
        sub = make_subgraph_from_indices(data, sub_idx, add_reverse_edges=False, adjacency_normalization=adj_norm)
        sub = sub.to(device)
        model.zero_grad()
        logits = model(sub)
        target = data.y[idx : idx + 1].to(device)
        loss = F.cross_entropy(logits[0:1], target)
        loss.backward()
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.pow(2).sum().item()
        norms.append(total ** 0.5)
    if not norms:
        return 1.0
    import numpy as np
    clip = float(np.percentile(norms, cfg.l2_norm_clip_percentile))
    scale = getattr(cfg, "clip_norm_scale", 1.0)
    return max(clip * scale, 1e-6)


def _per_example_gradients(
    model: torch.nn.Module,
    data: Data,
    subgraphs: torch.Tensor,
    batch_indices: torch.Tensor,
    cfg: Config,
    device: torch.device,
) -> dict:
    """Return dict param_name -> tensor [batch_size, *param_shape] (each grad / batch_size)."""
    model.train()
    pad_to = getattr(cfg, "pad_subgraphs_to", 100)
    adj_norm = getattr(cfg, "adjacency_normalization", "inverse-degree")
    B = batch_indices.size(0)
    grads_per_param = {name: [] for name, _ in model.named_parameters()}
    for i in range(B):
        idx = batch_indices[i].item()
        sub_idx = subgraphs[idx]
        sub = make_subgraph_from_indices(data, sub_idx, add_reverse_edges=False, adjacency_normalization=adj_norm)
        sub = sub.to(device)
        model.zero_grad()
        logits = model(sub)
        target = data.y[idx : idx + 1].to(device)
        loss = F.cross_entropy(logits[0:1], target)
        loss.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                grads_per_param[name].append(p.grad.detach().clone() / B)
            else:
                grads_per_param[name].append(torch.zeros_like(p, device=p.device))
    return {name: torch.stack(grads_per_param[name]) for name in grads_per_param}


def run(cfg: Config) -> None:
    torch.manual_seed(getattr(cfg, "rng_seed", cfg.seed))
    device = _device()
    log.info("Using device: %s", device)

    data, labels, masks = get_dataset(cfg)
    data = data.to(device)
    labels = labels.to(device)
    for k in masks:
        masks[k] = masks[k].to(device)

    in_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    cfg.num_classes = num_classes

    train_indices = data.train_mask.nonzero(as_tuple=True)[0]
    num_training_nodes = train_indices.size(0)
    batch_size = min(cfg.batch_size, num_training_nodes)

    model_cfg = _model_config(cfg, in_dim, num_classes)
    model = build_model(model_cfg).to(device)
    lr = getattr(cfg, "learning_rate", cfg.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dp_training = getattr(cfg, "differentially_private_training", True)
    pad_to = getattr(cfg, "pad_subgraphs_to", 100)
    subgraphs = get_subgraphs(data, pad_to=pad_to)
    clip_norm = cfg.clip_norm
    base_sensitivity = 1.0
    max_terms = 1

    if dp_training:
        base_sensitivity = compute_base_sensitivity(cfg)
        max_terms = compute_max_terms_per_node(cfg)
        n_est = getattr(cfg, "num_estimation_samples", min(5000, num_training_nodes))
        clip_norm = _estimate_clip_norm(
            model, data, subgraphs, train_indices, cfg, device, n_est
        )
        log.info("After clip_norm_scale=%.2f: clip_norm=%.4f", getattr(cfg, "clip_norm_scale", 1.0), clip_norm)

    def metrics_str(m: dict) -> str:
        return " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()))

    init_metrics = _compute_metrics(model, data, labels, masks, device)
    log.info("Init: %s", metrics_str(init_metrics))

    target_delta = getattr(cfg, "target_delta", 1e-5)
    if target_delta <= 0 or target_delta >= 1:
        target_delta = 1.0 / (10 * num_training_nodes)
    num_steps = getattr(cfg, "num_training_steps", 500)
    eval_every = getattr(cfg, "evaluate_every_steps", 50)
    ckpt_every = getattr(cfg, "checkpoint_every_steps", 50)
    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    for step in range(1, num_steps + 1):
        perm = torch.randperm(num_training_nodes, device=train_indices.device)
        batch_idx = train_indices[perm[:batch_size]]

        if dp_training and subgraphs is not None:
            grads_tree = _per_example_gradients(
                model, data, subgraphs, batch_idx, cfg, device
            )
            gen = torch.Generator(device=device).manual_seed(cfg.rng_seed + step)
            noisy = dp_step(
                grads_tree,
                clip_norm=clip_norm,
                noise_multiplier=cfg.training_noise_multiplier,
                batch_size=batch_size,
                base_sensitivity=base_sensitivity,
                generator=gen,
            )
            model.zero_grad()
            for name, p in model.named_parameters():
                if name in noisy:
                    p.grad = noisy[name].detach().clone().to(p.device)
            optimizer.step()
        else:
            model.zero_grad()
            adj_norm = getattr(cfg, "adjacency_normalization", "inverse-degree")
            loss_sum = torch.tensor(0.0, device=device)
            for i in range(batch_idx.size(0)):
                idx = batch_idx[i].item()
                sub = make_subgraph_from_indices(
                    data, subgraphs[idx], add_reverse_edges=False,
                    adjacency_normalization=adj_norm,
                )
                sub = sub.to(device)
                logits = model(sub)
                loss_sum = loss_sum + F.cross_entropy(logits[0:1], labels[idx : idx + 1])
            (loss_sum / batch_size).backward()
            optimizer.step()

        if step % eval_every == 0 or step == num_steps:
            m = _compute_metrics(model, data, labels, masks, device)
            eps = 0.0
            if dp_training:
                eps = get_epsilon(
                    steps=step,
                    noise_multiplier=cfg.training_noise_multiplier,
                    batch_size=batch_size,
                    target_delta=target_delta,
                    num_samples=num_training_nodes,
                    max_terms_per_node=max_terms,
                    use_multiterm=(cfg.model == "gcn"),
                )
            m["epsilon"] = eps
            log.info("Step %d: train_acc=%.4f val_acc=%.4f test_acc=%.4f epsilon=%.2f",
                     step, m.get("train_acc", 0), m.get("validation_acc", 0), m.get("test_acc", 0), m.get("epsilon", eps))
            if getattr(cfg, "max_training_epsilon", None) is not None and eps >= cfg.max_training_epsilon:
                log.info("Stopping: epsilon >= max_training_epsilon")
                break
        if step % ckpt_every == 0 or step == num_steps:
            torch.save(model.state_dict(), workdir / "model.pt")

    torch.save(model.state_dict(), workdir / "model.pt")
    log.info("Training done. Checkpoint saved to %s", workdir / "model.pt")
