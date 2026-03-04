"""Evaluation: load checkpoint, run on full graph, report metrics."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import Data

from .config import Config
from .input_pipeline import get_dataset
from .models import ModelConfig, build_model

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


def run(cfg: Config) -> None:
    device = _device()
    workdir = Path(cfg.workdir)
    ckpt_path = workdir / "model.pt"
    if not ckpt_path.exists():
        log.warning("No checkpoint at %s; skipping eval.", ckpt_path)
        return

    data, labels, masks = get_dataset(cfg)
    in_dim = data.x.size(1)
    num_classes = int(data.y.max().item()) + 1
    model_cfg = _model_config(cfg, in_dim, num_classes)
    model = build_model(model_cfg).to(device)
    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    data = data.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        logits = model(data)
    pred = logits.argmax(dim=-1)
    for name, mask in masks.items():
        m = mask.to(device)
        acc = (pred[m] == labels[m]).float().mean().item()
        log.info("%s accuracy: %.4f", name, acc)
