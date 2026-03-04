"""Configuration schema and load/save for DP-GNN reproduction.

Target: mirror reference repo config (flags / config files).
Paper: algorithm hyperparameters (clipping, noise, steps, etc.).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Global config schema. All fields have defaults for dry-run / skeleton."""

    # Data
    dataset: str = "ogbn-arxiv"
    data_dir: str = "data"
    # Sampling (Alg. 1–3 placeholders)
    num_layers: int = 2
    batch_size: int = 256
    num_neighbors: int = 10
    # DP (placeholder)
    clip_norm: float = 1.0
    noise_multiplier: float = 0.1
    # Training
    epochs: int = 100
    lr: float = 1e-3
    optimizer: str = "adam"
    # Privacy accounting (placeholder)
    target_delta: float = 1e-5
    # Run
    seed: int = 42
    workdir: str = "outputs/dev"
    debug: bool = False
    dry_run: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: str | Path) -> Config:
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str | Path) -> None:
        """Save config to JSON (e.g. config.resolved.json)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def merge_cli(self, workdir: str | None = None, debug: bool | None = None, dry_run: bool | None = None) -> None:
        """Update config from CLI flags (in-place)."""
        if workdir is not None:
            self.workdir = workdir
        if debug is not None:
            self.debug = debug
        if dry_run is not None:
            self.dry_run = dry_run
