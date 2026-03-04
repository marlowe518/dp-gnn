"""CLI entrypoint for DP-GNN reproduction.

Parses args, loads config, creates workdir, saves config.resolved.json.
Calls train/eval stubs; supports --dry_run to print stubs and exit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Config
from .logging import setup_logging
from . import train as train_mod
from . import eval as eval_mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DP-GNN reproduction (skeleton)")
    p.add_argument("--config", type=str, default="configs/base.json", help="Path to config JSON")
    p.add_argument("--workdir", type=str, default=None, help="Output directory (overrides config)")
    p.add_argument("--dataset", type=str, default=None, help="Dataset name (e.g. ogbn-arxiv, toy)")
    p.add_argument("--data_root", type=str, default=None, help="Data root directory")
    p.add_argument("--split_mode", type=str, default=None, choices=("transductive", "disjoint"), help="Split mode")
    p.add_argument("--seed", type=int, default=None, help="Random seed")
    p.add_argument("--feature_norm", type=str, default=None, choices=("none", "standard"), help="Feature normalization")
    p.add_argument("--adjacency_normalization", type=str, default=None, choices=("none", "inverse-degree", "inverse-sqrt-degree"), help="Adjacency normalization (for pipeline; applied later)")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--dry_run", action="store_true", help="Print stubs and exit without training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config.load(args.config)
    cfg.merge_cli(
        workdir=args.workdir,
        debug=args.debug,
        dry_run=args.dry_run,
        dataset=args.dataset,
        data_root=args.data_root,
        split_mode=args.split_mode,
        seed=args.seed,
        feature_norm=args.feature_norm,
        adjacency_normalization=args.adjacency_normalization,
    )

    workdir = Path(cfg.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    cfg.save(workdir / "config.resolved.json")

    log = setup_logging(debug=cfg.debug)
    log.info("Config: %s", cfg.workdir)
    if cfg.dry_run:
        log.info("DRY RUN: no training")
        log.info("Would call: train.run(cfg), then eval.run(cfg)")
        log.info("Stubs: data load -> sampling -> subgraph -> dp step -> accounting")
        if cfg.debug:
            from .data import load_dataset
            from .data.loader import log_data_debug
            data = load_dataset(cfg)
            log_data_debug(data, cfg)
        return

    if cfg.debug:
        from .data import load_dataset
        from .data.loader import log_data_debug
        data = load_dataset(cfg)
        log_data_debug(data, cfg)

    train_mod.run(cfg)
    eval_mod.run(cfg)


if __name__ == "__main__":
    main()
