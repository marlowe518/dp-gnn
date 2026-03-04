# DP-GNN Reproduction

Skeleton repository for reproducing **Differentially Private Graph Neural Networks** from [google-research/differentially_private_gnns](https://github.com/google-research/google-research/tree/master/differentially_private_gnns).

## Purpose

- Reproduce the reference implementation and paper results.
- Provide a clean, minimal project layout with placeholder modules and clear TODOs.
- Enable incremental implementation of sampling, subgraph extraction, DP-SGD/DP-Adam, and privacy accounting without changing structure.

## Scope

- **In scope:** Project structure, configs, CLI, logging, data loader interfaces, and stub modules for all algorithm components. Parity checklist and validation plan.
- **Out of scope (for now):** No algorithm logic is implemented yet. Sampling, subgraph extraction, DP-SGD, and privacy accounting are interfaces only.

## Non-Goals (Current Phase)

- No real neighbor sampling, subgraph extraction, or DP optimization.
- No dataset download or training; only dry-run and stub wiring.
- No extra ML libraries beyond PyTorch and PyG (as placeholder).

## Environment

- **Runtime:** Ubuntu 22.04, Python 3.10, PyTorch 2.2.0, CUDA 12.1.
- **Dependencies:** Pinned in `pyproject.toml`; install with `pip install -e ".[dev]"`.
- PyG may require separate wheels for your CUDA version; see project docs.

## Quickstart

```bash
# From repo root (recommended: install so CLI and tests see the package)
pip install -e ".[dev]"

# Dry run (no training; prints which stubs would be called)
python -m dpgnn_repro.cli --config configs/base.json --workdir outputs/dev --dry_run --debug

# With paper-like config
python -m dpgnn_repro.cli --config configs/ogbn_arxiv_dp_gcn_adam.json --workdir outputs/ogbn --dry_run
```

Without installing (e.g. no PyTorch/PyG yet): `PYTHONPATH=src python -m dpgnn_repro.cli --config configs/base.json --workdir outputs/dev --dry_run`.

Resolved config is written to `<workdir>/config.resolved.json`. The CLI creates `workdir` if missing.

## How We Will Validate Parity

1. **Sampling:** Match reference neighbor/subgraph sampling (Alg. 1–3).
2. **Gradient clipping & noise:** Same clipping norm and noise scale as paper/reference.
3. **Privacy accounting:** Same (ε, δ) for given steps and noise; compare to reference accounting.
4. **Metrics:** Reproduce reported accuracy / privacy curves on the same datasets (e.g. ogbn-arxiv).

See `REPRO_NOTES.md` for the full parity checklist.

## Roadmap

1. **Phase 1 (current):** Skeleton only — configs, CLI, placeholders, tests that assert imports and flags.
2. **Phase 2:** Implement sampling (Alg. 1–3) and subgraph extraction; unit tests.
3. **Phase 3:** Implement DP-SGD/DP-Adam and privacy accounting; integration tests.
4. **Phase 4:** Data loaders, training loop, and full pipeline; parity runs vs reference.
