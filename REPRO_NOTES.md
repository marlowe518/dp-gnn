# Reproduction Notes & Parity Checklist

Reference: [google-research/differentially_private_gnns](https://github.com/google-research/google-research/tree/master/differentially_private_gnns).

## Documentation template

- **Paper:** [cite paper title and venue]
- **Reference code:** `google-research/google-research` → `differentially_private_gnns/`
- **Datasets:** ogbn-arxiv (primary), others as in paper
- **Baselines:** Non-DP GCN; DP-GCN (Alg. 1–5), disjoint vs shared (if applicable)

## Parity checklist

Use this to validate our implementation against the reference and paper.

### Sampling (Alg. 1–3)

- [ ] Neighbor sampling matches reference (hop order, fanout, replacement).
- [ ] Subgraph extraction: same nodes/edges as reference for same seed and batch.
- [ ] Disjoint vs shared sampling (if applicable): behavior matches reference.

### Gradient clipping & noise

- [ ] Per-sample (or per-batch) gradient clipping: same norm bound as paper/reference.
- [ ] Noise scale: same σ (noise multiplier) and sensitivity; Gaussian noise added after clip.
- [ ] Clipping is applied before aggregation (same as reference).

### Privacy accounting

- [ ] Accounting method matches (RDP, moments, or as in reference).
- [ ] (ε, δ) for given (steps, σ, batch size, δ) match reference or paper Table/Figure.
- [ ] Theorem 1 (or paper’s main theorem) parameters reflected in our accounting.

### Metrics & reproducibility

- [ ] Same train/val/test splits and seeds as reference/paper.
- [ ] Accuracy (and other metrics) reproducible for fixed seed; match reference when implemented.
- [ ] Reported curves (accuracy vs ε or vs steps) can be reproduced (after full implementation).

## Implementation status

| Component        | Status   | Notes                    |
|-----------------|----------|--------------------------|
| Config / CLI    | Done     | Skeleton                 |
| Data loaders    | TODO     | Stub only                |
| Sampling (Alg.1–3) | TODO  | Stub only                |
| Subgraph        | TODO     | Stub only                |
| DP-SGD / DP-Adam| TODO     | Stub only                |
| Accounting      | TODO     | Stub only                |
| Train / Eval    | TODO     | Stub only                |

## Environment

- PyTorch 2.2.0, Python 3.10, CUDA 12.1, Ubuntu 22.04.
- See `pyproject.toml` for pinned dependencies.
