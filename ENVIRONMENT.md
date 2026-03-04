# Environment & Dependency Policy (Reproduction Project)

This repository is an **academic reproduction** project.

## Target Base Environment (Must Match)
- Base image: `pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`
- Python: `3.10`
- CUDA: `12.1`

**Hard rule:** The code must adapt to this environment.  
Do **NOT** modify/downgrade the environment to match the code.

---

## Preferred Dependency Stack (Mainstream, Minimal)

### Core
- `torch==2.2.0` (already provided by the base image; do not downgrade)
- `torch-geometric==2.5.3`

### PyG Operators (must match torch+CUDA)
Install from the official PyG wheel index for `torch-2.2.0+cu121`:
- `pyg-lib`
- `torch-scatter`
- `torch-sparse`
- `torch-cluster`
- `torch-spline-conv`

Wheel index:
- `https://data.pyg.org/whl/torch-2.2.0+cu121.html`

### Scientific Stack
- `numpy`
- `scipy`
- `pandas`
- `scikit-learn`

### Utilities
- `tqdm`
- `pyyaml`
- `matplotlib` (optional but recommended for debugging/plots)

---

## Installation Commands (Reference)
```bash
python -m pip install -U pip setuptools wheel

# Scientific + utilities
pip install numpy scipy pandas scikit-learn tqdm pyyaml matplotlib

# PyG core
pip install torch-geometric==2.5.3

# PyG operators (Torch 2.2.0 + CUDA 12.1 wheels)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
```
