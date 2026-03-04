#!/usr/bin/env bash
# Install dependencies for DP-GNN reproduction.
# Target base image: pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04
# See ENVIRONMENT.md for policy. Use only pip; no conda/poetry/uv.

set -e

# 1. Upgrade pip, setuptools, wheel
python -m pip install -U pip setuptools wheel

# 2. Scientific stack + utilities (numpy, scipy, pandas, scikit-learn, tqdm, pyyaml, matplotlib)
pip install numpy scipy pandas scikit-learn tqdm pyyaml matplotlib

# 3. PyTorch Geometric core (torch is already in base image; do not reinstall)
pip install torch-geometric==2.5.3

# 4. PyG operators: must match torch 2.2.0 + CUDA 12.1 (install from official wheel index)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
