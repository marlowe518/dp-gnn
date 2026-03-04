"""Dataset loader and download. Matches reference repo data loading and download layout."""

from .download_dataset import ensure_ogbn_arxiv
from .loader import load_dataset, log_data_debug

__all__ = ["load_dataset", "log_data_debug", "ensure_ogbn_arxiv"]
