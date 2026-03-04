"""Download OGBN-Arxiv to match reference repo layout. No extra dependencies (stdlib + pandas).

Reference: refrence_repo/differentially_private_gnns/download_datasets.py.
Target layout: data_dir/ogbn_arxiv/raw/, data_dir/ogbn_arxiv/split/time/.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

_OGB_MASTER_CSV = "https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv"
_OGB_DATASET_NAME = "ogbn-arxiv"


def ensure_ogbn_arxiv(data_dir: str) -> Path:
    """Download ogbn-arxiv if missing; return path to data_dir/ogbn_arxiv. Matches reference layout."""
    data_dir = Path(data_dir).resolve()
    target = data_dir / "ogbn_arxiv"
    if target.exists():
        raw_dir = target / "raw"
        split_dir = target / "split" / "time"
        if raw_dir.exists() and (split_dir.exists() or (target / "split").exists()):
            return target

    data_dir.mkdir(parents=True, exist_ok=True)
    with urlopen(_OGB_MASTER_CSV) as resp:
        master_df = pd.read_csv(BytesIO(resp.read()), index_col=0)
    if _OGB_DATASET_NAME not in master_df.columns:
        raise ValueError(
            f"Unknown dataset {_OGB_DATASET_NAME}. Columns: {list(master_df.columns)}"
        )
    meta = master_df[_OGB_DATASET_NAME]
    download_name = meta["download_name"]
    url = meta["url"]

    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "arxiv.zip"
        with urlopen(url) as resp:
            zip_path.write_bytes(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        extracted = data_dir / download_name
        if not extracted.exists():
            raise FileNotFoundError(
                f"After extract, expected {extracted}. Listing {data_dir}: {list(data_dir.iterdir())}"
            )
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(extracted), str(target))
    return target
