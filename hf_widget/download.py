"""Download models and datasets from Hugging Face Hub.

Uses snapshot_download with automatic resume (Range headers, .incomplete files),
concurrent downloads (max_workers), and pattern filtering.
"""

from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download

from hf_widget import MODELS_DIR, DATASETS_DIR, HF_DEFAULT_ORG


def _full_repo_id(repo_id: str, default_org: str) -> str:
    """Return repo_id with default org prefix if not already qualified."""
    if "/" in repo_id:
        return repo_id
    if default_org:
        return f"{default_org}/{repo_id}"
    return repo_id


def download_model(
    repo_id: str,
    token: str | None = None,
    endpoint: str | None = None,
    local_dir: Path | None = None,
    *,
    models_dir: Path | None = None,
    default_org: str | None = None,
    max_workers: int = 8,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    force_download: bool = False,
) -> Path:
    """Download a model from Hugging Face (resumable, concurrent)."""
    base = models_dir or MODELS_DIR
    full_id = _full_repo_id(repo_id, default_org or HF_DEFAULT_ORG)
    out_dir = local_dir or (base / Path(repo_id).name)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=full_id,
        repo_type="model",
        local_dir=str(out_dir),
        token=token,
        endpoint=endpoint,
        max_workers=max_workers,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force_download,
    )
    print(f"Downloaded {full_id} to {out_dir}")
    return out_dir


def download_dataset(
    repo_id: str,
    token: str | None = None,
    endpoint: str | None = None,
    local_dir: Path | None = None,
    *,
    datasets_dir: Path | None = None,
    default_org: str | None = None,
    max_workers: int = 8,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    force_download: bool = False,
) -> Path:
    """Download a dataset from Hugging Face (resumable, concurrent)."""
    base = datasets_dir or DATASETS_DIR
    full_id = _full_repo_id(repo_id, default_org or HF_DEFAULT_ORG)
    out_dir = local_dir or (base / Path(repo_id).name)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=full_id,
        repo_type="dataset",
        local_dir=str(out_dir),
        token=token,
        endpoint=endpoint,
        max_workers=max_workers,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        force_download=force_download,
    )
    print(f"Downloaded {full_id} to {out_dir}")
    return out_dir
