"""Upload models and datasets to Hugging Face Hub.

Uses upload_large_folder for resilient uploads: multi-worker parallelism,
automatic resume on interruption (metadata in .cache/huggingface/upload),
and validation against repo limits.
"""

from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from hf_widget import MODELS_DIR, DATASETS_DIR, HF_DEFAULT_ORG
from hf_widget.download import _full_repo_id


def upload_model(
    repo_id: str,
    token: str,
    endpoint: str | None = None,
    *,
    models_dir: Path | None = None,
    num_workers: Optional[int] = None,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    print_report_every: int = 30,
) -> None:
    """Upload a single model folder to Hugging Face (resumable, multi-worker)."""
    base = models_dir or MODELS_DIR
    full_id = _full_repo_id(repo_id, HF_DEFAULT_ORG)
    folder_path = base / full_id
    if not folder_path.is_dir():
        raise SystemExit(f"Folder not found: {folder_path}")

    api = HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)
    api.create_repo(repo_id=full_id, repo_type="model", exist_ok=True)
    api.upload_large_folder(
        folder_path=str(folder_path),
        repo_id=full_id,
        repo_type="model",
        num_workers=num_workers,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        print_report=True,
        print_report_every=print_report_every,
    )
    print(f"Uploaded {folder_path} to {full_id}")


def upload_dataset(
    folder_path: Path,
    repo_id: str,
    token: str,
    endpoint: str | None = None,
    *,
    num_workers: Optional[int] = None,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    print_report_every: int = 30,
) -> None:
    """Upload a dataset folder to Hugging Face (resumable, multi-worker)."""
    if not folder_path.is_dir():
        raise SystemExit(f"Folder not found: {folder_path}")

    api = HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        repo_type="dataset",
        num_workers=num_workers,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        print_report=True,
        print_report_every=print_report_every,
    )
    print(f"Uploaded {folder_path} to {repo_id}")
