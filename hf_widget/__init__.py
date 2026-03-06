"""Hugging Face Hub upload/download backend for the Streamlit widget."""

from pathlib import Path
import os

# Configurable base paths (from env or defaults relative to cwd)
MODELS_DIR = Path(os.getenv("HF_MODELS_DIR", "models"))
DATASETS_DIR = Path(os.getenv("HF_DATASETS_DIR", "datasets"))
HF_DEFAULT_ORG = os.getenv("HF_DEFAULT_ORG", "")

from hf_widget.upload import upload_model, upload_dataset
from hf_widget.download import download_model, download_dataset

__all__ = [
    "MODELS_DIR",
    "DATASETS_DIR",
    "HF_DEFAULT_ORG",
    "upload_model",
    "upload_dataset",
    "download_model",
    "download_dataset",
]
