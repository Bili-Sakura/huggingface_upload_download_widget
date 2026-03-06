"""Streamlit WebUI for Hugging Face models & datasets upload/download.

Share with community: pip install -e . && streamlit run app.py
"""

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import os

from hf_widget import (
    MODELS_DIR,
    DATASETS_DIR,
    HF_DEFAULT_ORG,
    upload_model,
    upload_dataset,
    download_model,
    download_dataset,
)

HF_ENDPOINT_DEFAULT = os.getenv("HF_ENDPOINT_DEFAULT", "https://hf-mirror.com")


class _TeeWriter:
    """Writes to both original stream and buffer. Safe minimal monitor: terminal + webui."""

    def __init__(self, original, buffer: io.StringIO):
        self._original = original
        self._buffer = buffer

    def write(self, text: str) -> None:
        if not text:
            return
        try:
            self._original.write(text)
            self._original.flush()
        except (OSError, ValueError):
            pass
        try:
            self._buffer.write(text)
        except (ValueError, TypeError):
            pass

    def flush(self) -> None:
        try:
            self._original.flush()
        except (OSError, ValueError):
            pass
        try:
            self._buffer.flush()
        except (ValueError, TypeError):
            pass

    def isatty(self) -> bool:
        return getattr(self._original, "isatty", lambda: False)()


@contextlib.contextmanager
def _tee_stdout_stderr():
    """Tee stdout/stderr to terminal and buffer. Yields buffer for log capture."""
    buf = io.StringIO()
    tee_out = _TeeWriter(sys.__stdout__, buf)
    tee_err = _TeeWriter(sys.__stderr__, buf)
    with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
        yield buf


def _run_with_log_capture(func, *args, **kwargs):
    """Run func, teeing stdout/stderr to terminal and buffer; return (result, log_text)."""
    with _tee_stdout_stderr() as buf:
        result = func(*args, **kwargs)
    return result, buf.getvalue()


def _show_log(log_text: str) -> None:
    """Render captured log in the web UI."""
    if log_text.strip():
        st.code(log_text.strip(), language=None)


def _patterns(val: str | None) -> list[str] | None:
    if not val or not val.strip():
        return None
    return [p.strip() for p in val.split(",") if p.strip()]


st.set_page_config(page_title="HF Hub", page_icon="🤗", layout="wide")
st.title("🤗 Hugging Face Hub")
st.caption("Upload & download models and datasets. Uploads/downloads support automatic resume on interruption.")

# Shared config
with st.sidebar:
    token = st.text_input(
        "HF Token",
        type="password",
        help="Required for upload; optional for public download. Note: For private repos, HF Token is required.",
    )
    endpoint = st.text_input(
        "HF Endpoint",
        value=HF_ENDPOINT_DEFAULT,
        help="Default: hf-mirror.com (for mirror access).",
    )
    endpoint = (endpoint or "").strip() or HF_ENDPOINT_DEFAULT
    default_org = st.text_input(
        "Default org (optional)",
        value=HF_DEFAULT_ORG,
        placeholder="your-username",
        help="Used when repo ID has no slash (e.g. 'my-model' -> org/my-model).",
    )

    with st.expander("Advanced (large repos, resume)", expanded=False):
        st.caption(
            "Upload: num_workers (fewer = better resume on slow links). Download: max_workers, force re-download."
        )
        num_workers = st.number_input(
            "Upload workers",
            min_value=1,
            max_value=32,
            value=4,
            help="Half of CPU cores by default. Lower for slow connections.",
        )
        max_workers = st.number_input(
            "Download workers",
            min_value=1,
            max_value=16,
            value=8,
            help="Concurrent file downloads.",
        )
        force_download = st.checkbox(
            "Force re-download",
            value=False,
            help="Ignore cache, re-download all files.",
        )
        allow_patterns_in = st.text_input(
            "Include patterns (comma)",
            placeholder="*.safetensors, *.bin",
            help="Glob patterns to include only.",
        )
        ignore_patterns_in = st.text_input(
            "Exclude patterns (comma)",
            placeholder="*.log, __pycache__",
            help="Glob patterns to exclude.",
        )

tab_models, tab_datasets = st.tabs(["Models", "Datasets"])

with tab_models:
    mod_upload, mod_download = st.columns(2)
    with mod_upload:
        st.subheader("Upload model")
        st.caption(f"From: {MODELS_DIR.resolve()}")
        model_repo = st.text_input("Repo ID", key="model_repo", placeholder="my-model or org/my-model")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload", key="upload_model"):
                if not token:
                    st.error("Set HF Token in sidebar.")
                elif not model_repo.strip():
                    st.error("Enter repo ID.")
                else:
                    with st.status("Uploading…", expanded=True) as status:
                        try:
                            _, log_txt = _run_with_log_capture(
                                upload_model,
                                model_repo.strip(),
                                token,
                                endpoint,
                                num_workers=num_workers,
                                allow_patterns=_patterns(allow_patterns_in),
                                ignore_patterns=_patterns(ignore_patterns_in),
                            )
                            _show_log(log_txt)
                            status.update(label="Done", state="complete")
                            st.success(f"Uploaded {model_repo}")
                        except SystemExit as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(str(e))
        with col2:
            if st.button("Upload all", key="upload_all_models"):
                if not token:
                    st.error("Set HF Token in sidebar.")
                else:
                    with st.status("Uploading all…", expanded=True) as status:
                        try:
                            upload_base = (
                                MODELS_DIR / HF_DEFAULT_ORG
                                if HF_DEFAULT_ORG
                                else MODELS_DIR
                            )
                            if not upload_base.is_dir():
                                st.error(
                                    f"Folder not found: {upload_base}"
                                    + (f" (models under {HF_DEFAULT_ORG}/)" if HF_DEFAULT_ORG else "")
                                )
                            else:
                                dirs = [d.name for d in upload_base.iterdir() if d.is_dir()]
                                with _tee_stdout_stderr() as buf:
                                    for name in dirs:
                                        repo_id = f"{HF_DEFAULT_ORG}/{name}" if HF_DEFAULT_ORG else name
                                        st.write(f"Uploading {repo_id}…")
                                        upload_model(
                                            repo_id,
                                            token,
                                            endpoint,
                                            num_workers=num_workers,
                                            allow_patterns=_patterns(allow_patterns_in),
                                            ignore_patterns=_patterns(ignore_patterns_in),
                                        )
                                _show_log(buf.getvalue())
                                status.update(label="Done", state="complete")
                                st.success(f"Uploaded {len(dirs)} models.")
                        except SystemExit as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(str(e))

    with mod_download:
        st.subheader("Download model")
        st.caption(f"To: {MODELS_DIR.resolve()}")
        dl_model_repo = st.text_input(
            "Repo ID", key="dl_model_repo", placeholder="my-model or org/my-model"
        )
        if st.button("Download", key="download_model"):
            if not dl_model_repo.strip():
                st.error("Enter repo ID.")
            else:
                with st.status("Downloading…", expanded=True) as status:
                    try:
                        out, log_txt = _run_with_log_capture(
                            download_model,
                            dl_model_repo.strip(),
                            token or None,
                            endpoint,
                            max_workers=max_workers,
                            allow_patterns=_patterns(allow_patterns_in),
                            ignore_patterns=_patterns(ignore_patterns_in),
                            force_download=force_download,
                            default_org=default_org.strip() or None,
                        )
                        _show_log(log_txt)
                        status.update(label="Done", state="complete")
                        st.success(f"Downloaded to {out}")
                    except SystemExit as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(str(e))

with tab_datasets:
    ds_upload, ds_download = st.columns(2)
    with ds_upload:
        st.subheader("Upload dataset")
        st.caption(f"From: {DATASETS_DIR.resolve()}")
        ds_path = st.text_input(
            "Folder path or name",
            key="ds_path",
            placeholder="my-dataset or /path/to/dataset",
        )
        if st.button("Upload dataset", key="upload_dataset"):
            if not token:
                st.error("Set HF Token in sidebar.")
            elif not ds_path.strip():
                st.error("Enter folder path or name.")
            else:
                with st.status("Uploading…", expanded=True) as status:
                    try:
                        path = Path(ds_path.strip())
                        if not path.is_absolute():
                            path = DATASETS_DIR / path
                        org = (default_org or "").strip()
                        repo_id = f"{org}/{path.name}" if org else path.name
                        _, log_txt = _run_with_log_capture(
                            upload_dataset,
                            path,
                            repo_id,
                            token,
                            endpoint,
                            num_workers=num_workers,
                            allow_patterns=_patterns(allow_patterns_in),
                            ignore_patterns=_patterns(ignore_patterns_in),
                        )
                        _show_log(log_txt)
                        status.update(label="Done", state="complete")
                        st.success(f"Uploaded {path} to {repo_id}")
                    except SystemExit as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(str(e))

    with ds_download:
        st.subheader("Download dataset")
        st.caption(f"To: {DATASETS_DIR.resolve()}")
        dl_ds_repo = st.text_input(
            "Repo ID", key="dl_ds_repo", placeholder="my-dataset or org/my-dataset"
        )
        if st.button("Download dataset", key="download_dataset"):
            if not dl_ds_repo.strip():
                st.error("Enter repo ID.")
            else:
                with st.status("Downloading…", expanded=True) as status:
                    try:
                        out, log_txt = _run_with_log_capture(
                            download_dataset,
                            dl_ds_repo.strip(),
                            token or None,
                            endpoint,
                            max_workers=max_workers,
                            allow_patterns=_patterns(allow_patterns_in),
                            ignore_patterns=_patterns(ignore_patterns_in),
                            force_download=force_download,
                            default_org=default_org.strip() or None,
                        )
                        _show_log(log_txt)
                        status.update(label="Done", state="complete")
                        st.success(f"Downloaded to {out}")
                    except SystemExit as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(str(e))
