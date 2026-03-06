# Hugging Face Upload/Download Widget

A Streamlit web UI for uploading and downloading Hugging Face models and datasets. Supports automatic resume on interruption, multi-worker parallelism, and configurable endpoints (including mirrors).

## Features

- **Models & datasets** – Upload and download both with the same interface
- **Resumable** – Interrupted uploads/downloads can be resumed
- **Progress & logs** – Output visible in both the web UI and terminal
- **Configurable** – Custom base directories, endpoint (e.g. hf-mirror.com), default org
- **Pattern filtering** – Include/exclude files by glob patterns

## Quick start

```bash
# Clone or copy this project, then:
cd huggingface_upload_download_widget
pip install -r requirements.txt
cp .env.example .env   # Edit .env with your HF_TOKEN
streamlit run app.py
```

## Configuration

| Env variable       | Description                         | Default   |
|--------------------|-------------------------------------|-----------|
| `HF_TOKEN`         | Hugging Face token (for uploads)    | -         |
| `HF_MODELS_DIR`    | Base directory for models            | `models`  |
| `HF_DATASETS_DIR`  | Base directory for datasets         | `datasets`|
| `HF_DEFAULT_ORG`   | Default org when repo ID has no `/` | (empty)   |
| `HF_ENDPOINT_DEFAULT` | Hub API endpoint                 | `https://hf-mirror.com` |

Create a `.env` file (see `.env.example`) or set these in your environment.

## Usage

1. **Upload model** – Enter repo ID (e.g. `my-model` or `org/my-model`). Files are read from `{HF_MODELS_DIR}/{repo_name}/`.
2. **Download model** – Enter repo ID. Files are saved to `{HF_MODELS_DIR}/{repo_name}/`.
3. **Upload dataset** – Enter folder path or name under `HF_DATASETS_DIR`. Repo ID is built from the default org and folder name.
4. **Download dataset** – Enter repo ID. Files are saved to `{HF_DATASETS_DIR}/{repo_name}/`.

**Upload all** – Uploads every subdirectory under `HF_MODELS_DIR` as separate model repos.

## Project structure

```
huggingface_upload_download_widget/
├── app.py              # Streamlit app
├── hf_widget/          # Backend (upload/download logic)
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
