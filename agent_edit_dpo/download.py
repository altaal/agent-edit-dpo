"""Download the exact public base-model revision recorded with this experiment."""

import json
from pathlib import Path


def main():
    from huggingface_hub import snapshot_download
    source = json.loads(Path("model-source.json").read_text())
    snapshot_download(source["model_id"], revision=source["revision"], local_dir=".model",
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "LICENSE"])


if __name__ == "__main__":
    main()
