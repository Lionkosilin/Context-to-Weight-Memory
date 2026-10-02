#!/usr/bin/env python3
"""Download a Hugging Face model into models/."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "Qwen/Qwen3-4B-Base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="defaults to models/<repo name>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = args.local_dir or (ROOT / "models" / args.repo_id.split("/")[-1])
    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=args.repo_id, local_dir=str(local_dir))
    print(f"downloaded {args.repo_id} -> {path}")


if __name__ == "__main__":
    main()
