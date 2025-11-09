#!/usr/bin/env python
"""Utility for pushing local artifacts to the Hugging Face Hub.

Examples
--------
Upload the structured adapter folder to a private model repo:

    python slm_swap/hf_upload.py \
        slm_swap/04_ft/adapter_structured \
        --repo-id your-username/slm-structured-adapter \
        --private

Upload both structured + tool-call adapters to different subfolders:

    python slm_swap/hf_upload.py \
        slm_swap/04_ft/adapter_structured \
        slm_swap/04_ft/adapter_toolcall \
        --repo-id your-username/slm-adapters \
        --auto-subdir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from huggingface_hub import HfApi, utils as hf_utils
from huggingface_hub.utils import HfHubHTTPError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create/update a Hugging Face repo and upload local files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files or directories to upload.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face repo identifier, e.g. `username/my-model`.",
    )
    parser.add_argument(
        "--repo-type",
        choices=("model", "dataset", "space"),
        default="model",
        help="Repo type to create/upload to (default: model).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/ensure the repo is private.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Hugging Face token. Falls back to $HUGGING_FACE_HUB_TOKEN or cached login.",
    )
    parser.add_argument(
        "--commit-message",
        default="sync local artifacts",
        help="Commit message to use for uploads.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        help="Optional branch name on the Hub (defaults to main).",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help="Upload everything under this subdirectory in the repo.",
    )
    parser.add_argument(
        "--auto-subdir",
        action="store_true",
        help=(
            "Place each provided path in its own subdirectory named after the file/folder. "
            "Ignored if --path-in-repo is set."
        ),
    )
    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[".git/*", "**/__pycache__/*"],
        help="Glob patterns to ignore during upload (applies to folders).",
    )
    return parser.parse_args()


def resolve_token(provided: Optional[str]) -> str:
    token = provided or os.environ.get("HUGGING_FACE_HUB_TOKEN") or hf_utils.get_token()
    if not token:
        raise SystemExit(
            "Missing Hugging Face token. Pass --token, set HUGGING_FACE_HUB_TOKEN, "
            "or run `huggingface-cli login`."
        )
    return token


def ensure_repo(api: HfApi, repo_id: str, repo_type: str, private: bool) -> None:
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        raise SystemExit(f"Failed to create repo {repo_id!r}: {exc}") from exc


def upload_path(
    api: HfApi,
    path: Path,
    repo_id: str,
    repo_type: str,
    commit_message: str,
    branch: Optional[str],
    path_in_repo: Optional[str],
    ignore_patterns: Iterable[str],
    token: str,
) -> None:
    if not path.exists():
        raise SystemExit(f"Path {path} does not exist.")

    resolved_target = path_in_repo.rstrip("/") if path_in_repo else None

    if path.is_file():
        destination = resolved_target or path.name
        print(f"Uploading file {path} -> {repo_id}:{destination}")
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=destination,
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=commit_message,
            revision=branch,
        )
    else:
        destination = resolved_target or ""
        if destination:
            print(f"Uploading folder {path} -> {repo_id}:{destination}/")
        else:
            print(f"Uploading folder {path} -> {repo_id}:/")

        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            revision=branch,
            commit_message=commit_message,
            path_in_repo=destination or None,
            ignore_patterns=list(ignore_patterns),
        )


def main() -> None:
    args = parse_args()
    token = resolve_token(args.token)
    api = HfApi(token=token)

    ensure_repo(api, args.repo_id, args.repo_type, args.private)

    for input_path in args.paths:
        path = Path(input_path).expanduser().resolve()

        if args.path_in_repo:
            target = args.path_in_repo
        elif args.auto_subdir:
            target = path.name
        else:
            target = None

        upload_path(
            api=api,
            path=path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
            branch=args.branch,
            path_in_repo=target,
            ignore_patterns=args.ignore,
            token=token,
        )

    print(f"Done. View your repo at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Aborted by user.")
