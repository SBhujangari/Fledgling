"""Compare LLM vs SLM to determine if fine-tuning is needed."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import openai
from clients import build_client
from eval import (
    EvalStepResult,
    StructuredMetricTracker,
    ToolcallMetricTracker,
    load_jsonl,
)
from langfuse_helpers import langfuse_trace, log_decision_event
from prompts import get_system_prompt


def run_comparison_eval(
    track: str,
    dataset_path: str,
    output_path: str,
    split: str,
    slm_adapter: str | None = None,
) -> Dict[str, any]:
    """Run evaluation comparing LLM (Azure) vs SLM."""
    data = load_jsonl(dataset_path)
    total = len(data)
    if not total:
        raise ValueError(f"Dataset is empty: {dataset_path}")

    # Build both clients
    print("Loading Azure LLM client...")
    llm_client = build_client("azure")
    print("Loading local SLM client...")
    slm_client = build_client("slm", adapter=slm_adapter)

    system_prompt = get_system_prompt(track)

    # Create separate trackers for each model
    llm_tracker = (
        StructuredMetricTracker() if track == "structured" else ToolcallMetricTracker()
    )
    slm_tracker = (
        StructuredMetricTracker() if track == "structured" else ToolcallMetricTracker()
    )

    comparison_results: List[Dict] = []

    for idx, example in enumerate(data, start=1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.prompt},
        ]

        # Get LLM prediction
        try:
            llm_prediction = llm_client.generate(messages)
        except openai.BadRequestError as e:
            if "content_filter" in str(e):
                print(
                    f"[{idx}/{total}] Skipped (LLM content filter): {example.prompt[:50]}..."
                )
                continue
            raise

        # Get SLM prediction
        try:
            slm_prediction = slm_client.generate(messages)
        except Exception as e:
            print(f"[{idx}/{total}] SLM error: {e}")
            slm_prediction = ""

        # Evaluate both
        llm_step = llm_tracker.update(llm_prediction, example.completion)
        slm_step = slm_tracker.update(slm_prediction, example.completion)

        # Compare predictions
        llm_snapshot = llm_tracker.summary()
        slm_snapshot = slm_tracker.summary()

        # Calculate performance gap
        if track == "structured":
            llm_score = llm_snapshot.get("exact_match_rate", 0.0)
            slm_score = slm_snapshot.get("exact_match_rate", 0.0)
        else:
            llm_score = llm_snapshot.get("args_exact_rate", 0.0)
            slm_score = slm_snapshot.get("args_exact_rate", 0.0)

        performance_gap = llm_score - slm_score

        comparison_results.append(
            {
                "prompt_index": idx,
                "prompt": example.prompt,
                "reference": example.completion,
                "llm_prediction": llm_prediction,
                "slm_prediction": slm_prediction,
                "llm_correct": llm_step.exact_match,
                "slm_correct": slm_step.exact_match,
                "llm_issues": llm_step.issues,
                "slm_issues": slm_step.issues,
                "performance_gap": performance_gap,
            }
        )

        # Log decision event
        log_decision_event(
            name=f"compare-{track}-prompt",
            metadata={
                "prompt_index": idx,
                "llm_metrics": llm_step.per_example_metrics,
                "slm_metrics": slm_step.per_example_metrics,
                "performance_gap": performance_gap,
                "track": track,
                "split": split,
            },
            level="WARNING" if performance_gap > 0.1 else "INFO",
        )

        # Progress output
        print(
            f"[{idx}/{total}] track={track} | "
            f"LLM: {llm_score:.2%} | SLM: {slm_score:.2%} | Gap: {performance_gap:+.2%}",
            flush=True
        )

    # Final metrics
    llm_metrics = llm_tracker.summary()
    slm_metrics = slm_tracker.summary()

    # Determine if fine-tuning is recommended
    if track == "structured":
        final_gap = (
            llm_metrics["exact_match_rate"] - slm_metrics["exact_match_rate"]
        )
    else:
        final_gap = llm_metrics["args_exact_rate"] - slm_metrics["args_exact_rate"]

    # Recommendation thresholds
    FINE_TUNE_THRESHOLD = 0.15  # 15% performance gap
    needs_fine_tuning = final_gap > FINE_TUNE_THRESHOLD

    summary = {
        "track": track,
        "split": split,
        "total_examples": total,
        "llm_metrics": llm_metrics,
        "slm_metrics": slm_metrics,
        "performance_gap": final_gap,
        "needs_fine_tuning": needs_fine_tuning,
        "recommendation": (
            "FINE-TUNE RECOMMENDED: LLM significantly outperforms SLM"
            if needs_fine_tuning
            else "SLM performance acceptable, fine-tuning optional"
        ),
    }

    # Save results
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save detailed comparison
    details_path = os.path.splitext(output_path)[0] + "_details.jsonl"
    with open(details_path, "w", encoding="utf-8") as f:
        for result in comparison_results:
            f.write(json.dumps(result) + "\n")

    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Track: {track}")
    print(f"LLM Performance: {llm_metrics}")
    print(f"SLM Performance: {slm_metrics}")
    print(f"Performance Gap: {final_gap:+.2%}")
    print(f"\n{summary['recommendation']}")
    print("=" * 80)

    return summary


def auto_upload_adapter(
    adapter_path: str,
    repo_id: str,
    repo_type: str,
    *,
    commit_message: str,
    path_in_repo: Optional[str],
    private: bool,
    auto_subdir: bool,
    branch: Optional[str],
    token: Optional[str],
    space_sdk: Optional[str],
) -> None:
    """Invoke hf_upload.py to publish the adapter once eval passes."""

    script_path = Path(__file__).with_name("hf_upload.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(f"Missing uploader script at {script_path}")

    adapter_abs = Path(adapter_path).expanduser().resolve()
    if not adapter_abs.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    args = [
        sys.executable,
        str(script_path),
        str(adapter_abs),
        "--repo-id",
        repo_id,
        "--repo-type",
        repo_type,
        "--commit-message",
        commit_message,
    ]

    if private:
        args.append("--private")

    if auto_subdir:
        args.append("--auto-subdir")

    if path_in_repo:
        args.extend(["--path-in-repo", path_in_repo])

    if branch:
        args.extend(["--branch", branch])

    if space_sdk:
        args.extend(["--space-sdk", space_sdk])

    env = os.environ.copy()
    if token:
        env["HUGGING_FACE_HUB_TOKEN"] = token

    print(
        f"[auto-upload] Publishing {adapter_abs} -> {repo_type}:{repo_id} "
        f"(path_in_repo={path_in_repo or '<default>'})"
    )
    result = subprocess.run(
        args,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Auto-upload failed with exit code "
            f"{result.returncode}:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    print("[auto-upload] Upload complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LLM vs SLM to determine if fine-tuning is needed."
    )
    parser.add_argument("--track", choices=("structured", "toolcall"), required=True)
    parser.add_argument("--split", default="eval100", help="Dataset split to use")
    parser.add_argument("--dataset", help="Override dataset path")
    parser.add_argument("--out", help="Path for comparison results JSON")
    parser.add_argument(
        "--adapter", help="Optional LoRA adapter directory for the SLM"
    )
    parser.add_argument(
        "--auto-upload-repo",
        help=(
            "Automatically push the evaluated adapter to this Hugging Face repo "
            "when the performance gap is within the threshold."
        ),
    )
    parser.add_argument(
        "--auto-upload-repo-type",
        choices=("model", "dataset", "space"),
        default="space",
        help="Repo type for automatic upload (default: space).",
    )
    parser.add_argument(
        "--auto-upload-gap-threshold",
        type=float,
        default=0.0,
        help=(
            "Upload adapters when performance_gap <= threshold "
            "(default 0.0 means upload once SLM matches/exceeds the LLM)."
        ),
    )
    parser.add_argument(
        "--auto-upload-path-in-repo",
        help="Optional destination subdirectory inside the repo.",
    )
    parser.add_argument(
        "--auto-upload-branch",
        help="Optional branch name to push to when uploading.",
    )
    parser.add_argument(
        "--auto-upload-commit-message",
        help="Override the default auto-upload commit message.",
    )
    parser.add_argument(
        "--auto-upload-private",
        action="store_true",
        help="Mark the repo as private when creating it (existing repos unaffected).",
    )
    parser.add_argument(
        "--auto-upload-auto-subdir",
        action="store_true",
        help="Mirror the local folder name in the repo (adds --auto-subdir).",
    )
    parser.add_argument(
        "--auto-upload-token",
        help="Explicit Hugging Face token to use for uploads (optional).",
    )
    parser.add_argument(
        "--auto-upload-space-sdk",
        choices=("gradio", "streamlit", "docker", "static"),
        help="SDK to set if a new Space repo must be created.",
    )
    args = parser.parse_args()

    # Resolve paths relative to this file so the script works from any CWD.
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = args.dataset or os.path.join(
        base_dir, "02_dataset", args.track, f"{args.split}.jsonl"
    )
    default_out = os.path.join(
        base_dir, "05_eval", f"comparison_{args.track}_{args.split}.json"
    )
    output_path = args.out or default_out

    with langfuse_trace(
        name=f"compare-llm-slm-{args.track}-{args.split}",
        metadata={
            "track": args.track,
            "dataset": dataset_path,
            "output": output_path,
            "split": args.split,
        },
    ) as trace:
        summary = run_comparison_eval(
            track=args.track,
            dataset_path=dataset_path,
            output_path=output_path,
            split=args.split,
            slm_adapter=args.adapter,
        )
        trace.update(metadata={"summary": summary})

    if args.auto_upload_repo and not args.adapter:
        raise SystemExit("--auto-upload-repo requires --adapter to be set.")

    if args.auto_upload_repo:
        final_gap = float(summary.get("performance_gap", 1.0))
        if final_gap <= args.auto_upload_gap_threshold:
            commit_message = (
                args.auto_upload_commit_message
                or (
                    f"auto upload {args.track} adapter after eval "
                    f"{datetime.now(timezone.utc).isoformat()}"
                )
            )
            auto_upload_adapter(
                adapter_path=args.adapter,
                repo_id=args.auto_upload_repo,
                repo_type=args.auto_upload_repo_type,
                commit_message=commit_message,
                path_in_repo=args.auto_upload_path_in_repo,
                private=args.auto_upload_private,
                auto_subdir=args.auto_upload_auto_subdir,
                branch=args.auto_upload_branch,
                token=args.auto_upload_token,
                space_sdk=args.auto_upload_space_sdk,
            )
        else:
            print(
                "[auto-upload] Skipped because performance gap "
                f"{final_gap:.2%} exceeds threshold "
                f"{args.auto_upload_gap_threshold:.2%}."
            )


if __name__ == "__main__":
    main()
