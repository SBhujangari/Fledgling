"""End-to-end dummy pipeline that exercises the Langfuse → dataset → dry-run fine-tune loop."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
SLM_SWAP_DIR = REPO_ROOT / "python-pipeline" / "slm_swap"
PROGRESS_FILE = REPO_ROOT / "slm_swap" / "logs" / "finetune_progress.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dummy agent → dataset → dry-run FT loop.")
    parser.add_argument(
        "--requests",
        type=Path,
        default=REPO_ROOT / "datasets" / "dummy_agent_requests.jsonl",
        help="Request JSONL consumed by dummy_agent_workflow.py",
    )
    parser.add_argument(
        "--traces",
        type=Path,
        default=REPO_ROOT / "storage" / "dummy_langfuse_traces.jsonl",
        help="Where to store generated trace dictionaries.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=SLM_SWAP_DIR / "02_dataset" / "dummy_langfuse",
        help="Output folder for structured/toolcall splits.",
    )
    parser.add_argument(
        "--adapter-root",
        type=Path,
        default=SLM_SWAP_DIR / "04_ft" / "dummy_langfuse",
        help="Where dry-run adapters/metadata should be written.",
    )
    parser.add_argument("--agent-id", default="dummy-agent", help="Agent id stored in trace metadata.")
    parser.add_argument("--count", type=int, default=16, help="Number of dummy traces to synthesize.")
    parser.add_argument("--clean", action="store_true", help="Delete traces/datasets/adapters before running.")
    return parser.parse_args()


def run(cmd: list[str]):
    display = " ".join(cmd)
    print(f"\n>>> {display}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def remove_path(path: Path):
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def load_track_stats(adapter_root: Path, track: str) -> Dict[str, float]:
    summary_path = adapter_root / f"{track}_adapter" / f"dry_run_{track}.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def write_progress_snapshot(track_stats: Dict[str, Dict[str, float]], duration: float, dataset_root: Path):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    total_examples = sum(int(stats.get("train_examples", 0)) for stats in track_stats.values())
    now = datetime.now(timezone.utc)
    elapsed_seconds = round(duration, 2)
    steps_per_minute = (total_examples / duration * 60) if duration > 0 else None
    snapshot = {
        "status": "completed",
        "message": f"Dummy dry-run finished for tracks: {', '.join(sorted(track_stats)) or 'n/a'}",
        "updatedAt": now.isoformat(),
        "logUpdatedAt": now.isoformat(),
        "logFile": str(dataset_root),
        "sourcePath": str(PROGRESS_FILE),
        "pid": None,
        "pidAlive": False,
        "hardware": {
            "summary": "Dummy dry-run (no GPU required)",
            "gpuCount": 0,
            "gpus": [],
            "cudaVisibleDevices": None,
        },
        "currentStep": total_examples,
        "totalSteps": total_examples,
        "percentComplete": 100.0 if total_examples else 0.0,
        "elapsedSeconds": elapsed_seconds,
        "remainingSeconds": 0,
        "remainingDisplay": "0s",
        "avgStepSeconds": (duration / total_examples) if total_examples else None,
        "stepTimeDisplay": format_duration(duration / total_examples) if total_examples else None,
        "stepsPerMinute": steps_per_minute,
        "eta": now.isoformat(),
        "etaTimestamp": int(now.timestamp()),
        "rawLine": f"dummy-run duration={format_duration(duration)}",
    }
    with PROGRESS_FILE.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2)
    print(f"Wrote training snapshot -> {PROGRESS_FILE}")


def main():
    args = parse_args()
    start_time = time.perf_counter()
    if args.clean:
        remove_path(args.traces)
        remove_path(args.dataset_root)
        remove_path(args.adapter_root)

    args.dataset_root.mkdir(parents=True, exist_ok=True)
    args.adapter_root.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            str(SLM_SWAP_DIR / "dummy_agent_workflow.py"),
            "--requests",
            str(args.requests),
            "--out-json",
            str(args.traces),
            "--agent-id",
            args.agent_id,
            "--count",
            str(args.count),
        ]
    )

    run(
        [
            sys.executable,
            str(SLM_SWAP_DIR / "langfuse_dataset.py"),
            "--trace-json",
            str(args.traces),
            "--agent-id",
            args.agent_id,
            "--output-root",
            str(args.dataset_root),
            "--limit",
            str(args.count),
        ]
    )

    track_stats: Dict[str, Dict[str, float]] = {}
    for track in ("structured", "toolcall"):
        run(
            [
                sys.executable,
                str(SLM_SWAP_DIR / "train_unsloth.py"),
                "--track",
                track,
                "--train",
                str(args.dataset_root / track / "train.jsonl"),
                "--val",
                str(args.dataset_root / track / "val.jsonl"),
                "--out",
                str(args.adapter_root / f"{track}_adapter"),
                "--dry-run",
            ]
        )
        try:
            stats = load_track_stats(args.adapter_root, track)
            track_stats[track] = stats
        except FileNotFoundError:
            print(f"Warning: could not locate dry-run summary for track '{track}'")

    duration = time.perf_counter() - start_time
    if track_stats:
        write_progress_snapshot(track_stats, duration, args.dataset_root)

    print(
        "\nDummy pipeline finished. Inspect "
        f"{args.dataset_root} for datasets and {args.adapter_root} for dry-run metadata."
    )


if __name__ == "__main__":
    main()
