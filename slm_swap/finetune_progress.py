#!/usr/bin/env python3
"""
Fine-tuning progress monitor.

Reads the latest progress information from a training log (tqdm output),
derives ETA/throughput based on the observed step time, captures GPU details,
prints a friendly dashboard in the terminal, and writes a JSON snapshot that the
web dashboard/backend can consume.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = Path(__file__).resolve().parent / "logs" / "train_llama_cep_pure.log"
DEFAULT_PID = Path(__file__).resolve().parent / "logs" / "train_llama_cep_pure.pid"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "logs" / "finetune_progress.json"

PROGRESS_RE = re.compile(
    r"(\d+)%\|[^\|]*\|\s*(\d+)/(\d+)\s*\[(\d+(?::\d+){1,2})<(\d+(?::\d+){1,2}),\s*([\d\.]+)s/it\]"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track fine-tuning progress + ETA.")
    parser.add_argument(
        "--log-file",
        default=str(DEFAULT_LOG),
        help="Training log with tqdm progress (default: slm_swap/logs/train_llama_cep_pure.log).",
    )
    parser.add_argument(
        "--pid-file",
        default=str(DEFAULT_PID),
        help="Optional PID file to confirm whether a training process is still alive.",
    )
    parser.add_argument(
        "--output-json",
        default=str(DEFAULT_OUTPUT),
        help="Path where the latest progress snapshot JSON should be written.",
    )
    parser.add_argument(
        "--min-total",
        type=int,
        default=10,
        help="Ignore tqdm bars with fewer total steps than this (filters eval loops).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=15.0,
        help="Polling interval in seconds when --watch is enabled.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously refresh progress instead of emitting a single snapshot.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress terminal output (still writes JSON snapshot).",
    )
    return parser.parse_args()


def read_tail(path: Path, bytes_back: int = 65536) -> Optional[str]:
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(size - bytes_back, 0))
            chunk = fh.read().decode(errors="ignore")
            return chunk.replace("\r", "\n")
    except FileNotFoundError:
        return None


def parse_hms(value: str) -> int:
    parts = value.split(":")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return seconds


def parse_progress(log_text: str, min_total: int) -> Optional[Dict[str, Any]]:
    match: Optional[re.Match[str]] = None
    for candidate in PROGRESS_RE.finditer(log_text):
        total = int(candidate.group(3))
        if total < min_total:
            continue
        match = candidate
    if not match:
        return None
    completed = int(match.group(2))
    total = int(match.group(3))
    percent = (completed / total) * 100 if total else 0.0
    elapsed_seconds = parse_hms(match.group(4))
    remaining_seconds = parse_hms(match.group(5))
    seconds_per_it = float(match.group(6))
    return {
        "currentStep": completed,
        "totalSteps": total,
        "percentComplete": percent,
        "elapsedSeconds": elapsed_seconds,
        "remainingSeconds": remaining_seconds,
        "avgStepSeconds": seconds_per_it,
        "rawLine": match.group(0),
    }


def read_pid(pid_path: Path) -> Optional[int]:
    try:
        content = pid_path.read_text().strip()
        return int(content)
    except (FileNotFoundError, ValueError):
        return None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def gather_gpu_info() -> Dict[str, Any]:
    if not shutil.which("nvidia-smi"):
        return {"summary": "nvidia-smi not available", "gpus": [], "gpuCount": 0}
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
        output = subprocess.check_output(cmd, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"summary": "Failed to query GPUs", "gpus": [], "gpuCount": 0}

    gpus: List[Dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if not parts:
            continue
        name = parts[0]
        memory_gb = float(parts[1]) / 1024 if len(parts) > 1 else None
        gpus.append({"name": name, "memoryTotalGb": round(memory_gb, 2) if memory_gb else None})

    summary = " / ".join(
        f"{gpu['name']} ({gpu['memoryTotalGb']} GB)" if gpu.get("memoryTotalGb") else gpu["name"]
        for gpu in gpus
    )
    return {
        "gpus": gpus,
        "gpuCount": len(gpus),
        "summary": summary or "No GPUs detected",
        "cudaVisibleDevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def build_snapshot(args: argparse.Namespace) -> Dict[str, Any]:
    log_path = Path(args.log_file).expanduser()
    pid_path = Path(args.pid_file).expanduser()
    output_path = Path(args.output_json).expanduser()

    log_text = read_tail(log_path)
    progress = parse_progress(log_text, args.min_total) if log_text else None
    pid = read_pid(pid_path) if pid_path.exists() else None
    pid_running = pid_alive(pid) if pid is not None else False
    hardware = gather_gpu_info()

    now = datetime.now(timezone.utc)
    log_mtime = None
    log_age = None
    if log_path.exists():
        log_mtime = datetime.fromtimestamp(log_path.stat().st_mtime, tz=timezone.utc)
        log_age = (now - log_mtime).total_seconds()

    status = "idle"
    message = "No progress detected yet."
    if progress:
        if pid_running:
            status = "running"
            message = "Fine-tuning in progress."
        elif log_age is not None and log_age < 120:
            status = "recent"
            message = "Recent progress recorded, waiting for next update."
        else:
            status = "stalled"
            message = "No active PID detected; last update may be stale."
    elif pid_running:
        status = "starting"
        message = "Training process running, awaiting first progress update."

    eta = (
        now + timedelta(seconds=progress["remainingSeconds"])
        if progress and progress["remainingSeconds"] is not None
        else None
    )

    snapshot = {
        "status": status,
        "message": message,
        "logFile": str(log_path),
        "pid": pid,
        "pidAlive": pid_running,
        "hardware": hardware,
        "updatedAt": now.isoformat(),
        "logUpdatedAt": log_mtime.isoformat() if log_mtime else None,
        "logStalenessSeconds": log_age,
    }
    if progress:
        snapshot.update(progress)
        snapshot["eta"] = eta.isoformat() if eta else None
        snapshot["etaTimestamp"] = int(eta.timestamp()) if eta else None
        snapshot["remainingDisplay"] = format_duration(progress["remainingSeconds"])
        snapshot["stepTimeDisplay"] = f"{progress['avgStepSeconds']:.2f}s/step"
        snapshot["stepsPerMinute"] = 60.0 / progress["avgStepSeconds"] if progress["avgStepSeconds"] else None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(snapshot, indent=2))

    return snapshot


def render(snapshot: Dict[str, Any]) -> None:
    status_line = f"[{snapshot.get('status', 'unknown').upper()}] {snapshot.get('message', '')}"
    print(status_line)
    if snapshot.get("currentStep") is not None:
        pct = snapshot.get("percentComplete", 0.0)
        print(
            f"Progress: {snapshot['currentStep']}/{snapshot['totalSteps']} steps "
            f"({pct:.2f}%) – {snapshot.get('stepTimeDisplay', '?')} "
            f"• Remaining ~ {snapshot.get('remainingDisplay', '?')}"
        )
        if snapshot.get("eta"):
            print(f"ETA: {snapshot['eta']} (UTC)")
    else:
        print("Progress data unavailable – waiting for first training update.")

    if snapshot.get("hardware"):
        summary = snapshot["hardware"].get("summary")
        gpu_count = snapshot["hardware"].get("gpuCount")
        devices = snapshot["hardware"].get("cudaVisibleDevices")
        print(f"Hardware: {summary} • GPUs: {gpu_count} • CUDA_VISIBLE_DEVICES={devices}")

    if snapshot.get("logFile"):
        print(f"Log: {snapshot['logFile']}")
    if snapshot.get("pid") is not None:
        state = "alive" if snapshot.get("pidAlive") else "not running"
        print(f"PID: {snapshot['pid']} ({state})")
    print(f"Last refresh: {snapshot.get('updatedAt')}")


def main() -> None:
    args = parse_args()
    if args.watch:
        try:
            while True:
                snapshot = build_snapshot(args)
                if not args.quiet:
                    print("\033[2J\033[H", end="")  # Clear screen
                    render(snapshot)
                time.sleep(max(args.interval, 1.0))
        except KeyboardInterrupt:
            print("\nStopped progress monitor.")
    else:
        snapshot = build_snapshot(args)
        if not args.quiet:
            render(snapshot)


if __name__ == "__main__":
    main()
