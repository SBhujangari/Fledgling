"""Dummy agentic workflow that emits Langfuse-style traces compatible with our eval datasets.

Usage:
    python dummy_agent_workflow.py \
        --requests ../../datasets/dummy_agent_requests.jsonl \
        --out-json ../../storage/dummy_langfuse_traces.jsonl \
        --agent-id dummy-agent

Pass `--emit-to-langfuse` if you also want the traces pushed to a live Langfuse project.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langfuse_helpers import get_langfuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dummy agent traces for Langfuse ingestion.")
    parser.add_argument(
        "--requests",
        default=Path(__file__).resolve().parents[2] / "datasets" / "dummy_agent_requests.jsonl",
        type=Path,
        help="Path to JSONL file with {query, tool_name, arguments} rows.",
    )
    parser.add_argument(
        "--out-json",
        default=Path(__file__).resolve().parents[2] / "storage" / "dummy_langfuse_traces.jsonl",
        type=Path,
        help="Where to write generated trace dictionaries (JSONL).",
    )
    parser.add_argument("--agent-id", default="dummy-agent", help="Agent identifier injected into Langfuse metadata.")
    parser.add_argument("--count", type=int, default=16, help="Number of traces to emit (cycles through the dataset).")
    parser.add_argument(
        "--emit-to-langfuse",
        action="store_true",
        help="If set, also send traces to the Langfuse API using LANGFUSE_* variables.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed used for deterministic timestamps + sampling.")
    return parser.parse_args()


def load_requests(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No requests found in {path}")
    return rows


def iso_timestamp(offset_seconds: int) -> str:
    base = datetime.now(timezone.utc) - timedelta(seconds=offset_seconds)
    return base.isoformat().replace("+00:00", "Z")


def structured_completion(example: Dict[str, Any]) -> str:
    return json.dumps(
        {
            "query": example["query"],
            "tool_name": example["tool_name"],
            "arguments": example["arguments"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def toolcall_completion(example: Dict[str, Any]) -> str:
    return f'<tool_call name="{example["tool_name"]}">{json.dumps(example["arguments"], sort_keys=True, separators=(",", ":"))}</tool_call>'


def build_trace(example: Dict[str, Any], agent_id: str, idx: int) -> Dict[str, Any]:
    ts_offset = idx * 11
    timestamp = iso_timestamp(ts_offset)
    trace_id = f"{agent_id}-{idx}-{uuid.uuid4().hex[:8]}"
    messages = [
        {"role": "system", "content": "You are a structured function caller. Return only JSON or <tool_call> payloads."},
        {"role": "user", "content": example["query"]},
    ]
    structured = structured_completion(example)
    tool_payload = {"status": "success", "output": example["arguments"]}
    observations = [
        {
            "id": str(uuid.uuid4()),
            "type": "generation",
            "name": "dummy-structured",
            "startTime": timestamp,
            "input": messages,
            "output": structured,
            "model": "dummy-agent-v1",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "generation",
            "name": "dummy-toolcall",
            "startTime": timestamp,
            "input": messages,
            "output": toolcall_completion(example),
            "model": "dummy-agent-v1",
        },
        {
            "id": str(uuid.uuid4()),
            "type": "tool",
            "name": example["tool_name"],
            "startTime": timestamp,
            "input": example["arguments"],
            "output": tool_payload,
            "metadata": {"category": "tool"},
        },
    ]
    return {
        "id": trace_id,
        "name": "dummy-agent-trace",
        "metadata": {"agent_id": agent_id, "source": "dummy_agent_workflow"},
        "timestamp": timestamp,
        "input": messages,
        "output": structured,
        "observations": observations,
    }


def write_traces(path: Path, traces: Iterable[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace) + "\n")


def emit_to_langfuse(traces: Iterable[Dict[str, Any]]):
    try:
        client = get_langfuse()
    except EnvironmentError as exc:
        raise SystemExit(f"Cannot emit to Langfuse: {exc}") from exc

    for trace in traces:
        lf_trace = client.trace(id=trace["id"], name=trace["name"], metadata=trace.get("metadata"))
        for obs in trace["observations"]:
            obs_type = (obs.get("type") or "").lower()
            if obs_type == "generation":
                client.generation(
                    trace_id=lf_trace.id,
                    name=obs.get("name"),
                    input=obs.get("input"),
                    output=obs.get("output"),
                    model=obs.get("model", "dummy-agent-v1"),
                )
            elif obs_type in ("tool", "span"):
                client.span(
                    trace_id=lf_trace.id,
                    name=obs.get("name"),
                    input=obs.get("input"),
                    output=obs.get("output"),
                    metadata=obs.get("metadata"),
                )


def main():
    args = parse_args()
    requests = load_requests(args.requests)
    rng = random.Random(args.seed)
    traces: List[Dict[str, Any]] = []
    for idx in range(args.count):
        example = requests[idx % len(requests)]
        # Introduce minor permutation so repeats still look unique
        if rng.random() < 0.3:
            example = {
                **example,
                "query": example["query"].replace("  ", " ").replace("  ", " "),
            }
        traces.append(build_trace(example, args.agent_id, idx))

    write_traces(args.out_json, traces)
    if args.emit_to_langfuse:
        emit_to_langfuse(traces)
    print(f"Generated {len(traces)} dummy traces -> {args.out_json}")
    if args.emit_to_langfuse:
        print("Also sent traces to Langfuse (check your workspace).")


if __name__ == "__main__":
    main()
