"""Build structured/tool-call datasets directly from Langfuse traces.

This script connects to Langfuse using the same environment variables required
for tracing (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`),
filters traces for a specific agent, and emits JSONL splits that match the
`prompt`/`completion` schema used by the rest of the SLM swap pipeline.

Example:
    python langfuse_dataset.py --agent-id demo-agent --limit 200 \\
        --output-root 02_dataset/langfuse --train-ratio 0.7 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import os
import random
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from langfuse import Langfuse

from langfuse_helpers import langfuse_trace

OFFLINE_TYPE_KEYS = ("category", "observationType", "type")


@dataclass
class ToolExample:
    trace_id: str
    agent_id: str
    query: str
    tool_name: str
    arguments: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Langfuse traces into structured/tool-call datasets."
    )
    parser.add_argument(
        "--agent-id",
        required=True,
        help="Filter traces by the agent_id stored in Langfuse metadata.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Maximum number of trace details to download.",
    )
    parser.add_argument(
        "--since",
        help="ISO8601 timestamp (inclusive). Skip traces updated before this time.",
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join("02_dataset", "langfuse"),
        help="Root directory for structured/toolcall/ split folders.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.75,
        help="Portion of data assigned to train split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.125,
        help="Portion of data assigned to validation split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling examples.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=50,
        help="Pagination size when fetching traces from Langfuse.",
    )
    parser.add_argument(
        "--trace-json",
        help="Offline mode: use traces from the given JSONL file instead of hitting Langfuse.",
    )
    return parser.parse_args()


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def extract_agent_id(metadata: Any) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    for key in ("agent_id", "agentId", "agentID"):
        raw = metadata.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def ensure_object(value: Any) -> Optional[Dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def flatten_content(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            text = value["text"].strip()
            return text or None
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [flatten_content(item) or "" for item in value]
        merged = "\n".join(part for part in parts if part)
        return merged or None
    return str(value)


def normalize_messages(payload: Any) -> List[Dict[str, str]]:
    def coerce(item: Any) -> Optional[Dict[str, str]]:
        if isinstance(item, dict):
            role = str(item.get("role") or "user")
            content = item.get("content")
            text = ""
            if isinstance(content, list):
                text = "\n".join(
                    part.get("text", "").strip()
                    for part in content
                    if isinstance(part, dict) and part.get("text")
                ).strip()
            elif isinstance(content, dict):
                text = flatten_content(content) or ""
            elif isinstance(content, str):
                text = content.strip()
            elif "text" in item and isinstance(item["text"], str):
                text = item["text"].strip()
            if not text and content is not None:
                text = flatten_content(content) or ""
            if not text:
                return None
            return {"role": role, "content": text}
        if isinstance(item, str):
            stripped = item.strip()
            if stripped:
                return {"role": "user", "content": stripped}
        return None

    items: List[Any] = []
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("messages"), list):
            items = payload["messages"]  # type: ignore[assignment]
        elif "role" in payload or "content" in payload:
            items = [payload]
    elif isinstance(payload, str):
        items = [payload]

    normalized: List[Dict[str, str]] = []
    for entry in items:
        message = coerce(entry)
        if message:
            normalized.append(message)
    return normalized


def infer_signature(tool_name: str, arguments: Dict[str, Any]) -> str:
    def type_name(value: Any) -> str:
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, list):
            return "list"
        if isinstance(value, dict):
            return "object"
        return "str"

    params = [f"{key}:{type_name(val)}" for key, val in arguments.items()]
    joined = ", ".join(params) if params else "..."
    return f"{tool_name}({joined})"


def extract_query(observations: List[Dict[str, Any]], trace_input: Any) -> Optional[str]:
    for obs in observations:
        if obs.get("type", "").lower() != "generation":
            continue
        messages = normalize_messages(obs.get("input"))
        for message in reversed(messages):
            if message["role"] == "user" and message["content"]:
                return message["content"]
    # Fallback to trace-level input
    fallback_messages = normalize_messages(trace_input)
    for message in reversed(fallback_messages):
        if message["role"] == "user" and message["content"]:
            return message["content"]
    return flatten_content(trace_input)


def extract_tool_calls(observations: List[Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    calls: List[Tuple[str, Dict[str, Any]]] = []
    for obs in observations:
        if not _is_tool_observation(obs):
            continue
        args = ensure_object(obs.get("input"))
        if not args:
            continue
        metadata = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
        metadata = metadata or {}
        raw_name = obs.get("name") or metadata.get("toolName") or metadata.get("name")
        tool_name = str(raw_name).strip() if raw_name else "tool"
        calls.append((tool_name, args))
    return calls


def _is_tool_observation(obs: Dict[str, Any]) -> bool:
    obs_type = (obs.get("type") or "").lower()
    if obs_type == "tool":
        return True
    metadata = obs.get("metadata") if isinstance(obs.get("metadata"), dict) else {}
    if metadata:
        for key in OFFLINE_TYPE_KEYS:
            raw = metadata.get(key)
            if isinstance(raw, str) and raw.lower() == "tool":
                return True
    if obs_type == "span":
        name = str(obs.get("name") or "").lower()
        if "tool" in name:
            return True
    return False


def tool_example_from_trace(trace: Dict[str, Any]) -> List[ToolExample]:
    observations = sorted(
        [obs if isinstance(obs, dict) else obs.dict() for obs in trace.get("observations", [])],
        key=lambda obs: obs.get("startTime") or "",
    )
    query = extract_query(observations, trace.get("input"))
    if not query:
        return []
    agent_id = extract_agent_id(trace.get("metadata")) or "unknown"
    results: List[ToolExample] = []
    for tool_name, arguments in extract_tool_calls(observations):
        results.append(
            ToolExample(
                trace_id=trace["id"],
                agent_id=agent_id,
                query=query,
                tool_name=tool_name,
                arguments=arguments,
            )
        )
    return results


def canonical_json(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def build_structured_row(example: ToolExample) -> Dict[str, Any]:
    prompt = (
        "Return a JSON object with keys query, tool_name, arguments describing the exact tool call.\n"
        f"Query: {example.query}\n"
        "Arguments must mirror the tool invocation without commentary."
    )
    completion = canonical_json(
        {"query": example.query, "tool_name": example.tool_name, "arguments": example.arguments}
    )
    return {
        "prompt": prompt,
        "completion": completion,
        "trace_id": example.trace_id,
        "agent_id": example.agent_id,
    }


def build_toolcall_row(example: ToolExample) -> Dict[str, Any]:
    signature = infer_signature(example.tool_name, example.arguments)
    prompt = (
        "Respond with exactly one <tool_call name=\"...\">{...}</tool_call> wrapper.\n"
        f"Tool signature: {signature}\n"
        f"Request: {example.query}"
    )
    completion = f'<tool_call name="{example.tool_name}">{canonical_json(example.arguments)}</tool_call>'
    return {
        "prompt": prompt,
        "completion": completion,
        "trace_id": example.trace_id,
        "agent_id": example.agent_id,
    }


def split_examples(
    examples: List[ToolExample], train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, List[ToolExample]]:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must satisfy: 0 < train_ratio < 1 and train_ratio + val_ratio < 1.")
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    if not train and shuffled:
        train = [shuffled[0]]
    if not val:
        pool = [item for item in shuffled if item not in train]
        if pool:
            val = [pool[0]]
    if not test:
        pool = [item for item in shuffled if item not in train and item not in val]
        if pool:
            test = [pool[0]]

    return {"train": train, "val": val, "test": test}


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_traces(
    client: Langfuse,
    agent_id: str,
    limit: int,
    since: Optional[datetime],
    page_size: int,
) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    page = 1
    while len(traces) < limit:
        response = client.fetch_traces(
            page=page,
            limit=page_size,
            order_by="timestamp.desc",
        )
        data = response.data or []
        if not data:
            break
        for summary in data:
            summary_dict = summary.dict()
            timestamp = parse_datetime(summary_dict.get("timestamp"))
            if since and timestamp and timestamp < since:
                continue
            detail = client.fetch_trace(summary_dict["id"]).data.dict()
            trace_agent = extract_agent_id(detail.get("metadata"))
            if trace_agent != agent_id:
                continue
            traces.append(detail)
            if len(traces) >= limit:
                break
        if len(data) < page_size:
            break
        page += 1
    return traces


def load_traces_from_file(path: str, agent_id: str, limit: int) -> List[Dict[str, Any]]:
    traces: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if extract_agent_id(obj.get("metadata")) != agent_id:
                continue
            traces.append(obj)
            if len(traces) >= limit:
                break
    return traces


def main():
    args = parse_args()
    since_dt = parse_datetime(args.since) if args.since else None
    trace_context = (
        langfuse_trace(
            name="dataset-from-langfuse",
            metadata={
                "agent_id": args.agent_id,
                "limit": args.limit,
                "since": args.since,
                "output_root": args.output_root,
                "offline": bool(args.trace_json),
            },
        )
        if not args.trace_json
        else nullcontext()
    )
    with trace_context:
        if args.trace_json:
            traces = load_traces_from_file(args.trace_json, args.agent_id, args.limit)
        else:
            client = Langfuse()
            traces = fetch_traces(
                client=client,
                agent_id=args.agent_id,
                limit=args.limit,
                since=since_dt,
                page_size=args.page_size,
            )
        examples: List[ToolExample] = []
        for trace in traces:
            examples.extend(tool_example_from_trace(trace))

        if not examples:
            raise SystemExit("No eligible tool calls found for the requested agent/time window.")

        splits = split_examples(examples, args.train_ratio, args.val_ratio, args.seed)
        structured_root = os.path.join(args.output_root, "structured")
        toolcall_root = os.path.join(args.output_root, "toolcall")
        for split_name, rows in splits.items():
            write_jsonl(
                os.path.join(structured_root, f"{split_name}.jsonl"),
                (build_structured_row(row) for row in rows),
            )
            write_jsonl(
                os.path.join(toolcall_root, f"{split_name}.jsonl"),
                (build_toolcall_row(row) for row in rows),
            )

        print(
            f"Wrote {len(examples)} examples "
            f"({len(splits['train'])} train / {len(splits['val'])} val / {len(splits['test'])} test) "
            f"to {args.output_root}"
        )


if __name__ == "__main__":
    main()
