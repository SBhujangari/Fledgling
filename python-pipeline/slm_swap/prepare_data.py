"""Create tiny structured/tool-call splits from the xLAM function-calling dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


XLAM_FILENAME = "xlam_function_calling_60k.json"


@dataclass
class Example:
    structured_prompt: str
    structured_completion: str
    tool_prompt: str
    tool_completion: str


def _read_dataset_file(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _ensure_json(obj):
    if isinstance(obj, str):
        return json.loads(obj)
    return obj


def _format_tool_signature(name: str, parameters: Dict[str, dict]) -> str:
    parts = []
    for param_name, spec in parameters.items():
        parts.append(f"{param_name}:{spec.get('type', 'str')}")
    joined = ", ".join(parts)
    return f"{name}({joined})"


def _build_example(entry: dict) -> Example | None:
    query = entry.get("query")
    tools = _ensure_json(entry.get("tools"))
    answers = _ensure_json(entry.get("answers"))
    if not query or not tools or not answers:
        return None

    first_answer = answers[0]
    tool_name = first_answer.get("name")
    arguments = first_answer.get("arguments") or {}
    target_tool = None
    for tool in tools:
        if tool.get("name") == tool_name:
            target_tool = tool
            break
    if not tool_name or target_tool is None:
        return None

    signature = _format_tool_signature(
        target_tool["name"], target_tool.get("parameters", {})
    )
    completion_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
    tool_prompt = (
        "Respond with exactly one <tool_call name=\"...\">{...}</tool_call> wrapper.\n"
        f"Tool signature: {signature}\n"
        f"Tool description: {target_tool.get('description','')}\n"
        f"Request: {query}"
    )
    tool_completion = f'<tool_call name="{tool_name}">{completion_args}</tool_call>'

    structured_prompt = (
        "Return a JSON object with keys query, tool_name, arguments describing the API call.\n"
        f"Query: {query}\n"
        f"Chosen tool: {tool_name}\n"
        "Arguments should mirror the assistant's recommendation."
    )
    structured_completion = json.dumps(
        {
            "query": query,
            "tool_name": tool_name,
            "arguments": arguments,
        },
        sort_keys=True,
    )
    return Example(
        structured_prompt=structured_prompt,
        structured_completion=structured_completion,
        tool_prompt=tool_prompt,
        tool_completion=tool_completion,
    )


def _write_jsonl(path: str, rows: List[Tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for prompt, completion in rows:
            f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sample tiny splits for structured/tool-call tracks from xLAM."
    )
    parser.add_argument(
        "--dataset-path",
        default=os.path.join("raw_xlam", XLAM_FILENAME),
        help="Path to xlam_function_calling_60k.json.",
    )
    parser.add_argument("--train-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=8)
    parser.add_argument("--test-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {args.dataset_path}. "
            "Download gated dataset manually via huggingface-cli after getting access."
        )

    entries = _read_dataset_file(args.dataset_path)
    examples: List[Example] = []
    for entry in entries:
        example = _build_example(entry)
        if example:
            examples.append(example)

    total_needed = args.train_size + args.val_size + args.test_size
    if len(examples) < total_needed:
        raise ValueError(
            f"Not enough examples ({len(examples)}) for requested split size {total_needed}."
        )

    rng = random.Random(args.seed)
    rng.shuffle(examples)

    def take(n: int) -> List[Example]:
        result = examples[:n]
        del examples[:n]
        return result

    train = take(args.train_size)
    val = take(args.val_size)
    test = take(args.test_size)

    _write_jsonl(
        os.path.join("02_dataset", "structured", "train.jsonl"),
        [(ex.structured_prompt, ex.structured_completion) for ex in train],
    )
    _write_jsonl(
        os.path.join("02_dataset", "structured", "val.jsonl"),
        [(ex.structured_prompt, ex.structured_completion) for ex in val],
    )
    _write_jsonl(
        os.path.join("02_dataset", "structured", "test.jsonl"),
        [(ex.structured_prompt, ex.structured_completion) for ex in test],
    )

    _write_jsonl(
        os.path.join("02_dataset", "toolcall", "train.jsonl"),
        [(ex.tool_prompt, ex.tool_completion) for ex in train],
    )
    _write_jsonl(
        os.path.join("02_dataset", "toolcall", "val.jsonl"),
        [(ex.tool_prompt, ex.tool_completion) for ex in val],
    )
    _write_jsonl(
        os.path.join("02_dataset", "toolcall", "test.jsonl"),
        [(ex.tool_prompt, ex.tool_completion) for ex in test],
    )


if __name__ == "__main__":
    main()
