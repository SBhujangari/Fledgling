"""Evaluation harness for structured JSON and tool-call tracks."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from clients import build_client
from langfuse_helpers import langfuse_trace
from prompts import get_system_prompt

TOOLCALL_RE = re.compile(
    r'^<tool_call name="(?P<name>[^"]+)">\s*(?P<body>\{.*\})\s*</tool_call>\s*$',
    re.DOTALL,
)


@dataclass
class Example:
    prompt: str
    completion: str


def load_jsonl(path: str) -> List[Example]:
    data: List[Example] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(Example(prompt=obj["prompt"], completion=obj["completion"]))
    return data


def canonical_json(text: str) -> Tuple[bool, str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return False, ""
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return True, canonical


def eval_structured(preds: List[str], refs: List[str]) -> Dict[str, float]:
    valid = 0
    matches = 0
    for pred, ref in zip(preds, refs):
        ok, canonical_pred = canonical_json(pred)
        if ok:
            valid += 1
        ok_ref, canonical_ref = canonical_json(ref)
        if ok and ok_ref and canonical_pred == canonical_ref:
            matches += 1
    total = len(refs) or 1
    return {
        "json_valid_rate": valid / total,
        "exact_match_rate": matches / total,
    }


def parse_tool_call(text: str) -> Tuple[bool, str, str]:
    match = TOOLCALL_RE.match(text.strip())
    if not match:
        return False, "", ""
    name = match.group("name")
    body = match.group("body")
    ok, canonical = canonical_json(body)
    if not ok:
        return False, "", ""
    return True, name, canonical


def eval_toolcall(preds: List[str], refs: List[str]) -> Dict[str, float]:
    valid_calls = 0
    name_matches = 0
    args_matches = 0
    for pred, ref in zip(preds, refs):
        pred_ok, pred_name, pred_args = parse_tool_call(pred)
        ref_ok, ref_name, ref_args = parse_tool_call(ref)
        if pred_ok:
            valid_calls += 1
        if pred_ok and ref_ok and pred_name == ref_name:
            name_matches += 1
        if pred_ok and ref_ok and pred_args == ref_args:
            args_matches += 1
    total = len(refs) or 1
    return {
        "valid_call_rate": valid_calls / total,
        "name_match_rate": name_matches / total,
        "args_exact_rate": args_matches / total,
    }


def run_eval(
    track: str,
    model_kind: str,
    dataset_path: str,
    output_path: str,
    split: str,
    adapter: str | None,
):
    data = load_jsonl(dataset_path)
    client = build_client(model_kind, adapter)
    system_prompt = get_system_prompt(track)
    predictions: List[str] = []
    for example in data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example.prompt},
        ]
        prediction = client.generate(messages)
        predictions.append(prediction)
    refs = [ex.completion for ex in data]
    if track == "structured":
        metrics = eval_structured(predictions, refs)
    else:
        metrics = eval_toolcall(predictions, refs)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate models on structured/tool-call tracks.")
    parser.add_argument("--track", choices=("structured", "toolcall"), required=True)
    parser.add_argument("--model-kind", choices=("slm", "azure"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--dataset",
        help="Override dataset path.",
    )
    parser.add_argument(
        "--out",
        help="Path for metrics JSON (default under 05_eval/).",
    )
    parser.add_argument(
        "--adapter",
        help="Optional LoRA adapter directory for the local SLM.",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or os.path.join(
        "02_dataset", args.track, f"{args.split}.jsonl"
    )
    default_out = os.path.join(
        "05_eval", f"{args.track}_{args.model_kind}_{args.split}.json"
    )
    output_path = args.out or default_out
    with langfuse_trace(
        name=f"eval-{args.track}-{args.model_kind}-{args.split}",
        metadata={
            "track": args.track,
            "model_kind": args.model_kind,
            "dataset": dataset_path,
            "output": output_path,
            "split": args.split,
        },
    ) as trace:
        metrics = run_eval(
            track=args.track,
            model_kind=args.model_kind,
            dataset_path=dataset_path,
            output_path=output_path,
            split=args.split,
            adapter=args.adapter,
        )
        trace.update(
            metadata={
                "metrics": metrics,
            }
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
