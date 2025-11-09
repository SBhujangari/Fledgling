"""Evaluation harness for structured JSON and tool-call tracks."""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai
from clients import build_client
from langfuse_helpers import langfuse_trace, log_decision_event
from prompts import get_system_prompt

TOOLCALL_RE = re.compile(
    r'^<tool_call name="(?P<name>[^"]+)">\s*(?P<body>\{.*\})\s*</tool_call>\s*$',
    re.DOTALL,
)


@dataclass
class Example:
    prompt: str
    completion: str


@dataclass
class EvalStepResult:
    valid: bool
    exact_match: bool
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    per_example_metrics: Dict[str, float] = field(default_factory=dict)


def _safe_json_load(text: str) -> Tuple[bool, Any]:
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def _flatten(obj: Any, prefix: str = "") -> Dict[str, str]:
    flat: Dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            flat.update(_flatten(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            flat.update(_flatten(value, new_prefix))
    else:
        flat[prefix or "$"] = json.dumps(obj, sort_keys=True)
    return flat


def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return precision, recall, f1_score


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def _get_embedding(text: str, azure_client: Any) -> List[float]:
    """Get embedding from Azure OpenAI."""
    try:
        # Use Azure client to get embeddings
        response = azure_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"  # Azure embedding model
        )
        return response.data[0].embedding
    except Exception as e:
        # If embedding fails, return zero vector
        print(f"Warning: Embedding failed: {e}")
        return [0.0] * 1536  # text-embedding-ada-002 dimension


def compute_semantic_similarity(pred: str, ref: str, azure_client: Any) -> float:
    """Compute semantic similarity between prediction and reference using embeddings."""
    pred_embedding = _get_embedding(pred, azure_client)
    ref_embedding = _get_embedding(ref, azure_client)
    return _cosine_similarity(pred_embedding, ref_embedding)


def llm_judge_score(
    pred: str,
    ref: str,
    track: str,
    azure_client: Any
) -> Tuple[float, str]:
    """
    Use Azure LLM as judge to score prediction quality on 1-7 scale.
    Returns normalized score [0,1] and reasoning.
    """
    if track == "structured":
        rubric = """
Score the prediction on a 1-7 scale:
7 = Perfect - All fields correct, properly formatted, semantically accurate
6 = Excellent - Minor formatting differences, content fully correct
5 = Good - Semantically correct, some non-critical field mismatches
4 = Acceptable - Core information correct, some missing/extra fields
3 = Partial - Some correct fields, notable gaps or errors
2 = Poor - Mostly incorrect or incomplete
1 = Failed - Invalid or completely wrong output
"""
    else:  # toolcall
        rubric = """
Score the tool call prediction on a 1-7 scale:
7 = Perfect - Correct tool name and all arguments with exact values
6 = Excellent - Correct tool and arguments, minor formatting differences
5 = Good - Correct tool, semantically correct arguments
4 = Acceptable - Correct tool, some argument mismatches
3 = Partial - Correct tool, significant argument errors
2 = Poor - Wrong tool or mostly incorrect arguments
1 = Failed - Invalid format or completely wrong call
"""

    judge_prompt = f"""You are an expert evaluator. Compare the prediction against the reference and score it.

{rubric}

Reference (correct answer):
{ref}

Prediction (to be scored):
{pred}

Respond with JSON: {{"score": <1-7>, "reasoning": "<brief explanation>"}}"""

    try:
        response = azure_client.chat.completions.create(
            model="gpt-4",  # Will use Azure deployment
            messages=[
                {"role": "system", "content": "You are a precise evaluator. Always respond with valid JSON."},
                {"role": "user", "content": judge_prompt}
            ],
            temperature=0.0,
            max_tokens=200
        )

        result_text = response.choices[0].message.content.strip()
        result = json.loads(result_text)
        score = float(result.get("score", 1))
        reasoning = result.get("reasoning", "")

        # Normalize to [0, 1]
        normalized_score = (score - 1) / 6.0  # 1->0, 7->1
        return normalized_score, reasoning

    except Exception as e:
        print(f"Warning: LLM judge failed: {e}")
        return 0.0, f"Error: {str(e)}"


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


class StructuredMetricTracker:
    def __init__(
        self,
        use_semantic_similarity: bool = False,
        use_llm_judge: bool = False,
        azure_client: Optional[Any] = None,
        track: str = "structured"
    ):
        self.processed = 0
        self.valid = 0
        self.exact = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.use_semantic_similarity = use_semantic_similarity
        self.use_llm_judge = use_llm_judge
        self.azure_client = azure_client
        self.track = track
        self.semantic_similarity_sum = 0.0
        self.judge_score_sum = 0.0

    def update(self, pred: str, ref: str) -> EvalStepResult:
        self.processed += 1
        issues: List[str] = []
        details: Dict[str, Any] = {}
        pred_ok, pred_obj = _safe_json_load(pred)
        ref_ok, ref_obj = _safe_json_load(ref)
        if pred_ok:
            self.valid += 1
        exact_match = bool(pred_ok and ref_ok and pred_obj == ref_obj)
        if exact_match:
            self.exact += 1
        pred_items = _flatten(pred_obj) if pred_ok else {}
        ref_items = _flatten(ref_obj) if ref_ok else {}
        pred_set = set(pred_items.items()) if pred_ok and isinstance(pred_items, dict) else set()
        ref_set = set(ref_items.items()) if ref_ok and isinstance(ref_items, dict) else set()
        if ref_ok and pred_ok:
            tp = len(pred_set & ref_set)
            fp = len(pred_set - ref_set)
            fn = len(ref_set - pred_set)
        elif ref_ok and not pred_ok:
            tp = 0
            fp = 0
            fn = len(ref_set)
        else:
            tp = 0
            fp = 0
            fn = 0
        self.tp += tp
        self.fp += fp
        self.fn += fn
        if not pred_ok:
            issues.append("prediction_not_valid_json")
        elif not exact_match:
            missing = sorted(path for path, _ in (ref_set - pred_set))
            extra = sorted(path for path, _ in (pred_set - ref_set))
            mismatched_fields = sorted(
                key
                for key in set(pred_items.keys()) & set(ref_items.keys())
                if pred_items.get(key) != ref_items.get(key)
            )
            if missing:
                details["missing_fields"] = missing[:5]
                issues.append("missing_fields")
            if extra:
                details["extra_fields"] = extra[:5]
                issues.append("extra_fields")
            if mismatched_fields:
                preview = [
                    {
                        "field": key,
                        "expected": ref_items.get(key),
                        "observed": pred_items.get(key),
                    }
                    for key in mismatched_fields[:5]
                ]
                details["value_mismatches"] = preview
                issues.append("value_mismatch")
        per_metrics: Dict[str, float] = {
            "json_valid": 1.0 if pred_ok else 0.0,
            "exact_match": 1.0 if exact_match else 0.0,
        }
        precision, recall, f1_score = _f1(tp, fp, fn)
        per_metrics.update(
            {
                "field_precision": precision,
                "field_recall": recall,
                "field_f1": f1_score,
            }
        )

        # Optional: Compute semantic similarity
        if self.use_semantic_similarity and self.azure_client:
            sem_sim = compute_semantic_similarity(pred, ref, self.azure_client)
            self.semantic_similarity_sum += sem_sim
            per_metrics["semantic_similarity"] = sem_sim
            details["semantic_similarity"] = sem_sim

        # Optional: Compute LLM judge score
        if self.use_llm_judge and self.azure_client:
            judge_score, reasoning = llm_judge_score(pred, ref, self.track, self.azure_client)
            self.judge_score_sum += judge_score
            per_metrics["judge_score"] = judge_score
            details["judge_reasoning"] = reasoning

        return EvalStepResult(
            valid=pred_ok,
            exact_match=exact_match,
            issues=issues,
            details=details,
            per_example_metrics=per_metrics,
        )

    def summary(self) -> Dict[str, float]:
        denom = self.processed or 1
        precision, recall, f1_score = _f1(self.tp, self.fp, self.fn)
        result = {
            "json_valid_rate": self.valid / denom,
            "exact_match_rate": self.exact / denom,
            "field_precision": precision,
            "field_recall": recall,
            "field_f1": f1_score,
        }

        # Include optional metrics if enabled
        if self.use_semantic_similarity:
            result["semantic_similarity_avg"] = self.semantic_similarity_sum / denom

        if self.use_llm_judge:
            result["judge_score_avg"] = self.judge_score_sum / denom

        return result


def eval_structured(preds: List[str], refs: List[str]) -> Dict[str, float]:
    tracker = StructuredMetricTracker()
    for pred, ref in zip(preds, refs):
        tracker.update(pred, ref)
    return tracker.summary()


def parse_tool_call(text: str) -> Tuple[bool, str, str, Dict[str, Any]]:
    match = TOOLCALL_RE.match(text.strip())
    if not match:
        return False, "", "", {}
    name = match.group("name")
    body = match.group("body")
    ok, parsed = _safe_json_load(body)
    if not ok:
        return False, "", "", {}
    canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
    return True, name, canonical, parsed


class ToolcallMetricTracker:
    def __init__(
        self,
        use_semantic_similarity: bool = False,
        use_llm_judge: bool = False,
        azure_client: Optional[Any] = None,
        track: str = "toolcall"
    ):
        self.processed = 0
        self.valid_calls = 0
        self.name_matches = 0
        self.args_matches = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.use_semantic_similarity = use_semantic_similarity
        self.use_llm_judge = use_llm_judge
        self.azure_client = azure_client
        self.track = track
        self.semantic_similarity_sum = 0.0
        self.judge_score_sum = 0.0

    def update(self, pred: str, ref: str) -> EvalStepResult:
        self.processed += 1
        issues: List[str] = []
        details: Dict[str, Any] = {}
        pred_ok, pred_name, pred_args_str, pred_args_obj = parse_tool_call(pred)
        ref_ok, ref_name, ref_args_str, ref_args_obj = parse_tool_call(ref)
        if pred_ok:
            self.valid_calls += 1
        name_match = bool(pred_ok and ref_ok and pred_name == ref_name)
        args_match = bool(pred_ok and ref_ok and pred_args_str == ref_args_str)
        if name_match:
            self.name_matches += 1
        if args_match:
            self.args_matches += 1
        pred_flat = _flatten(pred_args_obj) if pred_ok else {}
        ref_flat = _flatten(ref_args_obj) if ref_ok else {}
        pred_set = set(pred_flat.items()) if pred_ok else set()
        ref_set = set(ref_flat.items()) if ref_ok else set()
        if ref_ok and pred_ok:
            tp = len(pred_set & ref_set)
            fp = len(pred_set - ref_set)
            fn = len(ref_set - pred_set)
        elif ref_ok and not pred_ok:
            tp = 0
            fp = 0
            fn = len(ref_set)
        else:
            tp = 0
            fp = 0
            fn = 0
        self.tp += tp
        self.fp += fp
        self.fn += fn
        if not pred_ok:
            issues.append("invalid_tool_call_format")
        else:
            if not name_match:
                issues.append("tool_name_mismatch")
                details["expected_tool"] = ref_name
                details["predicted_tool"] = pred_name
            if not args_match and ref_ok:
                args_diff = sorted(path for path, _ in (ref_set - pred_set))
                if args_diff:
                    details["missing_args"] = args_diff[:5]
                extra_args = sorted(path for path, _ in (pred_set - ref_set))
                if extra_args:
                    details["extra_args"] = extra_args[:5]
                if not args_diff and not extra_args and ref_ok:
                    mismatches = [
                        {
                            "field": key,
                            "expected": ref_flat.get(key),
                            "observed": pred_flat.get(key),
                        }
                        for key in (set(ref_flat.keys()) & set(pred_flat.keys()))
                        if ref_flat.get(key) != pred_flat.get(key)
                    ]
                    if mismatches:
                        details["arg_value_mismatches"] = mismatches[:5]
                if not args_match:
                    issues.append("arguments_mismatch")
        per_metrics = {
            "valid_call": 1.0 if pred_ok else 0.0,
            "name_match": 1.0 if name_match else 0.0,
            "args_exact": 1.0 if args_match else 0.0,
        }
        precision, recall, f1_score = _f1(tp, fp, fn)
        per_metrics.update(
            {
                "args_precision": precision,
                "args_recall": recall,
                "args_f1": f1_score,
            }
        )

        # Optional: Compute semantic similarity (on arguments)
        if self.use_semantic_similarity and self.azure_client and pred_ok and ref_ok:
            # Compare argument strings
            sem_sim = compute_semantic_similarity(pred_args_str, ref_args_str, self.azure_client)
            self.semantic_similarity_sum += sem_sim
            per_metrics["semantic_similarity"] = sem_sim
            details["semantic_similarity"] = sem_sim

        # Optional: Compute LLM judge score
        if self.use_llm_judge and self.azure_client:
            judge_score, reasoning = llm_judge_score(pred, ref, self.track, self.azure_client)
            self.judge_score_sum += judge_score
            per_metrics["judge_score"] = judge_score
            details["judge_reasoning"] = reasoning

        return EvalStepResult(
            valid=pred_ok,
            exact_match=args_match,
            issues=issues,
            details=details,
            per_example_metrics=per_metrics,
        )

    def summary(self) -> Dict[str, float]:
        denom = self.processed or 1
        precision, recall, f1_score = _f1(self.tp, self.fp, self.fn)
        result = {
            "valid_call_rate": self.valid_calls / denom,
            "name_match_rate": self.name_matches / denom,
            "args_exact_rate": self.args_matches / denom,
            "args_precision": precision,
            "args_recall": recall,
            "args_f1": f1_score,
        }

        # Include optional metrics if enabled
        if self.use_semantic_similarity:
            result["semantic_similarity_avg"] = self.semantic_similarity_sum / denom

        if self.use_llm_judge:
            result["judge_score_avg"] = self.judge_score_sum / denom

        return result


def eval_toolcall(preds: List[str], refs: List[str]) -> Dict[str, float]:
    tracker = ToolcallMetricTracker()
    for pred, ref in zip(preds, refs):
        tracker.update(pred, ref)
    return tracker.summary()


def run_eval(
    track: str,
    model_kind: str,
    dataset_path: str,
    output_path: str,
    split: str,
    adapter: str | None,
    details_path: str | None,
    batch_size: int = 1,
    use_semantic_similarity: bool = False,
    use_llm_judge: bool = False,
):
    data = load_jsonl(dataset_path)
    total = len(data)
    if not total:
        raise ValueError(f"Dataset is empty: {dataset_path}")
    client = build_client(model_kind, adapter)
    system_prompt = get_system_prompt(track)

    # Create Azure client for optional metrics if needed
    azure_client = None
    if use_semantic_similarity or use_llm_judge:
        azure_client = build_client("azure", None)

    # Initialize tracker with optional metrics
    if track == "structured":
        tracker = StructuredMetricTracker(
            use_semantic_similarity=use_semantic_similarity,
            use_llm_judge=use_llm_judge,
            azure_client=azure_client,
            track=track
        )
    else:
        tracker = ToolcallMetricTracker(
            use_semantic_similarity=use_semantic_similarity,
            use_llm_judge=use_llm_judge,
            azure_client=azure_client,
            track=track
        )

    detailed_rows: List[Dict[str, Any]] = []

    # Process in batches for SLM (parallel across 4 GPUs), sequentially for Azure
    use_batching = model_kind == "slm" and batch_size > 1 and hasattr(client, "generate_batch")

    if use_batching:
        # Batch processing for SLM
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_examples = data[batch_start:batch_end]

            # Prepare batch of messages
            batch_messages = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example.prompt},
                ]
                for example in batch_examples
            ]

            # Generate predictions in parallel
            predictions = client.generate_batch(batch_messages)

            # Process each prediction in the batch
            for i, (example, prediction) in enumerate(zip(batch_examples, predictions)):
                idx = batch_start + i + 1
                step = tracker.update(prediction, example.completion)
                snapshot = tracker.summary()
                log_decision_event(
                    name=f"eval-{track}-prompt",
                    metadata={
                        "prompt_index": idx,
                        "prompt": example.prompt,
                        "prediction": prediction,
                        "reference": example.completion,
                        "issues": step.issues,
                        "details": step.details,
                        "per_example_metrics": step.per_example_metrics,
                        "running_metrics": snapshot,
                        "track": track,
                        "model_kind": model_kind,
                        "split": split,
                    },
                    level="ERROR" if step.issues else "INFO",
                )
                detailed_rows.append(
                    {
                        "prompt_index": idx,
                        "prompt": example.prompt,
                        "prediction": prediction,
                        "reference": example.completion,
                        "issues": step.issues,
                        "details": step.details,
                        "per_example_metrics": step.per_example_metrics,
                    }
                )
                progress_note = (
                    f"[{idx}/{total}] track={track} model={model_kind} batch={batch_size} "
                    f"json_valid_rate={snapshot.get('json_valid_rate', snapshot.get('valid_call_rate', 0.0)):.2%} "
                    f"exact_match_rate={snapshot.get('exact_match_rate', snapshot.get('args_exact_rate', 0.0)):.2%}"
                )
                print(progress_note, flush=True)
    else:
        # Sequential processing for Azure or SLM with batch_size=1
        for idx, example in enumerate(data, start=1):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example.prompt},
            ]
            try:
                prediction = client.generate(messages)
            except openai.BadRequestError as e:
                # Handle Azure content filter errors gracefully
                if "content_filter" in str(e):
                    print(f"[{idx}/{len(data)}] Skipped (content filter): {example.prompt[:50]}...")
                    continue
                raise
            step = tracker.update(prediction, example.completion)
            snapshot = tracker.summary()
            log_decision_event(
                name=f"eval-{track}-prompt",
                metadata={
                    "prompt_index": idx,
                    "prompt": example.prompt,
                    "prediction": prediction,
                    "reference": example.completion,
                    "issues": step.issues,
                    "details": step.details,
                    "per_example_metrics": step.per_example_metrics,
                    "running_metrics": snapshot,
                    "track": track,
                    "model_kind": model_kind,
                    "split": split,
                },
                level="ERROR" if step.issues else "INFO",
            )
            detailed_rows.append(
                {
                    "prompt_index": idx,
                    "prompt": example.prompt,
                    "prediction": prediction,
                    "reference": example.completion,
                    "issues": step.issues,
                    "details": step.details,
                    "per_example_metrics": step.per_example_metrics,
                }
            )
            progress_note = (
                f"[{idx}/{total}] track={track} model={model_kind} "
                f"json_valid_rate={snapshot.get('json_valid_rate', snapshot.get('valid_call_rate', 0.0)):.2%} "
                f"exact_match_rate={snapshot.get('exact_match_rate', snapshot.get('args_exact_rate', 0.0)):.2%}"
            )
            print(progress_note, flush=True)

    metrics = tracker.summary()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    if details_path:
        os.makedirs(os.path.dirname(details_path), exist_ok=True)
        with open(details_path, "w", encoding="utf-8") as f:
            for row in detailed_rows:
                f.write(json.dumps(row) + "\n")
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
        "--details-out",
        help="Optional JSONL file to store per-prompt diagnostics.",
    )
    parser.add_argument(
        "--adapter",
        help="Optional LoRA adapter directory for the local SLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for SLM inference (parallel across 4 GPUs). Default: 8. Use 1 for sequential.",
    )
    parser.add_argument(
        "--use-semantic-similarity",
        action="store_true",
        help="Enable semantic similarity metric using Azure embeddings (OpenAI Cookbook).",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Enable LLM-as-Judge scoring on 1-7 scale using Azure LLM (OpenAI Cookbook).",
    )
    args = parser.parse_args()

    dataset_path = args.dataset or os.path.join(
        "02_dataset", args.track, f"{args.split}.jsonl"
    )
    default_out = os.path.join(
        "05_eval", f"{args.track}_{args.model_kind}_{args.split}.json"
    )
    output_path = args.out or default_out
    details_default = os.path.splitext(output_path)[0] + "_details.jsonl"
    details_path = args.details_out or details_default
    with langfuse_trace(
        name=f"eval-{args.track}-{args.model_kind}-{args.split}",
        metadata={
            "track": args.track,
            "model_kind": args.model_kind,
            "dataset": dataset_path,
            "output": output_path,
            "split": args.split,
            "use_semantic_similarity": args.use_semantic_similarity,
            "use_llm_judge": args.use_llm_judge,
        },
    ) as trace:
        metrics = run_eval(
            track=args.track,
            model_kind=args.model_kind,
            dataset_path=dataset_path,
            output_path=output_path,
            split=args.split,
            adapter=args.adapter,
            details_path=details_path,
            batch_size=args.batch_size,
            use_semantic_similarity=args.use_semantic_similarity,
            use_llm_judge=args.use_llm_judge,
        )
        trace.update(
            metadata={
                "metrics": metrics,
                "details_path": details_path,
            }
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
