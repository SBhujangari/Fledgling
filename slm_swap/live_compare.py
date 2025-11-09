"""Live terminal viewer that streams Azure vs SLM evaluation metrics."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from clients import build_client
from compare import TRACK_METRICS
from eval import (
    Example,
    StructuredMetricTracker,
    ToolcallMetricTracker,
    eval_structured,
    eval_toolcall,
    load_jsonl,
)
from langfuse_helpers import langfuse_trace
from prompts import get_system_prompt


def _default_output_path(out_dir: str, track: str, model_kind: str, split: str) -> str:
    filename = f"{track}_{model_kind}_{split}.json"
    return os.path.join(out_dir, filename)


TRACKER_FACTORY = {
    "structured": StructuredMetricTracker,
    "toolcall": ToolcallMetricTracker,
}


@dataclass
class ModelState:
    kind: str
    client: object
    tracker: object
    predictions: List[str]
    output_path: Optional[str]
    trace: object

    @property
    def display_name(self) -> str:
        return self.kind.upper()


def render_dashboard(
    track: str,
    split: str,
    processed: int,
    total: int,
    states: List[ModelState],
    clear_screen: bool,
):
    metric_names = TRACK_METRICS[track]
    lines = []
    lines.append(f"Live evaluation | track={track} | split={split} | progress={processed}/{total}")
    lines.append("-" * 72)
    header = "model".ljust(10) + "".join(name.rjust(20) for name in metric_names)
    lines.append(header)
    state_snapshots: Dict[str, Dict[str, float]] = {}
    for state in states:
        snapshot = state.tracker.summary()
        state_snapshots[state.kind] = snapshot
        metrics = "".join(f"{snapshot.get(name, 0.0)*100:>19.2f}%" for name in metric_names)
        lines.append(state.display_name.ljust(10) + metrics)
    if {"azure", "slm"}.issubset(state_snapshots.keys()):
        azure_metrics = state_snapshots["azure"]
        slm_metrics = state_snapshots["slm"]
        lines.append("")
        lines.append("Δ (Azure - SLM)")
        delta = "".join(
            f"{(azure_metrics.get(name, 0.0) - slm_metrics.get(name, 0.0))*100:>19.2f}%"
            for name in metric_names
        )
        lines.append("".ljust(10) + delta)
    lines.append("")
    lines.append("Press Ctrl+C to stop. Metrics update after every example.")
    output = "\n".join(lines)
    if clear_screen:
        sys.stdout.write("\033[2J\033[H")
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


def run_live(
    track: str,
    split: str = "test",
    dataset_path: Optional[str] = None,
    adapter: Optional[str] = None,
    models: Sequence[str] = ("azure", "slm"),
    save_metrics: bool = False,
    out_dir: str = "05_eval",
    clear_screen: bool = True,
    use_semantic_similarity: bool = False,
    use_llm_judge: bool = False,
):
    """Run the live evaluation dashboard programmatically."""

    dataset = dataset_path or os.path.join("02_dataset", track, f"{split}.jsonl")
    data: List[Example] = load_jsonl(dataset)
    total = len(data)
    if not total:
        raise ValueError(f"Dataset is empty: {dataset}")

    tracker_cls = TRACKER_FACTORY[track]
    results: Dict[str, Dict[str, float]] = {}

    # Create Azure client for optional metrics if needed
    azure_client_for_metrics = None
    if use_semantic_similarity or use_llm_judge:
        azure_client_for_metrics = build_client("azure", None)

    with ExitStack() as stack:
        states: List[ModelState] = []
        for model_kind in models:
            client = build_client(model_kind, adapter=adapter if model_kind == "slm" else None)
            output_path = None
            if save_metrics:
                output_path = _default_output_path(out_dir, track, model_kind, split)
            trace = stack.enter_context(
                langfuse_trace(
                    name=f"live-eval-{track}-{model_kind}-{split}",
                    metadata={
                        "track": track,
                        "model_kind": model_kind,
                        "split": split,
                        "dataset": dataset,
                        "output": output_path,
                        "mode": "live",
                        "use_semantic_similarity": use_semantic_similarity,
                        "use_llm_judge": use_llm_judge,
                    },
                )
            )
            states.append(
                ModelState(
                    kind=model_kind,
                    client=client,
                    tracker=tracker_cls(
                        use_semantic_similarity=use_semantic_similarity,
                        use_llm_judge=use_llm_judge,
                        azure_client=azure_client_for_metrics,
                        track=track
                    ),
                    predictions=[],
                    output_path=output_path,
                    trace=trace,
                )
            )

        system_prompt = get_system_prompt(track)
        refs = [example.completion for example in data]
        try:
            for idx, example in enumerate(data, 1):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example.prompt},
                ]
                for state in states:
                    prediction = state.client.generate(messages)
                    state.predictions.append(prediction)
                    state.tracker.update(prediction, example.completion)
                render_dashboard(
                    track=track,
                    split=split,
                    processed=idx,
                    total=total,
                    states=states,
                    clear_screen=clear_screen,
                )
        except KeyboardInterrupt:
            print("\nInterrupted by user. Finalizing results...")

        eval_fn = eval_structured if track == "structured" else eval_toolcall
        for state in states:
            metrics = eval_fn(state.predictions, refs)
            state.trace.update(metadata={"metrics": metrics})
            if state.output_path:
                os.makedirs(os.path.dirname(state.output_path), exist_ok=True)
                with open(state.output_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
            print(
                f"Final metrics for {state.display_name}: "
                + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
            )
            results[state.kind] = metrics
    return results


def main():
    parser = argparse.ArgumentParser(description="Stream Azure vs SLM evaluation metrics in real time.")
    parser.add_argument("--track", choices=("structured", "toolcall"), required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", help="Override dataset path.")
    parser.add_argument("--adapter", help="Optional LoRA adapter for the SLM client.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("azure", "slm"),
        default=("azure", "slm"),
        help="Which models to run (default: azure slm)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist final metrics JSONs under 05_eval/ like eval.py does.",
    )
    parser.add_argument(
        "--out-dir",
        default="05_eval",
        help="Directory for metrics JSON when --save is enabled.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Disable ANSI screen clearing between refreshes.",
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

    run_live(
        track=args.track,
        split=args.split,
        dataset_path=args.dataset,
        adapter=args.adapter,
        models=args.models,
        save_metrics=args.save,
        out_dir=args.out_dir,
        clear_screen=not args.no_clear,
        use_semantic_similarity=args.use_semantic_similarity,
        use_llm_judge=args.use_llm_judge,
    )


if __name__ == "__main__":
    main()
