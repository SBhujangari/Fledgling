"""Compare LLM vs SLM to determine if fine-tuning is needed."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

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


if __name__ == "__main__":
    main()
