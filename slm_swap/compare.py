"""Compare Azure vs SLM metrics and decide whether to fine-tune."""

from __future__ import annotations

import argparse
import json
from typing import Dict

from langfuse_helpers import log_decision_event

TRACK_METRICS = {
    "structured": ("json_valid_rate", "exact_match_rate"),
    "toolcall": ("valid_call_rate", "name_match_rate", "args_exact_rate"),
}


def load_metrics(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def needs_fine_tune(track: str, azure: Dict[str, float], slm: Dict[str, float], delta: float):
    deltas = {}
    for metric in TRACK_METRICS[track]:
        azure_value = azure.get(metric, 0.0)
        slm_value = slm.get(metric, 0.0)
        gap = azure_value - slm_value
        deltas[metric] = gap
        if gap > delta:
            return True, deltas
    return False, deltas


def main():
    parser = argparse.ArgumentParser(description="Decide whether to fine-tune the SLM.")
    parser.add_argument("--track", choices=TRACK_METRICS.keys(), required=True)
    parser.add_argument("--azure", required=True, help="Azure metrics JSON.")
    parser.add_argument("--slm", required=True, help="SLM metrics JSON.")
    parser.add_argument("--delta", type=float, required=True, help="Allowed metric gap.")
    args = parser.parse_args()

    azure_metrics = load_metrics(args.azure)
    slm_metrics = load_metrics(args.slm)
    fine_tune, metric_deltas = needs_fine_tune(
        args.track, azure_metrics, slm_metrics, args.delta
    )
    decision_value = 1 if fine_tune else 0
    log_decision_event(
        name=f"compare-{args.track}",
        metadata={
            "track": args.track,
            "azure_metrics_path": args.azure,
            "slm_metrics_path": args.slm,
            "delta_threshold": args.delta,
            "metric_deltas": metric_deltas,
            "fine_tune": decision_value,
        },
        level="INFO",
    )
    print(f"FINE_TUNE={decision_value}")
    if fine_tune:
        raise SystemExit(10)


if __name__ == "__main__":
    main()
