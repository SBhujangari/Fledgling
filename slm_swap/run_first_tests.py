"""Bootstrap per-track 50-case test splits and launch live eval for both tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence
import random

from live_compare import run_live
from prepare_data import _build_example, _read_dataset_file, Example as PrepExample


def _write_jsonl(path: Path, rows: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for prompt, completion in rows:
            json.dump({"prompt": prompt, "completion": completion}, f)
            f.write("\n")

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "raw_xlam" / "xlam_function_calling_60k.json"


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _ensure_test_cases(dataset_path: Path, test_size: int, seed: int) -> None:
    structured = BASE_DIR / "02_dataset" / "structured" / "test.jsonl"
    toolcall = BASE_DIR / "02_dataset" / "toolcall" / "test.jsonl"
    structured_count = _jsonl_count(structured)
    tool_count = _jsonl_count(toolcall)
    if structured_count == tool_count == test_size:
        print(f"Existing test splits already pinned to {test_size} rows per track; skipping re-sample.")
        return

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Download xLAM first or pass --dataset-path."
        )

    print(
        f"Sampling {test_size} unique prompts for Structured and another {test_size} for Toolcall from {dataset_path}"
    )
    entries = _read_dataset_file(str(dataset_path))
    examples: list[PrepExample] = []
    for entry in entries:
        example = _build_example(entry)
        if example:
            examples.append(example)
    if len(examples) < test_size * 2:
        raise RuntimeError(
            f"Dataset only produced {len(examples)} usable rows; need at least {test_size * 2} to keep tracks distinct."
        )

    rng = random.Random(seed)
    rng.shuffle(examples)
    structured_examples = examples[:test_size]
    tool_examples = examples[test_size : test_size * 2]

    _write_jsonl(
        structured,
        [(ex.structured_prompt, ex.structured_completion) for ex in structured_examples],
    )
    _write_jsonl(
        toolcall,
        [(ex.tool_prompt, ex.tool_completion) for ex in tool_examples],
    )
    print(
        f"Structured test set now has {_jsonl_count(structured)} rows; toolcall has {_jsonl_count(toolcall)} rows."
    )


def _run_track(
    track: str,
    models: Sequence[str],
    adapter: str | None,
    save_metrics: bool,
    out_dir: str,
    clear_screen: bool,
):
    dataset_path = BASE_DIR / "02_dataset" / track / "test.jsonl"
    count = _jsonl_count(dataset_path)
    print(f"\n=== {track.title()} track ({count} cases) ===")
    run_live(
        track=track,
        split="test",
        dataset_path=str(dataset_path),
        adapter=adapter,
        models=models,
        save_metrics=save_metrics,
        out_dir=out_dir,
        clear_screen=clear_screen,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample exactly N test cases (default 50) and run the live evaluation dashboard "
            "for Structured and Toolcall tracks back-to-back."
        )
    )
    parser.add_argument("--dataset-path", default=str(DEFAULT_DATASET), help="Path to xLAM JSON dump.")
    parser.add_argument("--test-size", type=int, default=50, help="Number of test rows per track.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--adapter", help="Optional LoRA adapter directory for the SLM.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("azure", "slm"),
        default=("azure", "slm"),
        help="Models to compare (default runs both).",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist metrics JSONs under 05_eval/.",
    )
    parser.add_argument(
        "--out-dir",
        default="05_eval",
        help="Metrics directory when --save is set.",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Disable ANSI clear between progress updates.",
    )
    args = parser.parse_args()

    _ensure_test_cases(Path(args.dataset_path), args.test_size, args.seed)

    for track in ("structured", "toolcall"):
        _run_track(
            track=track,
            models=args.models,
            adapter=args.adapter,
            save_metrics=args.save,
            out_dir=args.out_dir,
            clear_screen=not args.no_clear,
        )


if __name__ == "__main__":
    main()
