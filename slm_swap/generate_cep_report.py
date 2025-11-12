#!/usr/bin/env python3
"""Generate comprehensive CEP evaluation report with recommendations."""

import argparse
import json
from typing import Dict, Any
from pathlib import Path


def load_metrics(path: str) -> Dict[str, float]:
    """Load metrics from JSON file."""
    with open(path) as f:
        return json.load(f)


def calculate_improvement(baseline: float, cep: float) -> Dict[str, Any]:
    """Calculate improvement metrics."""
    absolute_delta = cep - baseline
    if baseline == 0:
        relative_delta = float('inf') if cep > 0 else 0.0
    else:
        relative_delta = (cep - baseline) / baseline

    return {
        "baseline": baseline,
        "cep": cep,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "improved": cep > baseline
    }


def assess_track(metrics: Dict[str, float], track: str, azure_baseline: Dict[str, float]) -> Dict[str, Any]:
    """Assess performance of a single track."""
    if track == "structured":
        primary_metric = "field_f1"
        target = 0.60  # Match Azure baseline
        azure_value = azure_baseline.get(primary_metric, 0.60)
    else:  # toolcall
        primary_metric = "args_f1"
        target = 0.74  # Match Azure baseline
        azure_value = azure_baseline.get(primary_metric, 0.74)

    cep_value = metrics.get(primary_metric, 0.0)

    # Determine success level
    if cep_value >= azure_value:
        status = "EXCEEDS"
        emoji = "🚀"
    elif cep_value >= target:
        status = "SUCCESS"
        emoji = "✅"
    elif cep_value >= target * 0.9:
        status = "PARTIAL"
        emoji = "🟡"
    else:
        status = "FAILURE"
        emoji = "❌"

    return {
        "status": status,
        "emoji": emoji,
        "primary_metric": primary_metric,
        "value": cep_value,
        "target": target,
        "azure_baseline": azure_value,
        "delta_to_target": cep_value - target,
        "delta_to_azure": cep_value - azure_value,
        "meets_target": cep_value >= target
    }


def generate_recommendations(
    struct_assessment: Dict[str, Any],
    tool_assessment: Dict[str, Any],
    struct_improvement: Dict[str, Any],
    tool_improvement: Dict[str, Any]
) -> Dict[str, Any]:
    """Generate recommendations based on results."""

    both_success = struct_assessment["meets_target"] and tool_assessment["meets_target"]
    one_success = struct_assessment["meets_target"] or tool_assessment["meets_target"]

    if both_success:
        priority = "DEPLOY"
        actions = [
            "Train Phi-4 with CEP (smaller, faster model)",
            "Integrate into production workflow tracer",
            "Deploy gradual LLM → SLM swap",
            "Monitor continuous improvement loop",
            "Consider scaling up training data (100 → 1000 examples)"
        ]
        next_model = "microsoft/Phi-4"
    elif one_success:
        priority = "ITERATE"
        failed_track = "toolcall" if struct_assessment["meets_target"] else "structured"
        actions = [
            f"Analyze failure patterns in {failed_track} track details.jsonl",
            f"Refine CEP for {failed_track} track (track-specific variant)",
            "Retrain with refined CEP",
            "Consider increasing training data or epochs",
            "Re-evaluate after refinement"
        ]
        next_model = "unsloth/llama-3.1-8b-instruct-bnb-4bit"
    else:
        priority = "TROUBLESHOOT"
        actions = [
            "Verify CEP was applied during training (check logs)",
            "Verify adapter loaded correctly during evaluation",
            "Analyze error patterns (compare pre-CEP vs CEP)",
            "Try inference-time CEP (no retraining)",
            "Consider alternative approaches (different base model, more data)",
            "Review CEP design for potential issues"
        ]
        next_model = None

    return {
        "priority": priority,
        "actions": actions,
        "next_model": next_model,
        "both_success": both_success,
        "one_success": one_success
    }


def format_report(data: Dict[str, Any]) -> str:
    """Format report as markdown."""
    lines = []

    lines.append("# CEP Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {data['timestamp']}")
    lines.append("")

    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    struct = data["structured"]
    tool = data["toolcall"]

    lines.append(f"**Structured Track:** {struct['assessment']['emoji']} {struct['assessment']['status']}")
    lines.append(f"- CEP F1: {struct['assessment']['value']:.1%}")
    lines.append(f"- Target: {struct['assessment']['target']:.1%}")
    lines.append(f"- Azure Baseline: {struct['assessment']['azure_baseline']:.1%}")
    lines.append(f"- Delta to Target: {struct['assessment']['delta_to_target']:+.1%}")
    lines.append("")

    lines.append(f"**Toolcall Track:** {tool['assessment']['emoji']} {tool['assessment']['status']}")
    lines.append(f"- CEP F1: {tool['assessment']['value']:.1%}")
    lines.append(f"- Target: {tool['assessment']['target']:.1%}")
    lines.append(f"- Azure Baseline: {tool['assessment']['azure_baseline']:.1%}")
    lines.append(f"- Delta to Target: {tool['assessment']['delta_to_target']:+.1%}")
    lines.append("")

    # Recommendations
    rec = data["recommendations"]
    lines.append(f"## Recommendation: {rec['priority']}")
    lines.append("")
    for i, action in enumerate(rec["actions"], 1):
        lines.append(f"{i}. {action}")
    lines.append("")

    # Detailed Metrics
    lines.append("## Detailed Metrics")
    lines.append("")

    lines.append("### Structured Track")
    lines.append("")
    lines.append("| Metric | Baseline | CEP | Delta | Improvement |")
    lines.append("|--------|----------|-----|-------|-------------|")

    struct_metrics = [
        ("Valid JSON", "json_valid_rate"),
        ("Exact Match", "exact_match_rate"),
        ("Field Precision", "field_precision"),
        ("Field Recall", "field_recall"),
        ("Field F1", "field_f1")
    ]

    for label, key in struct_metrics:
        imp = struct["improvement"][key]
        lines.append(
            f"| {label} | {imp['baseline']:.1%} | {imp['cep']:.1%} | "
            f"{imp['absolute_delta']:+.1%} | {imp['relative_delta']:+.1%} |"
        )
    lines.append("")

    lines.append("### Toolcall Track")
    lines.append("")
    lines.append("| Metric | Baseline | CEP | Delta | Improvement |")
    lines.append("|--------|----------|-----|-------|-------------|")

    tool_metrics = [
        ("Valid Calls", "valid_call_rate"),
        ("Name Match", "name_match_rate"),
        ("Args Exact", "args_exact_rate"),
        ("Args Precision", "args_precision"),
        ("Args Recall", "args_recall"),
        ("Args F1", "args_f1")
    ]

    for label, key in tool_metrics:
        imp = tool["improvement"][key]
        if imp['baseline'] == 0 and imp['cep'] == 0:
            rel_str = "0.0%"
        elif imp['relative_delta'] == float('inf'):
            rel_str = "∞"
        else:
            rel_str = f"{imp['relative_delta']:+.1%}"

        lines.append(
            f"| {label} | {imp['baseline']:.1%} | {imp['cep']:.1%} | "
            f"{imp['absolute_delta']:+.1%} | {rel_str} |"
        )
    lines.append("")

    # Next Steps
    lines.append("## Next Steps")
    lines.append("")

    if rec["priority"] == "DEPLOY":
        lines.append("### Train Phi-4 with CEP")
        lines.append("```bash")
        lines.append("python slm_swap/train_phi_cep_v2.py \\")
        lines.append("  --model-id microsoft/Phi-4 \\")
        lines.append("  --train paper_approach/datasets/hermes_train.jsonl \\")
        lines.append("  --val paper_approach/datasets/hermes_val.jsonl \\")
        lines.append("  --output slm_swap/04_ft/adapter_phi_cep \\")
        lines.append("  --cep-type compact")
        lines.append("```")
    elif rec["priority"] == "ITERATE":
        failed_track = "toolcall" if struct['assessment']['meets_target'] else "structured"
        lines.append(f"### Analyze {failed_track.title()} Track Failures")
        lines.append("```bash")
        lines.append(f"# Review failure patterns")
        lines.append(f"grep '\"issues\":' slm_swap/05_eval/{failed_track}_slm_cep_test_details.jsonl | head -20")
        lines.append("")
        lines.append("# Refine CEP based on errors")
        lines.append("vim slm_swap/cep_config.py")
        lines.append("")
        lines.append("# Retrain")
        lines.append("./slm_swap/resume_llama_cep.sh")
        lines.append("```")
    else:  # TROUBLESHOOT
        lines.append("### Verify CEP Application")
        lines.append("```bash")
        lines.append("# Check training logs")
        lines.append("grep 'formatting_rules' slm_swap/logs/train_llama_cep_pure.log")
        lines.append("")
        lines.append("# Compare error patterns")
        lines.append("diff \\")
        lines.append("  <(grep '\"issues\":' slm_swap/05_eval/toolcall_slm_test_details.jsonl | sort | uniq -c) \\")
        lines.append("  <(grep '\"issues\":' slm_swap/05_eval/toolcall_slm_cep_test_details.jsonl | sort | uniq -c)")
        lines.append("```")

    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate CEP evaluation report")
    parser.add_argument("--baseline-structured", required=True, help="Pre-CEP structured metrics JSON")
    parser.add_argument("--baseline-toolcall", required=True, help="Pre-CEP toolcall metrics JSON")
    parser.add_argument("--cep-structured", required=True, help="CEP structured metrics JSON")
    parser.add_argument("--cep-toolcall", required=True, help="CEP toolcall metrics JSON")
    parser.add_argument("--azure-structured", required=True, help="Azure structured metrics JSON")
    parser.add_argument("--azure-toolcall", required=True, help="Azure toolcall metrics JSON")
    parser.add_argument("--out", help="Output path for report (JSON and MD)")
    args = parser.parse_args()

    # Load all metrics
    baseline_struct = load_metrics(args.baseline_structured)
    baseline_tool = load_metrics(args.baseline_toolcall)
    cep_struct = load_metrics(args.cep_structured)
    cep_tool = load_metrics(args.cep_toolcall)
    azure_struct = load_metrics(args.azure_structured)
    azure_tool = load_metrics(args.azure_toolcall)

    # Calculate improvements for structured track
    struct_improvement = {}
    for key in baseline_struct:
        struct_improvement[key] = calculate_improvement(
            baseline_struct[key],
            cep_struct[key]
        )

    # Calculate improvements for toolcall track
    tool_improvement = {}
    for key in baseline_tool:
        tool_improvement[key] = calculate_improvement(
            baseline_tool[key],
            cep_tool[key]
        )

    # Assess each track
    struct_assessment = assess_track(cep_struct, "structured", azure_struct)
    tool_assessment = assess_track(cep_tool, "toolcall", azure_tool)

    # Generate recommendations
    recommendations = generate_recommendations(
        struct_assessment,
        tool_assessment,
        struct_improvement,
        tool_improvement
    )

    # Build report data
    from datetime import datetime
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "structured": {
            "baseline": baseline_struct,
            "cep": cep_struct,
            "azure": azure_struct,
            "improvement": struct_improvement,
            "assessment": struct_assessment
        },
        "toolcall": {
            "baseline": baseline_tool,
            "cep": cep_tool,
            "azure": azure_tool,
            "improvement": tool_improvement,
            "assessment": tool_assessment
        },
        "recommendations": recommendations
    }

    # Save JSON report
    out_path = args.out or "slm_swap/reports/cep_evaluation_report.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)

    print(f"JSON report saved to: {out_path}")

    # Save markdown report
    md_path = out_path.replace(".json", ".md")
    with open(md_path, "w") as f:
        f.write(format_report(report_data))

    print(f"Markdown report saved to: {md_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("CEP EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nStructured Track: {struct_assessment['emoji']} {struct_assessment['status']}")
    print(f"  F1: {struct_assessment['value']:.1%} (target: {struct_assessment['target']:.1%})")
    print(f"\nToolcall Track: {tool_assessment['emoji']} {tool_assessment['status']}")
    print(f"  F1: {tool_assessment['value']:.1%} (target: {tool_assessment['target']:.1%})")
    print(f"\nRecommendation: {recommendations['priority']}")
    print("\nNext Actions:")
    for i, action in enumerate(recommendations['actions'], 1):
        print(f"  {i}. {action}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
