#!/usr/bin/env python3
"""
Unified Comparison Framework
Compares Fledgling approach vs Paper approach across multiple dimensions
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime


class ApproachComparator:
    """Compare two fine-tuning approaches"""

    def __init__(self, fledgling_results_dir: str, paper_results_dir: str):
        self.fledgling_results_dir = Path(fledgling_results_dir)
        self.paper_results_dir = Path(paper_results_dir)

    def load_results(self, approach: str, model: str, track: str = None) -> Dict:
        """Load evaluation results for an approach"""
        if approach == "fledgling":
            # Load from slm_swap/05_eval/
            if track == "structured":
                results_path = self.fledgling_results_dir / f"structured_{model}_test.json"
            elif track == "toolcall":
                results_path = self.fledgling_results_dir / f"toolcall_{model}_test.json"
            else:
                raise ValueError(f"Unknown track: {track}")
        elif approach == "paper":
            # Load from paper_approach/eval_results/
            results_path = self.paper_results_dir / f"{model}_hermes_results.json"
        else:
            raise ValueError(f"Unknown approach: {approach}")

        if not results_path.exists():
            return None

        with open(results_path, "r") as f:
            return json.load(f)

    def compare_single_turn_performance(self) -> pd.DataFrame:
        """Compare single-turn function calling performance"""
        comparisons = []

        # Fledgling: structured track, SLM
        fledgling_structured = self.load_results("fledgling", "slm", "structured")
        # Paper: Phi-4-mini
        paper_phi = self.load_results("paper", "phi")
        # Paper: Llama-3.1-8B
        paper_llama = self.load_results("paper", "llama")

        if fledgling_structured:
            comparisons.append({
                "Approach": "Fledgling",
                "Model": "Phi-4-mini",
                "Track": "Structured",
                "JSON Valid Rate": fledgling_structured.get("metrics", {}).get("json_valid_rate", 0.0),
                "Exact Match Rate": fledgling_structured.get("metrics", {}).get("exact_match_rate", 0.0),
                "Field F1": fledgling_structured.get("metrics", {}).get("field_f1", 0.0),
            })

        if paper_phi:
            comparisons.append({
                "Approach": "Paper",
                "Model": "Phi-4-mini",
                "Track": "Function Calling",
                "JSON Valid Rate": paper_phi.get("metrics", {}).get("valid_json_rate", 0.0),
                "Exact Match Rate": paper_phi.get("metrics", {}).get("args_exact_match_rate", 0.0),
                "Field F1": paper_phi.get("metrics", {}).get("args_field_f1", 0.0),
            })

        if paper_llama:
            comparisons.append({
                "Approach": "Paper",
                "Model": "Llama-3.1-8B",
                "Track": "Function Calling",
                "JSON Valid Rate": paper_llama.get("metrics", {}).get("valid_json_rate", 0.0),
                "Exact Match Rate": paper_llama.get("metrics", {}).get("args_exact_match_rate", 0.0),
                "Field F1": paper_llama.get("metrics", {}).get("args_field_f1", 0.0),
            })

        return pd.DataFrame(comparisons)

    def compare_multi_turn_performance(self) -> pd.DataFrame:
        """Compare multi-turn conversation performance"""
        # This requires multi-turn examples in the dataset
        # For now, return placeholder
        return pd.DataFrame([{
            "Approach": "Fledgling",
            "Model": "Phi-4-mini",
            "Multi-turn F1": "N/A - not tested yet"
        }])

    def compare_json_structured_output(self) -> pd.DataFrame:
        """Compare structured JSON output quality"""
        comparisons = []

        # Both approaches should have JSON validity metrics
        fledgling_structured = self.load_results("fledgling", "slm", "structured")
        paper_phi = self.load_results("paper", "phi")
        paper_llama = self.load_results("paper", "llama")

        if fledgling_structured:
            metrics = fledgling_structured.get("metrics", {})
            comparisons.append({
                "Approach": "Fledgling",
                "Model": "Phi-4-mini",
                "JSON Valid Rate": metrics.get("json_valid_rate", 0.0),
                "Field Precision": metrics.get("field_precision", 0.0),
                "Field Recall": metrics.get("field_recall", 0.0),
            })

        if paper_phi:
            metrics = paper_phi.get("metrics", {})
            comparisons.append({
                "Approach": "Paper",
                "Model": "Phi-4-mini",
                "JSON Valid Rate": metrics.get("valid_json_rate", 0.0),
                "Field Precision": metrics.get("args_field_precision", 0.0),
                "Field Recall": metrics.get("args_field_recall", 0.0),
            })

        if paper_llama:
            metrics = paper_llama.get("metrics", {})
            comparisons.append({
                "Approach": "Paper",
                "Model": "Llama-3.1-8B",
                "JSON Valid Rate": metrics.get("valid_json_rate", 0.0),
                "Field Precision": metrics.get("args_field_precision", 0.0),
                "Field Recall": metrics.get("args_field_recall", 0.0),
            })

        return pd.DataFrame(comparisons)

    def compare_domain_diversity(self) -> pd.DataFrame:
        """Compare performance across diverse domains"""
        # This requires domain-specific breakdown in results
        # Placeholder for now
        return pd.DataFrame([{
            "Domain": "Weather",
            "Fledgling": 0.0,
            "Paper-Phi": 0.0,
            "Paper-Llama": 0.0,
        }])

    def compare_to_llm_baseline(self) -> pd.DataFrame:
        """Compare SLM approaches to Azure LLM baseline"""
        comparisons = []

        # Load Azure baseline
        fledgling_azure_structured = self.load_results("fledgling", "azure", "structured")
        fledgling_azure_toolcall = self.load_results("fledgling", "azure", "toolcall")

        # Load SLM results
        fledgling_slm_structured = self.load_results("fledgling", "slm", "structured")
        fledgling_slm_toolcall = self.load_results("fledgling", "slm", "toolcall")

        paper_phi = self.load_results("paper", "phi")
        paper_llama = self.load_results("paper", "llama")

        # Structured track comparison
        if fledgling_azure_structured and fledgling_slm_structured:
            azure_f1 = fledgling_azure_structured.get("metrics", {}).get("field_f1", 0.0)
            slm_f1 = fledgling_slm_structured.get("metrics", {}).get("field_f1", 0.0)
            delta = slm_f1 - azure_f1

            comparisons.append({
                "Track": "Structured",
                "Approach": "Fledgling",
                "Model": "Phi-4-mini",
                "Azure Baseline F1": azure_f1,
                "SLM F1": slm_f1,
                "Delta": delta,
                "Within 10%": abs(delta) <= 0.10,
                "Surpasses LLM": delta > 0,
            })

        # Paper approach (no direct Azure comparison, but can compare absolute scores)
        if paper_phi:
            phi_f1 = paper_phi.get("metrics", {}).get("args_field_f1", 0.0)
            comparisons.append({
                "Track": "Function Calling",
                "Approach": "Paper",
                "Model": "Phi-4-mini",
                "Azure Baseline F1": "N/A",
                "SLM F1": phi_f1,
                "Delta": "N/A",
                "Within 10%": "N/A",
                "Surpasses LLM": "N/A",
            })

        if paper_llama:
            llama_f1 = paper_llama.get("metrics", {}).get("args_field_f1", 0.0)
            comparisons.append({
                "Track": "Function Calling",
                "Approach": "Paper",
                "Model": "Llama-3.1-8B",
                "Azure Baseline F1": "N/A",
                "SLM F1": llama_f1,
                "Delta": "N/A",
                "Within 10%": "N/A",
                "Surpasses LLM": "N/A",
            })

        return pd.DataFrame(comparisons)

    def generate_comprehensive_report(self, output_path: str):
        """Generate comprehensive comparison report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "fledgling_results_dir": str(self.fledgling_results_dir),
            "paper_results_dir": str(self.paper_results_dir),
        }

        print("=" * 80)
        print("COMPREHENSIVE COMPARISON REPORT")
        print("Fledgling vs Paper Approach")
        print("=" * 80)

        # Single-turn performance
        print("\n1. SINGLE-TURN FUNCTION CALLING PERFORMANCE")
        print("-" * 80)
        single_turn_df = self.compare_single_turn_performance()
        print(single_turn_df.to_string(index=False))
        report["single_turn"] = single_turn_df.to_dict(orient="records")

        # JSON structured output
        print("\n2. STRUCTURED JSON OUTPUT QUALITY")
        print("-" * 80)
        json_df = self.compare_json_structured_output()
        print(json_df.to_string(index=False))
        report["json_quality"] = json_df.to_dict(orient="records")

        # LLM baseline comparison
        print("\n3. COMPARISON TO LLM BASELINE (Azure GPT)")
        print("-" * 80)
        llm_comparison_df = self.compare_to_llm_baseline()
        print(llm_comparison_df.to_string(index=False))
        report["llm_comparison"] = llm_comparison_df.to_dict(orient="records")

        # Multi-turn (placeholder)
        print("\n4. MULTI-TURN CONVERSATION (Not yet implemented)")
        print("-" * 80)
        multi_turn_df = self.compare_multi_turn_performance()
        print(multi_turn_df.to_string(index=False))

        # Domain diversity (placeholder)
        print("\n5. DOMAIN DIVERSITY (Not yet implemented)")
        print("-" * 80)
        domain_df = self.compare_domain_diversity()
        print(domain_df.to_string(index=False))

        print("\n" + "=" * 80)

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {output_path}")

        # Also save as CSV
        csv_dir = output_path.parent / "csv"
        csv_dir.mkdir(exist_ok=True)

        single_turn_df.to_csv(csv_dir / "single_turn.csv", index=False)
        json_df.to_csv(csv_dir / "json_quality.csv", index=False)
        llm_comparison_df.to_csv(csv_dir / "llm_comparison.csv", index=False)

        print(f"CSV exports saved to: {csv_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare Fledgling vs Paper approaches")
    parser.add_argument("--fledgling-results", type=str,
                        default="slm_swap/05_eval",
                        help="Fledgling evaluation results directory")
    parser.add_argument("--paper-results", type=str,
                        default="paper_approach/eval_results",
                        help="Paper approach evaluation results directory")
    parser.add_argument("--output", type=str,
                        default="comparison_framework/reports/comparison_report.json",
                        help="Output path for comparison report")

    args = parser.parse_args()

    comparator = ApproachComparator(args.fledgling_results, args.paper_results)
    comparator.generate_comprehensive_report(args.output)


if __name__ == "__main__":
    main()
