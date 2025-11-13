#!/usr/bin/env python3
"""
LLM-as-a-Judge Evaluation
Uses a powerful LLM to evaluate and compare model outputs
"""

import argparse
import json
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from azure_client import AzureClient
from env_loader import ensure_env_loaded


JUDGE_PROMPT_TEMPLATE = """You are an expert AI evaluator. Your task is to compare two model outputs and determine which is better based on specific criteria.

PROMPT:
{prompt}

MODEL A OUTPUT:
{model_a_output}

MODEL B OUTPUT:
{model_b_output}

EVALUATION CRITERIA:
{criteria}

For each criterion, provide:
1. A score from 1-10 for each model
2. Clear reasoning for your scores
3. Which model performs better on that criterion

Finally, determine the overall winner and your confidence level (0-1).

Respond in this JSON format:
{{
  "scores": [
    {{
      "criterion": "criterion_name",
      "score_a": 8,
      "score_b": 7,
      "reasoning": "Model A is better because..."
    }}
  ],
  "overall_score_a": 8.5,
  "overall_score_b": 7.2,
  "winner": "A",
  "confidence": 0.85,
  "reasoning_summary": "Overall, Model A provides..."
}}"""


def evaluate_with_judge(
    prompt: str,
    model_a_output: str,
    model_b_output: str,
    criteria: list[str],
    judge_model: str = "gpt-4o-mini"
) -> dict:
    """
    Use LLM-as-a-judge to evaluate two model outputs.

    Args:
        prompt: Original user prompt
        model_a_output: Output from model A
        model_b_output: Output from model B
        criteria: List of evaluation criteria
        judge_model: Model to use as judge

    Returns:
        Evaluation results as dict
    """
    ensure_env_loaded()

    # Format criteria as bullet points
    criteria_text = "\n".join([f"- {c.capitalize()}" for c in criteria])

    # Prepare judge prompt
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        prompt=prompt,
        model_a_output=model_a_output,
        model_b_output=model_b_output,
        criteria=criteria_text
    )

    # Use Azure client for judging (can be configured to use other models)
    client = AzureClient()

    messages = [
        {"role": "system", "content": "You are an expert AI evaluator. Provide detailed, fair assessments."},
        {"role": "user", "content": judge_prompt}
    ]

    try:
        response = client.generate(messages)

        # Try to extract JSON from response
        if "```json" in response:
            json_match = response.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_match)
        elif "{" in response and "}" in response:
            # Try to parse as raw JSON
            start = response.index("{")
            end = response.rindex("}") + 1
            result = json.loads(response[start:end])
        else:
            # Fallback: create structured response
            result = {
                "scores": [],
                "overall_score_a": 5.0,
                "overall_score_b": 5.0,
                "winner": "tie",
                "confidence": 0.5,
                "reasoning_summary": response
            }

        return result

    except Exception as e:
        print(f"ERROR: Judge evaluation failed: {e}", file=sys.stderr)
        return {
            "scores": [],
            "overall_score_a": 0,
            "overall_score_b": 0,
            "winner": "tie",
            "confidence": 0,
            "reasoning_summary": f"Evaluation failed: {e}"
        }


def main():
    parser = argparse.ArgumentParser(description="LLM-as-a-Judge Evaluation")
    parser.add_argument("--prompt", required=True, help="Original user prompt")
    parser.add_argument("--model-a-output", required=True, help="Output from model A")
    parser.add_argument("--model-b-output", required=True, help="Output from model B")
    parser.add_argument("--criteria", default="accuracy,helpfulness,conciseness,safety",
                        help="Comma-separated list of evaluation criteria")
    parser.add_argument("--judge-model", default="gpt-4o-mini", help="Model to use as judge")

    args = parser.parse_args()

    criteria = [c.strip() for c in args.criteria.split(",")]

    print("Running LLM-as-a-judge evaluation...", file=sys.stderr)

    result = evaluate_with_judge(
        prompt=args.prompt,
        model_a_output=args.model_a_output,
        model_b_output=args.model_b_output,
        criteria=criteria,
        judge_model=args.judge_model
    )

    # Output result with markers for easy parsing
    print("<<<JUDGE_RESULT>>>")
    print(json.dumps(result, indent=2))
    print("<<<END_RESULT>>>")


if __name__ == "__main__":
    main()
