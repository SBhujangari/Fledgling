#!/usr/bin/env python3
"""
Example SLM Agent - Demonstrates fine-tuned model with full tracing and metrics
This agent runs API generation tasks and logs all results for frontend display
"""

import json
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configuration
import os

BASE_MODEL = "unsloth/llama-3.1-8b-instruct-bnb-4bit"

# Use HuggingFace model by default, fall back to local if USE_LOCAL_MODEL is set
USE_LOCAL_MODEL = os.getenv("USE_LOCAL_MODEL", "false").lower() == "true"
ADAPTER = "slm_swap/04_ft/adapter_llama_structured" if USE_LOCAL_MODEL else "kineticdrive/llama-structured-api-adapter"

AGENT_ID = "slm-api-generator"
AGENT_NAME = "Fine-tuned API Generator"
OUTPUT_DIR = Path("backend/src/data/slm_traces")

print(f"📦 Loading adapter from: {'Local' if USE_LOCAL_MODEL else 'HuggingFace'} ({ADAPTER})")

class SLMAgent:
    def __init__(self):
        print(f"🤖 Initializing {AGENT_NAME}...")
        self.agent_id = AGENT_ID
        self.model = None
        self.tokenizer = None
        self.traces = []
        self.metrics = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "avg_latency_ms": 0,
            "total_cost_usd": 0,
            "accuracy": 0.40,  # From evaluation
            "tool_name_accuracy": 0.98,
            "query_preservation": 0.92,
            "json_validity": 1.0,
            "functional_correctness": 0.71,
            "semantic_correctness": 0.75,
        }

    def load_model(self):
        """Load the fine-tuned model"""
        print("📦 Loading model...")
        start = time.time()

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            load_in_4bit=True
        )
        self.model = PeftModel.from_pretrained(self.model, ADAPTER)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_time = time.time() - start
        print(f"✅ Model loaded in {load_time:.2f}s")

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate API call from prompt with full tracing"""
        trace_id = f"trace-{int(time.time() * 1000)}"
        start_time = datetime.utcnow().isoformat()
        start_ms = time.time()

        # Create trace structure
        trace = {
            "traceId": trace_id,
            "agentId": self.agent_id,
            "name": "api-generation",
            "status": "running",
            "startedAt": start_time,
            "input": prompt,
            "metadata": {
                "model": "llama-3.1-8b-structured",
                "adapter": ADAPTER,
                "task": "structured_api_generation"
            }
        }

        try:
            # Generate using chat template
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=256,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            completion = self.tokenizer.decode(
                outputs[0][inputs.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            # Calculate metrics
            latency_ms = (time.time() - start_ms) * 1000

            # Parse result
            parsed_result = None
            is_valid_json = False
            try:
                parsed_result = json.loads(completion)
                is_valid_json = True
            except json.JSONDecodeError:
                pass

            # Update trace with results
            trace.update({
                "status": "completed",
                "completedAt": datetime.utcnow().isoformat(),
                "output": completion,
                "latencyMs": latency_ms,
                "costUsd": 0.0,  # On-premise, no API cost
                "metadata": {
                    **trace["metadata"],
                    "valid_json": is_valid_json,
                    "parsed_result": parsed_result,
                    "latency_ms": latency_ms,
                }
            })

            # Create observation
            observation = {
                "observationId": f"{trace_id}-obs-1",
                "traceId": trace_id,
                "type": "generation",
                "name": "slm-inference",
                "status": "OK",
                "startedAt": start_time,
                "completedAt": datetime.utcnow().isoformat(),
                "input": prompt,
                "output": completion,
                "metadata": {
                    "model": "llama-3.1-8b-structured",
                    "valid_json": is_valid_json,
                    "has_required_fields": self._check_required_fields(parsed_result),
                }
            }

            # Create generation record
            generation = {
                "generationId": f"{trace_id}-gen-1",
                "traceId": trace_id,
                "observationId": f"{trace_id}-obs-1",
                "model": "llama-3.1-8b-structured",
                "prompt": messages,
                "completion": completion,
                "usage": {
                    "inputTokens": inputs.shape[-1],
                    "outputTokens": len(completion.split()),
                    "totalTokens": inputs.shape[-1] + len(completion.split())
                },
                "metadata": {
                    "inference_time_ms": latency_ms,
                    "adapter_path": ADAPTER
                }
            }

            # Update agent metrics
            self.metrics["total_runs"] += 1
            self.metrics["successful_runs"] += 1
            total_latency = self.metrics["avg_latency_ms"] * (self.metrics["total_runs"] - 1)
            self.metrics["avg_latency_ms"] = (total_latency + latency_ms) / self.metrics["total_runs"]

            # Store trace
            self.traces.append({
                "trace": trace,
                "observation": observation,
                "generation": generation
            })

            return {
                "success": True,
                "traceId": trace_id,
                "result": completion,
                "parsed": parsed_result,
                "latencyMs": latency_ms,
                "metrics": {
                    "valid_json": is_valid_json,
                    "has_required_fields": self._check_required_fields(parsed_result)
                }
            }

        except Exception as e:
            # Handle error
            trace.update({
                "status": "error",
                "completedAt": datetime.utcnow().isoformat(),
                "metadata": {
                    **trace["metadata"],
                    "error": str(e)
                }
            })

            self.metrics["total_runs"] += 1
            self.metrics["failed_runs"] += 1

            return {
                "success": False,
                "traceId": trace_id,
                "error": str(e)
            }

    def _check_required_fields(self, parsed: Dict | None) -> bool:
        """Check if parsed result has required fields"""
        if not isinstance(parsed, dict):
            return False
        required = {"query", "tool_name", "arguments"}
        return required.issubset(parsed.keys())

    def save_traces(self):
        """Save traces to file for backend to serve"""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Prepare data in format expected by backend
        runs = [t["trace"] for t in self.traces]
        observations = [t["observation"] for t in self.traces]
        generations = [t["generation"] for t in self.traces]

        # Create finetune samples
        samples = []
        for t in self.traces:
            trace = t["trace"]
            gen = t["generation"]
            samples.append({
                "traceId": trace["traceId"],
                "agentId": self.agent_id,
                "conversation": gen["prompt"],
                "steps": [
                    {
                        "type": "generation",
                        "content": gen["completion"]
                    }
                ],
                "finalResponse": gen["completion"]
            })

        # Save traces
        traces_file = OUTPUT_DIR / "traces.json"
        with open(traces_file, 'w') as f:
            json.dump({
                "runs": runs,
                "observations": observations,
                "generations": generations,
                "samples": samples
            }, f, indent=2)

        print(f"💾 Saved {len(self.traces)} traces to {traces_file}")

        # Save metrics
        metrics_file = OUTPUT_DIR / "metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

        print(f"📊 Saved metrics to {metrics_file}")

        # Save detailed evaluation results
        eval_results_file = Path("eval_structured_detailed_results.json")
        if eval_results_file.exists():
            detailed_metrics_file = OUTPUT_DIR / "detailed_metrics.json"
            with open(eval_results_file) as f_in:
                with open(detailed_metrics_file, 'w') as f_out:
                    detailed_data = json.load(f_in)
                    json.dump(detailed_data, f_out, indent=2)
            print(f"📈 Saved detailed metrics to {detailed_metrics_file}")

def main():
    """Run example agent with sample prompts"""
    agent = SLMAgent()
    agent.load_model()

    # Test prompts from evaluation set
    test_prompts = [
        {
            "prompt": "Return a JSON object with keys query, tool_name, arguments describing the API call.\nQuery: Fetch the first 100 countries in ascending order.\nChosen tool: getallcountry\nArguments should mirror the assistant's recommendation.",
            "expected": {"arguments": {"limit": 100, "order": "ASC", "page": 1}, "query": "Fetch the first 100 countries in ascending order.", "tool_name": "getallcountry"}
        },
        {
            "prompt": "Return a JSON object with keys query, tool_name, arguments describing the API call.\nQuery: What is the word frequency in the text 'Hello world, hello universe, world'?\nChosen tool: word_frequency\nArguments should mirror the assistant's recommendation.",
            "expected": {"arguments": {"text": "Hello world, hello universe, world"}, "query": "What is the word frequency in the text 'Hello world, hello universe, world'?", "tool_name": "word_frequency"}
        },
        {
            "prompt": "Return a JSON object with keys query, tool_name, arguments describing the API call.\nQuery: I need to find the least common multiple of 24 and 36 for my math homework, can you help?\nChosen tool: least_common_multiple\nArguments should mirror the assistant's recommendation.",
            "expected": {"arguments": {"a": 24, "b": 36}, "query": "I need to find the least common multiple of 24 and 36 for my math homework, can you help?", "tool_name": "least_common_multiple"}
        },
        {
            "prompt": "Return a JSON object with keys query, tool_name, arguments describing the API call.\nQuery: What events are scheduled for basketball on March 30, 2023?\nChosen tool: schedule_by_date\nArguments should mirror the assistant's recommendation.",
            "expected": {"arguments": {"date": "2023-03-30", "sport_id": 2}, "query": "What events are scheduled for basketball on March 30, 2023?", "tool_name": "schedule_by_date"}
        },
        {
            "prompt": "Return a JSON object with keys query, tool_name, arguments describing the API call.\nQuery: Determine if 'icloud.com' is a disposable domain.\nChosen tool: domain\nArguments should mirror the assistant's recommendation.",
            "expected": {"arguments": {"domain": "icloud.com"}, "query": "Determine if 'icloud.com' is a disposable domain.", "tool_name": "domain"}
        }
    ]

    print(f"\n🚀 Running {len(test_prompts)} example API generation tasks...\n")

    for i, test in enumerate(test_prompts, 1):
        print(f"📝 Test {i}/{len(test_prompts)}")
        print(f"Query: {test['prompt'][:100]}...")

        result = agent.generate(test["prompt"])

        if result["success"]:
            print(f"✅ Success (latency: {result['latencyMs']:.0f}ms)")
            if result["parsed"]:
                print(f"   Tool: {result['parsed'].get('tool_name', 'N/A')}")
                print(f"   Args: {result['parsed'].get('arguments', {})}")

                # Check accuracy
                if result["parsed"] == test["expected"]:
                    print(f"   💯 Exact match!")
                elif result["parsed"].get("tool_name") == test["expected"].get("tool_name"):
                    print(f"   ⚠️  Partial match (correct tool)")
            print()
        else:
            print(f"❌ Failed: {result['error']}\n")

        # Small delay between requests
        time.sleep(0.5)

    # Print summary
    print("=" * 80)
    print("📊 AGENT PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"Total Runs:              {agent.metrics['total_runs']}")
    print(f"Successful:              {agent.metrics['successful_runs']}")
    print(f"Failed:                  {agent.metrics['failed_runs']}")
    print(f"Success Rate:            {agent.metrics['successful_runs'] / agent.metrics['total_runs'] * 100:.1f}%")
    print(f"Avg Latency:             {agent.metrics['avg_latency_ms']:.0f}ms")
    print(f"Total Cost:              ${agent.metrics['total_cost_usd']:.4f} (on-premise)")
    print()
    print("🎯 Model Quality Metrics (from evaluation):")
    print(f"Exact Match Accuracy:    {agent.metrics['accuracy'] * 100:.1f}%")
    print(f"Tool Name Accuracy:      {agent.metrics['tool_name_accuracy'] * 100:.1f}%")
    print(f"Query Preservation:      {agent.metrics['query_preservation'] * 100:.1f}%")
    print(f"JSON Validity:           {agent.metrics['json_validity'] * 100:.1f}%")
    print(f"Functional Correctness:  {agent.metrics['functional_correctness'] * 100:.1f}%")
    print(f"Semantic Correctness:    {agent.metrics['semantic_correctness'] * 100:.1f}%")
    print("=" * 80)

    # Save results
    agent.save_traces()

    print(f"\n✅ Example agent run complete!")
    print(f"📂 Data saved to: {OUTPUT_DIR}")
    print(f"🌐 View in frontend at: http://localhost:5173/metrics")

if __name__ == "__main__":
    main()
