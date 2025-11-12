#!/usr/bin/env python3
"""
Automatic Workflow Tracer for Continuous SLM Learning

Captures LLM calls from existing workflows and automatically:
1. Builds training datasets
2. Detects when enough data is collected
3. Triggers automatic retraining
4. Evaluates SLM vs LLM performance
5. Gradually swaps LLM → SLM when ready

FULLY AUTOMATIC after initial setup.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import subprocess


@dataclass
class WorkflowCall:
    """Single LLM/SLM call captured from workflow"""
    timestamp: str
    agent_type: str  # "orchestrator" or "sub_agent"
    agent_name: str  # e.g. "retrieval_agent", "processing_agent"
    model_used: str  # "gpt-4", "llama-slm", "phi-slm"
    system_prompt: str
    user_query: str
    assistant_response: str
    tool_calls: Optional[List[Dict]] = None
    success: bool = True
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    cost_usd: float = 0.0


class WorkflowTracer:
    """
    Automatically captures and learns from workflow executions

    Usage:
        tracer = WorkflowTracer()

        # In your existing workflow:
        with tracer.trace_call("orchestrator", "main"):
            result = llm.call(query)

        # Automatic learning happens in background
    """

    def __init__(self, storage_dir="slm_swap/workflow_traces", min_examples=100):
        self.storage_dir = storage_dir
        self.min_examples = min_examples
        self.calls = []

        os.makedirs(storage_dir, exist_ok=True)

        # Load existing traces
        self._load_existing_traces()

        print(f"WorkflowTracer initialized. {len(self.calls)} existing traces loaded.")
        if len(self.calls) >= self.min_examples:
            print(f"✅ Ready for retraining! ({len(self.calls)} >= {self.min_examples})")

    def _load_existing_traces(self):
        """Load previously captured traces"""
        trace_file = os.path.join(self.storage_dir, "all_traces.jsonl")
        if os.path.exists(trace_file):
            with open(trace_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.calls.append(WorkflowCall(**json.loads(line)))

    def capture_call(
        self,
        agent_type: str,
        agent_name: str,
        model_used: str,
        system_prompt: str,
        user_query: str,
        assistant_response: str,
        tool_calls: Optional[List[Dict]] = None,
        success: bool = True,
        error: Optional[str] = None,
        execution_time_ms: float = 0.0,
        cost_usd: float = 0.0
    ):
        """
        Capture a single workflow call

        Example:
            tracer.capture_call(
                agent_type="orchestrator",
                agent_name="main_orchestrator",
                model_used="gpt-4",
                system_prompt="You are an orchestrator...",
                user_query="Analyze sales data",
                assistant_response='<tool_call name="query_db">...</tool_call>',
                success=True
            )
        """
        call = WorkflowCall(
            timestamp=datetime.now().isoformat(),
            agent_type=agent_type,
            agent_name=agent_name,
            model_used=model_used,
            system_prompt=system_prompt,
            user_query=user_query,
            assistant_response=assistant_response,
            tool_calls=tool_calls,
            success=success,
            error=error,
            execution_time_ms=execution_time_ms,
            cost_usd=cost_usd
        )

        self.calls.append(call)

        # Save to disk
        trace_file = os.path.join(self.storage_dir, "all_traces.jsonl")
        with open(trace_file, "a") as f:
            f.write(json.dumps(asdict(call)) + "\n")

        # Check if ready for retraining
        if len(self.calls) >= self.min_examples and len(self.calls) % 50 == 0:
            print(f"\n🔔 {len(self.calls)} traces collected. Consider retraining!")
            self._check_auto_retrain()

    def generate_training_dataset(self, output_path: Optional[str] = None) -> str:
        """
        Convert captured calls into Hermes-format training dataset

        Returns:
            Path to generated dataset
        """
        if output_path is None:
            output_path = os.path.join(
                self.storage_dir,
                f"auto_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )

        # Convert to Hermes format
        hermes_examples = []
        for call in self.calls:
            if not call.success:
                continue  # Skip failed calls

            hermes_example = {
                "conversations": [
                    {"from": "system", "value": call.system_prompt},
                    {"from": "human", "value": call.user_query},
                    {"from": "gpt", "value": call.assistant_response}
                ],
                "metadata": {
                    "agent_type": call.agent_type,
                    "agent_name": call.agent_name,
                    "captured_from": call.model_used,
                    "timestamp": call.timestamp
                }
            }
            hermes_examples.append(hermes_example)

        # Write to file
        with open(output_path, "w") as f:
            for example in hermes_examples:
                f.write(json.dumps(example) + "\n")

        print(f"✅ Training dataset generated: {output_path}")
        print(f"   {len(hermes_examples)} examples")

        return output_path

    def _check_auto_retrain(self):
        """Check if automatic retraining should be triggered"""
        # Only retrain if we have enough NEW examples since last training
        last_retrain_file = os.path.join(self.storage_dir, "last_retrain.txt")

        if os.path.exists(last_retrain_file):
            with open(last_retrain_file, "r") as f:
                last_retrain_count = int(f.read().strip())

            new_examples = len(self.calls) - last_retrain_count

            if new_examples < self.min_examples:
                print(f"   Not enough new examples yet ({new_examples}/{self.min_examples})")
                return

        print("\n" + "=" * 70)
        print("🚀 AUTOMATIC RETRAINING TRIGGERED")
        print("=" * 70)

        # Generate dataset
        dataset_path = self.generate_training_dataset()

        # Mark retrain point
        with open(last_retrain_file, "w") as f:
            f.write(str(len(self.calls)))

        print(f"\n📊 Dataset ready: {dataset_path}")
        print(f"👉 Run training with: python slm_swap/train_llama_cep.py --dataset {dataset_path}")
        print("=" * 70)

    def get_statistics(self) -> Dict:
        """Get statistics about captured calls"""
        stats = {
            "total_calls": len(self.calls),
            "successful_calls": sum(1 for c in self.calls if c.success),
            "failed_calls": sum(1 for c in self.calls if not c.success),
            "by_agent_type": defaultdict(int),
            "by_model": defaultdict(int),
            "total_cost_usd": sum(c.cost_usd for c in self.calls),
            "avg_execution_time_ms": sum(c.execution_time_ms for c in self.calls) / len(self.calls) if self.calls else 0
        }

        for call in self.calls:
            stats["by_agent_type"][call.agent_type] += 1
            stats["by_model"][call.model_used] += 1

        return dict(stats)

    def print_statistics(self):
        """Print statistics about captured workflow"""
        stats = self.get_statistics()

        print("\n" + "=" * 70)
        print("WORKFLOW TRACER STATISTICS")
        print("=" * 70)
        print(f"Total Calls: {stats['total_calls']}")
        print(f"Successful: {stats['successful_calls']}")
        print(f"Failed: {stats['failed_calls']}")
        print(f"\nBy Agent Type:")
        for agent_type, count in stats["by_agent_type"].items():
            print(f"  {agent_type}: {count}")
        print(f"\nBy Model:")
        for model, count in stats["by_model"].items():
            print(f"  {model}: {count}")
        print(f"\nTotal Cost: ${stats['total_cost_usd']:.4f}")
        print(f"Avg Execution Time: {stats['avg_execution_time_ms']:.2f}ms")
        print("=" * 70)


# Global tracer instance for easy integration
_global_tracer: Optional[WorkflowTracer] = None


def get_tracer() -> WorkflowTracer:
    """Get or create global tracer instance"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = WorkflowTracer()
    return _global_tracer


# Decorator for automatic tracing
def trace_workflow_call(agent_type: str, agent_name: str):
    """
    Decorator to automatically trace function calls

    Usage:
        @trace_workflow_call("orchestrator", "main")
        def my_llm_function(query):
            return llm.call(query)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracer = get_tracer()
            import time
            start = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start) * 1000

                # Extract call details (assumes specific format)
                # TODO: Customize based on your LLM interface
                tracer.capture_call(
                    agent_type=agent_type,
                    agent_name=agent_name,
                    model_used="gpt-4",  # TODO: Detect automatically
                    system_prompt="",  # TODO: Extract from args
                    user_query=str(args[0]) if args else "",
                    assistant_response=str(result),
                    success=True,
                    execution_time_ms=execution_time
                )

                return result

            except Exception as e:
                execution_time = (time.time() - start) * 1000
                tracer.capture_call(
                    agent_type=agent_type,
                    agent_name=agent_name,
                    model_used="gpt-4",
                    system_prompt="",
                    user_query=str(args[0]) if args else "",
                    assistant_response="",
                    success=False,
                    error=str(e),
                    execution_time_ms=execution_time
                )
                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # Demo usage
    tracer = WorkflowTracer()

    # Simulate captured workflow calls
    print("\nSimulating workflow capture...")

    for i in range(5):
        tracer.capture_call(
            agent_type="orchestrator",
            agent_name="main_orchestrator",
            model_used="gpt-4",
            system_prompt="You are a function calling orchestrator.",
            user_query=f"Process request {i+1}",
            assistant_response=f'<tool_call name="process">{{id": {i+1}}}</tool_call>',
            success=True,
            execution_time_ms=150.5,
            cost_usd=0.002
        )

    # Show statistics
    tracer.print_statistics()

    # Generate dataset
    dataset_path = tracer.generate_training_dataset()
    print(f"\n✅ Demo dataset generated: {dataset_path}")
