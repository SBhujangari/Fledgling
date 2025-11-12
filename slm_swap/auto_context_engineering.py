#!/usr/bin/env python3
"""
Automatic Context Engineering for SLM Function Calling
Optimizes prompts dynamically based on:
1. Error patterns from SLM predictions
2. Format validation feedback
3. Self-reflection on failed examples

Works in tandem with fine-tuning to maximize performance.
"""

import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PromptTemplate:
    """A prompt template with placeholders"""
    name: str
    template: str
    success_rate: float = 0.0
    usage_count: int = 0


class AutoContextEngineer:
    """
    Automatically optimizes prompts for SLM function calling.

    Strategy:
    1. Error Analysis: Identify common failure patterns
    2. Template Evolution: Generate improved prompt variants
    3. A/B Testing: Test templates and track success rates
    4. Self-Correction: Add few-shot examples from failures
    """

    def __init__(self):
        self.templates = {
            "structured_json": [],
            "toolcall_xml": []
        }
        self.error_patterns = defaultdict(int)
        self.best_templates = {}

        # Initialize base templates
        self._init_base_templates()

    def _init_base_templates(self):
        """Initialize baseline prompt templates"""

        # Structured JSON templates
        self.templates["structured_json"] = [
            PromptTemplate(
                name="strict_json_v1",
                template="""You are a precise API call generator. Return ONLY a valid JSON object with these exact keys: query, tool_name, arguments.

CRITICAL RULES:
1. NO markdown code blocks (no ```json```)
2. NO additional text before or after JSON
3. Match parameter names EXACTLY as shown in the tool signature
4. Use proper JSON types (arrays as [], strings as "")

Query: {query}
Tool: {tool_name}
Tool Signature: {tool_signature}

Output (raw JSON only):"""
            ),

            PromptTemplate(
                name="few_shot_json_v1",
                template="""Generate an API call as JSON. Follow these examples EXACTLY:

Example 1:
Query: Search for cities starting with 'San' in US
Tool: autocomplete_places(q:str, country:str, limit:int)
Output: {{"query": "Search for cities", "tool_name": "autocomplete_places", "arguments": {{"q": "San", "country": "US", "limit": 5}}}}

Example 2:
Query: Get chart for BTC with dark theme
Tool: mini_chart(symbol:str, theme:str, interval:str)
Output: {{"query": "Get chart for BTC", "tool_name": "mini_chart", "arguments": {{"symbol": "BINANCE:BTC", "theme": "dark", "interval": "1D"}}}}

Now generate for:
Query: {query}
Tool: {tool_name}({tool_signature})
Output:"""
            ),

            PromptTemplate(
                name="self_correcting_json_v1",
                template="""You are an API call generator. You previously made these mistakes - AVOID THEM:

Common Errors to Avoid:
- Using "query" instead of "q" as parameter name
- Using array ["US", "CA"] instead of comma-separated "US,CA"
- Adding markdown code blocks around JSON
- Missing required fields like "format", "width", "height"

Task:
Query: {query}
Tool: {tool_name}
Signature: {tool_signature}

Generate ONLY the JSON object (no markdown, no extra text):"""
            )
        ]

        # Toolcall XML templates
        self.templates["toolcall_xml"] = [
            PromptTemplate(
                name="strict_xml_v1",
                template="""Generate a function call in XML format. Follow this EXACT format:

<tool_call name="FUNCTION_NAME">{{"arg1": "value1", "arg2": "value2"}}</tool_call>

CRITICAL RULES:
1. Opening tag: <tool_call name="...">
2. JSON arguments inside: {{"key": "value"}}
3. Closing tag: </tool_call>
4. NO self-closing tags like <tool_call ... />
5. NO "arguments=" attribute
6. NO extra text before or after

Query: {query}
Function: {tool_name}({tool_signature})

Output (exact XML format):"""
            ),

            PromptTemplate(
                name="few_shot_xml_v1",
                template="""Generate tool calls exactly like these examples:

Example 1:
Query: Get pit stop data for F1 2023 round 1
Function: pitstopdataforarace(year:str, round:str)
Output: <tool_call name="pitstopdataforarace">{{"year": "2023", "round": "1"}}</tool_call>

Example 2:
Query: Get song details for ID 54321 in French
Function: songs_v2_get_details(is_id:str, l:str)
Output: <tool_call name="songs_v2_get_details">{{"is_id": "54321", "l": "fr-FR"}}</tool_call>

Now generate:
Query: {query}
Function: {tool_name}({tool_signature})
Output:"""
            ),

            PromptTemplate(
                name="error_aware_xml_v1",
                template="""You previously failed at XML tool calls. These were WRONG:

❌ WRONG: 2023\\n  rounds: ["1"]\\n</tool_call>  (missing opening tag)
❌ WRONG: <tool_call name="func" arguments='{{...}}'>  (using "arguments=" attribute)
❌ WRONG: <tool_call name="func" {{...}} />  (self-closing tag)

✅ CORRECT FORMAT:
<tool_call name="FUNCTION_NAME">{{"arg": "value"}}</tool_call>

Task:
Query: {query}
Function: {tool_name}({tool_signature})

Generate (follow correct format exactly):"""
            )
        ]

    def analyze_error(self, prediction: str, reference: str, track: str, issues: List[str]) -> Dict:
        """Analyze what went wrong with a prediction"""
        error_info = {
            "track": track,
            "issues": issues,
            "patterns": []
        }

        # Detect common patterns
        if track == "structured_json":
            if "```json" in prediction or "```" in prediction:
                error_info["patterns"].append("markdown_wrapper")
                self.error_patterns["markdown_wrapper"] += 1

            if re.search(r'"query":\s*"[^"]*"', prediction) and '"q":' in reference:
                error_info["patterns"].append("param_name_mismatch")
                self.error_patterns["param_name_mismatch"] += 1

            if re.search(r'\[["\w",\s]+\]', prediction):
                error_info["patterns"].append("array_instead_of_string")
                self.error_patterns["array_instead_of_string"] += 1

        elif track == "toolcall_xml":
            if not prediction.startswith("<tool_call"):
                error_info["patterns"].append("missing_opening_tag")
                self.error_patterns["missing_opening_tag"] += 1

            if "arguments=" in prediction:
                error_info["patterns"].append("using_arguments_attribute")
                self.error_patterns["using_arguments_attribute"] += 1

            if "/>" in prediction:
                error_info["patterns"].append("self_closing_tag")
                self.error_patterns["self_closing_tag"] += 1

        return error_info

    def select_best_template(self, track: str, context: Dict) -> PromptTemplate:
        """
        Select the best-performing template for given context.
        Uses Thompson Sampling for exploration/exploitation.
        """
        templates = self.templates[track]

        # If no usage data, return strict template
        if all(t.usage_count == 0 for t in templates):
            return templates[0]  # Return strict version

        # Return template with highest success rate (with some exploration)
        import random
        if random.random() < 0.1:  # 10% exploration
            return random.choice(templates)

        return max(templates, key=lambda t: t.success_rate if t.usage_count > 0 else 0.0)

    def format_prompt(self, template: PromptTemplate, context: Dict) -> str:
        """Format a template with context variables"""
        return template.template.format(**context)

    def update_template_performance(self, template_name: str, track: str, success: bool):
        """Update template success rate based on outcome"""
        for template in self.templates[track]:
            if template.name == template_name:
                template.usage_count += 1
                # Running average
                alpha = 1.0 / template.usage_count
                template.success_rate = (1 - alpha) * template.success_rate + alpha * (1.0 if success else 0.0)
                break

    def generate_improved_template(self, track: str, error_patterns: Dict) -> Optional[PromptTemplate]:
        """
        Generate a new template variant based on observed errors.
        Uses error patterns to create targeted corrections.
        """
        if track == "structured_json":
            if error_patterns.get("markdown_wrapper", 0) > 5:
                return PromptTemplate(
                    name=f"no_markdown_v{len(self.templates[track]) + 1}",
                    template="""CRITICAL: Output ONLY raw JSON. NO markdown code blocks.

BAD (DO NOT DO THIS):
```json
{{"key": "value"}}
```

GOOD (DO THIS):
{{"key": "value"}}

Query: {query}
Tool: {tool_name}({tool_signature})

Raw JSON output:"""
                )

        elif track == "toolcall_xml":
            if error_patterns.get("missing_opening_tag", 0) > 5:
                return PromptTemplate(
                    name=f"strict_opening_tag_v{len(self.templates[track]) + 1}",
                    template="""Start your response with: <tool_call name="FUNCTION_NAME">

TEMPLATE:
<tool_call name="FUNCTION_NAME">{{"arg1": "val1", "arg2": "val2"}}</tool_call>

Query: {query}
Function: {tool_name}({tool_signature})

Start with <tool_call name="{tool_name}">:"""
                )

        return None

    def optimize_for_workflow(self, workflow_data: List[Dict]) -> Dict[str, PromptTemplate]:
        """
        Optimize prompts based on actual workflow execution data.

        Args:
            workflow_data: List of {query, tool_name, prediction, reference, success}

        Returns:
            Best templates per track
        """
        for item in workflow_data:
            track = item.get("track", "structured_json")

            # Analyze errors
            if not item["success"]:
                error_info = self.analyze_error(
                    item["prediction"],
                    item["reference"],
                    track,
                    item.get("issues", [])
                )

            # Update template performance
            if "template_used" in item:
                self.update_template_performance(
                    item["template_used"],
                    track,
                    item["success"]
                )

        # Generate improved templates if error patterns detected
        for track in ["structured_json", "toolcall_xml"]:
            new_template = self.generate_improved_template(track, self.error_patterns)
            if new_template:
                self.templates[track].append(new_template)
                print(f"Generated new template for {track}: {new_template.name}")

        # Return best templates
        best = {}
        for track in ["structured_json", "toolcall_xml"]:
            best[track] = self.select_best_template(track, {})

        return best

    def get_error_report(self) -> Dict:
        """Generate error pattern report"""
        return {
            "error_patterns": dict(self.error_patterns),
            "template_performance": {
                track: [
                    {
                        "name": t.name,
                        "success_rate": t.success_rate,
                        "usage_count": t.usage_count
                    }
                    for t in templates
                ]
                for track, templates in self.templates.items()
            }
        }


# Integration with fine-tuned models
class HybridSLM:
    """
    Combines fine-tuned SLM with automatic context engineering.

    Strategy:
    1. Fine-tuned model learns general patterns
    2. Auto context engineering optimizes prompts per use case
    3. Continuous improvement from workflow feedback
    """

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.context_engineer = AutoContextEngineer()

    def predict(self, query: str, tool_name: str, tool_signature: str, track: str) -> Tuple[str, str]:
        """
        Generate prediction with optimized prompt.

        Returns:
            (prediction, template_name)
        """
        # Select best template
        context = {
            "query": query,
            "tool_name": tool_name,
            "tool_signature": tool_signature
        }
        template = self.context_engineer.select_best_template(track, context)

        # Format prompt
        optimized_prompt = self.context_engineer.format_prompt(template, context)

        # TODO: Run inference with fine-tuned model
        # prediction = self.model.generate(optimized_prompt)

        return "PLACEHOLDER_PREDICTION", template.name

    def update_from_workflow(self, workflow_results: List[Dict]):
        """Update context engineering based on workflow outcomes"""
        best_templates = self.context_engineer.optimize_for_workflow(workflow_results)

        print("Context Engineering Update:")
        print(json.dumps(self.context_engineer.get_error_report(), indent=2))

        return best_templates


if __name__ == "__main__":
    # Example: Test context engineering
    engineer = AutoContextEngineer()

    # Simulate workflow data
    workflow_data = [
        {
            "track": "structured_json",
            "query": "Find San cities",
            "tool_name": "autocomplete_places",
            "prediction": "```json\n{\"query\": \"San\"}```",
            "reference": "{\"q\": \"San\"}",
            "success": False,
            "issues": ["markdown_wrapper", "param_name_mismatch"],
            "template_used": "strict_json_v1"
        },
        {
            "track": "toolcall_xml",
            "query": "Get pit stops",
            "tool_name": "pitstopdataforarace",
            "prediction": "2023\n</tool_call>",
            "reference": "<tool_call name=\"pitstopdataforarace\">{...}</tool_call>",
            "success": False,
            "issues": ["missing_opening_tag"],
            "template_used": "strict_xml_v1"
        }
    ]

    best_templates = engineer.optimize_for_workflow(workflow_data)

    print("\n=== Best Templates ===")
    for track, template in best_templates.items():
        print(f"\n{track}: {template.name}")
        print(f"Success Rate: {template.success_rate:.2%}")
        print(f"Preview:\n{template.template[:200]}...")

    print("\n=== Error Report ===")
    print(json.dumps(engineer.get_error_report(), indent=2))
