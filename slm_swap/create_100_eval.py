"""Create 100-example evaluation datasets from xLAM source."""

import json
import random
import os

SEED = 42
random.seed(SEED)

def load_xlam_dataset(path: str = None):
    """Load the xLAM dataset."""
    if path is None:
        # Default to parent directory
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_dir, "xlam_function_calling_60k.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_structured_example(item):
    """Convert xLAM item to structured JSON task."""
    tools = json.loads(item["tools"])
    answers = json.loads(item["answers"])

    # Create a prompt asking for structured JSON
    tool_desc = "\n".join([f"- {t['name']}: {t.get('description', '')}" for t in tools])
    prompt = f"{item['query']}\n\nAvailable tools:\n{tool_desc}\n\nReturn a JSON object with the tool calls needed."

    # Expected completion is the answers as JSON
    completion = json.dumps(answers, separators=(",", ":"))

    return {"prompt": prompt, "completion": completion}

def create_toolcall_example(item):
    """Convert xLAM item to tool call task."""
    tools = json.loads(item["tools"])
    answers = json.loads(item["answers"])

    if not answers or not tools:
        return None

    # Use the first tool call
    answer = answers[0]
    tool_name = answer["name"]
    tool_args = answer.get("arguments", {})

    # Find matching tool
    matching_tool = next((t for t in tools if t["name"] == tool_name), None)
    if not matching_tool:
        return None

    # Create prompt
    params_desc = matching_tool.get("parameters", {})
    params_str = json.dumps(params_desc, indent=2)
    prompt = f"{item['query']}\n\nTool: {tool_name}\nParameters: {params_str}\n\nCall this tool with appropriate arguments."

    # Expected completion in tool call format
    args_json = json.dumps(tool_args, separators=(",", ":"))
    completion = f'<tool_call name="{tool_name}">{args_json}</tool_call>'

    return {"prompt": prompt, "completion": completion}

def main():
    print("Loading xLAM dataset...")
    data = load_xlam_dataset()
    print(f"Loaded {len(data)} examples")

    # Shuffle and take samples
    random.shuffle(data)

    # Create 100 examples for each track
    structured_examples = []
    toolcall_examples = []

    for item in data:
        if len(structured_examples) < 100:
            try:
                ex = create_structured_example(item)
                structured_examples.append(ex)
            except (json.JSONDecodeError, KeyError):
                continue

        if len(toolcall_examples) < 100:
            try:
                ex = create_toolcall_example(item)
                if ex:
                    toolcall_examples.append(ex)
            except (json.JSONDecodeError, KeyError):
                continue

        if len(structured_examples) >= 100 and len(toolcall_examples) >= 100:
            break

    # Save datasets - use paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    structured_dir = os.path.join(script_dir, "02_dataset/structured")
    toolcall_dir = os.path.join(script_dir, "02_dataset/toolcall")

    os.makedirs(structured_dir, exist_ok=True)
    os.makedirs(toolcall_dir, exist_ok=True)

    structured_path = os.path.join(structured_dir, "eval100.jsonl")
    toolcall_path = os.path.join(toolcall_dir, "eval100.jsonl")

    with open(structured_path, "w", encoding="utf-8") as f:
        for ex in structured_examples:
            f.write(json.dumps(ex) + "\n")

    with open(toolcall_path, "w", encoding="utf-8") as f:
        for ex in toolcall_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Created {len(structured_examples)} structured examples -> {structured_path}")
    print(f"Created {len(toolcall_examples)} toolcall examples -> {toolcall_path}")

if __name__ == "__main__":
    main()
