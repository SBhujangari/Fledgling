"""Shared system prompts used for evaluation and training."""

SYSTEM_PROMPTS = {
    "structured": "You return valid JSON objects only. Do not add prose or code fences.",
    "toolcall": (
        "You must reply with exactly one <tool_call name=\"...\">{...}</tool_call> wrapper "
        "matching the tool signature in the prompt. Arguments must be JSON."
    ),
}


def get_system_prompt(track: str) -> str:
    if track not in SYSTEM_PROMPTS:
        raise ValueError(f"Unknown track: {track}")
    return SYSTEM_PROMPTS[track]
