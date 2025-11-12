#!/usr/bin/env python3
"""
Context Engineering Prefix (CEP) Configuration

Universal prefix that enforces rigid structure for both:
- Structured JSON outputs
- Tool call XML outputs

Inspired by FTP (Fine-Tuning Prefix) from the paper, but generalized.
"""

# Universal CEP that works for BOTH structured and toolcall tracks
UNIVERSAL_CEP = """<|formatting_rules|>
CRITICAL OUTPUT RULES - FOLLOW EXACTLY:

1. JSON Outputs:
   - Output ONLY raw JSON object
   - NO markdown blocks (no ```json or ```)
   - NO explanatory text before or after
   - Match parameter names EXACTLY as in signature
   - Use proper JSON types: strings "...", numbers 123, arrays [...]

2. XML Tool Calls:
   - Format: <tool_call name="FUNC_NAME">{"arg": "value"}</tool_call>
   - MUST have opening tag: <tool_call name="...">
   - MUST have closing tag: </tool_call>
   - NO self-closing tags (NOT <tool_call ... />)
   - NO "arguments=" attribute
   - JSON dict inside tags

3. Common Errors to AVOID:
   ❌ ```json{"key": "value"}``` (markdown wrapper)
   ❌ <tool_call arguments='...'> (wrong attribute)
   ❌ <tool_call name="func" ... /> (self-closing)
   ❌ {"query": "x"} when signature says "q" (param mismatch)
   ❌ {"country": ["US","CA"]} when expecting "US,CA" (array vs string)

4. Parameter Mapping:
   - "q" NOT "query"
   - "is_id" NOT "id"
   - Check signature for EXACT names
   - Include ALL required fields
   - Omit optional fields if not needed

VERIFY: Does output match format rules? Check before responding.
<|end_formatting_rules|>

"""

# Shorter version for efficiency (same rules, condensed)
COMPACT_CEP = """<|fmt|>
OUTPUT RULES:
- JSON: Raw only, no ```json, exact param names
- XML: <tool_call name="F">{"a":"v"}</tool_call>, no />
- NO: markdown, arguments=, param mismatches
VERIFY format before output.
<|/fmt|>

"""

# Track-specific prefixes (if needed for specialization)
STRUCTURED_JSON_CEP = """<|json_rules|>
Return ONLY valid JSON with exact keys: query, tool_name, arguments
- NO ```json wrappers
- Match parameter names from signature EXACTLY
- Use correct types (arrays as [], not comma-separated)
<|/json_rules|>

"""

TOOLCALL_XML_CEP = """<|xml_rules|>
Return tool call as: <tool_call name="FUNCTION">{"arg":"val"}</tool_call>
- MUST start with <tool_call name="...">
- MUST end with </tool_call>
- NO self-closing <tool_call/>
- NO arguments= attribute
<|/xml_rules|>

"""


class CEPFormatter:
    """
    Applies Context Engineering Prefix to training examples
    """

    def __init__(self, cep_type="universal"):
        """
        Args:
            cep_type: "universal", "compact", "structured", or "toolcall"
        """
        self.cep_type = cep_type
        self.cep_map = {
            "universal": UNIVERSAL_CEP,
            "compact": COMPACT_CEP,
            "structured": STRUCTURED_JSON_CEP,
            "toolcall": TOOLCALL_XML_CEP
        }
        self.cep = self.cep_map.get(cep_type, UNIVERSAL_CEP)

    def apply_to_system_prompt(self, system_message: str) -> str:
        """
        Prepend CEP to system message

        Args:
            system_message: Original system message

        Returns:
            Enhanced system message with CEP
        """
        return self.cep + system_message

    def apply_to_example(self, example_text: str, format_type: str = "llama") -> str:
        """
        Apply CEP to a formatted conversation

        Args:
            example_text: Pre-formatted conversation text
            format_type: "llama", "phi", or "generic"

        Returns:
            Conversation with CEP injected
        """
        if format_type == "llama":
            # Llama-3.1 format: inject after <|begin_of_text|><|start_header_id|>system<|end_header_id|>
            return example_text.replace(
                "<|start_header_id|>system<|end_header_id|>\n\n",
                f"<|start_header_id|>system<|end_header_id|>\n\n{self.cep}"
            )
        elif format_type == "phi":
            # Phi format: inject after <|system|>
            return example_text.replace(
                "<|system|>\n",
                f"<|system|>\n{self.cep}"
            )
        else:
            # Generic: prepend to start
            return self.cep + example_text

    def get_cep(self) -> str:
        """Get the current CEP"""
        return self.cep


# Automatic CEP selection based on error patterns
def auto_select_cep(error_patterns: dict) -> str:
    """
    Automatically select best CEP based on observed errors

    Args:
        error_patterns: Dict of error type -> count

    Returns:
        Best CEP type
    """
    # If mostly JSON errors, use structured CEP
    json_errors = error_patterns.get("markdown_wrapper", 0) + \
                  error_patterns.get("param_name_mismatch", 0)

    # If mostly XML errors, use toolcall CEP
    xml_errors = error_patterns.get("missing_opening_tag", 0) + \
                 error_patterns.get("self_closing_tag", 0)

    # If both or neither, use universal
    if json_errors > xml_errors and json_errors > 5:
        return "structured"
    elif xml_errors > json_errors and xml_errors > 5:
        return "toolcall"
    else:
        return "universal"


if __name__ == "__main__":
    # Demo
    formatter = CEPFormatter("universal")

    print("=" * 70)
    print("Context Engineering Prefix (CEP) Demo")
    print("=" * 70)

    print("\nUniversal CEP:")
    print(formatter.get_cep())

    print("\n" + "=" * 70)
    print("Example Usage:")
    print("=" * 70)

    system_msg = "You are a function calling AI model."
    enhanced = formatter.apply_to_system_prompt(system_msg)

    print("\nOriginal System Message:")
    print(system_msg)

    print("\nEnhanced with CEP:")
    print(enhanced[:200] + "...")

    print("\n" + "=" * 70)
    print("Auto CEP Selection Demo:")
    print("=" * 70)

    error_patterns = {
        "markdown_wrapper": 10,
        "param_name_mismatch": 8,
        "missing_opening_tag": 2
    }

    best_cep = auto_select_cep(error_patterns)
    print(f"\nError patterns: {error_patterns}")
    print(f"Auto-selected CEP: {best_cep}")
