#!/usr/bin/env python3
"""
Lab Autologger - Extract ISA-Tab compliant metadata from unstructured lab notes
Uses LLM to parse and structure experimental data
"""

import argparse
import json
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))

from azure_client import AzureClient
from env_loader import ensure_env_loaded


EXTRACTION_PROMPT_TEMPLATE = """You are an expert bioinformatics assistant specialized in ISA-Tab (Investigation/Study/Assay) metadata standards.

Extract structured, ISA-Tab compliant metadata from the following unstructured lab notes.

LAB NOTES:
{lab_notes}

Extract the following information and return it in the specified JSON format:

1. Study Information:
   - study_identifier: Unique identifier (if not provided, generate one based on context)
   - study_title: Descriptive title of the study
   - study_description: Brief description of the study goals
   - study_submission_date: Date of submission (YYYY-MM-DD format)
   - study_public_release_date: Expected release date (YYYY-MM-DD format)
   - study_file_name: Suggested filename for the study (e.g., "s_study001.txt")

2. Study Factors (experimental variables):
   - name: Factor name (e.g., "temperature", "treatment")
   - type: Factor type (e.g., "environmental", "biological")
   - values: Array of values tested

3. Study Assays (measurements performed):
   - measurement_type: What was measured (e.g., "growth rate", "gene expression")
   - technology_type: Technology used (e.g., "spectrophotometry", "RNA-seq")
   - technology_platform: Specific platform/instrument

4. Study Protocols:
   - name: Protocol name
   - type: Protocol type (e.g., "growth", "sample collection", "data transformation")
   - description: Brief description
   - parameters: Array of parameter names

5. Study Contacts:
   - name: Person's name
   - affiliation: Institution/department
   - email: Email address
   - role: Role in study (e.g., "principal investigator", "researcher")

Respond ONLY with valid JSON in this exact format:
{
  "study_identifier": "STUDY001",
  "study_title": "Effect of Temperature on Bacterial Growth",
  "study_description": "Investigation of growth rate variation...",
  "study_submission_date": "2024-10-15",
  "study_public_release_date": "2025-01-15",
  "study_file_name": "s_temperature_growth.txt",
  "study_factors": [
    {
      "name": "temperature",
      "type": "environmental",
      "values": ["25°C", "30°C", "37°C"]
    }
  ],
  "study_assays": [
    {
      "measurement_type": "growth rate",
      "technology_type": "spectrophotometry",
      "technology_platform": "Spectrophotometer Model X"
    }
  ],
  "study_protocols": [
    {
      "name": "Growth Protocol",
      "type": "growth",
      "description": "Bacterial growth in LB medium",
      "parameters": ["temperature", "duration", "medium"]
    }
  ],
  "study_contacts": [
    {
      "name": "Dr. Sarah Chen",
      "affiliation": "MIT Biology Dept",
      "email": "sarah.chen@mit.edu",
      "role": "principal investigator"
    }
  ]
}

If information is not explicitly provided, make reasonable inferences based on the context. Return ONLY the JSON, no additional text."""


def extract_isatab_metadata(lab_notes: str, model: str = "gpt-4o-mini") -> dict:
    """
    Extract ISA-Tab metadata from unstructured lab notes using LLM.

    Args:
        lab_notes: Raw lab notes text
        model: LLM model to use

    Returns:
        Extracted metadata as dict
    """
    ensure_env_loaded()

    # Prepare extraction prompt
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(lab_notes=lab_notes)

    # Use Azure client for extraction
    client = AzureClient()

    messages = [
        {
            "role": "system",
            "content": "You are an expert bioinformatics assistant. Extract structured ISA-Tab metadata accurately. Return only valid JSON."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        response = client.generate(messages)

        # Try to extract JSON from response
        if "```json" in response:
            json_match = response.split("```json")[1].split("```")[0].strip()
            result = json.loads(json_match)
        elif "```" in response:
            # Try generic code block
            json_match = response.split("```")[1].split("```")[0].strip()
            result = json.loads(json_match)
        elif "{" in response and "}" in response:
            # Try to parse as raw JSON
            start = response.index("{")
            end = response.rindex("}") + 1
            result = json.loads(response[start:end])
        else:
            raise ValueError("No valid JSON found in response")

        # Validate required fields
        required_fields = [
            "study_identifier", "study_title", "study_description",
            "study_submission_date", "study_public_release_date", "study_file_name"
        ]

        for field in required_fields:
            if field not in result:
                result[field] = "Not specified"

        # Ensure arrays exist
        for field in ["study_factors", "study_assays", "study_protocols", "study_contacts"]:
            if field not in result:
                result[field] = []

        return result

    except Exception as e:
        print(f"ERROR: Extraction failed: {e}", file=sys.stderr)
        # Return minimal valid structure
        return {
            "study_identifier": "ERROR",
            "study_title": "Extraction Failed",
            "study_description": f"Error: {str(e)}",
            "study_submission_date": "2024-01-01",
            "study_public_release_date": "2024-01-01",
            "study_file_name": "error.txt",
            "study_factors": [],
            "study_assays": [],
            "study_protocols": [],
            "study_contacts": []
        }


def main():
    parser = argparse.ArgumentParser(
        description="Extract ISA-Tab metadata from unstructured lab notes"
    )
    parser.add_argument(
        "--lab-notes",
        required=True,
        help="Lab notes text to extract metadata from"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="LLM model to use for extraction"
    )

    args = parser.parse_args()

    print("Extracting ISA-Tab metadata from lab notes...", file=sys.stderr)

    result = extract_isatab_metadata(
        lab_notes=args.lab_notes,
        model=args.model
    )

    # Calculate confidence score based on completeness
    total_fields = 6  # Required fields
    filled_fields = sum(
        1 for f in ["study_identifier", "study_title", "study_description",
                    "study_submission_date", "study_public_release_date", "study_file_name"]
        if result.get(f) and result[f] != "Not specified"
    )
    confidence_score = filled_fields / total_fields

    # Output result with markers for easy parsing
    output = {
        "data": result,
        "confidence_score": confidence_score
    }

    print("<<<EXTRACTION_RESULT>>>")
    print(json.dumps(output, indent=2))
    print("<<<END_RESULT>>>")


if __name__ == "__main__":
    main()
