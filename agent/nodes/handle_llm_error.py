"""
Node 5 — handle_llm_error

Reached only when generate_report sets generation_error (LLM call failed
or JSON parsing failed completely).

Responsibilities:
  1. Log / preserve the error message.
  2. Insert a mock report as structured_report so validate_output always
     has something to work with and the UI never shows a blank page.
"""

from __future__ import annotations

from typing import Any

from utils.mock_llm import generate_mock_report


def handle_llm_error(state: dict[str, Any]) -> dict[str, Any]:
    error_msg: str = state.get("generation_error", "Unknown LLM error.")

    # Generate a mock report from real computed stats
    mock = generate_mock_report(state)

    return {
        # Preserve original error for display in the UI
        "generation_error": error_msg,
        # Provide the mock as the structured report so downstream nodes proceed
        "structured_report": mock.model_dump(mode="json"),
    }
