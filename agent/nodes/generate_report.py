"""
Node 4 — generate_report

Calls the Groq LLM (llama-3.1-8b-instant) with the assembled prompt and attempts
to parse the response into a structured dict. Falls back to mock_llm when
GROQ_API_KEY is absent or the LLM call raises an exception.

Parse strategy (3-attempt fallback chain):
  1. json.loads() on the raw response text
  2. Regex extraction of ```json ... ``` block → json.loads()
  3. generation_error set; handle_llm_error node takes over
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from agent.prompts import SYSTEM_PROMPT, build_human_prompt
from utils.mock_llm import generate_mock_report


def generate_report(state: dict[str, Any]) -> dict[str, Any]:
    api_key = os.getenv("GROQ_API_KEY", "").strip()

    # ------------------------------------------------------------------
    # Fast-path: no API key → Demo Mode immediately
    # ------------------------------------------------------------------
    if not api_key:
        mock = generate_mock_report(state)
        return {
            "raw_llm_output": "[Demo Mode — GROQ_API_KEY not set]",
            "structured_report": mock.model_dump(mode="json"),
            "generation_error": None,
        }

    # ------------------------------------------------------------------
    # Live LLM path
    # ------------------------------------------------------------------
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        return {
            "raw_llm_output": "",
            "structured_report": None,
            "generation_error": (
                "langchain-groq is not installed. Run: pip install langchain-groq langchain-core"
            ),
        }

    human_prompt = build_human_prompt(state)

    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            max_tokens=2048,
            api_key=api_key,
        )
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=human_prompt),
        ])
        raw_text: str = response.content

    except Exception as exc:
        return {
            "raw_llm_output": "",
            "structured_report": None,
            "generation_error": f"LLM call failed: {type(exc).__name__}: {exc}",
        }

    # ------------------------------------------------------------------
    # Parse attempt 1: direct json.loads()
    # ------------------------------------------------------------------
    parsed = _try_json_loads(raw_text)
    if parsed is not None:
        return {
            "raw_llm_output": raw_text,
            "structured_report": parsed,
            "generation_error": None,
        }

    # ------------------------------------------------------------------
    # Parse attempt 2: extract ```json ... ``` block
    # ------------------------------------------------------------------
    extracted = _extract_json_block(raw_text)
    if extracted is not None:
        return {
            "raw_llm_output": raw_text,
            "structured_report": extracted,
            "generation_error": None,
        }

    # ------------------------------------------------------------------
    # Parse attempt 3: total failure → signal handle_llm_error
    # ------------------------------------------------------------------
    return {
        "raw_llm_output": raw_text,
        "structured_report": None,
        "generation_error": (
            "Failed to parse LLM response as JSON after 2 attempts. "
            f"Response preview: {raw_text[:200]!r}"
        ),
    }


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------

def _try_json_loads(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text.strip())
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _extract_json_block(text: str) -> dict[str, Any] | None:
    """Extract the first ```json ... ``` or ``` ... ``` fenced block."""
    # Try ```json first, then plain ```
    patterns = [
        r"```json\s*([\s\S]+?)\s*```",
        r"```\s*([\s\S]+?)\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            result = _try_json_loads(candidate)
            if result is not None:
                return result
    return None
