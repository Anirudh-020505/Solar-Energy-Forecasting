"""
Node 2 — retrieve_guidelines

Builds two targeted queries from state (variability + storage context),
runs both against the FAISSStore, merges and deduplicates results,
and writes retrieved_chunks + source_references into state.

The FAISSStore is built once at import time via @st.cache_resource
(or on first call outside Streamlit). Using a module-level singleton
avoids rebuilding the index on every graph invocation.
"""

from __future__ import annotations

from typing import Any

from rag.knowledge_base import load_and_chunk_documents
from rag.faiss_store import FAISSStore

# ---------------------------------------------------------------------------
# Module-level singleton — built once per process
# ---------------------------------------------------------------------------
_store: FAISSStore | None = None


def _get_store() -> FAISSStore:
    global _store
    if _store is None or not _store._built:
        chunks = load_and_chunk_documents()
        _store = FAISSStore()
        _store.build(chunks)
    return _store


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def retrieve_guidelines(state: dict[str, Any]) -> dict[str, Any]:
    variability_score: float = state.get("variability_score", 0.0)
    risk_periods: list[dict] = state.get("risk_periods", [])
    scenario_name: str = state.get("scenario_name", "baseline")

    store = _get_store()

    # ------------------------------------------------------------------ #
    # Build query strings from state context
    # ------------------------------------------------------------------ #

    # Query 1: variability + risk-specific terms
    risk_levels = [r["risk_level"] for r in risk_periods]
    has_high = "High" in risk_levels
    var_label = "high" if variability_score > 0.66 else ("moderate" if variability_score > 0.33 else "low")
    q1 = (
        f"solar variability {var_label} ramp rate frequency regulation "
        f"{'curtailment spinning reserve' if has_high else 'voltage control'} "
        f"grid stability {scenario_name}"
    )

    # Query 2: storage + demand-side terms
    q2 = (
        "battery storage dispatch state of charge round trip efficiency "
        "demand response load shifting time of use tariff export revenue "
        "depth of discharge IEC 62619"
    )

    results_q1 = store.query(q1, k=4)
    results_q2 = store.query(q2, k=4)

    # ------------------------------------------------------------------ #
    # Merge and deduplicate by chunk_id (preserve insertion order)
    # ------------------------------------------------------------------ #
    seen_ids: set[str] = set()
    merged: list[dict[str, Any]] = []
    for chunk in results_q1 + results_q2:
        cid = chunk["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(chunk)

    # Sort merged list by score descending so the best chunks come first
    merged.sort(key=lambda c: c.get("score", 0.0), reverse=True)

    # Cap at 6 total chunks (fits comfortably in an 8k context window)
    merged = merged[:6]

    source_references = sorted({c["source"] for c in merged})

    return {
        "retrieved_chunks": merged,
        "source_references": source_references,
    }
