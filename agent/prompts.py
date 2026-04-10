"""
All LLM prompt templates for the Grid Optimizer agent.

SYSTEM_PROMPT  — establishes role, embeds JSON schema, enforces output format
build_human_prompt(state) — assembles per-invocation context from state values
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# JSON schema embedded verbatim so the LLM knows exact field names and types
# ---------------------------------------------------------------------------
_JSON_SCHEMA = """
{
  "forecast_summary": {
    "total_energy_kwh": float,
    "peak_power_kw": float,
    "avg_power_kw": float,
    "daylight_hours": int,
    "variability_category": "Low" | "Moderate" | "High",
    "narrative": "string (1-3 sentences)"
  },
  "variability_analysis": {
    "variability_score": float (0.0-1.0),
    "risk_periods": [
      {
        "hour_index": int,
        "risk_level": "Low" | "Medium" | "High",
        "reason": "string",
        "recommended_action": "string"
      }
    ],
    "ramp_rate_concern": bool,
    "narrative": "string (2-4 sentences)"
  },
  "grid_balancing": {
    "recommendations": [
      {
        "category": "Frequency Regulation" | "Voltage Control" | "Ramp Management" | "Curtailment" | "Storage Dispatch" | "Demand Response" | "Export Management",
        "priority": "Critical" | "High" | "Medium" | "Low",
        "action": "string",
        "expected_benefit": "string",
        "references": ["source_filename.txt"]
      }
    ],
    "storage_dispatch_summary": "string",
    "export_window": "HH:MM-HH:MM",
    "narrative": "string (2-3 sentences)"
  },
  "energy_utilization": {
    "strategies": [
      {
        "strategy_name": "string",
        "description": "string",
        "estimated_savings_pct": float (0-100),
        "applicable_hours": [int]
      }
    ],
    "peak_shaving_potential_kw": float,
    "self_consumption_rate_pct": float (0-100)
  },
  "supporting_references": ["source_filename.txt"],
  "confidence_level": "High" | "Medium" | "Low"
}
"""

SYSTEM_PROMPT = f"""You are a senior grid optimization analyst specialising in solar energy integration and battery storage dispatch.

Your task is to analyse the provided solar forecast data, risk periods, battery schedule, and grid management guidelines, then produce a structured optimization report.

CRITICAL RULES:
1. Return ONLY valid JSON wrapped in triple backticks (```json ... ```). No prose before or after.
2. Every recommendation MUST cite at least one reference from the provided knowledge base chunks.
3. Do NOT make operational claims without a supporting source citation.
4. All numerical values must be consistent with the input statistics provided (±5% tolerance for rounding).
5. risk_periods in your output must be a subset of the detected risk periods provided — do not invent new ones.
6. confidence_level should reflect the quality of RAG support: High = strong matches, Low = sparse or weak matches.

OUTPUT JSON SCHEMA:
{_JSON_SCHEMA}

Return your response as:
```json
{{ ... }}
```
"""


def build_human_prompt(state: dict[str, Any]) -> str:
    """
    Assemble the per-invocation human message from state values.

    Includes: forecast stats table, risk periods list, battery summary,
    RAG chunks (labeled with source), and scenario context.
    """
    stats: dict[str, Any] = state.get("forecast_stats", {})
    risk_periods: list[dict] = state.get("risk_periods", [])
    battery_summary: dict[str, Any] = state.get("battery_summary", {})
    chunks: list[dict[str, Any]] = state.get("retrieved_chunks", [])
    scenario_name: str = state.get("scenario_name", "baseline")
    variability_score: float = state.get("variability_score", 0.0)
    total_energy: float = state.get("total_energy_kwh", 0.0)
    daylight_fraction: float = state.get("daylight_fraction", 0.0)

    # -- Forecast stats table --
    stats_block = f"""
FORECAST STATISTICS (Scenario: {scenario_name})
================================================
Total Energy (kWh):      {total_energy:.2f}
Peak Power (kW):         {stats.get('peak', 0):.2f}
Average Power (kW):      {stats.get('mean', 0):.2f}
Std Dev (kW):            {stats.get('std', 0):.2f}
Coefficient of Variation:{stats.get('cv', 0):.4f}
Daylight Hours:          {stats.get('daylight_hours', 0)}
Daylight Fraction:       {daylight_fraction:.2%}
Peak Generation Hour:    Hour {stats.get('peak_hour', 0):02d}:00
Variability Score:       {variability_score:.4f} (0=stable, 1=highly variable)
"""

    # -- Risk periods --
    if risk_periods:
        risk_lines = "\n".join(
            f"  Hour {r['hour_index']:02d} | {r['risk_level']:6s} | {r['reason'][:80]}"
            for r in risk_periods[:10]  # cap at 10 to fit context
        )
        risk_block = f"\nDETECTED RISK PERIODS ({len(risk_periods)} total, showing top 10)\n{'='*60}\n{risk_lines}\n"
    else:
        risk_block = "\nDETECTED RISK PERIODS\n" + "="*60 + "\nNone — forecast is stable.\n"

    # -- Battery summary --
    battery_block = f"""
BATTERY DISPATCH SUMMARY
========================
Total Charged (kWh):     {battery_summary.get('total_charged_kwh', 0):.2f}
Total Discharged (kWh):  {battery_summary.get('total_discharged_kwh', 0):.2f}
Total Exported (kWh):    {battery_summary.get('total_exported_kwh', 0):.2f}
Total Imported (kWh):    {battery_summary.get('total_imported_kwh', 0):.2f}
Peak Shaving (kW):       {battery_summary.get('peak_shaving_kw', 0):.2f}
Self-Sufficiency (%):    {battery_summary.get('self_sufficiency_pct', 0):.1f}
Cycles Used:             {battery_summary.get('cycles_used', 0):.3f}
Min SoC (kWh):           {battery_summary.get('min_soc_kwh', 0):.2f}
Max SoC (kWh):           {battery_summary.get('max_soc_kwh', 0):.2f}
"""

    # -- RAG knowledge base chunks --
    if chunks:
        chunk_lines = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("source", "unknown")
            score = chunk.get("score", 0.0)
            text = chunk.get("text", "")[:400]  # truncate long chunks
            chunk_lines.append(
                f"\n[SOURCE {i}: {source} | relevance={score:.3f}]\n{text}"
            )
        rag_block = "\nKNOWLEDGE BASE GUIDELINES\n" + "="*60 + "\n".join(chunk_lines)
    else:
        rag_block = "\nKNOWLEDGE BASE GUIDELINES\n" + "="*60 + "\nNo guidelines retrieved.\n"

    instructions = """
TASK
====
Using all data above, produce the structured JSON optimization report.
- grid_balancing must contain 3-5 recommendations ordered by priority.
- energy_utilization must contain 2-4 strategies.
- Every recommendation's "references" list must name the source .txt file from the knowledge base.
- variability_analysis.risk_periods must use ONLY hour indices from the detected risk periods above.
- Keep all narratives factual and grounded in the numbers provided.
"""

    return stats_block + risk_block + battery_block + rag_block + instructions
