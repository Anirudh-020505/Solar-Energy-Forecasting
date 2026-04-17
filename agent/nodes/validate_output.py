"""
Node 6 — validate_output

Validates structured_report (a raw dict from either the LLM or mock_llm)
against the GridOptimizationReport Pydantic model.

Checks performed:
  1. Pydantic model_validate() — type coercion + field constraints
  2. All 4 required sections are non-empty (redundant with Pydantic but explicit)
  3. Numeric consistency: total_energy_kwh within ±5% of forecast_stats["total"]

Sets:
  is_valid            bool
  validation_warnings list[str]   — non-fatal issues (UI can show a banner)
  final_report        GridOptimizationReport | None
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from schemas.report_schema import GridOptimizationReport


def validate_output(state: dict[str, Any]) -> dict[str, Any]:
    raw: dict[str, Any] | None = state.get("structured_report")
    stats: dict[str, Any] = state.get("forecast_stats", {})
    scenario_name: str = state.get("scenario_name", "baseline")
    warnings: list[str] = []

    if raw is None:
        return {
            "is_valid": False,
            "validation_warnings": ["structured_report is None — no report to validate."],
            "final_report": None,
        }

    # ------------------------------------------------------------------
    # 1. Pydantic validation
    # ------------------------------------------------------------------
    try:
        # Inject scenario_name if the LLM omitted it
        if "scenario_name" not in raw or not raw.get("scenario_name"):
            raw = {**raw, "scenario_name": scenario_name}

        report = GridOptimizationReport.model_validate(raw)

    except ValidationError as exc:
        # Collect all field errors into warnings and return invalid
        error_msgs = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
        return {
            "is_valid": False,
            "validation_warnings": error_msgs,
            "final_report": None,
        }

    # ------------------------------------------------------------------
    # 2. Required sections non-empty (belt-and-suspenders)
    # ------------------------------------------------------------------
    if not report.grid_balancing.recommendations:
        warnings.append("grid_balancing.recommendations is empty.")
    if not report.energy_utilization.strategies:
        warnings.append("energy_utilization.strategies is empty.")
    if not report.variability_analysis.risk_periods and state.get("risk_periods"):
        warnings.append(
            "variability_analysis.risk_periods is empty but compute detected "
            f"{len(state['risk_periods'])} risk period(s)."
        )

    # ------------------------------------------------------------------
    # 3. Numeric consistency: total_energy_kwh within ±5% of computed total
    # ------------------------------------------------------------------
    computed_total = stats.get("total", 0.0)
    reported_total = report.forecast_summary.total_energy_kwh
    if computed_total > 0:
        deviation = abs(reported_total - computed_total) / computed_total
        if deviation > 0.05:
            warnings.append(
                f"total_energy_kwh deviation {deviation:.1%} "
                f"(reported={reported_total:.1f}, computed={computed_total:.1f}). "
                "Overriding with computed value."
            )
            # Correct the value in-place via a re-validated copy
            summary_dict = report.forecast_summary.model_dump()
            summary_dict["total_energy_kwh"] = round(computed_total, 2)
            from schemas.report_schema import ForecastSummary
            try:
                corrected_summary = ForecastSummary.model_validate(summary_dict)
                report = report.model_copy(update={"forecast_summary": corrected_summary})
            except ValidationError:
                pass  # leave original if correction itself fails

    # ------------------------------------------------------------------
    # Merge source references from recommendations into top-level list
    # ------------------------------------------------------------------
    all_refs = report.all_section_references()
    if all_refs != sorted(report.supporting_references):
        report = report.model_copy(update={"supporting_references": all_refs})

    return {
        "is_valid": True,
        "validation_warnings": warnings,
        "final_report": report,
    }
