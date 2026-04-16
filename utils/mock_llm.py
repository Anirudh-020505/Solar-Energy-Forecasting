"""
Demo Mode fallback report generator.

generate_mock_report(state) is called by:
  - agent/nodes/generate_report.py  when GROQ_API_KEY is absent
  - agent/nodes/handle_llm_error.py when the LLM call fails at runtime

The report uses real computed stats from state so numbers are accurate,
but all narrative text is canned. Every narrative field is prefixed with
"[Demo Mode]" so users always know this is not a live LLM response.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from schemas.report_schema import (
    EnergyUtilizationSection,
    ForecastSummary,
    GridBalancingSection,
    GridOptimizationReport,
    GridRecommendation,
    RiskPeriod,
    UtilizationStrategy,
    VariabilityAnalysis,
)
from utils.forecast_stats import variability_category

if TYPE_CHECKING:
    pass  # avoid circular imports; state is passed as plain dict


def generate_mock_report(state: dict[str, Any]) -> GridOptimizationReport:
    """
    Build a GridOptimizationReport from pre-computed state values.

    Uses real stats (total_energy_kwh, peak_power_kw, variability_score,
    risk_periods, battery_summary) so metrics tables are accurate.
    Narratives are canned but clearly labelled "[Demo Mode]".

    Parameters
    ----------
    state : GridOptimizerState dict; expects the following keys to be set
            by earlier nodes (analyze_forecast, battery_optimizer):
                forecast_stats, variability_score, risk_periods,
                total_energy_kwh, daylight_fraction, battery_summary,
                scenario_name

    Returns
    -------
    GridOptimizationReport — fully valid Pydantic model ready for the UI.
    """
    stats: dict[str, Any] = state.get("forecast_stats", {})
    variability_score: float = state.get("variability_score", 0.0)
    risk_periods_raw: list[dict] = state.get("risk_periods", [])
    battery_summary: dict[str, Any] = state.get("battery_summary", {})
    scenario_name: str = state.get("scenario_name", "baseline")

    total_energy = stats.get("total", 0.0)
    peak_power = stats.get("peak", 0.0)
    avg_power = stats.get("mean", 0.0)
    daylight_hours = stats.get("daylight_hours", 0)
    ramp_concern = any(
        r.get("risk_level") in ("Medium", "High") for r in risk_periods_raw
    )
    var_cat = variability_category(variability_score)

    # -- Build RiskPeriod models (cap at 5 to keep report concise) --
    risk_period_models = [
        RiskPeriod(
            hour_index=r["hour_index"],
            risk_level=r["risk_level"],
            reason=r["reason"],
            recommended_action=r["recommended_action"],
        )
        for r in risk_periods_raw[:5]
    ]

    # -- Forecast Summary --
    forecast_summary = ForecastSummary(
        total_energy_kwh=round(total_energy, 2),
        peak_power_kw=round(peak_power, 2),
        avg_power_kw=round(avg_power, 2),
        daylight_hours=min(daylight_hours, 24),
        variability_category=var_cat,
        narrative=(
            f"[Demo Mode] The {scenario_name} forecast predicts {total_energy:.0f} kWh "
            f"of total generation with a peak of {peak_power:.0f} kW over "
            f"{daylight_hours} daylight hours. "
            f"Overall variability is classified as {var_cat.lower()}, "
            f"suggesting {'stable conditions suitable for standard dispatch.' if var_cat == 'Low' else 'active grid management will be required.'}"
        ),
    )

    # -- Variability Analysis --
    variability_analysis = VariabilityAnalysis(
        variability_score=round(variability_score, 4),
        risk_periods=risk_period_models,
        ramp_rate_concern=ramp_concern,
        narrative=(
            f"[Demo Mode] Variability score is {variability_score:.2f} ({var_cat}). "
            f"{len(risk_periods_raw)} risk period(s) were detected in the forecast horizon. "
            f"{'Ramp rate concerns are present; spinning reserve activation is advised.' if ramp_concern else 'No significant ramp rate concerns detected.'} "
            f"Battery storage can mitigate the majority of identified risk events."
        ),
    )

    # -- Grid Balancing --
    export_kwh = battery_summary.get("total_exported_kwh", 0.0)
    import_kwh = battery_summary.get("total_imported_kwh", 0.0)
    peak_shaving = battery_summary.get("peak_shaving_kw", 0.0)

    high_risk_hours = [r["hour_index"] for r in risk_periods_raw if r["risk_level"] == "High"]
    export_window = (
        f"{min(high_risk_hours):02d}:00–{max(high_risk_hours) + 1:02d}:00"
        if high_risk_hours
        else "10:00–14:00"
    )

    grid_balancing = GridBalancingSection(
        recommendations=[
            GridRecommendation(
                category="Storage Dispatch",
                priority="High",
                action=(
                    "Pre-charge battery to 80% SoC before the first detected risk period. "
                    "Discharge at maximum rated power during deficit hours."
                ),
                expected_benefit=f"Reduces grid import by up to {peak_shaving:.0f} kW per hour during deficit periods.",
                references=["battery_standards.txt", "grid_management.txt"],
            ),
            GridRecommendation(
                category="Ramp Management",
                priority="High" if ramp_concern else "Medium",
                action=(
                    "Activate fast-response storage discharge within 10 seconds of any "
                    "ramp-down exceeding 30 kW/hr to maintain frequency within ±0.5 Hz."
                ),
                expected_benefit="Prevents under-frequency load shedding events and avoids AGC saturation.",
                references=["grid_management.txt"],
            ),
            GridRecommendation(
                category="Export Management",
                priority="Medium",
                action=(
                    f"Schedule grid export between {export_window} when surplus is highest. "
                    "Coordinate with DSO to avoid voltage rise above 1.05 pu."
                ),
                expected_benefit=f"Exports {export_kwh:.0f} kWh to grid, reducing curtailment losses.",
                references=["renewable_integration.txt"],
            ),
            GridRecommendation(
                category="Demand Response",
                priority="Low",
                action=(
                    "Notify demand response aggregators 1 hour before forecast deficit periods "
                    "to pre-emptively shed 10–15% of flexible load."
                ),
                expected_benefit=f"Reduces grid import by up to {import_kwh * 0.15:.0f} kWh over the forecast horizon.",
                references=["demand_response.txt"],
            ),
        ],
        storage_dispatch_summary=(
            f"[Demo Mode] Battery charged {battery_summary.get('total_charged_kwh', 0):.1f} kWh "
            f"and discharged {battery_summary.get('total_discharged_kwh', 0):.1f} kWh. "
            f"Peak shaving of {peak_shaving:.1f} kW achieved. "
            f"{battery_summary.get('cycles_used', 0):.2f} equivalent full cycles consumed."
        ),
        export_window=export_window,
        narrative=(
            f"[Demo Mode] Grid balancing strategy prioritises storage dispatch and ramp management. "
            f"Net grid export of {max(export_kwh - import_kwh, 0):.0f} kWh is forecast. "
            f"Demand response provides a secondary buffer for deficit periods."
        ),
    )

    # -- Energy Utilization --
    peak_shaving_potential = min(peak_shaving * 1.1, peak_power * 0.2)
    self_consumption = battery_summary.get("self_sufficiency_pct", 70.0)

    # Find the top daylight hours for applicable_hours
    daylight_hour_list = list(range(6, 6 + min(daylight_hours, 12)))

    energy_utilization = EnergyUtilizationSection(
        strategies=[
            UtilizationStrategy(
                strategy_name="Peak Load Shifting",
                description=(
                    "Reschedule deferrable loads (HVAC pre-cooling, EV charging, industrial processes) "
                    "to coincide with peak solar generation hours (typically 09:00–14:00). "
                    "This maximises self-consumption and reduces TOU tariff exposure."
                ),
                estimated_savings_pct=12.5,
                applicable_hours=daylight_hour_list[:6],
            ),
            UtilizationStrategy(
                strategy_name="Battery Arbitrage",
                description=(
                    "Charge battery during surplus hours at near-zero marginal cost "
                    "and discharge during evening peak demand when grid tariffs are highest. "
                    "Target SoC of 90% by end of peak generation window."
                ),
                estimated_savings_pct=8.0,
                applicable_hours=daylight_hour_list,
            ),
            UtilizationStrategy(
                strategy_name="Export Revenue Maximisation",
                description=(
                    "Delay export until Time-of-Use export premium windows (typically 16:00–20:00). "
                    "Use battery to store surplus from 11:00–14:00 and export at the premium rate."
                ),
                estimated_savings_pct=5.5,
                applicable_hours=list(range(16, 20)),
            ),
        ],
        peak_shaving_potential_kw=round(peak_shaving_potential, 2),
        self_consumption_rate_pct=round(self_consumption, 1),
    )

    return GridOptimizationReport(
        scenario_name=scenario_name,
        generated_at=datetime.utcnow(),
        forecast_summary=forecast_summary,
        variability_analysis=variability_analysis,
        grid_balancing=grid_balancing,
        energy_utilization=energy_utilization,
        supporting_references=[
            "grid_management.txt",
            "renewable_integration.txt",
            "battery_standards.txt",
            "demand_response.txt",
        ],
        confidence_level="Low",  # always Low for demo mode
        disclaimer=(
            "[Demo Mode — No GROQ_API_KEY detected or LLM call failed] "
            "This report was generated using pre-programmed rules, not a live AI model. "
            "Numerical values are computed from real forecast data; narratives are illustrative only. "
            "Set GROQ_API_KEY in your .env file to enable full AI-generated analysis."
        ),
    )
