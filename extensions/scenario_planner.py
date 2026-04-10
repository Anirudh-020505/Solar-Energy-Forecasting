"""
Step 10 — Scenario Planner (extensions/scenario_planner.py)

Four named scenarios that modify the base forecast before running the
full LangGraph agent pipeline.

Public API
----------
SCENARIOS       dict[str, dict]  — scenario metadata + params
run_scenario()  → full agent state dict
compare_scenarios() → pd.DataFrame for charting
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from extensions.battery_optimizer import BatteryConfig

# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict[str, Any]] = {
    "baseline": {
        "label": "Baseline",
        "description": "Nominal solar production, standard demand and battery.",
        "color": "#4ECDC4",
        "params": {
            "scale_factor": 1.0,
            "cloud_ramp_noise": False,
            "demand_multiplier": 1.0,
            "battery_capacity_multiplier": 1.0,
        },
    },
    "cloudy_day": {
        "label": "Cloudy Day",
        "description": "45% production, random 20-40% midday drops (hours 10-14).",
        "color": "#95A5A6",
        "params": {
            "scale_factor": 0.45,
            "cloud_ramp_noise": True,
            "demand_multiplier": 1.0,
            "battery_capacity_multiplier": 1.0,
        },
    },
    "high_demand": {
        "label": "High Demand",
        "description": "Standard solar, 35% higher electricity demand.",
        "color": "#E74C3C",
        "params": {
            "scale_factor": 1.0,
            "cloud_ramp_noise": False,
            "demand_multiplier": 1.35,
            "battery_capacity_multiplier": 1.0,
        },
    },
    "battery_upgrade": {
        "label": "Battery Upgrade",
        "description": "Standard solar, 50% larger battery capacity.",
        "color": "#3498DB",
        "params": {
            "scale_factor": 1.0,
            "cloud_ramp_noise": False,
            "demand_multiplier": 1.0,
            "battery_capacity_multiplier": 1.5,
        },
    },
}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_scenario(
    key: str,
    base_predictions: list[float],
    base_battery_config: BatteryConfig | dict | None = None,
    graph: Any | None = None,
    time_value: int = 24,
    time_unit: str = "hours",
) -> dict[str, Any]:
    """
    Run a single named scenario through the full LangGraph agent pipeline.

    Parameters
    ----------
    key                 Scenario key from SCENARIOS (e.g. "cloudy_day").
    base_predictions    Raw LSTM predictions (list of floats, kW).
    base_battery_config BatteryConfig or dict or None (defaults are used if None).
    graph               Compiled LangGraph graph. If None, build_graph() is called.
    time_value / time_unit  Forecast horizon metadata (for state).

    Returns
    -------
    Full agent state dict after graph.invoke() completes, with an extra
    "scenario_key" field injected for downstream use.
    """
    if graph is None:
        from agent.graph import build_graph
        graph = build_graph()

    scenario_meta = SCENARIOS.get(key, SCENARIOS["baseline"])
    params = scenario_meta["params"]

    # Resolve battery config fields
    if isinstance(base_battery_config, BatteryConfig):
        cfg = base_battery_config
    elif isinstance(base_battery_config, dict):
        cfg = BatteryConfig(**{
            k: v for k, v in base_battery_config.items()
            if k in BatteryConfig.__dataclass_fields__
        })
    else:
        cfg = BatteryConfig()

    initial_state: dict[str, Any] = {
        # Forecast inputs
        "predictions": base_predictions,
        "time_value": time_value,
        "time_unit": time_unit,
        # Scenario metadata
        "scenario_name": key,
        "scenario_params": params,
        # Battery config (analyze_forecast may scale these)
        "battery_capacity_kwh": cfg.capacity_kwh,
        "battery_max_charge_kw": cfg.max_charge_kw,
        "battery_max_discharge_kw": cfg.max_discharge_kw,
        "battery_initial_soc_kwh": cfg.initial_soc_kwh,
        "battery_min_soc_kwh": cfg.min_soc_kwh,
        "battery_demand_baseline_kw": cfg.demand_baseline_kw,
    }

    result: dict[str, Any] = graph.invoke(initial_state)
    result["scenario_key"] = key
    return result


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------

def compare_scenarios(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from multiple scenario results.

    Parameters
    ----------
    results   dict mapping scenario key → agent state dict (from run_scenario).

    Returns
    -------
    pd.DataFrame with one row per scenario and these columns:
        scenario_key, label, total_energy_kwh, peak_power_kw, avg_power_kw,
        variability_score, daylight_hours, self_sufficiency_pct,
        total_charged_kwh, total_discharged_kwh, total_exported_kwh,
        total_imported_kwh, peak_shaving_kw, cycles_used,
        is_valid, confidence_level
    """
    rows = []
    for key, state in results.items():
        meta = SCENARIOS.get(key, {"label": key, "color": "#888"})
        stats: dict = state.get("forecast_stats", {})
        batt: dict = state.get("battery_summary", {})
        report = state.get("final_report")

        row: dict[str, Any] = {
            "scenario_key": key,
            "label": meta["label"],
            "color": meta.get("color", "#888"),
            # Forecast metrics
            "total_energy_kwh": state.get("total_energy_kwh", stats.get("total", 0.0)),
            "peak_power_kw": stats.get("peak", 0.0),
            "avg_power_kw": stats.get("mean", 0.0),
            "variability_score": state.get("variability_score", 0.0),
            "daylight_hours": stats.get("daylight_hours", 0),
            # Battery metrics
            "self_sufficiency_pct": batt.get("self_sufficiency_pct", 0.0),
            "total_charged_kwh": batt.get("total_charged_kwh", 0.0),
            "total_discharged_kwh": batt.get("total_discharged_kwh", 0.0),
            "total_exported_kwh": batt.get("total_exported_kwh", 0.0),
            "total_imported_kwh": batt.get("total_imported_kwh", 0.0),
            "peak_shaving_kw": batt.get("peak_shaving_kw", 0.0),
            "cycles_used": batt.get("cycles_used", 0.0),
            # Report quality
            "is_valid": state.get("is_valid", False),
            "confidence_level": (
                report.confidence_level if report is not None else "N/A"
            ),
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Preserve scenario order as defined in SCENARIOS
    order = {k: i for i, k in enumerate(SCENARIOS)}
    df["_order"] = df["scenario_key"].map(lambda k: order.get(k, 99))
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)
    return df
