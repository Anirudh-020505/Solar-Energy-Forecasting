"""
Node 1 — analyze_forecast

Responsibilities:
  1. Apply scenario_params transforms to raw predictions.
  2. Compute stats, risk periods, and variability score via forecast_stats helpers.
  3. Write results back into state for downstream nodes.

Scenario transforms (applied in order):
  scale_factor          float  — multiply every prediction (e.g. 0.45 for cloudy_day)
  cloud_ramp_noise      bool   — add random 20–40% drops on hours 10–14 (midday shadow)
  demand_multiplier     float  — written to battery_demand_baseline_kw for battery node
"""

from __future__ import annotations

import random
from typing import Any

from utils.forecast_stats import (
    compute_stats,
    compute_variability_score,
    find_risk_periods,
)


def analyze_forecast(state: dict[str, Any]) -> dict[str, Any]:
    predictions: list[float] = state.get("predictions", [])
    params: dict[str, Any] = state.get("scenario_params", {})

    # ------------------------------------------------------------------ #
    # 1. Apply scenario transforms
    # ------------------------------------------------------------------ #
    scale = float(params.get("scale_factor", 1.0))
    effective = [max(0.0, p * scale) for p in predictions]

    if params.get("cloud_ramp_noise", False):
        # Simulate intermittent cloud shadow on midday hours 10–14
        rng = random.Random(42)  # fixed seed for reproducibility within a session
        for i in range(min(10, len(effective)), min(15, len(effective))):
            drop = rng.uniform(0.20, 0.40)
            effective[i] = max(0.0, effective[i] * (1.0 - drop))

    # Demand multiplier — write updated baseline so battery node picks it up
    demand_multiplier = float(params.get("demand_multiplier", 1.0))
    current_baseline = float(state.get("battery_demand_baseline_kw", 80.0))
    new_baseline = current_baseline * demand_multiplier

    # Battery capacity multiplier
    cap_multiplier = float(params.get("battery_capacity_multiplier", 1.0))
    current_cap = float(state.get("battery_capacity_kwh", 200.0))
    new_cap = current_cap * cap_multiplier

    # ------------------------------------------------------------------ #
    # 2. Compute statistics
    # ------------------------------------------------------------------ #
    stats = compute_stats(effective)
    risk_periods = find_risk_periods(effective)
    variability_score = compute_variability_score(stats, risk_periods)

    return {
        "effective_predictions": effective,
        "forecast_stats": stats,
        "risk_periods": risk_periods,
        "variability_score": variability_score,
        "peak_generation_hour": stats["peak_hour"],
        "total_energy_kwh": stats["total"],
        "daylight_fraction": stats["daylight_fraction"],
        # Propagate (possibly scaled) battery params for downstream nodes
        "battery_demand_baseline_kw": new_baseline,
        "battery_capacity_kwh": new_cap,
        "battery_max_discharge_kw": state.get("battery_max_discharge_kw", 50.0),
        "battery_max_charge_kw": state.get("battery_max_charge_kw", 50.0),
        "battery_initial_soc_kwh": min(
            state.get("battery_initial_soc_kwh", new_cap * 0.5), new_cap
        ),
        "battery_min_soc_kwh": new_cap * 0.10,  # 10% DoD floor
    }
