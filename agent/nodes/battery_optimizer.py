"""
Node 3 — battery_optimizer

Thin wrapper around extensions/battery_optimizer.run_battery_optimizer().
Reads battery config fields from state, delegates to the greedy algorithm,
and writes battery_schedule + battery_summary back into state.
"""

from __future__ import annotations

from typing import Any

from extensions.battery_optimizer import BatteryConfig, run_battery_optimizer


def battery_optimizer(state: dict[str, Any]) -> dict[str, Any]:
    predictions = state.get("effective_predictions") or state.get("predictions", [])

    cfg = BatteryConfig(
        capacity_kwh=float(state.get("battery_capacity_kwh", 200.0)),
        max_charge_kw=float(state.get("battery_max_charge_kw", 50.0)),
        max_discharge_kw=float(state.get("battery_max_discharge_kw", 50.0)),
        initial_soc_kwh=float(state.get("battery_initial_soc_kwh", 100.0)),
        min_soc_kwh=float(state.get("battery_min_soc_kwh", 20.0)),
        demand_baseline_kw=float(state.get("battery_demand_baseline_kw", 80.0)),
        round_trip_efficiency=0.92,
    )

    result = run_battery_optimizer(predictions, cfg)

    return {
        "battery_schedule": result["schedule"],
        "battery_summary": result["summary"],
    }
