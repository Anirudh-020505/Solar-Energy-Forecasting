"""
Greedy rule-based battery scheduling algorithm.

No LP solver or scipy required. The algorithm makes a single forward pass
over the hourly predictions and decides charge / discharge / idle per step
based on the net surplus/deficit against a flat demand baseline.

Called by agent/nodes/battery_optimizer.py, which writes the result into
GridOptimizerState. Also importable directly for standalone use or testing.

Public API
----------
run_battery_optimizer(predictions, config) -> dict
    Returns {"schedule": [...], "summary": {...}}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class BatteryConfig:
    """
    Physical and operational parameters for the battery system.

    All power values in kW, energy in kWh.
    """
    capacity_kwh: float = 200.0          # Usable energy capacity
    max_charge_kw: float = 50.0          # Maximum charge power
    max_discharge_kw: float = 50.0       # Maximum discharge power
    initial_soc_kwh: float = 100.0       # Starting state of charge
    min_soc_kwh: float = 20.0            # Lower SoC bound (DoD protection ~90%)
    max_soc_kwh: float | None = None     # Upper SoC bound; defaults to capacity_kwh
    round_trip_efficiency: float = 0.92  # IEC 62619 typical LFP round-trip efficiency
    demand_baseline_kw: float = 80.0     # Flat demand assumption when no demand curve

    def __post_init__(self) -> None:
        if self.max_soc_kwh is None:
            self.max_soc_kwh = self.capacity_kwh
        self._validate()

    def _validate(self) -> None:
        if not (0.0 < self.round_trip_efficiency <= 1.0):
            raise ValueError("round_trip_efficiency must be in (0, 1].")
        if self.min_soc_kwh < 0:
            raise ValueError("min_soc_kwh must be >= 0.")
        if self.max_soc_kwh > self.capacity_kwh:
            raise ValueError("max_soc_kwh cannot exceed capacity_kwh.")
        if self.initial_soc_kwh < self.min_soc_kwh or self.initial_soc_kwh > self.max_soc_kwh:
            raise ValueError(
                f"initial_soc_kwh ({self.initial_soc_kwh}) must be within "
                f"[{self.min_soc_kwh}, {self.max_soc_kwh}]."
            )


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def run_battery_optimizer(
    predictions: list[float],
    config: BatteryConfig | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Run a single-pass greedy battery scheduler over hourly solar predictions.

    Algorithm (per hour t)
    ----------------------
    net = solar[t] - demand_baseline_kw

    if net > 0  (solar surplus):
        charge = min(net, max_charge_kw, headroom / charge_efficiency)
        soc   += charge * charge_efficiency   # energy actually stored
        export = net - charge                 # residual sent to grid

    elif net < 0  (solar deficit):
        discharge = min(|net|, max_discharge_kw, soc - min_soc)
        soc       -= discharge                # energy drawn from battery
        grid_import = |net| - discharge       # residual pulled from grid

    else: idle

    Parameters
    ----------
    predictions : list of hourly solar power values (kW); negatives clamped to 0
    config      : BatteryConfig instance, or a dict of BatteryConfig fields,
                  or None (uses defaults)

    Returns
    -------
    dict with two keys:

    "schedule" — list of per-hour dicts:
        hour            int    0-based hour index
        solar_kw        float  clamped solar prediction
        demand_kw       float  demand baseline used
        net_kw          float  solar - demand
        action          str    "charge" | "discharge" | "idle"
        charge_kw       float  power flowing into battery (kW)
        discharge_kw    float  power flowing out of battery (kW)
        soc_kwh         float  SoC at end of this hour
        soc_pct         float  SoC as % of capacity
        grid_export_kw  float  power exported to grid (kW)
        grid_import_kw  float  power imported from grid (kW)

    "summary" — aggregate metrics dict:
        total_charged_kwh    float
        total_discharged_kwh float
        total_exported_kwh   float
        total_imported_kwh   float
        peak_shaving_kw      float  max demand reduction achieved in a single hour
        self_sufficiency_pct float  % of demand met by solar+battery (not grid)
        cycles_used          float  equivalent full cycles (charged / capacity)
        min_soc_kwh          float  lowest SoC reached
        max_soc_kwh          float  highest SoC reached
    """
    # Resolve config
    if config is None:
        cfg = BatteryConfig()
    elif isinstance(config, dict):
        cfg = BatteryConfig(**config)
    else:
        cfg = config

    # Charge efficiency: energy_stored = charge_power * sqrt(RTE)
    # Discharge efficiency: energy_delivered = discharge_power * sqrt(RTE)
    # Using one-way efficiency = sqrt(round_trip) for each direction.
    import math
    one_way_eff = math.sqrt(cfg.round_trip_efficiency)

    soc = cfg.initial_soc_kwh
    schedule: list[dict[str, Any]] = []

    # Accumulators for summary
    total_charged = 0.0
    total_discharged = 0.0
    total_exported = 0.0
    total_imported = 0.0
    total_demand = 0.0
    total_demand_met_locally = 0.0
    peak_shaving = 0.0
    soc_min = soc
    soc_max = soc

    for t, raw_solar in enumerate(predictions):
        solar = max(0.0, float(raw_solar))  # clamp inverse-scaler negatives
        demand = cfg.demand_baseline_kw
        net = solar - demand

        charge_kw = 0.0
        discharge_kw = 0.0
        grid_export_kw = 0.0
        grid_import_kw = 0.0
        action = "idle"

        if net > 0:
            # --- Surplus: charge battery, export remainder ---
            headroom = cfg.max_soc_kwh - soc          # energy space available
            # Max charge limited by: power cap, headroom (accounting for efficiency)
            max_chargeable_kw = min(
                net,
                cfg.max_charge_kw,
                headroom / one_way_eff if one_way_eff > 0 else 0.0,
            )
            charge_kw = max(0.0, max_chargeable_kw)
            soc += charge_kw * one_way_eff            # energy stored
            grid_export_kw = net - charge_kw          # remainder to grid

            if charge_kw > 0:
                action = "charge"
            elif grid_export_kw > 0:
                action = "idle"  # no room to charge, just exporting

        elif net < 0:
            # --- Deficit: discharge battery, import remainder ---
            deficit = abs(net)
            available = soc - cfg.min_soc_kwh         # energy available to discharge
            max_dischargeable_kw = min(
                deficit,
                cfg.max_discharge_kw,
                available / one_way_eff if one_way_eff > 0 else 0.0,
            )
            discharge_kw = max(0.0, max_dischargeable_kw)
            soc -= discharge_kw / one_way_eff         # energy drawn from storage
            grid_import_kw = deficit - discharge_kw   # remainder from grid

            if discharge_kw > 0:
                action = "discharge"
                # Peak shaving: how much grid import was avoided?
                peak_shaving = max(peak_shaving, discharge_kw)

        # Clamp SoC within hard bounds (floating-point guard)
        soc = max(cfg.min_soc_kwh, min(cfg.max_soc_kwh, soc))

        # Track SoC extremes
        soc_min = min(soc_min, soc)
        soc_max = max(soc_max, soc)

        # Accumulate
        total_charged += charge_kw
        total_discharged += discharge_kw
        total_exported += grid_export_kw
        total_imported += grid_import_kw
        total_demand += demand
        total_demand_met_locally += demand - grid_import_kw

        schedule.append({
            "hour": t,
            "solar_kw": round(solar, 2),
            "demand_kw": round(demand, 2),
            "net_kw": round(net, 2),
            "action": action,
            "charge_kw": round(charge_kw, 2),
            "discharge_kw": round(discharge_kw, 2),
            "soc_kwh": round(soc, 2),
            "soc_pct": round(soc / cfg.capacity_kwh * 100, 1),
            "grid_export_kw": round(grid_export_kw, 2),
            "grid_import_kw": round(grid_import_kw, 2),
        })

    # Build summary
    self_sufficiency_pct = (
        (total_demand_met_locally / total_demand * 100)
        if total_demand > 0
        else 0.0
    )
    cycles_used = total_charged / cfg.capacity_kwh if cfg.capacity_kwh > 0 else 0.0

    summary: dict[str, Any] = {
        "total_charged_kwh": round(total_charged, 2),
        "total_discharged_kwh": round(total_discharged, 2),
        "total_exported_kwh": round(total_exported, 2),
        "total_imported_kwh": round(total_imported, 2),
        "peak_shaving_kw": round(peak_shaving, 2),
        "self_sufficiency_pct": round(self_sufficiency_pct, 1),
        "cycles_used": round(cycles_used, 3),
        "min_soc_kwh": round(soc_min, 2),
        "max_soc_kwh": round(soc_max, 2),
    }

    return {"schedule": schedule, "summary": summary}
