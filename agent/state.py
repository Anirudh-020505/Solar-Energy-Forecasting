"""
GridOptimizerState — the single shared state dict that flows through
every node in the LangGraph pipeline.

All fields are Optional so each node only needs to populate its own
outputs; LangGraph merges state dicts between nodes automatically.

Lifecycle
---------
  Initial state   — caller populates "inputs" group fields
  analyze_forecast — writes forecast_stats group
  retrieve_guidelines — writes rag group
  battery_optimizer — writes battery group
  generate_report  — writes llm group
  handle_llm_error — overwrites generation_error + structured_report
  validate_output  — writes validation group + final_report
"""

from __future__ import annotations

from typing import Any, Optional
from typing_extensions import TypedDict


class GridOptimizerState(TypedDict, total=False):
    # ------------------------------------------------------------------
    # Inputs  (set by the caller before graph.invoke())
    # ------------------------------------------------------------------
    predictions: list[float]
    """Raw LSTM predictions in kW (list of hourly values)."""

    time_unit: str
    """'hours' | 'days' | 'weeks' — forecast horizon unit."""

    time_value: int
    """Numeric horizon length (e.g. 24 for 24 hours)."""

    battery_capacity_kwh: float
    """Usable battery capacity in kWh."""

    battery_max_charge_kw: float
    """Maximum charge power in kW."""

    battery_max_discharge_kw: float
    """Maximum discharge power in kW."""

    battery_initial_soc_kwh: float
    """Starting state of charge in kWh."""

    battery_min_soc_kwh: float
    """Minimum allowed SoC in kWh (depth-of-discharge protection)."""

    battery_demand_baseline_kw: float
    """Flat hourly demand assumption used by the battery scheduler."""

    scenario_name: str
    """Human-readable scenario label (e.g. 'baseline', 'cloudy_day')."""

    scenario_params: dict[str, Any]
    """
    Scenario transform parameters consumed by analyze_forecast node.
    Supported keys:
        scale_factor          float  — multiplier applied to all predictions
        demand_multiplier     float  — multiplier applied to demand_baseline_kw
        cloud_ramp_noise      bool   — add random 20–40% drops to hours 10–14
        battery_capacity_multiplier float — scale battery capacity
    """

    # ------------------------------------------------------------------
    # After analyze_forecast
    # ------------------------------------------------------------------
    forecast_stats: dict[str, Any]
    """Output of utils.forecast_stats.compute_stats() on the (transformed) predictions."""

    variability_score: float
    """Composite score 0–1 from utils.forecast_stats.compute_variability_score()."""

    risk_periods: list[dict[str, Any]]
    """List of risk-period dicts from utils.forecast_stats.find_risk_periods()."""

    peak_generation_hour: int
    """0-based index of the hour with maximum predicted power."""

    total_energy_kwh: float
    """Sum of all predictions (kWh, assuming 1-hr timesteps)."""

    daylight_fraction: float
    """Fraction of hours with predicted power > 0."""

    effective_predictions: list[float]
    """Predictions after scenario transforms are applied (used by downstream nodes)."""

    # ------------------------------------------------------------------
    # After retrieve_guidelines
    # ------------------------------------------------------------------
    retrieved_chunks: list[dict[str, Any]]
    """Deduplicated RAG chunks returned by FAISSStore.query()."""

    source_references: list[str]
    """Unique source filenames cited in retrieved_chunks."""

    # ------------------------------------------------------------------
    # After battery_optimizer
    # ------------------------------------------------------------------
    battery_schedule: list[dict[str, Any]]
    """Per-hour battery dispatch schedule (list of dicts with 11 fields each)."""

    battery_summary: dict[str, Any]
    """Aggregate battery metrics (total_charged_kwh, self_sufficiency_pct, etc.)."""

    # ------------------------------------------------------------------
    # After generate_report
    # ------------------------------------------------------------------
    raw_llm_output: str
    """Raw string response from the Groq LLM (before JSON parsing)."""

    structured_report: Optional[dict[str, Any]]
    """Parsed JSON dict from the LLM response; None on parse failure."""

    generation_error: Optional[str]
    """Error message if LLM call or JSON parsing failed; None on success."""

    # ------------------------------------------------------------------
    # After validate_output
    # ------------------------------------------------------------------
    is_valid: bool
    """True when Pydantic validation of structured_report succeeded."""

    validation_warnings: list[str]
    """Non-fatal warnings from the validation step (e.g. minor numeric drift)."""

    final_report: Optional[Any]
    """
    The validated GridOptimizationReport Pydantic model instance.
    This is what pages/2_Grid_Optimizer.py reads to render the UI.
    """
