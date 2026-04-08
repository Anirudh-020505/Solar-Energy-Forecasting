"""
Pydantic v2 models for the structured GridOptimizationReport output.

All LLM JSON output is parsed into these models by validate_output node.
The JSON schema is embedded verbatim in the LLM system prompt (agent/prompts.py)
so the model knows the exact shape expected.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-section: Forecast Summary
# ---------------------------------------------------------------------------

class ForecastSummary(BaseModel):
    """High-level numeric summary of the raw LSTM forecast."""

    total_energy_kwh: float = Field(
        ..., description="Sum of all predicted hourly power values (kWh = kW × 1 hr)."
    )
    peak_power_kw: float = Field(
        ..., description="Maximum predicted power in the forecast horizon."
    )
    avg_power_kw: float = Field(
        ..., description="Mean predicted power across all time steps."
    )
    daylight_hours: int = Field(
        ..., ge=0, le=24, description="Number of hours with predicted power > 0."
    )
    variability_category: Literal["Low", "Moderate", "High"] = Field(
        ..., description="Human-readable bucket derived from variability_score."
    )
    narrative: str = Field(
        ..., min_length=20, description="1–3 sentence plain-English summary of forecast."
    )


# ---------------------------------------------------------------------------
# Sub-section: Risk Periods
# ---------------------------------------------------------------------------

class RiskPeriod(BaseModel):
    """A single hour flagged as operationally risky."""

    hour_index: int = Field(..., ge=0, description="0-based index into the predictions list.")
    risk_level: Literal["Low", "Medium", "High"] = Field(
        ..., description="Severity of the risk at this hour."
    )
    reason: str = Field(
        ..., min_length=5, description="Short explanation (e.g. 'rapid ramp-down', 'cloud intermittency')."
    )
    recommended_action: str = Field(
        ..., min_length=5, description="Concise operator action for this specific hour."
    )


class VariabilityAnalysis(BaseModel):
    """Quantitative assessment of forecast variability and risk."""

    variability_score: float = Field(
        ..., ge=0.0, le=1.0, description="Composite score: 0 = stable, 1 = highly variable."
    )
    risk_periods: list[RiskPeriod] = Field(
        default_factory=list,
        description="Hours flagged by find_risk_periods(); may be empty for stable forecasts.",
    )
    ramp_rate_concern: bool = Field(
        ..., description="True when any inter-hour ramp exceeds 30 kW/hr."
    )
    narrative: str = Field(
        ..., min_length=20, description="2–4 sentence plain-English variability assessment."
    )


# ---------------------------------------------------------------------------
# Sub-section: Grid Balancing
# ---------------------------------------------------------------------------

class GridRecommendation(BaseModel):
    """A single actionable grid-balancing recommendation."""

    category: Literal[
        "Frequency Regulation",
        "Voltage Control",
        "Ramp Management",
        "Curtailment",
        "Storage Dispatch",
        "Demand Response",
        "Export Management",
    ] = Field(..., description="Operational category this recommendation addresses.")
    priority: Literal["Critical", "High", "Medium", "Low"] = Field(
        ..., description="Implementation urgency."
    )
    action: str = Field(
        ..., min_length=10, description="Specific operator action."
    )
    expected_benefit: str = Field(
        ..., min_length=10, description="Quantified or qualified outcome if action is taken."
    )
    references: list[str] = Field(
        default_factory=list,
        description="Source document names from the RAG knowledge base that support this recommendation.",
    )


class GridBalancingSection(BaseModel):
    """Grid balancing recommendations + battery dispatch summary."""

    recommendations: list[GridRecommendation] = Field(
        ..., min_length=1, description="Ordered list of grid recommendations (highest priority first)."
    )
    storage_dispatch_summary: str = Field(
        ..., min_length=10,
        description="Plain-English summary of the battery schedule produced by the battery optimizer.",
    )
    export_window: str = Field(
        ..., description="Hours (e.g. '10:00–14:00') during which grid export is recommended."
    )
    narrative: str = Field(
        ..., min_length=20, description="2–3 sentence overall grid balancing narrative."
    )


# ---------------------------------------------------------------------------
# Sub-section: Energy Utilization
# ---------------------------------------------------------------------------

class UtilizationStrategy(BaseModel):
    """A single demand-side or self-consumption strategy."""

    strategy_name: str = Field(..., min_length=3, description="Short strategy label.")
    description: str = Field(
        ..., min_length=20, description="What the strategy entails and how to implement it."
    )
    estimated_savings_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Estimated cost or energy savings as a percentage (0–100).",
    )
    applicable_hours: list[int] = Field(
        ..., min_length=1,
        description="0-based hour indices when this strategy is most effective.",
    )


class EnergyUtilizationSection(BaseModel):
    """Demand-side management and self-consumption strategies."""

    strategies: list[UtilizationStrategy] = Field(
        ..., min_length=1, description="List of utilization strategies, most impactful first."
    )
    peak_shaving_potential_kw: float = Field(
        ..., ge=0.0,
        description="Estimated peak demand reduction achievable via battery + demand response (kW).",
    )
    self_consumption_rate_pct: float = Field(
        ..., ge=0.0, le=100.0,
        description="Estimated percentage of generated energy consumed on-site (vs. exported).",
    )


# ---------------------------------------------------------------------------
# Top-level Report
# ---------------------------------------------------------------------------

class GridOptimizationReport(BaseModel):
    """
    Complete structured output of the LangGraph grid optimization agent.

    Composed by generate_report node (Groq LLM) and validated by validate_output node.
    Serialised to JSON for the PDF exporter and the Streamlit report UI.
    """

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this report run.",
    )
    scenario_name: str = Field(
        ..., description="Name of the scenario that produced this report (e.g. 'baseline', 'cloudy_day')."
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of report generation.",
    )

    # Core sections — all required
    forecast_summary: ForecastSummary
    variability_analysis: VariabilityAnalysis
    grid_balancing: GridBalancingSection
    energy_utilization: EnergyUtilizationSection

    # Provenance
    supporting_references: list[str] = Field(
        default_factory=list,
        description="Deduplicated list of RAG source document names cited across all sections.",
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        ...,
        description=(
            "LLM self-assessed confidence. High = strong RAG support + low variability; "
            "Low = sparse RAG hits or high forecast uncertainty."
        ),
    )
    disclaimer: str = Field(
        default=(
            "This report is generated by an AI assistant for informational purposes only. "
            "All recommendations should be reviewed by a qualified grid operator before implementation. "
            "Forecasts are based on historical LSTM model outputs and do not constitute operational guarantees."
        ),
        description="Standard liability disclaimer appended to every report.",
    )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator("scenario_name")
    @classmethod
    def scenario_name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("scenario_name must not be blank.")
        return v.strip()

    @model_validator(mode="after")
    def check_sections_non_empty(self) -> GridOptimizationReport:
        """Ensure every required section has at least one substantive item."""
        if not self.grid_balancing.recommendations:
            raise ValueError("grid_balancing.recommendations must contain at least one item.")
        if not self.energy_utilization.strategies:
            raise ValueError("energy_utilization.strategies must contain at least one item.")
        return self

    @model_validator(mode="after")
    def check_energy_consistency(self) -> GridOptimizationReport:
        """
        Validate basic physical plausibility:
        - avg_power_kw must not exceed peak_power_kw
        - total_energy_kwh must be non-negative

        Note: avg_power_kw is the mean over ALL time steps (including nighttime zeros),
        so avg × daylight_hours != total_energy; only avg × total_steps == total_energy.
        We don't store total_steps here, so we skip that cross-check.
        """
        s = self.forecast_summary
        if s.avg_power_kw > s.peak_power_kw + 1e-3:
            raise ValueError(
                f"avg_power_kw ({s.avg_power_kw}) cannot exceed peak_power_kw ({s.peak_power_kw})."
            )
        if s.total_energy_kwh < 0:
            raise ValueError("total_energy_kwh must be non-negative.")
        return self

    def all_section_references(self) -> list[str]:
        """Collect all cited references from every recommendation for deduplication."""
        refs: list[str] = list(self.supporting_references)
        for rec in self.grid_balancing.recommendations:
            refs.extend(rec.references)
        return sorted(set(refs))
