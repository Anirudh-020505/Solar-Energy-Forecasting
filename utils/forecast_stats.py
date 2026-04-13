"""
Pure forecast statistics helpers — no external dependencies beyond numpy.

These functions are called by agent/nodes/analyze_forecast.py and also
directly by utils/mock_llm.py for the demo-mode fallback path.

All inputs are a flat list/array of hourly power predictions in kW.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# compute_stats
# ---------------------------------------------------------------------------

def compute_stats(predictions: list[float]) -> dict[str, Any]:
    """
    Compute aggregate statistics over a flat list of hourly kW predictions.

    Returns
    -------
    dict with keys:
        mean            float  — average power across all hours (kW)
        std             float  — standard deviation of power (kW)
        peak            float  — maximum predicted power (kW)
        total           float  — sum of predictions (≡ total energy in kWh for 1-hr steps)
        cv              float  — coefficient of variation (std / mean); 0 if mean == 0
        daylight_fraction float — fraction of hours with power > 0
        daylight_hours  int    — count of hours with power > 0
        min_nonzero     float  — minimum power among daylight hours; 0.0 if no daylight
        peak_hour       int    — 0-based index of the hour with maximum power
        total_hours     int    — total number of predictions
    """
    if not predictions:
        return {
            "mean": 0.0, "std": 0.0, "peak": 0.0, "total": 0.0,
            "cv": 0.0, "daylight_fraction": 0.0, "daylight_hours": 0,
            "min_nonzero": 0.0, "peak_hour": 0, "total_hours": 0,
        }

    arr = np.array(predictions, dtype=float)
    arr = np.maximum(arr, 0.0)  # clamp negatives produced by inverse-scaler noise

    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    peak_val = float(np.max(arr))
    total_val = float(np.sum(arr))

    cv = std_val / mean_val if mean_val > 0.0 else 0.0

    daylight_mask = arr > 0.0
    daylight_hours = int(np.sum(daylight_mask))
    daylight_fraction = daylight_hours / len(arr)

    nonzero_vals = arr[daylight_mask]
    min_nonzero = float(np.min(nonzero_vals)) if len(nonzero_vals) > 0 else 0.0

    peak_hour = int(np.argmax(arr))

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "peak": round(peak_val, 4),
        "total": round(total_val, 4),
        "cv": round(cv, 4),
        "daylight_fraction": round(daylight_fraction, 4),
        "daylight_hours": daylight_hours,
        "min_nonzero": round(min_nonzero, 4),
        "peak_hour": peak_hour,
        "total_hours": len(arr),
    }


# ---------------------------------------------------------------------------
# find_risk_periods
# ---------------------------------------------------------------------------

def find_risk_periods(predictions: list[float]) -> list[dict[str, Any]]:
    """
    Scan the prediction sequence and flag hours that pose grid stability risk.

    Flagging criteria (any one is sufficient to flag an hour):
        1. Rapid ramp-down: power drops > 30 kW from the previous hour.
        2. Intermittent cloud shadow: power > 0 but < 5 % of peak (brief partial output).
        3. Sharp 3-hour context drop: current value is < 60 % of the 3-hour rolling mean
           (i.e., > 40 % drop from local average).

    Risk level assignment:
        High   — criteria 1 AND (2 or 3) simultaneously, OR ramp > 80 kW/hr
        Medium — criteria 1 (ramp 30–80 kW/hr), OR criterion 3 alone
        Low    — criterion 2 alone (minor intermittency)

    Returns
    -------
    List of dicts with keys:
        hour_index          int
        risk_level          "Low" | "Medium" | "High"
        reason              str
        recommended_action  str
    """
    if not predictions:
        return []

    arr = np.array(predictions, dtype=float)
    arr = np.maximum(arr, 0.0)

    peak = float(np.max(arr)) if np.any(arr > 0) else 0.0
    intermittency_threshold = peak * 0.05  # 5 % of peak

    risk_periods: list[dict[str, Any]] = []
    seen: set[int] = set()

    for i in range(len(arr)):
        reasons: list[str] = []
        actions: list[str] = []
        severity_score = 0  # 0=none, 1=Low, 2=Medium, 3=High

        # --- Criterion 1: ramp-down from previous hour ---
        if i > 0:
            ramp = arr[i - 1] - arr[i]  # positive = drop
            if ramp > 30.0:
                if ramp > 80.0:
                    reasons.append(f"severe ramp-down of {ramp:.1f} kW/hr from previous hour")
                    actions.append("activate spinning reserve and fast-response storage discharge")
                    severity_score = max(severity_score, 3)
                else:
                    reasons.append(f"ramp-down of {ramp:.1f} kW/hr from previous hour")
                    actions.append("ramp up grid reserve or initiate battery discharge")
                    severity_score = max(severity_score, 2)

        # --- Criterion 2: intermittent cloud shadow ---
        if 0 < arr[i] < intermittency_threshold and peak > 0:
            reasons.append(
                f"intermittent output ({arr[i]:.1f} kW, < 5% of peak {peak:.1f} kW)"
            )
            actions.append("monitor frequency deviation; prepare demand response")
            severity_score = max(severity_score, 1)

        # --- Criterion 3: >40% drop from 3-hour rolling mean ---
        if i >= 2:
            window_mean = float(np.mean(arr[max(0, i - 3):i]))
            if window_mean > 0 and arr[i] < 0.60 * window_mean:
                drop_pct = (1 - arr[i] / window_mean) * 100
                reasons.append(
                    f"{drop_pct:.0f}% drop from 3-hour average ({window_mean:.1f} kW)"
                )
                actions.append("schedule dispatchable backup generation or curtail flexible loads")
                severity_score = max(severity_score, 2)

        if severity_score == 0:
            continue

        # Escalate if multiple criteria triggered simultaneously
        if len(reasons) >= 2:
            severity_score = min(severity_score + 1, 3)

        level_map = {1: "Low", 2: "Medium", 3: "High"}
        risk_level = level_map[severity_score]

        risk_periods.append({
            "hour_index": i,
            "risk_level": risk_level,
            "reason": "; ".join(reasons),
            "recommended_action": "; ".join(dict.fromkeys(actions)),  # dedup order-preserving
        })
        seen.add(i)

    # Sort by severity descending, then by hour ascending
    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    risk_periods.sort(key=lambda r: (severity_order[r["risk_level"]], r["hour_index"]))

    return risk_periods


# ---------------------------------------------------------------------------
# compute_variability_score
# ---------------------------------------------------------------------------

def compute_variability_score(
    stats: dict[str, Any],
    risk_periods: list[dict[str, Any]],
) -> float:
    """
    Compute a composite variability score in [0, 1].

    Formula (weighted sum, then sigmoid-like normalisation):
        raw = 0.4 * cv_norm + 0.4 * ramp_score + 0.2 * intermittency_score

    Components
    ----------
    cv_norm         — coefficient of variation normalised to [0, 1] by capping at CV = 1.5
    ramp_score      — fraction of risk periods that are "High" or "Medium" severity,
                      weighted by their share of total hours (0 if no risk periods)
    intermittency_score — fraction of total hours that are flagged as any risk level

    The result is clamped to [0.0, 1.0] and rounded to 4 decimal places.

    Parameters
    ----------
    stats        : dict returned by compute_stats()
    risk_periods : list returned by find_risk_periods()

    Returns
    -------
    float in [0.0, 1.0]
    """
    total_hours = stats.get("total_hours", 0)
    if total_hours == 0:
        return 0.0

    # Component 1: CV normalised to [0, 1] (cap at 1.5 → score = 1.0)
    cv = stats.get("cv", 0.0)
    cv_norm = min(cv / 1.5, 1.0)

    # Component 2: ramp severity score
    # Count medium+high risk periods weighted by 1.0 for High, 0.6 for Medium
    severity_weights = {"High": 1.0, "Medium": 0.6, "Low": 0.2}
    weighted_risk_sum = sum(
        severity_weights.get(rp["risk_level"], 0.0) for rp in risk_periods
    )
    # Normalise: max possible weighted sum if every hour were High = total_hours * 1.0
    ramp_score = min(weighted_risk_sum / max(total_hours, 1), 1.0)

    # Component 3: intermittency — fraction of hours flagged at all
    flagged_hours = len(risk_periods)
    intermittency_score = min(flagged_hours / max(total_hours, 1), 1.0)

    raw = 0.4 * cv_norm + 0.4 * ramp_score + 0.2 * intermittency_score

    return round(max(0.0, min(raw, 1.0)), 4)


# ---------------------------------------------------------------------------
# Convenience: variability category label
# ---------------------------------------------------------------------------

def variability_category(score: float) -> str:
    """Map a variability score to a human-readable category for ForecastSummary."""
    if score < 0.33:
        return "Low"
    if score < 0.66:
        return "Moderate"
    return "High"
