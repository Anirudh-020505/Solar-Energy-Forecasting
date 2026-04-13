"""
Step 11 — PDF Report Exporter (extensions/pdf_exporter.py)

Generates a 6-section PDF from a GridOptimizationReport using ReportLab Platypus.
Charts are rendered as PNG via plotly + kaleido (gracefully omitted if kaleido absent).

Public API
----------
generate_pdf_report(
    report          GridOptimizationReport
    forecast_predictions  list[float]          raw kW values
    battery_schedule      list[dict] | None    per-hour schedule from battery node
    scenario_name         str
) -> bytes   (ready to pass to st.download_button)
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

# ---------------------------------------------------------------------------
# ReportLab imports — hard dependency
# ---------------------------------------------------------------------------
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ---------------------------------------------------------------------------
# Kaleido / plotly (optional — charts omitted if unavailable)
# ---------------------------------------------------------------------------
try:
    import plotly.graph_objects as go
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False

_KALEIDO_AVAILABLE = False
if _PLOTLY_AVAILABLE:
    try:
        import kaleido  # noqa: F401
        _KALEIDO_AVAILABLE = True
    except ImportError:
        pass

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_SOLAR_ORANGE = colors.HexColor("#FF6B35")
_SOLAR_TEAL   = colors.HexColor("#4ECDC4")
_SOLAR_DARK   = colors.HexColor("#1A1A2E")
_LIGHT_GREY   = colors.HexColor("#F4F4F4")
_MID_GREY     = colors.HexColor("#CCCCCC")

# Priority colour map
_PRIORITY_COLORS = {
    "Critical": colors.HexColor("#E74C3C"),
    "High":     colors.HexColor("#E67E22"),
    "Medium":   colors.HexColor("#F1C40F"),
    "Low":      colors.HexColor("#2ECC71"),
}

# Risk colour map
_RISK_COLORS = {
    "High":   colors.HexColor("#E74C3C"),
    "Medium": colors.HexColor("#E67E22"),
    "Low":    colors.HexColor("#2ECC71"),
}


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _get_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontSize=28,
            fontName="Helvetica-Bold",
            textColor=_SOLAR_ORANGE,
            alignment=TA_CENTER,
            spaceAfter=12,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontSize=14,
            fontName="Helvetica",
            textColor=colors.white,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            fontSize=16,
            fontName="Helvetica-Bold",
            textColor=_SOLAR_ORANGE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "sub_heading": ParagraphStyle(
            "sub_heading",
            fontSize=12,
            fontName="Helvetica-Bold",
            textColor=_SOLAR_DARK,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=10,
            fontName="Helvetica",
            textColor=colors.black,
            leading=14,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "small",
            fontSize=8,
            fontName="Helvetica",
            textColor=colors.grey,
            leading=12,
        ),
        "disclaimer": ParagraphStyle(
            "disclaimer",
            fontSize=8,
            fontName="Helvetica-Oblique",
            textColor=colors.grey,
            alignment=TA_CENTER,
            leading=11,
        ),
        "right": ParagraphStyle(
            "right",
            fontSize=9,
            fontName="Helvetica",
            textColor=colors.grey,
            alignment=TA_RIGHT,
        ),
    }


def _hr(color=_MID_GREY, thickness=0.5) -> HRFlowable:
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=6)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _forecast_chart_png(predictions: list[float], width_px: int = 600) -> bytes | None:
    """Render forecast line chart to PNG bytes. Returns None if unavailable."""
    if not (_PLOTLY_AVAILABLE and _KALEIDO_AVAILABLE):
        return None
    hours = list(range(len(predictions)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=predictions,
        mode="lines",
        line=dict(color="#FF6B35", width=2),
        fill="tozeroy",
        fillcolor="rgba(255,107,53,0.2)",
        name="Power (kW)",
    ))
    fig.update_layout(
        template="plotly_white",
        height=250,
        width=width_px,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Hour",
        yaxis_title="kW",
        showlegend=False,
    )
    try:
        return fig.to_image(format="png")
    except Exception:
        return None


def _battery_chart_png(schedule: list[dict], width_px: int = 600) -> bytes | None:
    """Render battery SoC + charge/discharge chart to PNG bytes."""
    if not (_PLOTLY_AVAILABLE and _KALEIDO_AVAILABLE) or not schedule:
        return None
    hours = [row.get("hour", i) for i, row in enumerate(schedule)]
    soc   = [row.get("soc_end_kwh", 0) for row in schedule]
    charge  = [row.get("charge_kw", 0) for row in schedule]
    discharge = [-row.get("discharge_kw", 0) for row in schedule]  # negative for visual

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=charge,    name="Charge kW",    marker_color="#4ECDC4"))
    fig.add_trace(go.Bar(x=hours, y=discharge, name="Discharge kW", marker_color="#E74C3C"))
    fig.add_trace(go.Scatter(
        x=hours, y=soc,
        mode="lines", name="SoC (kWh)",
        line=dict(color="#F39C12", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        template="plotly_white",
        barmode="relative",
        height=250, width=width_px,
        margin=dict(l=40, r=60, t=20, b=40),
        xaxis_title="Hour",
        yaxis_title="kW",
        yaxis2=dict(title="kWh", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.3),
    )
    try:
        return fig.to_image(format="png")
    except Exception:
        return None


def _png_to_image_flowable(png_bytes: bytes, max_width_cm: float = 16.0) -> Image:
    buf = io.BytesIO(png_bytes)
    img = Image(buf)
    # Scale to fit page width while maintaining aspect ratio
    scale = (max_width_cm * cm) / img.drawWidth
    img.drawWidth  *= scale
    img.drawHeight *= scale
    return img


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _cover_section(
    elements: list,
    styles: dict,
    report: Any,
    scenario_name: str,
) -> None:
    elements.append(Spacer(1, 3 * cm))
    elements.append(Paragraph("Solar Grid Optimization Report", styles["cover_title"]))
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph(
        f"Scenario: <b>{scenario_name.replace('_', ' ').title()}</b>",
        ParagraphStyle("cs", fontSize=16, fontName="Helvetica-Bold",
                       textColor=_SOLAR_TEAL, alignment=TA_CENTER),
    ))
    elements.append(Spacer(1, 0.6 * cm))
    elements.append(Paragraph(
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        styles["cover_sub"],
    ))
    if hasattr(report, "report_id"):
        elements.append(Paragraph(f"Report ID: {report.report_id}", styles["small"]))
    elements.append(Spacer(1, 1.5 * cm))
    # Confidence badge
    conf_color = {"High": "#2ECC71", "Medium": "#F39C12", "Low": "#E74C3C"}.get(
        report.confidence_level, "#888888"
    )
    elements.append(Paragraph(
        f'Confidence Level: <font color="{conf_color}"><b>{report.confidence_level}</b></font>',
        ParagraphStyle("conf", fontSize=13, fontName="Helvetica-Bold",
                       alignment=TA_CENTER, textColor=_SOLAR_DARK),
    ))
    elements.append(PageBreak())


def _forecast_summary_section(
    elements: list,
    styles: dict,
    report: Any,
    predictions: list[float],
) -> None:
    elements.append(Paragraph("1. Forecast Summary", styles["section_heading"]))
    elements.append(_hr())

    fs = report.forecast_summary
    data = [
        ["Metric", "Value"],
        ["Total Energy", f"{fs.total_energy_kwh:.2f} kWh"],
        ["Peak Power", f"{fs.peak_power_kw:.2f} kW"],
        ["Average Power", f"{fs.avg_power_kw:.2f} kW"],
        ["Daylight Hours", str(fs.daylight_hours)],
        ["Variability", fs.variability_category],
    ]
    t = Table(data, colWidths=[8 * cm, 8 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _SOLAR_DARK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GREY, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.3, _MID_GREY),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 0.4 * cm))
    elements.append(Paragraph(fs.narrative, styles["body"]))

    png = _forecast_chart_png(predictions)
    if png:
        elements.append(Spacer(1, 0.3 * cm))
        elements.append(Paragraph("Hourly Forecast Profile", styles["sub_heading"]))
        elements.append(_png_to_image_flowable(png))
    elements.append(PageBreak())


def _variability_section(elements: list, styles: dict, report: Any) -> None:
    elements.append(Paragraph("2. Variability Analysis", styles["section_heading"]))
    elements.append(_hr())

    va = report.variability_analysis
    elements.append(Paragraph(
        f"Variability Score: <b>{va.variability_score:.3f}</b> &nbsp;&nbsp; "
        f"Ramp Rate Concern: <b>{'Yes' if va.ramp_rate_concern else 'No'}</b>",
        styles["body"],
    ))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(va.narrative, styles["body"]))

    if va.risk_periods:
        elements.append(Spacer(1, 0.4 * cm))
        elements.append(Paragraph("Detected Risk Periods", styles["sub_heading"]))
        data = [["Hour", "Risk Level", "Reason", "Recommended Action"]]
        for rp in va.risk_periods:
            risk_col = _RISK_COLORS.get(rp.risk_level, colors.grey)
            data.append([
                str(rp.hour_index),
                Paragraph(f'<font color="{risk_col.hexval()}">'
                          f'<b>{rp.risk_level}</b></font>',
                          ParagraphStyle("rpl", fontSize=9, fontName="Helvetica-Bold")),
                Paragraph(rp.reason, styles["small"]),
                Paragraph(rp.recommended_action, styles["small"]),
            ])
        t = Table(data, colWidths=[1.5 * cm, 2.5 * cm, 6.5 * cm, 5.5 * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _SOLAR_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GREY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, _MID_GREY),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t)
    elements.append(PageBreak())


def _grid_balancing_section(elements: list, styles: dict, report: Any) -> None:
    elements.append(Paragraph("3. Grid Balancing Recommendations", styles["section_heading"]))
    elements.append(_hr())

    gb = report.grid_balancing
    elements.append(Paragraph(f"Export Window: <b>{gb.export_window}</b>", styles["body"]))
    elements.append(Paragraph(gb.storage_dispatch_summary, styles["body"]))
    elements.append(Spacer(1, 0.3 * cm))
    elements.append(Paragraph(gb.narrative, styles["body"]))
    elements.append(Spacer(1, 0.4 * cm))

    for rec in gb.recommendations:
        pri_color = _PRIORITY_COLORS.get(rec.priority, colors.grey)
        elements.append(Paragraph(
            f'<font color="{pri_color.hexval()}"><b>[{rec.priority}]</b></font> '
            f'<b>{rec.category}</b>',
            styles["sub_heading"],
        ))
        elements.append(Paragraph(f"Action: {rec.action}", styles["body"]))
        elements.append(Paragraph(f"Expected Benefit: {rec.expected_benefit}", styles["body"]))
        if rec.references:
            refs = ", ".join(rec.references)
            elements.append(Paragraph(f"Sources: {refs}", styles["small"]))
        elements.append(Spacer(1, 0.2 * cm))
    elements.append(PageBreak())


def _energy_utilization_section(elements: list, styles: dict, report: Any) -> None:
    elements.append(Paragraph("4. Energy Utilization Strategies", styles["section_heading"]))
    elements.append(_hr())

    eu = report.energy_utilization
    elements.append(Paragraph(
        f"Peak Shaving Potential: <b>{eu.peak_shaving_potential_kw:.1f} kW</b> &nbsp;&nbsp; "
        f"Self-Consumption Rate: <b>{eu.self_consumption_rate_pct:.1f}%</b>",
        styles["body"],
    ))
    elements.append(Spacer(1, 0.4 * cm))

    for strat in eu.strategies:
        elements.append(Paragraph(f"Strategy: <b>{strat.strategy_name}</b>", styles["sub_heading"]))
        elements.append(Paragraph(strat.description, styles["body"]))
        hours_str = ", ".join(str(h) for h in strat.applicable_hours[:12])
        if len(strat.applicable_hours) > 12:
            hours_str += f" ... (+{len(strat.applicable_hours) - 12} more)"
        elements.append(Paragraph(
            f"Estimated Savings: <b>{strat.estimated_savings_pct:.1f}%</b> &nbsp; "
            f"Applicable Hours: {hours_str}",
            styles["small"],
        ))
        elements.append(Spacer(1, 0.3 * cm))
    elements.append(PageBreak())


def _battery_section(
    elements: list,
    styles: dict,
    report: Any,
    battery_schedule: list[dict] | None,
) -> None:
    elements.append(Paragraph("5. Battery Dispatch Summary", styles["section_heading"]))
    elements.append(_hr())

    gb = report.grid_balancing
    elements.append(Paragraph(gb.storage_dispatch_summary, styles["body"]))
    elements.append(Spacer(1, 0.3 * cm))

    png = _battery_chart_png(battery_schedule or [])
    if png:
        elements.append(Paragraph("Battery Schedule (SoC + Charge/Discharge)", styles["sub_heading"]))
        elements.append(_png_to_image_flowable(png))
    elif battery_schedule:
        # Show table fallback (first 12 rows)
        elements.append(Paragraph("Battery Schedule (first 12 hours)", styles["sub_heading"]))
        data = [["Hr", "Solar kW", "Charge kW", "Discharge kW", "Export kW", "SoC kWh"]]
        for row in battery_schedule[:12]:
            data.append([
                str(row.get("hour", "")),
                f"{row.get('solar_kw', 0):.1f}",
                f"{row.get('charge_kw', 0):.1f}",
                f"{row.get('discharge_kw', 0):.1f}",
                f"{row.get('export_kw', 0):.1f}",
                f"{row.get('soc_end_kwh', 0):.1f}",
            ])
        t = Table(data, colWidths=[1.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm, 2.5*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _SOLAR_DARK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_LIGHT_GREY, colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.3, _MID_GREY),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]))
        elements.append(t)
    elements.append(PageBreak())


def _references_section(elements: list, styles: dict, report: Any) -> None:
    elements.append(Paragraph("6. Supporting References & Disclaimer", styles["section_heading"]))
    elements.append(_hr())

    if report.supporting_references:
        elements.append(Paragraph("Knowledge Base Sources", styles["sub_heading"]))
        for ref in report.supporting_references:
            elements.append(Paragraph(f"• {ref}", styles["body"]))
    else:
        elements.append(Paragraph("No external references cited.", styles["body"]))

    elements.append(Spacer(1, 1 * cm))
    elements.append(_hr(color=_SOLAR_ORANGE))
    elements.append(Paragraph(report.disclaimer, styles["disclaimer"]))


# ---------------------------------------------------------------------------
# Page-number callback
# ---------------------------------------------------------------------------

class _PageNumCanvas:
    """Mixin-style: not used directly — SimpleDocTemplate doesn't support it easily.
    We attach a simple onPage callback instead."""


def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    page_str = f"Page {canvas.getPageNumber()}"
    canvas.drawRightString(A4[0] - 1.5 * cm, 1 * cm, page_str)
    canvas.drawString(1.5 * cm, 1 * cm, "Solar Grid Optimizer — Confidential")
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf_report(
    report: Any,
    forecast_predictions: list[float],
    battery_schedule: list[dict] | None = None,
    scenario_name: str = "baseline",
) -> bytes:
    """
    Render a GridOptimizationReport to a multi-page PDF and return raw bytes.

    Parameters
    ----------
    report                GridOptimizationReport Pydantic model instance.
    forecast_predictions  Hourly kW values for the forecast chart.
    battery_schedule      Per-hour battery schedule dicts (optional).
    scenario_name         Human-readable scenario identifier.

    Returns
    -------
    bytes  — PDF file contents suitable for st.download_button.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title="Solar Grid Optimization Report",
        author="Solar Energy Forecasting Agent",
    )

    styles = _get_styles()
    elements: list = []

    # Build each section
    _cover_section(elements, styles, report, scenario_name)
    _forecast_summary_section(elements, styles, report, forecast_predictions)
    _variability_section(elements, styles, report)
    _grid_balancing_section(elements, styles, report)
    _energy_utilization_section(elements, styles, report)
    _battery_section(elements, styles, report, battery_schedule)
    _references_section(elements, styles, report)

    doc.build(elements, onFirstPage=_add_page_number, onLaterPages=_add_page_number)
    return buf.getvalue()
