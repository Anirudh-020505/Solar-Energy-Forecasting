"""
Page 2 — Grid Optimizer

Reads st.session_state["forecast_predictions"] set by page 1,
runs the full LangGraph agent pipeline, and displays the structured
optimization report in a 3-tab layout.

Tabs:
  1. Summary          — key metrics + forecast/battery KPIs
  2. Grid Balancing   — recommendations, risk periods, variability
  3. Report Details   — energy utilization, raw JSON, PDF download
"""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Grid Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Cached graph builder
# ---------------------------------------------------------------------------

@st.cache_resource
def get_graph():
    from agent.graph import build_graph
    return build_graph()


# ---------------------------------------------------------------------------
# Sidebar — battery configuration
# ---------------------------------------------------------------------------

def _battery_sidebar() -> dict:
    st.sidebar.header("Battery Configuration")
    with st.sidebar.expander("Parameters", expanded=True):
        capacity   = st.number_input("Capacity (kWh)", 50.0, 2000.0, 200.0, 25.0)
        max_chg    = st.number_input("Max Charge (kW)", 5.0, 500.0, 50.0, 5.0)
        max_dis    = st.number_input("Max Discharge (kW)", 5.0, 500.0, 50.0, 5.0)
        init_soc   = st.number_input("Initial SoC (kWh)", 0.0, capacity, capacity * 0.5, 10.0)
        min_soc    = st.number_input("Min SoC (kWh)", 0.0, capacity * 0.4, capacity * 0.1, 5.0)
        demand     = st.number_input("Demand Baseline (kW)", 10.0, 500.0, 80.0, 5.0)
    return {
        "battery_capacity_kwh": capacity,
        "battery_max_charge_kw": max_chg,
        "battery_max_discharge_kw": max_dis,
        "battery_initial_soc_kwh": init_soc,
        "battery_min_soc_kwh": min_soc,
        "battery_demand_baseline_kw": demand,
    }


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _soc_chart(schedule: list[dict]) -> go.Figure:
    hours     = [r.get("hour", i) for i, r in enumerate(schedule)]
    soc       = [r.get("soc_end_kwh", 0) for r in schedule]
    charge    = [r.get("charge_kw", 0) for r in schedule]
    discharge = [-r.get("discharge_kw", 0) for r in schedule]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=charge,    name="Charge kW",    marker_color="#4ECDC4"))
    fig.add_trace(go.Bar(x=hours, y=discharge, name="Discharge kW", marker_color="#E74C3C"))
    fig.add_trace(go.Scatter(
        x=hours, y=soc, mode="lines",
        name="SoC (kWh)", line=dict(color="#F39C12", width=2),
        yaxis="y2",
    ))
    fig.update_layout(
        barmode="relative",
        template="plotly_dark",
        height=300,
        xaxis_title="Hour",
        yaxis_title="kW",
        yaxis2=dict(title="kWh", overlaying="y", side="right"),
        legend=dict(orientation="h", y=-0.35),
        margin=dict(b=80),
    )
    return fig


def _risk_chart(risk_periods: list[dict]) -> go.Figure | None:
    if not risk_periods:
        return None
    df = pd.DataFrame(risk_periods)
    level_order = {"High": 3, "Medium": 2, "Low": 1}
    df["level_num"] = df["risk_level"].map(level_order)
    color_map = {"High": "#E74C3C", "Medium": "#E67E22", "Low": "#2ECC71"}
    fig = px.bar(
        df, x="hour_index", y="level_num",
        color="risk_level",
        color_discrete_map=color_map,
        labels={"hour_index": "Hour", "level_num": "Risk", "risk_level": "Level"},
        title="Risk Period Distribution",
        category_orders={"risk_level": ["High", "Medium", "Low"]},
        hover_data=["reason"],
    )
    fig.update_layout(
        template="plotly_dark",
        height=250,
        yaxis=dict(tickvals=[1, 2, 3], ticktext=["Low", "Medium", "High"]),
    )
    return fig


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------

PRIORITY_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
RISK_EMOJI     = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}


def _render_summary_tab(state: dict) -> None:
    report = state.get("final_report")
    batt   = state.get("battery_summary", {})

    if not state.get("is_valid") or report is None:
        st.error("Report validation failed. See Report Details tab for warnings.")
        return

    fs  = report.forecast_summary
    va  = report.variability_analysis

    # Forecast KPIs
    st.subheader("Forecast Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Energy",   f"{fs.total_energy_kwh:.1f} kWh")
    c2.metric("Peak Power",     f"{fs.peak_power_kw:.1f} kW")
    c3.metric("Avg Power",      f"{fs.avg_power_kw:.1f} kW")
    c4.metric("Daylight Hours", str(fs.daylight_hours))
    c5.metric("Variability",    fs.variability_category)

    st.caption(fs.narrative)
    st.divider()

    # Battery KPIs
    st.subheader("Battery Dispatch")
    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("Charged",        f"{batt.get('total_charged_kwh', 0):.1f} kWh")
    b2.metric("Discharged",     f"{batt.get('total_discharged_kwh', 0):.1f} kWh")
    b3.metric("Exported",       f"{batt.get('total_exported_kwh', 0):.1f} kWh")
    b4.metric("Imported",       f"{batt.get('total_imported_kwh', 0):.1f} kWh")
    b5.metric("Self-Sufficiency", f"{batt.get('self_sufficiency_pct', 0):.1f}%")
    b6.metric("Peak Shaving",   f"{batt.get('peak_shaving_kw', 0):.1f} kW")

    schedule = state.get("battery_schedule", [])
    if schedule:
        st.plotly_chart(_soc_chart(schedule), use_container_width=True)


def _render_grid_tab(state: dict) -> None:
    report = state.get("final_report")
    if report is None:
        st.warning("No valid report generated.")
        return

    gb = report.grid_balancing
    va = report.variability_analysis

    st.subheader("Grid Balancing")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.metric("Export Window", gb.export_window)
        st.caption(gb.storage_dispatch_summary)
    with col_b:
        st.metric("Confidence", report.confidence_level)
        st.metric("Variability Score", f"{va.variability_score:.3f}")

    st.markdown(f"**Narrative:** {gb.narrative}")
    st.divider()

    # Recommendations
    st.subheader("Recommendations")
    for rec in gb.recommendations:
        emoji = PRIORITY_EMOJI.get(rec.priority, "•")
        with st.expander(f"{emoji} [{rec.priority}] {rec.category}"):
            st.markdown(f"**Action:** {rec.action}")
            st.markdown(f"**Expected Benefit:** {rec.expected_benefit}")
            if rec.references:
                st.caption("Sources: " + ", ".join(rec.references))

    st.divider()
    st.subheader("Variability & Risk Periods")

    risk_fig = _risk_chart(state.get("risk_periods", []))
    if risk_fig:
        st.plotly_chart(risk_fig, use_container_width=True)

    if va.risk_periods:
        risk_data = pd.DataFrame([rp.model_dump() for rp in va.risk_periods])
        st.dataframe(risk_data, use_container_width=True)
    else:
        st.success("No significant risk periods detected.")

    st.markdown(f"**Analysis:** {va.narrative}")
    if va.ramp_rate_concern:
        st.warning("Ramp rate concern flagged — consider additional storage dispatch.")


def _render_details_tab(
    state: dict,
    predictions: list[float],
    scenario_name: str,
) -> None:
    report = state.get("final_report")
    if report is None:
        st.error("No valid report to show.")
        return

    eu = report.energy_utilization
    st.subheader("Energy Utilization")
    col_a, col_b = st.columns(2)
    col_a.metric("Peak Shaving Potential", f"{eu.peak_shaving_potential_kw:.1f} kW")
    col_b.metric("Self-Consumption Rate", f"{eu.self_consumption_rate_pct:.1f}%")

    for strat in eu.strategies:
        with st.expander(f"{strat.strategy_name} — {strat.estimated_savings_pct:.1f}% saving"):
            st.markdown(strat.description)
            hrs = ", ".join(str(h) for h in strat.applicable_hours[:12])
            if len(strat.applicable_hours) > 12:
                hrs += " …"
            st.caption(f"Applicable hours: {hrs}")

    st.divider()

    # Validation warnings
    warnings = state.get("validation_warnings", [])
    if warnings:
        with st.expander(f"Validation warnings ({len(warnings)})"):
            for w in warnings:
                st.warning(w)

    if state.get("generation_error"):
        st.error(f"LLM error (mock fallback used): {state['generation_error']}")

    # Raw JSON
    with st.expander("Raw structured report (JSON)"):
        raw = state.get("structured_report") or {}
        st.json(raw)

    st.divider()

    # PDF download
    st.subheader("Export Report")
    if st.button("Generate PDF", use_container_width=True):
        with st.spinner("Rendering PDF…"):
            try:
                from extensions.pdf_exporter import generate_pdf_report
                schedule = state.get("battery_schedule")
                pdf_bytes = generate_pdf_report(
                    report=report,
                    forecast_predictions=predictions,
                    battery_schedule=schedule,
                    scenario_name=scenario_name,
                )
                st.download_button(
                    "Download PDF Report",
                    pdf_bytes,
                    f"grid_report_{scenario_name}.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"PDF generation failed: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("⚡ Grid Optimizer")
    st.markdown("AI-powered grid optimization using the LangGraph agent pipeline.")

    # Check for forecast data
    if "forecast_predictions" not in st.session_state:
        st.warning("No forecast data found. Run the **Forecast** page first.")
        if st.button("Go to Forecast Page"):
            st.switch_page("pages/1_Forecast.py")
        return

    predictions: list[float] = st.session_state["forecast_predictions"]
    metadata: dict = st.session_state.get("forecast_metadata", {})
    time_value = metadata.get("time_value", 24)
    time_unit  = metadata.get("time_unit", "hours")

    st.sidebar.success(f"Forecast loaded: {len(predictions)} steps ({time_value} {time_unit})")

    # Battery sidebar
    battery_config = _battery_sidebar()

    scenario_name = st.sidebar.selectbox(
        "Scenario",
        ["baseline", "cloudy_day", "high_demand", "battery_upgrade"],
        format_func=lambda k: k.replace("_", " ").title(),
    )

    st.sidebar.divider()

    # ---- Run agent ----
    if st.sidebar.button("Run Grid Optimizer", type="primary", use_container_width=True):
        graph = get_graph()

        initial_state = {
            "predictions": predictions,
            "time_value": time_value,
            "time_unit": time_unit,
            "scenario_name": scenario_name,
            "scenario_params": {},  # no additional transforms for baseline run
            **battery_config,
        }

        with st.status("Running LangGraph agent pipeline…", expanded=True) as status:
            st.write("analyze_forecast…")
            st.write("retrieve_guidelines (RAG)…")
            st.write("battery_optimizer…")
            st.write("generate_report (LLM / Demo Mode)…")
            st.write("validate_output…")
            try:
                result = graph.invoke(initial_state)
                st.session_state["optimizer_result"] = result
                status.update(label="Pipeline complete!", state="complete")
            except Exception as exc:
                status.update(label=f"Error: {exc}", state="error")
                st.error(str(exc))
                return

    # ---- Display results ----
    if "optimizer_result" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Grid Optimizer**.")
        return

    state: dict = st.session_state["optimizer_result"]
    is_demo = state.get("raw_llm_output", "").startswith("[Demo Mode")

    if is_demo:
        st.info("Demo Mode — no GROQ_API_KEY set. Report generated from computed statistics.")
    if state.get("generation_error") and not is_demo:
        st.warning(f"LLM error (mock fallback): {state['generation_error']}")

    # ---- 3-tab layout ----
    tab1, tab2, tab3 = st.tabs(["Summary", "Grid Balancing", "Report Details"])
    with tab1:
        _render_summary_tab(state)
    with tab2:
        _render_grid_tab(state)
    with tab3:
        _render_details_tab(state, predictions, scenario_name)


main()
