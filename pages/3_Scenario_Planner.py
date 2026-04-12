"""
Page 3 — Scenario Planner

Runs multiple scenario variants of the LangGraph agent pipeline on the
base forecast from session state and presents a side-by-side comparison
with radar chart and bar charts.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Scenario Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_graph():
    from agent.graph import build_graph
    return build_graph()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _radar_chart(df: pd.DataFrame) -> go.Figure:
    """Spider / radar chart across 5 normalised KPIs."""
    metrics = [
        ("total_energy_kwh",    "Total Energy"),
        ("self_sufficiency_pct","Self-Sufficiency"),
        ("peak_shaving_kw",     "Peak Shaving"),
        ("total_exported_kwh",  "Exported"),
        ("variability_score",   "Variability"),
    ]
    # Normalise each metric to [0, 1] so radar is visually balanced
    norm_df = df.copy()
    for col, _ in metrics:
        mx = norm_df[col].max()
        if mx > 0:
            norm_df[col] = norm_df[col] / mx

    fig = go.Figure()
    labels = [lbl for _, lbl in metrics]
    for _, row in norm_df.iterrows():
        values = [float(row[col]) for col, _ in metrics]
        values.append(values[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill="toself",
            name=row["label"],
            line=dict(color=row.get("color", "#888"), width=2),
            opacity=0.6,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_dark",
        height=420,
        title="Scenario Comparison — Normalised KPIs",
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def _bar_chart(df: pd.DataFrame, col: str, title: str, unit: str) -> go.Figure:
    fig = px.bar(
        df,
        x="label",
        y=col,
        color="label",
        color_discrete_sequence=[row["color"] for _, row in df.iterrows()],
        title=title,
        labels={"label": "Scenario", col: unit},
        text_auto=".1f",
    )
    fig.update_layout(
        template="plotly_dark",
        height=300,
        showlegend=False,
        margin=dict(t=40, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# PDF helper (per-scenario)
# ---------------------------------------------------------------------------

def _pdf_download_button(state: dict, predictions: list[float], key: str) -> None:
    label = key.replace("_", " ").title()
    if st.button(f"Generate PDF — {label}", key=f"pdf_{key}"):
        report = state.get("final_report")
        if report is None:
            st.warning("No valid report for this scenario.")
            return
        with st.spinner("Rendering PDF…"):
            try:
                from extensions.pdf_exporter import generate_pdf_report
                pdf = generate_pdf_report(
                    report=report,
                    forecast_predictions=predictions,
                    battery_schedule=state.get("battery_schedule"),
                    scenario_name=key,
                )
                st.download_button(
                    f"Download — {label}",
                    pdf,
                    f"report_{key}.pdf",
                    "application/pdf",
                    key=f"dl_{key}",
                )
            except Exception as exc:
                st.error(f"PDF error: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("📊 Scenario Planner")
    st.markdown(
        "Compare how the four built-in scenarios affect grid optimization "
        "across energy, battery, and variability metrics."
    )

    # Guard: need forecast
    if "forecast_predictions" not in st.session_state:
        st.warning("No forecast data. Run the **Forecast** page first.")
        if st.button("Go to Forecast"):
            st.switch_page("pages/1_Forecast.py")
        return

    predictions: list[float] = st.session_state["forecast_predictions"]
    metadata: dict = st.session_state.get("forecast_metadata", {})
    time_value = metadata.get("time_value", 24)
    time_unit  = metadata.get("time_unit", "hours")

    # Sidebar scenario selector
    from extensions.scenario_planner import SCENARIOS
    all_keys = list(SCENARIOS.keys())

    st.sidebar.header("Scenario Selection")
    selected_keys: list[str] = st.sidebar.multiselect(
        "Scenarios to compare",
        options=all_keys,
        default=all_keys,
        format_func=lambda k: SCENARIOS[k]["label"],
    )
    if not selected_keys:
        st.info("Select at least one scenario in the sidebar.")
        return

    st.sidebar.divider()
    st.sidebar.subheader("Base Battery Config")
    capacity = st.sidebar.number_input("Capacity (kWh)", 50.0, 2000.0, 200.0, 25.0)
    max_chg  = st.sidebar.number_input("Max Charge (kW)", 5.0, 500.0, 50.0, 5.0)
    max_dis  = st.sidebar.number_input("Max Discharge (kW)", 5.0, 500.0, 50.0, 5.0)
    demand   = st.sidebar.number_input("Demand Baseline (kW)", 10.0, 500.0, 80.0, 5.0)

    from extensions.battery_optimizer import BatteryConfig
    base_cfg = BatteryConfig(
        capacity_kwh=capacity,
        max_charge_kw=max_chg,
        max_discharge_kw=max_dis,
        initial_soc_kwh=capacity * 0.5,
        min_soc_kwh=capacity * 0.1,
        demand_baseline_kw=demand,
    )

    # ---- Run scenarios ----
    run_btn = st.sidebar.button("Run All Scenarios", type="primary", use_container_width=True)
    if run_btn:
        from extensions.scenario_planner import run_scenario, compare_scenarios

        graph = get_graph()
        results: dict = {}

        progress = st.progress(0)
        for idx, key in enumerate(selected_keys):
            with st.status(f"Running {SCENARIOS[key]['label']}…") as s:
                try:
                    state = run_scenario(
                        key=key,
                        base_predictions=predictions,
                        base_battery_config=base_cfg,
                        graph=graph,
                        time_value=time_value,
                        time_unit=time_unit,
                    )
                    results[key] = state
                    s.update(label=f"{SCENARIOS[key]['label']} complete", state="complete")
                except Exception as exc:
                    s.update(label=f"Error: {exc}", state="error")
            progress.progress((idx + 1) / len(selected_keys))

        st.session_state["scenario_results"] = results
        st.success(f"Completed {len(results)} / {len(selected_keys)} scenarios.")

    # ---- Display results ----
    if "scenario_results" not in st.session_state:
        st.info("Click **Run All Scenarios** in the sidebar to begin.")
        return

    results: dict = st.session_state["scenario_results"]
    if not results:
        st.warning("No results available.")
        return

    # Filter to selected keys that were actually run
    active = {k: v for k, v in results.items() if k in selected_keys}
    if not active:
        st.info("Re-run scenarios with the current selection.")
        return

    from extensions.scenario_planner import compare_scenarios
    df = compare_scenarios(active)

    # ---- Comparison table ----
    st.subheader("Comparison Table")
    display_cols = [
        "label", "total_energy_kwh", "peak_power_kw", "variability_score",
        "self_sufficiency_pct", "total_charged_kwh", "total_exported_kwh",
        "peak_shaving_kw", "cycles_used", "confidence_level",
    ]
    display_df = df[[c for c in display_cols if c in df.columns]].copy()
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]
    st.dataframe(display_df, use_container_width=True)

    # ---- Radar chart ----
    st.divider()
    st.subheader("Radar Comparison")
    st.plotly_chart(_radar_chart(df), use_container_width=True)

    # ---- Bar charts ----
    st.divider()
    st.subheader("Individual Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            _bar_chart(df, "total_energy_kwh", "Total Energy", "kWh"),
            use_container_width=True,
        )
        st.plotly_chart(
            _bar_chart(df, "self_sufficiency_pct", "Self-Sufficiency", "%"),
            use_container_width=True,
        )
    with col2:
        st.plotly_chart(
            _bar_chart(df, "variability_score", "Variability Score", "0–1"),
            use_container_width=True,
        )
        st.plotly_chart(
            _bar_chart(df, "peak_shaving_kw", "Peak Shaving", "kW"),
            use_container_width=True,
        )

    # ---- Per-scenario expanders ----
    st.divider()
    st.subheader("Per-Scenario Reports")
    for key, state in active.items():
        meta = SCENARIOS[key]
        report = state.get("final_report")
        with st.expander(f"{meta['label']} — {meta['description']}"):
            if report:
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Confidence", report.confidence_level)
                col_b.metric("Variability", report.forecast_summary.variability_category)
                col_c.metric("Is Valid", "Yes" if state.get("is_valid") else "No")

                st.caption(f"Forecast: {report.forecast_summary.narrative}")

                warnings = state.get("validation_warnings", [])
                if warnings:
                    for w in warnings:
                        st.warning(w)
                if state.get("generation_error"):
                    st.info(f"Demo Mode (mock): {state['generation_error']}")
            else:
                st.error("No valid report for this scenario.")

            # PDF download
            _pdf_download_button(state, predictions, key)

    # ---- Download comparison CSV ----
    st.divider()
    csv = df.drop(columns=["color"], errors="ignore").to_csv(index=False).encode()
    st.download_button(
        "Download Comparison CSV",
        csv,
        "scenario_comparison.csv",
        "text/csv",
        use_container_width=True,
    )


main()
