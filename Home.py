"""
Home.py — Multipage app entry point (Milestone 2)

Loads environment variables, checks API key status, and presents the
landing page with navigation to the three main pages.
"""

import os

import streamlit as st

# Load .env before any other imports that might need env vars
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

st.set_page_config(
    page_title="Solar Grid Optimizer",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# API key / mode detection
# ---------------------------------------------------------------------------

_GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
_LIVE_MODE = bool(_GROQ_KEY)


def _status_banner() -> None:
    if _LIVE_MODE:
        st.sidebar.success("GROQ_API_KEY set — Live Mode")
    else:
        st.sidebar.warning("No GROQ_API_KEY — Demo Mode")
        st.sidebar.caption(
            "Add your key to `.env` as `GROQ_API_KEY=...` "
            "or set it as an environment variable to enable the live LLM."
        )


# ---------------------------------------------------------------------------
# Landing page
# ---------------------------------------------------------------------------

def main():
    _status_banner()

    st.title("🌞 Solar Energy Forecasting & Grid Optimization")
    st.markdown(
        "An end-to-end platform combining LSTM solar power forecasting with "
        "an AI-driven grid optimization agent (LangGraph + Groq LLM)."
    )

    if not _LIVE_MODE:
        st.info(
            "Running in **Demo Mode** — the AI report will be generated from "
            "computed statistics without an LLM call. Set `GROQ_API_KEY` in "
            "`.env` to enable the live Groq (llama-3.1-8b-instant) report."
        )

    st.divider()

    # ---- Page cards ----
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### ☀️ 1 — Solar Forecast")
        st.markdown(
            "Upload weather CSV data and generate LSTM-powered hourly solar "
            "power predictions. Results are saved for the optimizer."
        )
        if st.button("Open Forecast Page", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Forecast.py")

    with col2:
        st.markdown("### ⚡ 2 — Grid Optimizer")
        st.markdown(
            "Run the full LangGraph agent pipeline: RAG knowledge retrieval, "
            "battery scheduling, LLM report generation, and Pydantic validation."
        )
        ready = "forecast_predictions" in st.session_state
        if st.button(
            "Open Grid Optimizer",
            use_container_width=True,
            type="primary",
            disabled=not ready,
        ):
            st.switch_page("pages/2_Grid_Optimizer.py")
        if not ready:
            st.caption("Run the Forecast page first to enable this.")

    with col3:
        st.markdown("### 📊 3 — Scenario Planner")
        st.markdown(
            "Compare four scenarios (Baseline, Cloudy Day, High Demand, "
            "Battery Upgrade) across energy, battery, and variability KPIs."
        )
        if st.button(
            "Open Scenario Planner",
            use_container_width=True,
            type="primary",
            disabled=not ready,
        ):
            st.switch_page("pages/3_Scenario_Planner.py")
        if not ready:
            st.caption("Run the Forecast page first to enable this.")

    st.divider()

    # ---- Architecture overview ----
    with st.expander("Architecture overview"):
        st.markdown("""
**Milestone 1 (Forecast page)**
- LSTM model `(1, 24, 24)` — 24-hour lookback, 24 features, auto-regressive loop
- Wind decomposition: speed + direction → u/v components
- Cyclical time features: `hour_sin/cos`, `month_sin/cos`
- Graceful TF fallback: mock solar curve when TensorFlow is unavailable

**Milestone 2 (Optimizer & Planner pages)**
- LangGraph pipeline: `analyze_forecast → retrieve_guidelines → battery_optimizer → generate_report → validate_output`
- RAG: `all-MiniLM-L6-v2` embeddings + FAISS `IndexFlatIP` (cosine, no server)
- Greedy battery scheduler (charge/discharge/idle per hour, √RTE efficiency)
- Structured output: Pydantic v2 `GridOptimizationReport` with cross-field validation
- Demo Mode: `utils/mock_llm.py` generates a valid report from computed stats when no API key is set
- PDF export: ReportLab Platypus, 6 sections, Plotly+kaleido charts
        """)

    with st.expander("Session state"):
        keys = {k: type(v).__name__ for k, v in st.session_state.items()}
        if keys:
            st.json(keys)
        else:
            st.caption("Empty — run the Forecast page to populate.")


main()
