"""
Page 1 — Solar Energy Forecast

Migrated from streamlit_app.py (Milestone 1) and extended for Milestone 2:
- Writes st.session_state["forecast_predictions"] and ["forecast_metadata"]
- Provides a "Go to Grid Optimizer" button after generating predictions
"""

import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Try to import tensorflow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False

st.set_page_config(
    page_title="Solar Forecast",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_scalers():
    MODEL_DIR = "backend/models"
    _default_features = [
        "temperature_2_m_above_gnd", "relative_humidity_2_m_above_gnd",
        "mean_sea_level_pressure_msl", "total_precipitation_sfc",
        "snowfall_amount_sfc", "total_cloud_cover_sfc",
        "high_cloud_cover_high_cld_lay", "medium_cloud_cover_mid_cld_lay",
        "low_cloud_cover_low_cld_lay", "shortwave_radiation_backwards_sfc",
        "wind_gust_10_m_above_gnd", "angle_of_incidence", "zenith", "azimuth",
        "wind_u_10_m_above_gnd", "wind_v_10_m_above_gnd",
        "wind_u_80_m_above_gnd", "wind_v_80_m_above_gnd",
        "wind_u_900_mb", "wind_v_900_mb",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]

    if not TF_AVAILABLE:
        return None, None, None, _default_features

    try:
        lstm_model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "power_forecasting_lstm.h5"),
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
            compile=False,
        )
        x_scaler = joblib.load(os.path.join(MODEL_DIR, "X_scaler.pkl"))
        y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.pkl"))
        return lstm_model, x_scaler, y_scaler, list(x_scaler.feature_names_in_)
    except Exception as exc:
        st.error(f"Model load error: {exc}")
        return None, None, None, _default_features


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_csv(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.strip()

    if "generated_power_kw" in df.columns:
        df = df.drop(columns=["generated_power_kw"])

    wind_layers = [
        ("10_m_above_gnd", "wind_speed_10_m_above_gnd", "wind_direction_10_m_above_gnd"),
        ("80_m_above_gnd", "wind_speed_80_m_above_gnd", "wind_direction_80_m_above_gnd"),
        ("900_mb", "wind_speed_900_mb", "wind_direction_900_mb"),
    ]
    for layer, speed_col, dir_col in wind_layers:
        if dir_col in df.columns and speed_col in df.columns:
            rad = np.deg2rad(df[dir_col])
            df[f"wind_u_{layer}"] = df[speed_col] * np.cos(rad)
            df[f"wind_v_{layer}"] = df[speed_col] * np.sin(rad)
            df.drop(columns=[dir_col, speed_col], inplace=True)

    df = df.ffill().bfill()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2025-01-01", periods=len(df), freq="h")

    df["hour_sin"]   = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]   = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"]  = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"]  = np.cos(2 * np.pi * df.index.month / 12)

    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[expected_features]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_horizon(
    features_df: pd.DataFrame,
    time_value: int,
    time_unit: str,
    model, x_scaler, y_scaler,
) -> list[float]:
    multipliers = {"hours": 1, "days": 24, "weeks": 168}
    total_steps = time_value * multipliers.get(time_unit.lower(), 1)

    if model is None or x_scaler is None or y_scaler is None:
        # Mock solar curve
        predictions = []
        for i in range(total_steps):
            hour = i % 24
            if 6 <= hour <= 18:
                base = 4 * (hour - 6) * (18 - hour) / 36
                power = max(0, (base * np.random.uniform(0.7, 1.0) + np.random.normal(0, 0.1)) * 100)
            else:
                power = 0.0
            predictions.append(round(power, 2))
        return predictions

    current_seq = x_scaler.transform(features_df.tail(24).values)
    predictions = []
    progress = st.progress(0)
    status   = st.empty()

    for i in range(total_steps):
        progress.progress((i + 1) / total_steps)
        status.text(f"Predicting step {i + 1} of {total_steps}…")
        inp = np.expand_dims(current_seq, axis=0)
        pred_scaled = float(model(inp, training=False)[0][0].numpy())
        predictions.append(pred_scaled)
        new_row = current_seq[-1].reshape(1, -1)
        current_seq = np.vstack([current_seq[1:], new_row])

    progress.progress(1.0)
    status.text("Prediction complete!")

    results_kw = y_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()
    return [max(0.0, round(float(v), 2)) for v in results_kw]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _forecast_chart(predictions: list[float], time_value: int, time_unit: str) -> go.Figure:
    hours = list(range(1, len(predictions) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours, y=predictions,
        mode="lines+markers",
        name="Predicted Power",
        line=dict(color="#FF6B35", width=3),
        marker=dict(size=4),
    ))
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=predictions + [0] * len(predictions),
        fill="tonexty",
        fillcolor="rgba(255,107,53,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Solar Power Forecast — {time_value} {time_unit.title()}",
        xaxis_title="Time Step",
        yaxis_title="Power (kW)",
        template="plotly_dark",
        height=450,
        hovermode="x unified",
    )
    return fig


def _summary_chart(predictions: list[float], time_unit: str) -> go.Figure:
    if time_unit == "hours":
        data, labels = predictions, [f"H{i+1}" for i in range(len(predictions))]
    elif time_unit == "days":
        days = len(predictions) // 24
        data   = [sum(predictions[i*24:(i+1)*24]) for i in range(days)]
        labels = [f"Day {i+1}" for i in range(days)]
    else:
        weeks = len(predictions) // 168
        data   = [sum(predictions[i*168:(i+1)*168]) for i in range(weeks)]
        labels = [f"Wk {i+1}" for i in range(weeks)]

    fig = go.Figure(go.Bar(
        x=labels, y=data,
        marker_color="#4ECDC4",
        hovertemplate="<b>%{x}</b><br>%{y:.2f} kWh<extra></extra>",
    ))
    fig.update_layout(
        title=f"Energy by {time_unit[:-1].title()}",
        xaxis_title=time_unit[:-1].title(),
        yaxis_title="kWh",
        template="plotly_dark",
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("☀️ Solar Energy Forecast")
    st.markdown("Upload weather data and generate LSTM-powered solar power predictions.")

    lstm_model, x_scaler, y_scaler, expected_features = load_model_and_scalers()

    if not TF_AVAILABLE:
        st.warning("TensorFlow not available — mock predictions will be used.")
    else:
        st.success("LSTM model loaded.")

    # ---- Sidebar ----
    with st.sidebar:
        st.header("Configuration")

        # Data source selector
        st.subheader("Data Source")
        data_source = st.radio(
            "Choose input",
            ["Sample dataset", "Upload my own CSV"],
            help="Use a built-in sample or upload your own weather CSV.",
        )

        SAMPLES = {
            "☀️ Sunny Summer Day": "sample_data/sunny_summer_day.csv",
            "🌧️ Cloudy Winter Day": "sample_data/cloudy_winter_day.csv",
            "🌤️ Overcast Autumn Day (test_data)": "test_data.csv",
        }

        uploaded_file = None
        selected_sample_path = None

        if data_source == "Sample dataset":
            sample_choice = st.selectbox("Select sample", list(SAMPLES.keys()))
            selected_sample_path = SAMPLES[sample_choice]
        else:
            uploaded_file = st.file_uploader(
                "Upload Weather CSV",
                type=["csv"],
                help="At least 24 rows with required weather columns.",
            )

        st.subheader("Forecast Horizon")
        time_value = st.number_input("Value", min_value=1, max_value=336, value=24)
        time_unit  = st.selectbox("Unit", ["hours", "days", "weeks"])

        with st.expander("Required CSV columns"):
            for i, col in enumerate(expected_features, 1):
                st.text(f"{i:2d}. {col}")

    # ---- Load data ----
    # Resolve the DataFrame from either sample or upload
    raw_df = None

    if data_source == "Sample dataset":
        if os.path.exists(selected_sample_path):
            raw_df = pd.read_csv(selected_sample_path)
            st.info(f"Using built-in sample: **{sample_choice}**")
            # Download button for the sample
            with open(selected_sample_path) as f:
                st.sidebar.download_button(
                    "Download this sample CSV",
                    f.read(),
                    os.path.basename(selected_sample_path),
                    "text/csv",
                    use_container_width=True,
                )
        else:
            st.error(f"Sample file not found: `{selected_sample_path}`")
            return
    else:
        if uploaded_file is None:
            st.info("Upload a CSV file to begin, or switch to a sample dataset.")

            # Show sample download cards
            st.subheader("Or download a sample to explore the format:")
            c1, c2, c3 = st.columns(3)
            for col_widget, (label, path) in zip(
                [c1, c2, c3], SAMPLES.items()
            ):
                if os.path.exists(path):
                    col_widget.markdown(f"**{label}**")
                    with open(path) as f:
                        col_widget.download_button(
                            "Download CSV",
                            f.read(),
                            os.path.basename(path),
                            "text/csv",
                            use_container_width=True,
                        )

            with st.expander("Expected CSV column format"):
                sample_preview = {
                    "temperature_2_m_above_gnd": [28.5, 31.0],
                    "relative_humidity_2_m_above_gnd": [45, 40],
                    "total_cloud_cover_sfc": [10, 8],
                    "shortwave_radiation_backwards_sfc": [750, 920],
                    "wind_u_10_m_above_gnd": [3.5, 4.0],
                    "wind_v_10_m_above_gnd": [-1.5, -2.0],
                    "hour_sin": [0.87, 0.71],
                    "hour_cos": [-0.50, -0.71],
                    "...": ["...", "..."],
                }
                st.dataframe(pd.DataFrame(sample_preview))
            return

        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
            return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data preview")
        st.write(f"**Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} cols")
        st.dataframe(raw_df.head(), use_container_width=True)
    with col2:
        st.subheader("Data info")
        st.write(f"**Missing values:** {raw_df.isnull().sum().sum()}")
        st.write(f"**Columns:** {list(raw_df.columns)[:6]} …")

    if len(raw_df) < 24:
        st.error(f"CSV must have at least 24 rows (got {len(raw_df)}).")
        return

    try:
        with st.spinner("Preprocessing…"):
            processed_df = preprocess_csv(raw_df, expected_features)
        st.success("Data preprocessed — 24-row lookback window ready.")
    except ValueError as exc:
        st.error(str(exc))
        return

    # ---- Run forecast ----
    if st.button("Generate Forecast", type="primary", use_container_width=True):
        with st.spinner(f"Forecasting {time_value} {time_unit}…"):
            predictions = predict_horizon(processed_df, time_value, time_unit,
                                          lstm_model, x_scaler, y_scaler)

        # Write to session state for downstream pages
        st.session_state["forecast_predictions"] = predictions
        st.session_state["forecast_metadata"] = {
            "time_value": time_value,
            "time_unit": time_unit,
            "total_steps": len(predictions),
        }

        # ---- Metrics ----
        st.header("Forecast Results")
        total_e = sum(predictions)
        avg_p   = float(np.mean(predictions))
        peak_p  = max(predictions)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Energy", f"{total_e:.1f} kWh")
        c2.metric("Average Power", f"{avg_p:.1f} kW")
        c3.metric("Peak Power", f"{peak_p:.1f} kW")
        c4.metric("Time Steps", len(predictions))

        st.divider()

        # ---- Charts ----
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.plotly_chart(_forecast_chart(predictions, time_value, time_unit),
                            use_container_width=True)
        with col_b:
            st.plotly_chart(_summary_chart(predictions, time_unit),
                            use_container_width=True)

        # ---- Detailed table ----
        with st.expander(f"Detailed results ({len(predictions)} steps)"):
            results_df = pd.DataFrame({
                "Hour": range(1, len(predictions) + 1),
                "Power (kW)": predictions,
                "Cumulative Energy (kWh)": list(np.cumsum(predictions)),
            })
            st.dataframe(results_df, use_container_width=True, height=350)

        # ---- Download CSV ----
        csv_bytes = results_df.to_csv(index=False).encode()
        st.download_button(
            "Download Results CSV",
            csv_bytes,
            f"solar_forecast_{time_value}{time_unit}.csv",
            "text/csv",
        )

        # ---- Cross-page navigation ----
        st.divider()
        st.success("Forecast saved to session — navigate to Grid Optimizer to run the agent.")
        if st.button("Go to Grid Optimizer →", use_container_width=True):
            st.switch_page("pages/2_Grid_Optimizer.py")


main()
