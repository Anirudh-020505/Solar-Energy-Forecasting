import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Try to import tensorflow, fallback to None if not available
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    tf = None
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. Using mock predictions for demonstration.")

# Set page config
st.set_page_config(
    page_title="🌞 Solar Energy Forecasting",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache model loading
@st.cache_resource
def load_model_and_scalers():
    """Load the LSTM model and scalers"""
    try:
        MODEL_DIR = "backend/models"
        
        if not TF_AVAILABLE:
            st.warning("TensorFlow not available - using mock model for demonstration")
            return None, None, None, [
                'temperature_2_m_above_gnd', 'relative_humidity_2_m_above_gnd', 
                'mean_sea_level_pressure_msl', 'total_precipitation_sfc', 
                'snowfall_amount_sfc', 'total_cloud_cover_sfc', 
                'high_cloud_cover_high_cld_lay', 'medium_cloud_cover_mid_cld_lay', 
                'low_cloud_cover_low_cld_lay', 'shortwave_radiation_backwards_sfc', 
                'wind_gust_10_m_above_gnd', 'angle_of_incidence', 'zenith', 'azimuth',
                'wind_u_10_m_above_gnd', 'wind_v_10_m_above_gnd', 
                'wind_u_80_m_above_gnd', 'wind_v_80_m_above_gnd', 
                'wind_u_900_mb', 'wind_v_900_mb', 'hour_sin', 'hour_cos', 
                'month_sin', 'month_cos'
            ]
        
        # Load model with custom objects to handle compile warnings
        lstm_model = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, "power_forecasting_lstm.h5"),
            custom_objects={"mse": tf.keras.losses.MeanSquaredError()},
            compile=False
        )
        
        # Load scalers
        x_scaler = joblib.load(os.path.join(MODEL_DIR, "X_scaler.pkl"))
        y_scaler = joblib.load(os.path.join(MODEL_DIR, "y_scaler.pkl"))
        
        return lstm_model, x_scaler, y_scaler, list(x_scaler.feature_names_in_)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None

def preprocess_csv(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Preprocess the CSV data to match model requirements"""
    df = df.copy()
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower().str.strip()

    # Drop target column if present
    if "generated_power_kw" in df.columns:
        df = df.drop(columns=["generated_power_kw"])

    # Wind decomposition: speed + direction → u/v components
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

    # Forward/backward fill missing values
    df = df.ffill().bfill()

    # Add time features (synthetic hourly timeline)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range(start="2025-01-01", periods=len(df), freq="h")

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

    # Ensure we have all required columns
    missing = set(expected_features) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df[expected_features]

def predict_horizon(features_df: pd.DataFrame, time_value: int, time_unit: str, 
                   model, x_scaler, y_scaler):
    """Make predictions for the specified time horizon"""
    multipliers = {"hours": 1, "days": 24, "weeks": 168}
    total_steps = time_value * multipliers.get(time_unit.lower(), 1)

    # If model is None (TensorFlow not available), generate mock predictions
    if model is None or x_scaler is None or y_scaler is None:
        st.info("🔄 Generating mock predictions (TensorFlow not available)")
        
        # Generate realistic mock solar predictions
        predictions = []
        for i in range(total_steps):
            hour_of_day = i % 24
            # Simple solar curve: peak at noon, zero at night
            if 6 <= hour_of_day <= 18:  # Daylight hours
                # Parabolic curve peaking at hour 12
                base_power = 4 * (hour_of_day - 6) * (18 - hour_of_day) / 36
                # Add some randomness and cloud effects
                cloud_factor = np.random.uniform(0.7, 1.0)
                noise = np.random.normal(0, 0.1)
                power = max(0, (base_power * cloud_factor + noise) * 100)
            else:
                power = 0  # No solar power at night
            
            predictions.append(round(power, 2))
        
        return predictions

    # Use last 24 rows as lookback window
    current_seq = x_scaler.transform(features_df.tail(24).values)
    predictions = []

    # Always show progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(total_steps):
        # Update progress
        progress = (i + 1) / total_steps
        progress_bar.progress(progress)
        status_text.text(f"Predicting step {i + 1} of {total_steps}...")
            
        # Reshape for LSTM: (batch_size, timesteps, features)
        inp = np.expand_dims(current_seq, axis=0)
        # Use __call__ for faster inference than predict()
        pred_scaled = float(model(inp, training=False)[0][0].numpy())
        predictions.append(pred_scaled)
        
        # Slide window: drop oldest, repeat last weather
        new_row = current_seq[-1].reshape(1, -1)
        current_seq = np.vstack([current_seq[1:], new_row])
    
    progress_bar.progress(1.0)
    status_text.text("✅ Prediction complete!")

    # Convert back to real kW values
    results_kw = y_scaler.inverse_transform(
        np.array(predictions).reshape(-1, 1)
    ).flatten()

    return [max(0, round(float(v), 2)) for v in results_kw]

def create_prediction_chart(predictions, time_unit, time_value):
    """Create an interactive chart for the predictions"""
    hours = list(range(1, len(predictions) + 1))
    
    # Create time labels based on unit
    if time_unit == "hours":
        time_labels = [f"Hour {h}" for h in hours]
    elif time_unit == "days":
        time_labels = [f"Day {(h-1)//24 + 1}, Hour {((h-1)%24) + 1}" for h in hours]
    else:  # weeks
        time_labels = [f"Week {(h-1)//168 + 1}, Day {((h-1)%168)//24 + 1}, Hour {((h-1)%24) + 1}" for h in hours]
    
    # Create the main line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=predictions,
        mode='lines+markers',
        name='Predicted Power',
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=4),
        hovertemplate='<b>%{customdata}</b><br>' +
                     'Power: %{y:.2f} kW<extra></extra>',
        customdata=time_labels
    ))
    
    # Add area fill
    fig.add_trace(go.Scatter(
        x=hours + hours[::-1],
        y=predictions + [0] * len(predictions),
        fill='tonexty',
        fillcolor='rgba(255, 107, 53, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f"Solar Power Forecast - {time_value} {time_unit.title()}",
        xaxis_title="Time Step",
        yaxis_title="Power Generation (kW)",
        template="plotly_dark",
        height=500,
        hovermode="x unified"
    )
    
    return fig

def create_summary_chart(predictions, time_unit):
    """Create summary statistics chart"""
    if time_unit == "hours":
        # Hourly summary
        data = predictions
        labels = [f"Hour {i+1}" for i in range(len(predictions))]
    elif time_unit == "days":
        # Daily summary
        days = len(predictions) // 24
        data = [sum(predictions[i*24:(i+1)*24]) for i in range(days)]
        labels = [f"Day {i+1}" for i in range(days)]
    else:  # weeks
        # Weekly summary  
        weeks = len(predictions) // 168
        data = [sum(predictions[i*168:(i+1)*168]) for i in range(weeks)]
        labels = [f"Week {i+1}" for i in range(weeks)]
    
    fig = go.Figure(data=go.Bar(
        x=labels,
        y=data,
        marker_color='#4ECDC4',
        hovertemplate='<b>%{x}</b><br>Energy: %{y:.2f} kWh<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Energy Generation Summary by {time_unit.title()[:-1]}",
        xaxis_title=time_unit.title()[:-1],
        yaxis_title="Total Energy (kWh)",
        template="plotly_dark",
        height=400
    )
    
    return fig

# Main App
def main():
    st.title("🌞 Solar Energy Forecasting")
    st.markdown("Upload weather data CSV and get AI-powered solar energy predictions")
    
    # Load model and scalers
    lstm_model, x_scaler, y_scaler, expected_features = load_model_and_scalers()
    
    if lstm_model is None:
        st.error("❌ Could not load the ML model. Please check if model files exist in backend/models/")
        st.info("Required files: power_forecasting_lstm.h5, X_scaler.pkl, y_scaler.pkl")
        return
    
    st.success("✅ LSTM Model loaded successfully!")
    
    # Sidebar for inputs
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Weather CSV",
            type=['csv'],
            help=f"CSV must have at least 24 rows with {len(expected_features)} required columns"
        )
        
        # Forecast horizon
        st.subheader("📊 Forecast Settings")
        time_value = st.number_input("Forecast Value", min_value=1, max_value=336, value=24)
        time_unit = st.selectbox("Time Unit", ["hours", "days", "weeks"])
        
        # Show required columns
        with st.expander("📋 Required CSV Columns"):
            for i, col in enumerate(expected_features, 1):
                st.text(f"{i:2d}. {col}")
    
    # Main content
    if uploaded_file is not None:
        try:
            # Load and validate CSV
            raw_df = pd.read_csv(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📄 Uploaded Data")
                st.write(f"**Shape:** {raw_df.shape[0]} rows × {raw_df.shape[1]} columns")
                st.dataframe(raw_df.head(), use_container_width=True)
            
            with col2:
                st.subheader("🔍 Data Info") 
                st.write(f"**Columns:** {list(raw_df.columns)}")
                st.write(f"**Missing values:** {raw_df.isnull().sum().sum()}")
                st.write(f"**Data types:** {raw_df.dtypes.value_counts().to_dict()}")
            
            # Validate minimum rows
            if len(raw_df) < 24:
                st.error(f"❌ CSV must have at least 24 rows. You uploaded {len(raw_df)} rows.")
                return
            
            # Preprocess data
            with st.spinner("🔄 Preprocessing data..."):
                processed_df = preprocess_csv(raw_df, expected_features)
            
            st.success(f"✅ Data preprocessed successfully! Using last 24 rows as lookback window.")
            
            # Make predictions
            if st.button("🚀 Generate Forecast", type="primary", use_container_width=True):
                with st.spinner(f"🤖 Generating {time_value} {time_unit} forecast..."):
                    predictions = predict_horizon(
                        processed_df, time_value, time_unit, 
                        lstm_model, x_scaler, y_scaler
                    )
                
                # Results section
                st.header("📈 Forecast Results")
                
                # Key metrics
                total_energy = sum(predictions)
                avg_power = np.mean(predictions)
                max_power = max(predictions)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🔋 Total Energy", f"{total_energy:.1f} kWh")
                
                with col2:
                    st.metric("⚡ Average Power", f"{avg_power:.1f} kW")
                
                with col3:
                    st.metric("📊 Peak Power", f"{max_power:.1f} kW")
                    
                with col4:
                    st.metric("⏱️ Time Steps", len(predictions))
                
                st.divider()
                
                # Charts
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Main forecast chart
                    main_fig = create_prediction_chart(predictions, time_unit, time_value)
                    st.plotly_chart(main_fig, use_container_width=True)
                
                with col2:
                    # Summary chart
                    summary_fig = create_summary_chart(predictions, time_unit)
                    st.plotly_chart(summary_fig, use_container_width=True)
                
                # Detailed results table
                with st.expander(f"📊 Detailed Hourly Results ({len(predictions)} values)"):
                    results_df = pd.DataFrame({
                        'Hour': range(1, len(predictions) + 1),
                        'Power (kW)': predictions,
                        'Cumulative Energy (kWh)': np.cumsum(predictions)
                    })
                    st.dataframe(results_df, use_container_width=True, height=400)
                
                # Download results
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results CSV",
                    csv_data,
                    f"solar_forecast_{time_value}{time_unit}.csv",
                    "text/csv",
                    use_container_width=True
                )
                
        except ValueError as ve:
            st.error(f"❌ Data validation error: {str(ve)}")
        except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            
    else:
        # Show sample data format
        st.info("👆 Upload a CSV file to get started")
        
        with st.expander("📋 Sample Data Format"):
            st.write("Your CSV should look like this:")
            sample_data = {
                'temperature_2_m_above_gnd': [8.2, 7.5, 7.0],
                'relative_humidity_2_m_above_gnd': [82, 85, 87],
                'total_cloud_cover_sfc': [45, 50, 55],
                'shortwave_radiation_backwards_sfc': [0, 0, 0],
                'wind_speed_10_m_above_gnd': [3.2, 2.9, 2.7],
                'wind_direction_10_m_above_gnd': [180, 185, 190],
                '...': ['...', '...', '...']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
            
            st.download_button(
                "📥 Download Sample CSV",
                open("test_data.csv", "r").read(),
                "sample_weather_data.csv",
                "text/csv"
            )

if __name__ == "__main__":
    main()