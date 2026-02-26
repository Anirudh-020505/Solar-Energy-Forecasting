# ☀️ Solar Energy Power Generation Forecasting using LSTM

This project focuses on forecasting solar power generation using a deep learning model (LSTM) based on historical weather and environmental data. The goal is to help energy providers and researchers predict future solar energy output for better planning and smart grid management.

---

## 📌 Features
- Time-series forecasting using Long Short-Term Memory (LSTM)
- Weather-based feature engineering (temperature, humidity, cloud cover, wind, radiation, etc.)
- Cyclical encoding for temporal features (hour, month)
- Streamlit-based web interface for interactive predictions
- Easy deployment-ready structure

---

## 🗂️ Project Structure
project/
├── backend/
├── README.md
├── solarForecasting.ipynb
├── streamlit_app.py
└── test_data.csv

---

## ⚙️ Tech Stack
- Python 3.10+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Streamlit

---

## 📊 Dataset
The dataset contains hourly meteorological features such as:
- Temperature
- Relative Humidity
- Cloud Cover (low, medium, high layers)
- Wind Speed & Wind Direction
- Solar Radiation
- Zenith & Azimuth Angles

Target Variable:
- `generated_power_kw` (Solar Power in kW)

---

## 🧠 Model
We use an **LSTM (Long Short-Term Memory)** neural network for time-series forecasting.

- Look-back window: 24 hours  
- Optimizer: Adam  
- Loss function: Mean Squared Error (MSE)  
- Evaluation Metrics:
  - Mean Absolute Error (MAE): **461.33 kW**
  - Root Mean Squared Error (RMSE): **615.24 kW**

---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Anirudh-020505/Solar-Energy-Forecasting.git
cd Solar-Energy-Forecasting

⚙️ Installation & Run
2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit App
streamlit run streamlit_app.py
```


### 📈 Results

The LSTM model demonstrates strong performance in capturing temporal patterns in solar power generation:

MAE: 461.33 kW

RMSE: 615.24 kW


### 📦 Future Work

Extend the model into an AI assistant that provides grid optimization recommendations.

Add battery storage optimization and variability risk analysis.

Incorporate uncertainty-aware forecasting with confidence intervals.

Enable multi-site solar forecasting and scenario-based planning.

Deploy as a full-stack cloud application with API support.



