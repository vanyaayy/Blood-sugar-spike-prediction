import os
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from models import BrisT1DLSTM, BrisT1DDTransformer, BrisT1DTCN

# Streamlit config
st.set_page_config(page_title="Glucose Prediction Dashboard", layout="centered")
st.title("📈 Glucose Prediction Model Comparison")

# Model selection
model_name = st.selectbox("Select a Model", ["LSTM", "Transformer", "TCN"], index=0)

@st.cache_resource
def load_model(name):
    if name == "LSTM":
        model = BrisT1DLSTM()
        model.load_state_dict(torch.load(".export/checkpoint-aug-lstm.pt", map_location="cpu"))
    elif name == "Transformer":
        model = BrisT1DDTransformer()
        model.load_state_dict(torch.load(".export/checkpoint-aug-tf.pt", map_location="cpu"))
    else:
        model = BrisT1DTCN(num_inputs=6, num_channels=[64, 32])
        model.load_state_dict(torch.load(".export/checkpoint-aug-tcn.pt", map_location="cpu"))
    model.eval()
    return model

@st.cache_resource
def load_target_scaler():
    return joblib.load(".data/target_scaler_aug.pkl")

@st.cache_resource
def load_feature_scaler():
    return joblib.load(".data/feature_scaler_aug.pkl")

@st.cache_data
def load_scaled_data():
    return pd.read_csv("train__test_aug.csv")

# Load resources
model = load_model(model_name)
target_scaler = load_target_scaler()
feature_scaler = load_feature_scaler()
df_scaled = load_scaled_data()

# Feature info
feature_order = ["bg", "insulin", "carbs", "cals", "hr", "steps"]
time_steps = [f"{i//60}:{i%60:02d}" for i in range(115, -5, -5)]  # 1:55 to 0:00

# ---------------------
# Row Selection Section
# ---------------------
st.markdown("### 📄 Select a Row from Dataset to Predict")
row_idx = st.number_input("Row index (0 to {})".format(len(df_scaled) - 1),
                          min_value=0, max_value=len(df_scaled) - 1, value=0)

if st.button("Run Prediction on Selected Row"):
    row = df_scaled.iloc[row_idx]

    # Extract input features for 24x6 sequence
    x_scaled = row[[f"{f}-{t}" for t in time_steps for f in feature_order]].values.reshape(1, 24, 6)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    # Ground truth
    true_bg = row["bg+1:00"]

    # Predict
    with torch.no_grad():
        scaled_pred = model(x_tensor).item()
        pred_bg = target_scaler.inverse_transform([[scaled_pred]])[0][0]

    # Inverse transform input for display
    x_orig = feature_scaler.inverse_transform(x_scaled.reshape(-1, 6)).reshape(24, 6)
    df_display = pd.DataFrame(x_orig, columns=feature_order)
    df_display["Time"] = time_steps
    df_display = df_display[["Time"] + feature_order]

    # Display input
    st.markdown("#### 🔍 Input Features (Original Scale)")
    st.dataframe(df_display)

    # Display results
    st.markdown(f"### 📍 Predicted Glucose (bg+1:00): {pred_bg:.2f} mmol/L")
    st.markdown(f"### 🎯 Ground Truth Glucose: {true_bg:.2f} mmol/L")

    # Error metrics
    error = abs(pred_bg - true_bg)
    mae = mean_absolute_error([true_bg], [pred_bg])
    rmse = mean_squared_error([true_bg], [pred_bg], squared=False)

    st.markdown(f"**Absolute Error:** {error:.2f} mmol/L")
    st.markdown(f"**MAE:** {mae:.2f} | **RMSE:** {rmse:.2f}")
