import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier

# --- Configuration ---
MODEL_PATH = 'best_wine_model_combined.pkl'
SCALER_PATH = 'scaler_combined.pkl'

# --- Load Model and Scaler ---
@st.cache_resource
def load_assets():
    """Loads the model and scaler from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or Scaler file not found. Check if {MODEL_PATH} and {SCALER_PATH} are in the same directory.")
        return None, None

model, scaler = load_assets()

# --- Streamlit App ---

st.title("🍷 Wine Quality Prediction App")
st.markdown("Predict the wine quality label (1: Good >= 7, 0: Bad < 7) using an optimized XGBoost Classifier.")

if model and scaler:
    # --- Input Widgets for Wine Features (11 features) ---

    st.header("1. Wine Type")
    wine_type = st.selectbox("Select Wine Type:", ("White", "Red"))

    st.header("2. Physico-chemical Characteristics")

    # Layout using columns for better organization
    col1, col2, col3 = st.columns(3)

    # Column 1
    fixed_acidity = col1.slider("Fixed Acidity", 4.0, 16.0, 7.5, 0.1)
    volatile_acidity = col1.slider("Volatile Acidity", 0.08, 1.6, 0.35, 0.01)
    citric_acid = col1.slider("Citric Acid", 0.0, 1.7, 0.30, 0.01)
    residual_sugar = col1.slider("Residual Sugar", 0.6, 66.0, 5.8, 0.1)

    # Column 2
    chlorides = col2.slider("Chlorides", 0.01, 0.65, 0.05, 0.001)
    free_sulfur_dioxide = col2.slider("Free Sulfur Dioxide", 1.0, 290.0, 30.0, 1.0)
    total_sulfur_dioxide = col2.slider("Total Sulfur Dioxide", 6.0, 440.0, 110.0, 1.0)
    density = col2.slider("Density", 0.98, 1.04, 0.99, 0.0001)

    # Column 3
    pH = col3.slider("pH", 2.7, 4.0, 3.3, 0.01)
    sulphates = col3.slider("Sulphates", 0.2, 2.0, 0.60, 0.01)
    alcohol = col3.slider("Alcohol (%)", 8.0, 15.0, 10.5, 0.1)

    # --- Feature Engineering & Prediction ---

    # Map wine type to 'type_white' feature (1 for White, 0 for Red)
    type_white = 1 if wine_type == "White" else 0

    # Create a DataFrame for prediction (must match the training feature order)
    # The feature order from the notebook is:
    # fixed acidity, volatile acidity, citric acid, residual sugar, chlorides,
    # free sulfur dioxide, total sulfur dioxide, density, pH, sulphates,
    # alcohol, type_white
    input_data = pd.DataFrame([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
        alcohol, type_white
    ]], columns=[
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol', 'type_white'
    ])

    st.header("3. Prediction")
    if st.button("Predict Quality"):
        # Scale the input data
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        # Display results
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.success(f"**This wine is classified as GOOD Quality!** 🌟 (Quality >= 7)")
        else:
            st.warning(f"**This wine is classified as BAD Quality.** 😥 (Quality < 7)")

        st.markdown(f"*(Probability of **Good Quality (1)**: {prediction_proba[0][1]:.2f})*")
        st.markdown(f"*(Probability of **Bad Quality (0)**: {prediction_proba[0][0]:.2f})*")
