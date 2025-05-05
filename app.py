import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, preprocessing pipeline, and training columns
best_model = joblib.load("best_model.pkl")
preprocessing_pipeline = joblib.load("preprocessing_pipeline.pkl")
model_columns = joblib.load("model_columns.pkl")

# Extract unique place names from model columns
place_options = [col.replace("Place_", "") for col in model_columns if col.startswith("Place_")]
place_options.insert(0, "Other")  # For default/fallback

# Streamlit UI
st.title("🏠 House Price Prediction App")
st.write("Enter the details below to predict the house price (in Crores).")

# User Inputs
area = st.number_input("📏 Area (in cents)", min_value=0.0, value=0.0, step=0.1)
place = st.selectbox("📍 Place", options=place_options)

# Handle case when area is 0
if area == 0:
    st.warning("Area is 0. Prediction might not be meaningful.")
    st.success("💰 Estimated Price: ₹ 0.00 Cr")
else:
    # Build user input into DataFrame
    input_df = pd.DataFrame({
        "Area (in cents)": [area],
        "Place": [place]
    })

    # cOne-hot encode place like training data
    input_df_encoded = pd.get_dummies(input_df, columns=["Place"], prefix="Place")

    # Add missing columns (set to 0) to match training
    for col in model_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Reorder columns to match training
    input_df_encoded = input_df_encoded[model_columns]

    # Apply preprocessing pipeline
    input_prepared = preprocessing_pipeline.transform(input_df_encoded)

    # Make prediction
    prediction = best_model.predict(input_prepared)[0]

    # Display result
    st.success(f"💰 Estimated Price: ₹ {prediction:.2f} Cr")
