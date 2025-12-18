import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open("best_model_random_forest.pkl", "rb") as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model files not found! Please run Insurance.py first to train and save models.")
        return None

model = load_model()

st.title("🏥 Insurance Premium Prediction")
st.write("Enter your personal details to predict your insurance premium.")

if model is not None:
    # User input fields
    age = st.number_input("Age", min_value=18, max_value=80, value=30)
    sex = st.radio("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1)
    children = st.selectbox("Number of Children", [0, 1, 2, 3, 4, 5])
    smoker = st.radio("Smoker", ["No", "Yes"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    # One-hot encoding for region
    region_northwest, region_southeast, region_southwest = 0, 0, 0
    if region == "northwest":
        region_northwest = 1
    elif region == "southeast":
        region_southeast = 1
    elif region == "southwest":
        region_southwest = 1
    # northeast is the reference category (all zeros)

    if st.button("Predict Insurance Premium"):
        # Create DataFrame
        input_data = pd.DataFrame({
            "age": [age],
            "sex": [1 if sex == "Female" else 0],
            "bmi": [bmi],
            "children": [children],
            "smoker": [1 if smoker == "Yes" else 0],
            "region_northwest": [region_northwest],
            "region_southeast": [region_southeast],
            "region_southwest": [region_southwest]
        })
        
        # Prediction
        predicted_premium = model.predict(input_data)[0]
        
        st.write(f"### Predicted Insurance Premium: ${predicted_premium:,.2f}")
        
        # Risk assessment
        if smoker == "Yes":
            st.warning("🚬 Smoking significantly increases your premium")
        if bmi > 30:
            st.warning("⚖️ High BMI may increase costs")
        if age > 50:
            st.info("👴 Age factor affects premium")
        
        if predicted_premium < 10000:
            st.success("✅ Your premium is in the LOW range.")
        elif predicted_premium < 20000:
            st.info("ℹ️ Your premium is in the MODERATE range.")
        else:
            st.error("⚠️ Your premium is in the HIGH range.")