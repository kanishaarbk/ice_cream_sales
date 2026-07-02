import streamlit as st
import pickle
import numpy as np

# Load Polynomial Transformer
with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)

# Load Trained Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰",
    layout="centered"
)

# Title
st.title("💰 Salary Prediction")
st.write("Enter the employee level to predict the salary.")

# Input
level = st.number_input(
    "Employee Level",
    min_value=1.0,
    max_value=10.0,
    value=6.5,
    step=0.1
)
st.write(model.predict(poly.transform([[5]])))
# Prediction
if st.button("Predict Salary"):

    # Convert input to numpy array
    input_data = np.array([[level]])

    # Polynomial Features
    input_poly = poly.transform(input_data)

    # Predict
    prediction = model.predict(input_poly)

    # Display Result
    st.success(f"💰 Predicted Salary: ₹{prediction[0]:,.2f}")
