import streamlit as st
import pickle
import pandas as pd

# Load Polynomial Transformer
with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)

# Load Model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(
    page_title="Salary Prediction",
    page_icon="💰"
)

st.title("💰 Salary Prediction")
st.write("Enter the employee level to predict salary.")

level = st.number_input(
    "Employee Level",
    min_value=1.0,
    max_value=10.0,
    value=6.5,
    step=0.1
)

if st.button("Predict Salary"):

    # Create DataFrame with same column name as training
    input_df = pd.DataFrame({"Level": [level]})

    # Polynomial Transform
    input_poly = poly.transform(input_df)

    # Prediction
    prediction = model.predict(input_poly)

    st.success(f"💰 Predicted Salary: ₹{prediction[0]:,.2f}")
