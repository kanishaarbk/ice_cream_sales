import streamlit as st
import pickle

# Load transformer
with open("poly.pkl", "rb") as f:
    poly = pickle.load(f)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Prediction", page_icon="🍦")

st.title("🍦 Ice Cream Sales Prediction")
st.write("Enter the temperature to predict ice cream sales.")

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

if st.button("Predict Sales"):
    temp_poly = poly.transform([[temperature]])
    prediction = model.predict(temp_poly)

    st.success(f"🍦 Predicted Ice Cream Sales: {prediction[0]:.2f} units")
