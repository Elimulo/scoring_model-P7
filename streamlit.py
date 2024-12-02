import streamlit as st
import requests

st.title("Model Prediction")

user_input = st.text_input("Enter Input Data")
if st.button("Predict"):
    response = requests.post("https://<username>.pythonanywhere.com/predict", json={"data": user_input})
    st.write("Prediction:", response.json())
