import streamlit as st
import requests
import numpy as np

# Function to call the Flask API
def call_prediction_api(client_data):
    url = "http://127.0.0.1:5000/predict"  # Your Flask API endpoint
    payload = {'client_data': client_data}
    response = requests.post(url, json=payload)
    return response.json()

# Streamlit UI
st.title("Prédiction de la probabilité de défaut d'un client")

# Input form for client data
client_data = []
client_data.append(st.number_input("Feature 1"))
client_data.append(st.number_input("Feature 2"))
client_data.append(st.number_input("Feature 3"))
# Repeat for all features required for the model

if st.button("Prédire"):
    # Call the API with user input
    result = call_prediction_api(client_data)
    
    # Display the result
    if result:
        st.write(f"Probabilité de défaut : {result['probabilité_defaut']:.2f}")
        st.write(f"Classe du client : {result['classe']}")
