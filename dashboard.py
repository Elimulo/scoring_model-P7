import streamlit as st
import requests
import pandas as pd

st.title("Prédictions avec le modèle LightGBM")

# Formulaire d'entrée
st.write("Veuillez entrer les caractéristiques pour faire une prédiction :")
feature_input = st.text_input("Caractéristiques (séparées par des virgules, ex: 1.2,3.4,5.6)")

if st.button("Prédire"):
    # Préparer les données pour l'API Flask
    features = [[float(x) for x in feature_input.split(",")]]
    input_data = {"features": features}
    
    # Appeler l'API Flask
    response = requests.post("http://<votre-flask-app-url>/predict", json=input_data)
    
    if response.status_code == 200:
        predictions = response.json()["predictions"]
        st.success(f"Prédictions : {predictions}")
    else:
        st.error("Erreur lors de la requête à l'API.")