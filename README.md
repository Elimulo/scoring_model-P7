# Projet P7 - API de Prédiction de Défaut de Crédit

## Objectif du projet

L'objectif de ce projet est de créer une API Flask qui permet de prédire la probabilité qu'un client fasse défaut sur un prêt en fonction de ses caractéristiques. Le modèle utilisé pour les prédictions est un modèle LightGBM, entraîné et enregistré avec MLflow. L'API expose une route qui permet d'envoyer les données d'un client et de récupérer une prédiction quant à son acceptation ou son rejet, ainsi que la probabilité associée.

## Découpage des dossiers

Le projet est structuré de manière à séparer clairement les différentes fonctionnalités et composants :

- **flask_api/** : Ce dossier contient l'API Flask.
  - **api.py** : Le fichier principal contenant le code de l'API, qui charge le modèle et gère les requêtes.
  - **api_flask.ipynb** : Un notebook pour l'expérimentation et l'évaluation du modèle.
  
- **tests_unitaires/** : Ce dossier contient les tests unitaires pour le projet.
  - **tests_model.py** : Les tests unitaires pour vérifier le bon fonctionnement du modèle.

- **analyse_explo.ipynb** : Notebook issu du kaggle d'exploration des données

- **modelisations.ipynb** : Notebook des tests de modélisation

- **run_model.py** : Script qui charge et entraîne le modèle LightGBM.
  
- **data_drift_report.html** : Data drift report

## Installation

### Prérequis

Avant de démarrer l'application, assurez-vous d'avoir les éléments suivants installés sur votre machine :

- Python 3.10.6
- pip

### Installation des dépendances

Installez les dépendances nécessaires en exécutant la commande suivante :

```bash
pip install -r requirements.txt


###  Strucutre des fichiers:

p7/
│
├── flask_api/
│   ├── api.py                 # Code de l'API Flask
│   ├── api_flask.ipynb        # Notebook pour la prédiction d'un clien via l'api
│   ├──requirements.txt           # Liste des packages requis pour l'application
│
├── tests_unitaires/
│   ├── tests_model.py         # Tests unitaires du modèle
│
├── analyse_explo.ipynb        # Notebook issu du kaggle d'exploration des données
├── modelisations.ipynb        # Notebook des tests de modélisation
├── requirements.txt           # Liste des packages requis pour l'application
├── run_model.py               # Script pour entraîner le modèle
└── data_drift_report.html     # Data drift report