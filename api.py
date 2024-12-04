from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# Set tracking URI for MLflow (adjust it if necessary)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load the LightGBM model via mlflow.sklearn (since LightGBM is compatible with scikit-learn)
model_uri = 'runs:/65a069f4d14f4a37a59045771ae2a7f2/LightGBM_final'
model = mlflow.sklearn.load_model(model_uri)

# Helper function to predict default with probability
def predict_default(client_data, threshold=0.54):
    # Convert client_data into a pandas DataFrame
    client_df = pd.DataFrame([client_data])

    # Get probabilities using predict_proba method (for binary classification)
    prob = model.predict_proba(client_df)  # Returns probabilities for both classes

    # Assuming binary classification, prob will return probabilities for both class 0 and class 1
    default_probability = prob[0][1]  # Probability for the positive class (class 1)

    # Classify based on threshold
    client_class = "Accepted" if default_probability < threshold else "Rejected"

    return default_probability, client_class

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    client_data = request.json['client_data']

    # Get prediction probability and classification result
    probabilité_defaut, classe = predict_default(client_data)

    # Return the prediction as JSON
    return jsonify({'probabilité_defaut': probabilité_defaut, 'classe': classe})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
    

