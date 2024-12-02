from flask import Flask, request, jsonify
import mlflow.pyfunc

app = Flask(__name__)

# Set the remote tracking URI (MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Example model URI
model_uri = 'runs:/f8d1df805a2a4209b7ec662c6ab1dc00/lightgbm_model_final' 

# Load the model from the server
model = mlflow.pyfunc.load_model(model_uri)

# Use the model to make predictions
predictions = model.predict(your_input_data)  # Replace `your_input_data` with actual data

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données JSON
    input_data = request.get_json()
    features = input_data["features"]  # Assurez-vous que c'est un tableau 2D [[x1, x2, ...], ...]
    
    # Faire des prédictions
    predictions = model.predict(features)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)