from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)


mlflow.set_tracking_uri("http://127.0.0.1:5000")  

model_uri = 'runs:/f18925492b974af4aaae5fe4d38cd151/lightgbm_model_final'
model = mlflow.pyfunc.load_model(model_uri)

def predict_default(client_data, threshold=0.52):
    # Predict the probability of default
    default_probability = model.predict(pd.DataFrame([client_data]))[0]
    
    # Classification based on the threshold
    client_class = "Accepted" if default_probability < threshold else "Rejected"
    
    return default_probability, client_class

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request
    client_data = request.json['client_data']
    probabilité_defaut, classe = predict_default(client_data)
    
    # Return the prediction as JSON
    return jsonify({'probabilité_defaut': probabilité_defaut, 'classe': classe})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
