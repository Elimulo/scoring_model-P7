import mlflow
import mlflow.pyfunc

# Set the remote tracking URI (MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Example model URI
model_uri = 'runs:/f8d1df805a2a4209b7ec662c6ab1dc00/lightgbm_model_final' 

# Load the model from the server
model = mlflow.pyfunc.load_model(model_uri)

# Use the model to make predictions
predictions = model.predict(your_input_data)  # Replace `your_input_data` with actual data