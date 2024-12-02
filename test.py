import mlflow
from mlflow.tracking import MlflowClient

# Connect to the MLflow server (make sure the tracking URI is correct)
client = MlflowClient(tracking_uri="http://127.0.0.1:5000")

# List all registered models
models = client.get_registered_model('LightGBM_final ')

# Print details of each model and its versions
for model in models:
    print(f"Model: {model.name}")
    
    # List all versions of the model
    versions = client.get_model_versions(model.name)
    for version in versions:
        print(f"  Version: {version.version}, Status: {version.status}")