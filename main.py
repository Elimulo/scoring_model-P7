from typing import Union
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load model from MLflow
model_uri = "models:/LightGBM with SMOTE and RUS/2"  # Adjust this URI according to your MLflow model path
model = mlflow.sklearn.load_model(model_uri)

# Define request body structure
class PredictionRequest(BaseModel):
    data: List[float]

@app.post("/predict")
async def predict(request: PredictionRequest):
    prediction = model.predict([request.data])[0]
    return {"prediction": int(prediction)}