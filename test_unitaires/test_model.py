import pytest
import mlflow
import numpy as np

# Set the remote tracking URI (MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Example model URI
model_uri = 'runs:/ec3f9fc05b94404eb297e518416d9ea6/LightGBM_Final'

# This function is used to load the model from the remote server
@pytest.fixture(scope='module')
def loaded_model():
    model = mlflow.pyfunc.load_model(model_uri)  # Load model from the MLflow server using URI
    return model

# Test 1: Check if model loads successfully
def test_model_loads(loaded_model):
    assert loaded_model is not None, "Model did not load correctly"

# Test 2: Check if model prediction works 
def test_model_prediction(loaded_model):
    
    input_data = np.array([np.int64(2), np.float64(-395.75), np.float64(148746.825), np.float64(9.0), np.float64(56533.5), np.float64(12.1), np.True_, np.int64(-13099), np.float64(713889.0), 
                           np.float64(10494.135), np.float64(-109.0), np.float64(4557.195), np.int64(-5379), np.float64(-32.0), np.False_, np.float64(41014.755), np.float64(-848.0),
                           np.float64(0.0), np.float64(-25.0), np.float64(4557.195), np.float64(0.0), np.float64(0.022625), np.float64(-300.0), np.float64(50625.0), np.float64(0.0), 
                           np.float64(-1.0), np.float64(594987.3), np.int64(0), np.float64(-1374.0), np.float64(4266.0), np.float64(0.0), np.float64(455.7195), np.float64(661500.0), 
                           np.float64(12.0), np.float64(4557.195), np.float64(0.996594719195306), np.float64(0.079190882616205), np.float64(55381.5), np.float64(0.2), np.float64(-198.5), 
                           np.float64(671.22), np.float64(0.0647377662416978), np.float64(0.2836575433996041), np.float64(-46.0), np.float64(4266.0), np.float64(-154.9), 
                           np.float64(0.2791777777777778), np.float64(1.0), np.float64(-109.0), np.float64(0.0), np.float64(-204.5), np.float64(121.0), np.float64(110763.0), np.float64(15.0),
                           np.float64(0.9), np.float64(39.0), np.int64(1), np.float64(12.0), np.float64(0.993189438390612), np.float64(10494.135), np.float64(-935.0), np.float64(-1549.0), 
                           np.float64(0.2918181951717679), np.float64(0.3425288720742255), np.float64(3885.975), np.float64(4101.4755), np.float64(214740.0), np.float64(4557.195), 
                           np.float64(-300.0), np.float64(0.0), np.float64(-5.0), np.float64(340.0), np.True_, np.float64(0.993189438390612), np.float64(39.0)] )
    
    prediction = loaded_model.predict(input_data.reshape(1, -1))

    # Assert prediction shape or values depending on model output
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert prediction.shape[0] == 1, f"Expected 1 prediction, but got {prediction.shape[0]}"
    # Add more specific checks based on expected output (e.g., value range)

