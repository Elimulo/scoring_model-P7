import pytest
import mlflow
import numpy as np

# Set the remote tracking URI (MLflow server)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Example model URI
model_uri = 'runs:/f8d1df805a2a4209b7ec662c6ab1dc00/lightgbm_model_final'

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
    
    input_data = [263698,2,0.0,2,0.0,127039.5,114750.0,0.0,-1397.4615384615386,0.0,False,664636.4619230769,0.0,1.0,11.0,True,0.0,False,False,46080.0,0.0,118165.5,0.0,0,
                  0.0,221400.0,22.0,0.0,0,3483.0,173880.0,False,9814.5,False,False,False,0.0,False,False,0.0,0.0,False,False,15.0,0.0,10.125,0.0,False,0.125,0.0,-788.25,
                  0.0,0.0,False,True,0.0,False,False,-10916,10.0,False,0.0,0.0,0.0,True,0.0,0.0,False,0.0,247500.0,1,48280.5,12092.104285714286,0.0,0.0,False,119084.625,
                  0.0,0.0,False,False,False,-438.0,0.0,False,False,6038.595,False,-3534,True,False,0.0,-65.0,0.0,False,False,0.0,0.0,324801.225,False,0.0,0.0,False,0.5,
                  False,0.0,0.0,175261.5,False,0.0,False,2.0,0.0,False,False,0.0,17550.0,0.0,1.0,False,0.0,False,16309.935,0.6666666666666666,-810.0,0.0,0.0,0,False,0.0,
                  True,False,0.0,-293.0,0.2,36290.025,0.9752851711026616,0.0,False,False,0.015221,0.0,0.0,0.0,0.0,-1502.0,2.0,110700.0,False,False,False,0.0,
                  0.1428571428571428,-4.0,0.0,0.0,0.0,False,0.0,0.0,0.0,8640274.004999999,0.0,0.0,False,0,False,0.1111111111111111,0.0,0,False,0.0,0.0,0,0,
                  0.2857142857142857,-2710.0,False,False,0.0,0.0,0.1111111111111111,0,0,0.0,0.0,0,False,0.0,0.0,0.0,130372.93125,210465.0,0,0,0.5,False,0.0,0.0,0.0,0.0,
                  False,0.0,1.0,0.0,0.0,True,False,-532.0,0.0,0.0,0.0,0,0,1.0,0.0,0.0709825368811565,5850.0,0.0,18081.18,0.0,15.0,False,0.0,0.0,0.0,0.0,0.0,
                  0.6153846153846154,247500.0,12.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9713732472900456,False,0.0,False,False,0,0.0,106834.5,0.0,0.0,False,25.0,0.0,1,0.0,False,
                  0.0,0.0,0.0,0.3333333333333333,False,0.5,False,False,False,0.0,0.0396545454545454,0.0,0.0,722091.7390909091,False,0.0357142857142857,0.0769230769230769,
                  -944.1538461538462,False,-1487.0,False,2323.575,False,False,False,0.0,False,0.0,13.0,False,0.0,0.0,False,False,0.0,0.0,False,False,36.0,0.0,
                  0.0742030047636496,0.8945454545454545,False,0.0651575783195063,1.0,0.0,2480.923076923077,0.0,False,0.0,False,45000.0,5517.0,0,0,0.2857142857142857,
                  False,0.0,0.25,0.0,3354.75,0.0,False,-806.5657754010695,0.0,0.0443292682926829,0.0,0.0,0.25,0.0,0.0,0.0,0.0,False,186671.25,0.0,True,False,False,False,
                  0.0,0.2,0.0,1,0.3846153846153846,225000.0,0.9166666666666666,24.638298261174413,False,0.0,23188.5,False,0,0.0,0.0,0.0,7868929.5,0,-304.0,0.0,0.0,
                  334393.2,False,False,0,False,0.0,False,-942.0,267.0,0.0,0.25,0.0,7943009.13,False,False,0.0,False,1.0313090640151203,0.0,0,0.0,0.0,12.76923076923077,
                  0.0,False,1.0,0.0,1,True,12.0,49950.0,False,0.0,0.5,0.6666666666666666,0.8984493637044166,22.0,0.1042730008990579,0.0,11476.271250000002,0.0,False,
                  0.0,0.0,False,False,False,False,0.0,False,False,0,-2891.0,0.0,0.0,False,-21233.0,0,0.0,False,0.0,0.7777777777777778,0.5873646392985845,0.0,0.0,0.0,0.0,
                  False,0.6925590674998008,True,0.0,0.0,0.75,0,0.0,0.6923076923076923,0.0,0,0.0,0.0,0.0,1,0,0.3,0,0.0,0.0,0.0,False,False,0.0,1.0423496353783368,0.0,0.0,1,
                  False,0.2307692307692307,0.2,12404.115,0.0,0.0,30.0,False,0.0,0.0,False,0.0,0.7142857142857143,0,False,True,0.0,False,False,7875000.0,False,5750.775,13,
                  False,False,1.0,-3.0,False,37035.4725,0.0,12726.201315789473,-28.571428571428573,0.0,0.0769230769230769,0.2,0.0,0,31118.0,0.0,0.0,False,0.0,True,0.0,0,
                  0.0,False,0.1048607239885971,0.0,212493.6,False,0.0,False,0.9044993450513296,False,0.0,31.0]  
    prediction = loaded_model.predict(input_data)

    # Assert prediction shape or values depending on model output
    assert prediction is not None, "Prediction should not be None"
    assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
    assert prediction.shape[0] == 1, f"Expected 1 prediction, but got {prediction.shape[0]}"
    # Add more specific checks based on expected output (e.g., value range)

