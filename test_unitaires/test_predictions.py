import pytest
import requests

def test_prediction():
    url = "http://127.0.0.1:1234/invocations"
    payload = {"columns": ["feature1", "feature2"], "data": [[1, 2]]}
    response = requests.post(url, json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()
