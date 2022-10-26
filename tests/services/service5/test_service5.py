from fastapi import FastAPI
from fastapi.testclient import TestClient
from mapintel.services.service5.api.api_endpoint import app
import mlflow
import base64
import cloudpickle



client = TestClient(app)

def test_get_model():

    response = client.post("http://localhost:8001/get_model",json={"model_name":"sklearn_vectorizer"})
    model=cloudpickle.loads(base64.b64decode(response.json()["model"]))
    assert isinstance(model,mlflow.pyfunc.PyFuncModel)
    assert response.status_code == 200

def test_available_models():

    response = client.post("http://localhost:8001/available_models",json={})
    assert response.json()["models"][0]=="sklearn_vectorizer"
    assert response.status_code == 200

