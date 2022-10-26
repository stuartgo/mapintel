from fastapi import FastAPI
from fastapi.testclient import TestClient
from mapintel.services.service5.api.api_endpoint import app
import mlflow
import shutil




client = TestClient(app)

def test_get_model():
    response = client.post("http://localhost:8001/get_model",json={"model_name":"sklearn_vectorizer"})
    with open("./tests/services/service5/model.zip","wb") as f:
        f.write(response.content)
    shutil.unpack_archive("./tests/services/service5/model.zip", "./tests/services/service5/model/", "zip")
    model=mlflow.pyfunc.load_model("./tests/services/service5/model/")

    assert isinstance(model,mlflow.pyfunc.PyFuncModel)
    assert response.status_code == 200

def test_available_models():

    response = client.post("http://localhost:8001/available_models",json={})
    assert response.json()["models"][0]=="sklearn_vectorizer"
    assert response.status_code == 200
