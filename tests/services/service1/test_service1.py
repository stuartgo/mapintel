from fastapi import FastAPI
from fastapi.testclient import TestClient
from mapintel.services.service1.api.api_endpoint import app
import mlflow
from sklearn.feature_extraction.text import CountVectorizer
import base64
import cloudpickle

class Vectorizer(CountVectorizer):

    def predict(self,docs):
        return self.transform(docs).toarray()

client = TestClient(app)

def test_model():
    vectorizer=Vectorizer()
    vectorizer.fit(["test 123, this is a test sentence","Barack Obama was the president"])
    model_info = mlflow.sklearn.log_model(sk_model=vectorizer, artifact_path="model")
    sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
    response = client.post("http://localhost:8000/model",json={"model":base64.b64encode(cloudpickle.dumps(sklearn_pyfunc)).decode()})
    with open("./src/mapintel/services/service1/model.pickle","rb") as f:
        sklearn_pyfunc_alt = cloudpickle.load(f)
    assert sklearn_pyfunc.metadata==sklearn_pyfunc_alt.metadata
    assert response.status_code == 200

def test_vectorisation():
    docs=["this is a test 123"]
    response = client.post("http://localhost:8000/vectorisation",json={"docs":docs})
    assert response.status_code == 200
    assert isinstance(response.json()["embeddings"],list)
    assert isinstance(response.json()["embeddings"][0],list)
    assert isinstance(response.json()["embeddings"][0][0],(int,float))
    assert len(docs)==len(response.json()["embeddings"])
