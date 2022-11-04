from fastapi import FastAPI
from fastapi.testclient import TestClient
from mapintel.services.service1.api.api_endpoint import app
import mlflow
from sklearn.feature_extraction.text import CountVectorizer
import shutil,os


class Vectorizer(CountVectorizer):

    def predict(self,docs):
        return self.transform(docs).toarray()

client = TestClient(app)

def test_model():
    vectorizer=Vectorizer()
    vectorizer.fit(["test 123, this is a test sentence","Barack Obama was the president"])
    #removes data from previous tests
    if os.path.exists("./tests/services/service1/model/"):
        shutil.rmtree("./tests/services/service1/model/")
    if os.path.exists("./tests/services/service1/model_get/"):
        shutil.rmtree("./tests/services/service1/model_get/")
    if os.path.exists("./tests/services/service1/model.zip"):
        os.remove("./tests/services/service1/model.zip")
    if os.path.exists("./tests/services/service1/model_get.zip"):
        os.remove("./tests/services/service1/model_get.zip")
    #creates and saves mlflow model
    model_info = mlflow.sklearn.save_model(sk_model=vectorizer, path="./tests/services/service1/model/")
    #loads model from saved files
    sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri="./tests/services/service1/model/")
    #zips file
    shutil.make_archive("./tests/services/service1/model", 'zip', "./tests/services/service1/model/")
    with open("./tests/services/service1/model.zip", "rb") as f:
        response_post = client.post("http://localhost:8000/model",files={"file": ("filename", f, "application/zip")})
    #calls get request to fetch model that was just posted and stored
    response_get=client.get("http://localhost:8000/model",json={})
    with open("./tests/services/service1/model_get.zip", "wb") as f:
        f.write(response_get.content)
    shutil.unpack_archive("./tests/services/service1/model_get.zip", "./tests/services/service1/model_get/", "zip")
    sklearn_pyfunc_alt=mlflow.pyfunc.load_model(model_uri="./tests/services/service1/model_get/")
    assert sklearn_pyfunc.metadata==sklearn_pyfunc_alt.metadata
    assert response_post.status_code == 200
    assert response_get.status_code == 200

def test_vectorisation():
    docs=["test 123, this is a test sentence"]
    response = client.post("http://localhost:8000/model/vectors",json={"docs":docs})
    assert response.status_code == 200
    assert [[1, 0, 1, 0, 0, 1, 2, 0, 1, 0]]==response.json()["embeddings"]
    assert isinstance(response.json()["embeddings"],list)
    assert isinstance(response.json()["embeddings"][0],list)
    assert len(docs)==len(response.json()["embeddings"])

def test_info():
    response=client.get("http://localhost:8000/model/info",json={})
    metadata=response.json()["metadata"]
    assert metadata["flavors"]["python_function"]["loader_module"]=="mlflow.sklearn"


