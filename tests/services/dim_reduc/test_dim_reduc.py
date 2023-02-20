import os
import shutil

import mlflow
from fastapi.testclient import TestClient
from mapintel.services.dim_reduc.api.api_endpoint import app
from sklearn.decomposition import PCA


class DimReduction(PCA):
    def predict(self, docs):
        return self.transform(docs)


client = TestClient(app)


def test_model():
    reducer = DimReduction(n_components=2, random_state=8)
    reducer.fit([[-1, -1, 3, 2, 5, 2, 6, 32, 6, 2], [5, 8, 4, 3, 8, 3, 7, 3, 7, 9]])
    # removes data from previous tests
    if os.path.exists("./tests/services/dim_reduc/model/"):
        shutil.rmtree("./tests/services/dim_reduc/model/")
    if os.path.exists("./tests/services/dim_reduc/model_get/"):
        shutil.rmtree("./tests/services/dim_reduc/model_get/")
    if os.path.exists("./tests/services/dim_reduc/model.zip"):
        os.remove("./tests/services/dim_reduc/model.zip")
    if os.path.exists("./tests/services/dim_reduc/model_get.zip"):
        os.remove("./tests/services/dim_reduc/model_get.zip")
    # creates and saves mlflow model
    mlflow.sklearn.save_model(
        sk_model=reducer,
        path="./tests/services/dim_reduc/model/",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
    )
    # loads model from saved files
    sklearn_pyfunc = mlflow.pyfunc.load_model(model_uri="./tests/services/dim_reduc/model/")
    # zips file
    shutil.make_archive("./tests/services/dim_reduc/model", 'zip', "./tests/services/dim_reduc/model/")
    with open("./tests/services/dim_reduc/model.zip", "rb") as f:
        response_post = client.post(
            "http://localhost:8000/dim_reduc/model",
            files={"file": ("filename", f, "application/zip")},
        )
    # calls get request to fetch model that was just posted and stored
    response_get = client.get("http://localhost:8000/dim_reduc/model", json={})
    with open("./tests/services/dim_reduc/model_get.zip", "wb") as f:
        f.write(response_get.content)
    shutil.unpack_archive("./tests/services/dim_reduc/model_get.zip", "./tests/services/dim_reduc/model_get/", "zip")
    sklearn_pyfunc_alt = mlflow.pyfunc.load_model(model_uri="./tests/services/dim_reduc/model_get/")
    assert sklearn_pyfunc.metadata == sklearn_pyfunc_alt.metadata
    assert response_post.status_code == 200
    assert response_get.status_code == 200


def test_vectorisation():
    docs = [[-1, -1, 3, 2, 5, 5, 7, 4, 8, 5]]
    response = client.post("http://localhost:8000/model/vectors", json={"docs": docs})
    assert response.status_code == 200
    assert [[-10.280695658958365, -5.019751578009658]] == response.json()["embeddings"]
    assert isinstance(response.json()["embeddings"], list)
    assert isinstance(response.json()["embeddings"][0], list)
    assert len(docs) == len(response.json()["embeddings"])


def test_info():
    response = client.get("http://localhost:8000/model/info", json={})
    metadata = response.json()["metadata"]
    assert metadata["flavors"]["python_function"]["loader_module"] == "mlflow.sklearn"
