import os
import platform
import shutil
from pathlib import Path

import cloudpickle
import mlflow
import mlflow.pyfunc
import numpy as np
from fastapi.testclient import TestClient
from mapintel.services.service3.api.api_endpoint import app
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


class LDAWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.lda_model = cloudpickle.load(Path(context.artifacts["lda_model"]).open(mode="rb"))

    def predict(self, context, model_input):
        predictions = self.lda_model.transform(model_input)
        return [np.argmax(x) for x in predictions]


client = TestClient(app)


def test_model():
    model = Pipeline(
        [
            ("tf", CountVectorizer(max_df=0.95, min_df=2, max_features=50, stop_words='english')),
            ("LDA", LDA(random_state=8)),
        ],
    )
    model.fit(
        [
            "Obama was the president",
            "Biden is the president",
            "Greece is warmer than Norway",
            "Norway is cold",
            "Portugal is far west",
            "The president is elected",
            "Norway is far north",
            "Greece has many islands",
            "Europe is large",
            "The president has a 4 year term",
        ],
    )
    artifacts = {"lda_model": "./tests/services/service3/model.pkl"}
    cloudpickle.dump(model, Path(artifacts["lda_model"]).open(mode="wb"))
    # removes data from previous tests
    if os.path.exists("./tests/services/service3/model/"):
        shutil.rmtree("./tests/services/service3/model/")
    if os.path.exists("./tests/services/service3/model_get/"):
        shutil.rmtree("./tests/services/service3/model_get/")
    if os.path.exists("./tests/services/service3/model.zip"):
        os.remove("./tests/services/service3/model.zip")
    if os.path.exists("./tests/services/service3/model_get.zip"):
        os.remove("./tests/services/service3/model_get.zip")
    # creates and saves mlflow model
    mlflow_pyfunc_model_path = "./tests/services/service3/model/"
    conda_env = {
        'channels': ['defaults'],
        'dependencies': [
            f'python={platform.python_version()}',
            'pip',
            {
                'pip': [
                    'mlflow',
                    f'cloudpickle=={cloudpickle.__version__}',
                    'scikit-learn=={}'.format("1.1.2"),
                ],
            },
        ],
        'name': 'lda_env',
    }
    mlflow.pyfunc.save_model(
        path=mlflow_pyfunc_model_path,
        python_model=LDAWrapper(),
        artifacts=artifacts,
        conda_env=conda_env,
    )
    # loads model from saved files
    mlflow_model = mlflow.pyfunc.load_model("./tests/services/service3/model/")
    # zips file
    shutil.make_archive("./tests/services/service3/model", 'zip', "./tests/services/service3/model/")
    with open("./tests/services/service3/model.zip", "rb") as f:
        response_post = client.post("http://localhost:8000/model", files={"file": ("filename", f, "application/zip")})
    # calls get request to fetch model that was just posted and stored
    response_get = client.get("http://localhost:8000/model", json={})
    with open("./tests/services/service3/model_get.zip", "wb") as f:
        f.write(response_get.content)
    shutil.unpack_archive("./tests/services/service3/model_get.zip", "./tests/services/service3/model_get/", "zip")
    mlflow_model_get = mlflow.pyfunc.load_model("./tests/services/service3/model_get/")
    assert mlflow_model == mlflow_model_get
    assert response_post.status_code == 200
    assert response_get.status_code == 200


def test_topic():
    docs = ["Norway", "president"]
    response = client.post("http://localhost:8000/model/topic", json={"docs": docs})
    assert [3, 8] == response.json()["topics"]


def test_info():
    response = client.get("http://localhost:8000/model/info", json={})
    metadata = response.json()["metadata"]
    assert "lda_model" in metadata["flavors"]["python_function"]["artifacts"]
