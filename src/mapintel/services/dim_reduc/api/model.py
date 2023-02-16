import shutil
from typing import Any

import mlflow
from fastapi import APIRouter, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class Response(BaseModel):
    status: str


@router.post("/model", response_model=Response)
def model(file: bytes = File()):
    """Used to set model to be used for dimensionality reduction.

    Args:
        file: mlflow zipped model

    Returns:
        dict: status
    """
    with open("./src/mapintel/services/service2/model.zip", "wb") as f:
        f.write(file)

    return {"status": "Success"}


@router.get("/model")
def model(request: BaseModel):
    """Used to fetch model currently set for dimentionality reduction.

    Returns:
        bytes: zipped model
    """
    return FileResponse("./src/mapintel/services/service2/model.zip")


class Request_vectors(BaseModel):
    docs: list[list[float]]


class Response_vectors(BaseModel):
    status: str
    embeddings: Any

    class Config:
        arbitrary_types_allowed = True


@router.post("/vectors", response_model=Response_vectors)
def vectorisation(request: Request_vectors):
    """Transforms n dimensional vectors to two dimensions.

    Args:
        list: List of embeddings

    Returns:
        dict: returns 2d vectors
    """
    shutil.unpack_archive(
        "./src/mapintel/services/service2/model.zip",
        "./src/mapintel/services/service2/model/",
        "zip",
    )
    model = mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/service2/model/")
    return {
        "status": "Success",
        "embeddings": model.predict(request.docs).tolist(),
    }  # np array not serialisable so must be turned to list


class Response_info(BaseModel):
    status: str
    metadata: dict


@router.get("/model/info", response_model=Response_info)
def model_info(request: BaseModel):
    """Returns info about model currently in use
    Returns:
        dict: Metadata with info about model.
    """
    model = mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/service1/model/")
    metadata = model.metadata.to_dict()
    return {"status": "Success", "metadata": metadata}  # np array not serialisable so must be turned to list
