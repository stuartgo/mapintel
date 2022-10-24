import base64
import os
from typing import List

import cloudpickle
import mlflow
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Request(BaseModel):
    model_name: str


class Response(BaseModel):
    status: str
    model: str


class Response_aval_models(BaseModel):
    models: List[str]
    status: str


@router.post("/get_model", response_model=Response)
def model(request: Request):
    model = mlflow.pyfunc.load_model(model_uri="./models/" + request.model_name)
    return {"status": "Success", "model": base64.b64encode(cloudpickle.dumps(model)).decode()}


@router.post("/available_models", response_model=Response_aval_models)
def model(request: BaseModel):
    return {
        "status": "Success",
        "models": [name for name in os.listdir("./models") if os.path.isdir(os.path.join("./models", name))],
    }
