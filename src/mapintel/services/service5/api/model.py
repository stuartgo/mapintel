import base64
import os
from typing import List

import cloudpickle
import mlflow
from fastapi import APIRouter
from pydantic import BaseModel
import shutil
from fastapi.responses import FileResponse

router = APIRouter()


class Request(BaseModel):
    model_name: str


class Response_aval_models(BaseModel):
    models: List[str]
    status: str


@router.post("/get_model")
def model(request: Request):
    shutil.make_archive("./src/mapintel/services/service5/model", 'zip', "./src/mapintel/services/service5/models/" + request.model_name)
    return FileResponse("./src/mapintel/services/service5/model.zip")


@router.post("/available_models", response_model=Response_aval_models)
def model(request: BaseModel):
    return {
        "status": "Success",
        "models": [name for name in os.listdir("./src/mapintel/services/service5/models/") if os.path.isdir(os.path.join("./src/mapintel/services/service5/models/", name))],
    }
