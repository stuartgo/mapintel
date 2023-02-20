import os
import shutil

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class Request(BaseModel):
    model_name: str


class Response_aval_models(BaseModel):
    models: list[str]
    status: str


@router.post("/get_model")
def model(request: Request):
    """Fetches model from database.

    Args:
        string: Name of model. Alternatives can be seen by calling /available models

    Returns:
        file: Zipped model
    """
    shutil.make_archive(
        "./src/mapintel/services/model_database/model",
        'zip',
        "./src/mapintel/services/model_database/models/" + request.model_name,
    )
    return FileResponse("./src/mapintel/services/model_database/model.zip")


@router.post("/available_models", response_model=Response_aval_models)
def model(request: BaseModel):
    """Returns models stored in database.

    Returns:
        List: List of model names
    """
    return {
        "status": "Success",
        "models": [
            name
            for name in os.listdir("./src/mapintel/services/model_database/models/")
            if os.path.isdir(os.path.join("./src/mapintel/services/model_database/models/", name))
        ],
    }
