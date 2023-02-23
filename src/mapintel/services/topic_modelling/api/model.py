import shutil

import mlflow
from fastapi import APIRouter, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

router = APIRouter()


class Response(BaseModel):
    status: str


@router.post("/model", response_model=Response)
def model(file: bytes = File()):
    """Used to set model to be used for topic modelling.

    Args:
        file: Zipped file of model

    Returns:
        dict: Status
    """
    with open("./src/mapintel/services/topic_modelling/model.zip", "wb") as f:
        f.write(file)

    return {"status": "Success"}


@router.get("/model")
def model(request: BaseModel):
    """Used to fetch model being used.

    Returns:
        File: Zipped file of model
    """
    return FileResponse("./src/mapintel/services/topic_modelling/model.zip")


class Request_topic(BaseModel):
    docs: list[str]


class Response_topic(BaseModel):
    status: str
    topics: list[int]


@router.post("/", response_model=Response_topic)
def topic(request: Request_topic):
    """Predicts topic of docs sent.

    Args:
        List: List of strings/docs

    Returns:
        dict: Topics
    """
    shutil.unpack_archive(
        "./src/mapintel/services/topic_modelling/model.zip",
        "./src/mapintel/services/topic_modelling/model/",
        "zip",
    )
    model = mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/topic_modelling/model/")
    # print(model.predict(request.docs).tolist(),"shee")
    # print(type(model.predict(request.docs).tolist()[0]))
    return {
        "status": "Success",
        "topics": model.predict(request.docs),
    }

class Response_topic_names(BaseModel):
    status: str
    topic_names: list[str]

@router.get("/topic_names", response_model=Response_topic_names)
def topic_names(request: BaseModel):
    """Returns names of topics


    Returns:
        dict: Topics
    """
    shutil.unpack_archive(
        "./src/mapintel/services/topic_modelling/model.zip",
        "./src/mapintel/services/topic_modelling/model/",
        "zip",
    )
    model = mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/topic_modelling/model/")
    return {
        "status": "Success",
        "topic_names": model._model_impl.python_model.topics(),
    }


class Response_info(BaseModel):
    status: str
    metadata: dict


@router.get("/model/info", response_model=Response_info)
def vectorisation(request: BaseModel):
    """Info about model used.

    Returns:
        dict: Metadata about model
    """
    model = mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/topic_modelling/model/")
    metadata = model.metadata.to_dict()
    return {"status": "Success", "metadata": metadata}  # np array not serialisable so must be turned to list
