import base64

from typing import List,Any
from fastapi import APIRouter,File
from pydantic import BaseModel
from fastapi.responses import FileResponse
import shutil
import mlflow
router = APIRouter()


class Response(BaseModel):
    status: str


@router.post("/model", response_model=Response)
def model(file: bytes = File()):
    with open("./src/mapintel/services/service3/model.zip", "wb") as f:
        f.write(file)

    return {"status": "Success"}


@router.get("/model")
def model(request: BaseModel):
    return FileResponse("./src/mapintel/services/service3/model.zip")


class Request_topic(BaseModel):
    docs: List[str]



class Response_topic(BaseModel):
    status: str
    topics: List[int]



@router.post("/", response_model=Response_topic)
def topic(request: Request_topic):
    shutil.unpack_archive("./src/mapintel/services/service3/model.zip", "./src/mapintel/services/service3/model/", "zip")
    model=mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/service3/model/")
    # print(model.predict(request.docs).tolist(),"shee")
    # print(type(model.predict(request.docs).tolist()[0]))
    return {
        "status": "Success",
        "topics": model.predict(request.docs),
    } 






class Response_info(BaseModel):
    status: str
    metadata:dict




@router.get("/model/info", response_model=Response_info)
def vectorisation(request: BaseModel):
    model=mlflow.pyfunc.load_model(model_uri="./src/mapintel/services/service3/model/")
    metadata=model.metadata.to_dict()
    return {
        "status": "Success",
        "metadata":metadata
            }  # np array not serialisable so must be turned to list
