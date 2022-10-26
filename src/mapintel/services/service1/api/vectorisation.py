import pickle
from typing import Any, List, Union

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Request(BaseModel):
    docs: List[str]



class Response(BaseModel):
    status: str
    embeddings: Any

    class Config:
        arbitrary_types_allowed = True


@router.post("/vectorisation", response_model=Response)
def vectorisation(request: Request):
    with open("./src/mapintel/services/service1/model.pickle", "rb") as f:

        model = pickle.load(f)
    return {
        "status": "Success",
        "embeddings": model.predict(request.docs).tolist(),
    }  # np array not serialisable so must be turned to list
