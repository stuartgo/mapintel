import base64

import cloudpickle
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Request(BaseModel):
    model: str


class Response(BaseModel):
    status: str


@router.post("/model", response_model=Response)
def model(request: Request):

    model = cloudpickle.loads(base64.b64decode(request.model))
    with open("./src/mapintel/services/service1/model.pickle", "wb") as f:
        cloudpickle.dump(model, f)
    return {"status": "Success"}
