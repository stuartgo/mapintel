import model
from fastapi import APIRouter

router = APIRouter()

router.include_router(model.router, tags=["model"])
