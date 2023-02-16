from fastapi import APIRouter

from .model import router as model_router

router = APIRouter(prefix="/topic")


router.include_router(model_router, tags=["Topic modelling"])
