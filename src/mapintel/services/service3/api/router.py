from .model import router as model_router
from fastapi import APIRouter

router = APIRouter(prefix="/topic")


router.include_router(model_router, tags=["Topic modelling"])

