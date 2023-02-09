from .model import router as model_router
from fastapi import APIRouter

router = APIRouter(prefix="/dim_reduc")


router.include_router(model_router, tags=["Dimensionality reduction"])

