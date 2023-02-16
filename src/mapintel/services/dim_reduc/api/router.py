from fastapi import APIRouter

from .model import router as model_router

router = APIRouter(prefix="/dim_reduc")


router.include_router(model_router, tags=["Dimensionality reduction"])
