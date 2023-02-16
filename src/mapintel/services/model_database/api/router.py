from fastapi import APIRouter

from .model import router as model_router

router = APIRouter(prefix="/models")

router.include_router(model_router, tags=["Model database"])
