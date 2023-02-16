
from .model import router as model_router
from fastapi import APIRouter

router = APIRouter(prefix="/models")

router.include_router(model_router, tags=["Model database"])
