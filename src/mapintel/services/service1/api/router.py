from .model import router as model_router
from . vectorisation import router as vectorisation_router
from fastapi import APIRouter

router = APIRouter()


router.include_router(model_router, tags=["model"])
router.include_router(vectorisation_router, tags=["vectorisation"])

