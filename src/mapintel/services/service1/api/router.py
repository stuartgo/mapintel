import model
import vectorisation
from fastapi import APIRouter

router = APIRouter()

router.include_router(model.router, tags=["model"])
router.include_router(vectorisation.router, tags=["vectorisation"])
