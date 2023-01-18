
from .database import router as database_router
from fastapi import APIRouter

router = APIRouter()

router.include_router(database_router, tags=["Document database"])
