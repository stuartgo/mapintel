from fastapi import APIRouter

from .database import router as database_router

router = APIRouter()

router.include_router(database_router, tags=["Document database"])
