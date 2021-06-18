from fastapi import APIRouter

from api.controller import (
    file_upload, 
    news_upload, 
    search,
    feedback, 
    umap
)

router = APIRouter()

router.include_router(search.router, tags=["search"])
router.include_router(feedback.router, tags=["feedback"])
router.include_router(file_upload.router, tags=["file-upload"])
router.include_router(news_upload.router, tags=["news-upload"])
router.include_router(umap.router, tags=["umap"])
