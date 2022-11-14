import logging

import uvicorn
from fastapi import FastAPI

from .router import router as api_router

from starlette.middleware.cors import CORSMiddleware

logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p")
logger = logging.getLogger(__name__)


def get_application() -> FastAPI:
    application = FastAPI(title="Mapintel-API", debug=True, version="0.1", root_path="/")

    # This middleware enables allow all cross-domain requests to the API from a browser. For production
    # deployments, it could be made more restrictive.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    return application


app = get_application()

logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
