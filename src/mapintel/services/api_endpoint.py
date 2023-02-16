import uvicorn
from fastapi import FastAPI

from .database.api.router import router as s4_router
from .dim_reduc.api.router import router as s2_router
from .model_database.api.router import router as s5_router
from .topic_modelling.api.router import router as s3_router
from .vectorization.api.router import router as s1_router

app = FastAPI()


app.include_router(s1_router)
app.include_router(s2_router)
app.include_router(s3_router)
app.include_router(s4_router)
app.include_router(s5_router)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}


if __name__ == "__main__":
    uvicorn.run("mapintel.services.api_endpoint:app", host="0.0.0.0", port=30000, reload=True)
