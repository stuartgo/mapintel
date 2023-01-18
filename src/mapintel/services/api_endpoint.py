from fastapi import Depends, FastAPI
import uvicorn

from .service1.api.router import router  as s1_router
from .service2.api.router import router  as s2_router
from .service3.api.router import router  as s3_router
from .service4.api.router import router  as s4_router
from .service5.api.router import router  as s5_router
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
    uvicorn.run("mapintel.services.api_endpoint:app", host="0.0.0.0", port=30000,reload=True)