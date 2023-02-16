from fastapi import FastAPI, Depends

from vectorisation.config import get_settings, Settings

app = FastAPI()


@app.get('/index')
async def index(settings: Settings = Depends(get_settings)):
    return {
        'msg': 'Hello!',
        'environment': settings.environment,
        'testing': settings.testing,
    }