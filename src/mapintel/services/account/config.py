from functools import lru_cache
import os
import logging

from pydantic import BaseSettings, AnyUrl


log = logging.getLogger("uvicorn")


class Settings(BaseSettings):

    db_url: AnyUrl = os.getenv('DATABASE_URL')


@lru_cache
def get_settings() -> BaseSettings:
    log.info("Loading db service config settings from the environment")
    return Settings
