import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
import newsapi

from fastapi import APIRouter, UploadFile, HTTPException
from pydantic import BaseModel

from api.custom_components.custom_pipe import CustomPipeline
from api.custom_components.text_cleaner import documents_cleaner
from api.config import PIPELINE_YAML_PATH, INDEXING_NU_PIPELINE_NAME

logger = logging.getLogger(__name__)
router = APIRouter()

try:
    INDEXING_PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME)
except KeyError:
    INDEXING_PIPELINE = None
    logger.info("Indexing Pipeline not found in the YAML configuration. News Upload API will not be available.")

# Getting NewsAPI key and establish connection to NewsAPI
load_dotenv("/run/secrets/dotenv-file")  # hard-coded: path to secret passed through docker-compose
NEWSAPIKEY = os.environ.get("NEWSAPIKEY")


class Response(BaseModel):
    status: str


@router.post("/news-upload", response_model=Response)
def news_upload():
    try:
        # Open NewsApi connection
        news_client = newsapi.NewsApiClient(api_key=NEWSAPIKEY)

        if not INDEXING_PIPELINE:
            raise HTTPException(status_code=501, detail="Indexing Pipeline is not configured.")
        
        document_batch = []

        # Get top_headlines
        logger.info("Getting top_headlines results.")
        for categ in newsapi.const.categories:
            request = news_client.get_top_headlines(
                category=categ, 
                language="en", 
                country="us", 
                page_size=100
                )
            # Adding category field and updating source field to every document in articles
            for r in request["articles"]:
                r.update({"category": categ, "source": r["source"]["id"]})
            # getting source id
            document_batch += (request["articles"])

        # Get everything using source categories
        logger.info("Getting everything results.")
        for categ in newsapi.const.categories:
            # get sources with specified category
            sources = set(map(lambda x: x["id"], news_client.get_sources(category=categ, language="en")["sources"]))
            request = news_client.get_everything(
                sources=",".join(sources), 
                from_param=datetime.today() - timedelta(1),
                to=datetime.today(), 
                page_size=100
                )
            # Guaranteeing requests only come from requested sources
            request_sources = set(map(lambda x: x["source"]['id'], request["articles"]))
            diff_sources = request_sources.difference(sources)
            request["articles"] = list(filter(lambda x: x["source"]["id"] not in diff_sources, request["articles"]))
            # Adding category field and updating source field to every document in articles
            for r in request["articles"]:
                r.update({"category": categ, "source": r["source"]["id"]})
            document_batch += request["articles"]

        # Cleaning the documents
        dicts = documents_cleaner(document_batch)

        # Embeds the documents in dicts and writes them to the document store
        logger.info("Running indexing pipeline.")
        INDEXING_PIPELINE.run(
            documents=dicts
        )
    except:
        return {'status': 'Fail'}
    else:
        return {'status': 'Success'}
