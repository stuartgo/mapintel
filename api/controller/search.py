import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

from fastapi import APIRouter
from pydantic import BaseModel

from api.custom_components.custom_pipe import CustomPipeline
from api.config import PIPELINE_YAML_PATH, LOG_LEVEL, QUERY_PIPELINE_NAME
from api.controller.utils import RequestLimiter

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

router = APIRouter()


class Request(BaseModel):
    query: str
    filters: Optional[List[dict]] = None
    match: Optional[List[dict]] = None
    top_k_retriever: Optional[int]
    top_k_reader: Optional[int]


class Answer(BaseModel):
    answer: Optional[str]
    score: Optional[float] = None
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Optional[str]]]


class Response(BaseModel):
    query: str
    answers: List[Answer]


PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=QUERY_PIPELINE_NAME)
logger.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")
concurrency_limiter = RequestLimiter(4)


@router.post("/query", response_model=Response)
def query(request: Request):
    with concurrency_limiter.run():
        result = _process_request(PIPELINE, request)
        return result


def _process_request(pipeline, request) -> Response:
    start_time = time.time()

    result = pipeline.run(query=request.query, filters=request.filters, match=request.match,
        top_k_retriever=request.top_k_retriever, top_k_reader=request.top_k_reader)

    end_time = time.time()
    logger.info(json.dumps({"request": request.dict(), "response": str(result), "time": f"{(end_time - start_time):.2f}"}))

    return result
