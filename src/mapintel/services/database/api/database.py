import os
from typing import Any

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from fastapi import APIRouter
from pydantic import BaseModel

es = Elasticsearch(
    cloud_id="Testin123Mapintel:ZXVyb3BlLXdlc3QzLmdjcC5jbG91ZC5lcy5pbzo0NDMkNWZlZjhmODQ5YjViNDMwOGFjZGJiMTJiYzVhMmFmMjAkYzdkYTM0OTJiZjIyNDBlM2I2ODBjOTVmNDQyMTVkNWI=",
    basic_auth=('elastic', 'G3lzQNlF392UoiKQTeFxM3te'),
)
router = APIRouter()


API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:30000")


class Request_count(BaseModel):
    filters: Any


@router.post("/doc_count")
def count_docs(request: Request_count):
    """Returns the number of documents in the database.

    Args:
        filters: Filters are defined in the UI on the sidebar

    Returns:
        dict: dict containing number of documents
    """
    es.indices.refresh(index="docs")
    num_docs = es.cat.count(index="docs", params={"format": "json"})[0]["count"]
    return {"num_documents": int(num_docs)}


@router.get("/topic_names")
def topic_names():
    """Returns all topics in the database.

    Returns:
        dict: dict with list of topics_names
    """
    es.indices.refresh(index="topics")
    topics = es.search(index="topics", query={"match_all": {}})
    topics = [x["_source"]["topic"] for x in topics["hits"]["hits"]]
    return {"topic_names": topics}


class Request_query(BaseModel):
    query: Any
    filters: Any
    top_k_reader: Any
    top_k_retriever: Any


@router.post("/query")
def query(request: Request_query):
    """_summary_.

    Args:
        potto (_type_): _description_

    Returns:
        _type_: _description_
    """
    es.indices.refresh(index="topics")
    topics = es.search(index="topics", query={"match_all": {}})
    topics = [x["_source"]["topic"] for x in topics["hits"]["hits"]]
    docs = es.search(
        index="docs",
        query={"bool": {"filter": request.filters}},
        knn={
            "field": "meta.umap_embeddings",
            "query_vector": [0, 0],
            "k": 15,
            "num_candidates": request.top_k_retriever,
        },
    )
    return {"topic_names": topics, "answers": docs["hits"]["hits"]}


class Request_all(BaseModel):
    filters: Any
    batch_size: Any


@router.post("/all_docs_generator")
def all_docs_generator(request: Request_all):
    """fetches all the documents
    Returns:
        dict: contains generator that returns all documents.
    """
    results_gen = scan(
        es,
        query={"query": {"bool": {"filter": request.filters}}},
        index="docs",
    )
    # docs=es.search(index="docs", query={"match_all": {}})["hits"]["hits"]
    return {"generator": results_gen}


class Request_umap(BaseModel):
    query: Any


@router.post("/umap_query")
def umap_query(request: Request_umap):
    """When doing a search it updates the point rendered on the plot
    representing the search terms.

    Args:
        string: query

    Returns:
        dict: contains vecotrised query and query text
    """
    url = f"{API_ENDPOINT}/vectorisation/vectors"
    request_params = {"docs": [request.query]}
    response_raw = requests.post(url, json=request_params).json()
    embeddings = response_raw["embeddings"]
    url = f"{API_ENDPOINT}/dim_reduc/vectors"
    request_params = {"docs": embeddings}
    response_raw = requests.post(url, json=request_params).json()
    embedding_2d = response_raw["embeddings"][0]
    return {
        "status": "Success",
        "query_text": request.query,
        "query_umap": embedding_2d,
    }
