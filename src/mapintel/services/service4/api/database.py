import cloudpickle
import mlflow
from fastapi import APIRouter
from pydantic import BaseModel
import shutil
from fastapi.responses import FileResponse
from datetime import datetime
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan,bulk
from typing import List,Any
import os
import requests
es = Elasticsearch(cloud_id="Mapintel_testing:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyQ5ZjBiYjM5NzllMmQ0YTQ4YjQ4MGFmMjVlMDIyMjU5NSQxMDU1ODAyZTZkNzU0ZjRhOTlhODA4OTAzZmQ1ZTMwZg==",
basic_auth=('elastic', 'ojap2aCpyk3FiO7pRZfxUtIt'),)
router = APIRouter()



API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:30000")


class Request_count(BaseModel):
    filters: Any

@router.post("/doc_count")
def count_docs(request: Request_count):
    es.indices.refresh(index="docs")
    num_docs=es.cat.count(index="docs", params={"format": "json"})[0]["count"]
    return {"num_documents":int(num_docs)}


@router.get("/topic_names")
def topic_names():
    es.indices.refresh(index="topics")  #not sure if there is a poibt, might as well query documents and get topics from there?
    topics=es.search(index="topics", query={"match_all": {}})
    topics=list(map(lambda x: x["_source"]["topic"],topics["hits"]["hits"]))
    return {"topic_names":topics}

class Request_query(BaseModel):
    query: Any
    filters: Any
    top_k_reader: Any
    top_k_retriever: Any

@router.post("/query")
def query(request: Request_query):
    print(request)
    es.indices.refresh(index="topics")
    topics=es.search(index="topics",  query={"match_all": {}})
    topics=list(map(lambda x: x["_source"]["topic"],topics["hits"]["hits"]))
    docs=es.search(index="docs",
        query={"bool":{"filter": request.filters}},
        knn={"field":"meta.umap_embeddings","query_vector":[0,0],"k":15,"num_candidates":request.top_k_retriever},)
    return {"topic_names":topics,"answers":docs["hits"]["hits"]}


class Request_all(BaseModel):
    filters: Any
    batch_size:Any


@router.post("/all_docs_generator")
def all_docs_generator(request:Request_all):
    results_gen = scan(
        es,
        query={"query": {"bool":{"filter": request.filters}}},
        index="docs",
    )
    # docs=es.search(index="docs", query={"match_all": {}})["hits"]["hits"]
    return {"generator":results_gen}


class Request_umap(BaseModel):
    query:Any

@router.post("/umap_query")
def umap_query(request: Request_umap):
    url = f"{API_ENDPOINT}/vectorisation/vectors"
    request_params={"docs":[request.query]}
    response_raw = requests.post(url, json=request_params).json()
    embeddings=response_raw["embeddings"]
    url = f"{API_ENDPOINT}/dim_reduc/vectors"
    request_params={"docs":embeddings}
    response_raw = requests.post(url, json=request_params).json()
    embedding_2d=response_raw["embeddings"][0]
    return {
            "status": "Success",
            "query_text": request.query,
            "query_umap": embedding_2d, #needs to be fixed
        }
