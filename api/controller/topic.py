import logging
import pickle
from pathlib import Path
from typing import List, Optional
from tqdm.auto import tqdm
import numpy as np
from umap import UMAP
from elasticsearch.helpers import bulk
from haystack.utils import get_batches_from_generator
from api.custom_components.custom_pipe import CustomPipeline
from api.config import PIPELINE_YAML_PATH, LOG_LEVEL, INDEXING_NU_PIPELINE_NAME

from fastapi import APIRouter
from pydantic import BaseModel
from api.controller.utils import RequestLimiter

logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")

router = APIRouter()


class Request_query(BaseModel):
    query: str


class Response_query(BaseModel):
    status: str
    query_text: Optional[str]
    query_umap: Optional[List[float]]


class Request_training(BaseModel):
    umap_params: Optional[dict] = None


class Response(BaseModel):
    status: str


PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME)
logger.info(f"Loaded pipeline nodes: {PIPELINE.graph.nodes.keys()}")
concurrency_limiter = RequestLimiter(4)

saved_umap_path = "./outputs/saved_models/umap.pkl"
umap_field = "umap_embeddings"
batch_size = 60000


@router.post("/umap-query", response_model=Response_query)
def umap_query(request: Request_query):
    """UMAP Query endpoint.

    Loads the fitted UMAP model and calls transform() on the embedding of the query 
    string and returns the resulting 2 dimensional UMAP embeddings.
    """
    with concurrency_limiter.run():
        # Load fitted UMAP
        umap = _load_umap(path=saved_umap_path)
        if umap is None:
            return {'status': 'Fail', 'query_umap': None}

        # Get query 768 dimensional embedding
        logger.info("Getting the query embeddding using the Retriever component.")
        retriever = PIPELINE.get_node("Retriever")
        results = retriever.run(pipeline_type="Indexing", documents=[{"text": request.query}])
        X = np.array(list(map(lambda x: x['embedding'], results[0]['documents'])))

        # call umap.transform() to obtain the 2 dimensional representation of the query
        logger.info("Obtaining the UMAP embedding of the query.")
        query_umap = umap.transform(X)
        return {'status': 'Success', 'query_text': request.query, 'query_umap': query_umap[0].tolist()}
            

# TODO: Fine-tune the Top2Vec model!
@router.post("/top2vec-training", response_model=Response)
def top2vec_training(request: Request_training):
    """UMAP Training endpoint.

    Takes the 768 dimensional embeddings of each document and calls fit_transform()
    to generate the respective 2 dimensional embeddings while saving the fitted 
    model under outputs/saved_models. Note: this mode requires a considerable 
    amount of allocated memory to be performed.

    The 2 dimensional embeddings of each document are inserted in the document store 
    under the umap field. These 2 dimensional embeddings can be then obtained 
    through the search-binary endpoint, where we can specify any filter or get a random
    sample (see https://stackoverflow.com/questions/25887850/random-document-in-elasticsearch).
    """
    # Instantiate UMAP
    if request.umap_params:
        umap = UMAP(**request.umap_params)
    else:
        umap = UMAP()
    try:
        # Saving the umap embeddings in the Document Store and the fitted model to disk
        _update_umap_embeddings(umap, train=True)
        return {'status': 'Success'}
    except:
        return {'status': 'Fail'}


@router.post("/umap-inference", response_model=Response)
def top2vec_inference():
    """UMAP Inference endpoint.

    Loads the fitted UMAP model and calls transform() on any document in the database 
    that doesn't have a 2 dimensional embedding.

    The 2 dimensional embeddings of each document are inserted in the document store 
    under the umap field. These 2 dimensional embeddings can be then obtained 
    through the search-binary endpoint, where we can specify any filter or get a random
    sample (see https://stackoverflow.com/questions/25887850/random-document-in-elasticsearch).
    """
    # Load fitted UMAP
    umap = _load_umap(path=saved_umap_path)
    if umap is None:
        return {'status': 'Fail'}
    try:
        # Update the umap embeddings of the documents in the Document Store
        _update_umap_embeddings(umap, train=False)
        return {'status': 'Success'}
    except:
        return {'status': 'Fail'}


def _load_umap(path):
    try:
        with open(path,'rb') as f: 
            logger.info("Loading the fitted UMAP model.")
            umap = pickle.load(f)
        return umap
    except:
        logger.info("Couldn't load the fitted UMAP model.")
        return None


def _update_umap_embeddings(umap, train=False):
    """Updates UMAP 2 dimensional embeddings in the Document Store.
    If the UMAP embeddings don't yeat exist for a document, they are respectively added.
    """
    logger.info("Loading Document Store.")
    doc_store = PIPELINE.get_node("DocumentStore")

    if train:
        # Get all documents
        logger.info("Getting all documents from Document Store.")
        docs = doc_store.get_all_documents(return_embedding=True)
        logger.info(f"Number of documents in Document Store {len(docs)}.")
        X = np.array(list(map(lambda x: x.embedding, docs)))

        # Obtain the 2 dimensional embeddings of the documents
        logger.info("Fitting the UMAP instance and obtaining the embeddings.")
        projections = umap.fit_transform(X)

        # Saving the fitted UMAP model
        logger.info("Saving the fitted UMAP model.")
        with open(saved_umap_path,'wb') as f:
            pickle.dump(umap, f)
    else:
        # Get only documents without umap_embeddings value
        logger.info("Getting all documents without UMAP embedding from Document Store.")
        docs = doc_store.get_all_documents(
            return_embedding=True,
            embedding_field=umap_field,
            only_documents_without_embedding=True
        )
        if len(docs) == 0:
            logger.info("No documents in the database without UMAP embeddings.")
            return None
        
        logger.info(f"Number of documents in Document Store without UMAP embedding {len(docs)}.")
        X = np.array(list(map(lambda x: x.embedding, docs)))
       
        # Obtain the 2 dimensional embeddings of the documents
        logger.info("Obtaining the UMAP embeddings.")
        projections = umap.transform(X)
    
    # Update the umap_embeddings
    doc_updates = []
    for doc, emb in zip(docs, projections):
        update = {
            "_op_type": "update",
            "_index": "document",
            "_id": doc.id,
            "doc": {umap_field: emb.tolist()},
        }
        doc_updates.append(update)

    logger.info("Updating the UMAP embeddings in the Document Store.")
    with tqdm(total=len(doc_updates), position=0, unit="Updates", desc="Updating UMAP embeddings") as progress_bar:
        for batch in get_batches_from_generator(doc_updates, batch_size):
            bulk(doc_store.client, batch, request_timeout=300, refresh=doc_store.refresh_type)
            progress_bar.update(batch_size) 
