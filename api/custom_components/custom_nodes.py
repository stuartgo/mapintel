"""
See https://github.com/deepset-ai/haystack/issues/955 for further context
"""
import logging
from copy import deepcopy
from typing import Dict, Generator, List, Optional, Union

import numpy as np
from elasticsearch.helpers import bulk, scan
from tqdm.auto import tqdm
from haystack.utils import get_batches_from_generator
from haystack import Document
from haystack.document_store.base import BaseDocumentStore
from haystack.document_store.elasticsearch import OpenDistroElasticsearchDocumentStore
from haystack.retriever.base import BaseRetriever
from haystack.reader.base import BaseReader

from .top2vec import Top2Vec2

logger = logging.getLogger(__name__)


class Top2VecRetriever(BaseRetriever):
    def __init__(
        self,
        document_store: BaseDocumentStore,
        embedding_model: str,
        umap_args: dict = None,
        hdbscan_args: dict = None,
        top_k: int = 10,
        progress_bar: bool = True
    ):
        """
        :param document_store: An instance of DocumentStore from which to retrieve documents.
        :param embedding_model: Local path or name of model in Hugging Face's model hub such as ``'deepset/sentence_bert'``
        :param umap_args: Pass custom arguments to UMAP.
        :param hdbscan_args: Pass custom arguments to HDBSCAN.
        :param top_k: How many documents to return per query.
        :param progress_bar: If true displays progress bar during embedding.
        """

        # # save init parameters to enable export of component config as YAML
        # self.set_config(
        #     document_store=document_store, embedding_model=embedding_model, umap_args=umap_args, 
        #     hdbscan_args=hdbscan_args, top_k=top_k
        # )

        self.document_store = document_store
        self.embedding_model = embedding_model
        self.umap_args = umap_args
        self.hdbscan_args = hdbscan_args
        self.top_k = top_k
        self.progress_bar = progress_bar

        logger.info(f"Init retriever using embeddings of model {embedding_model}")
        self.embedding_encoder = _Top2VecEncoder(self)

    def retrieve(self, query: str, filters: dict = None, top_k: Optional[int] = None, index: str = None) -> List[Document]:
        """
        Scan through documents in DocumentStore and return a small number documents
        that are most relevant to the query.
        :param query: The query
        :param filters: A dictionary where the keys specify a metadata field and the value is a list of accepted values for that field
        :param top_k: How many documents to return per query.
        :param index: The name of the index in the DocumentStore from which to retrieve documents
        """
        if top_k is None:
            top_k = self.top_k
        if index is None:
            index = self.document_store.index
        query_emb = self.embed_queries(texts=[query])
        documents = self.document_store.query_by_embedding(query_emb=query_emb[0], filters=filters,
                                                           top_k=top_k, index=index)
        return documents

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for a list of queries.
        :param texts: Queries to embed
        :return: Embeddings, one per input queries
        """
        # for backward compatibility: cast pure str input
        if isinstance(texts, str):
            texts = [texts]
        assert isinstance(texts, list), "Expecting a list of texts, i.e. create_embeddings(texts=['text1',...])"
        return self.embedding_encoder.embed_queries(texts)

    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        """
        Create embeddings for a list of passages. Produces the original embeddings, the UMAP embeddings, 
        the topic number and the topic label of each document.
        :param docs: List of documents to embed
        :return: Embeddings, one per input passage
        """
        return self.embedding_encoder.embed_passages(docs)

    def run_indexing(self, documents: List[dict], **kwargs):
        documents = deepcopy(documents)
        document_objects = [Document.from_dict(doc) for doc in documents]
        embeddings, umap_embeddings, topic_numbers, topic_labels = self.embed_passages(document_objects)
        for doc, emb, umap_emb, topn, topl in zip(documents, embeddings, umap_embeddings, topic_numbers, topic_labels):
            doc["embedding"] = emb
            doc["umap_embeddings"] = umap_emb
            doc["topic_number"] = topn
            doc["topic_label"] = topl
        output = {**kwargs, "documents": documents}
        return output, "output_1"

    
class _Top2VecEncoder():
    def __init__(
            self,
            retriever: Top2VecRetriever
    ):  
        self.saved_top2vec_path = "/home/user/outputs/saved_models/top2vec.pkl"
        self.embedding_model = retriever.embedding_model
        self.umap_args = retriever.umap_args
        self.hdbscan_args = retriever.hdbscan_args
        self.show_progress_bar = retriever.progress_bar
        self.document_store = retriever.document_store

        if self.document_store.similarity != "cosine":
            logger.warning(
                f"You are using a Sentence Transformer with the {self.document_store.similarity} function. "
                f"We recommend using cosine instead. "
                f"This can be set when initializing the DocumentStore")

    def embed(self, texts: Union[List[List[str]], List[str], str]) -> List[np.ndarray]:        
        # texts can be a list of strings or a list of [title, text]
        # get back list of numpy embedding vectors
        self.top2vec_model._check_model_status()  # Setting the embed attribute based on the embedding_model
        emb = self.top2vec_model.embed(texts, batch_size=200, show_progress_bar=self.show_progress_bar)
        emb = [r for r in emb]
        return emb

    def embed_queries(self, texts: List[str]) -> List[np.ndarray]:
        # Initializing the top2vec model
        self.init_model()
            
        return self.embed(texts)     
    
    def embed_passages(self, docs: List[Document]) -> List[np.ndarray]:
        # Initializing the top2vec model
        self.init_model(docs)

        passages = [[d.meta["name"] if d.meta and "name" in d.meta else "", d.text] for d in docs]  # type: ignore
        embeddings = self.embed(passages)
        umap_embeddings = self.top2vec_model.get_umap().transform(embeddings)
        topic_numbers = self.top2vec_model.doc_top_reduced
        topic_labels = self.create_topic_labels()
        return [embeddings, umap_embeddings, topic_numbers, topic_labels]

    def create_topic_labels(self):
        # TODO: Give more importance to words with higher score and that are unique to a cluster.
        # Get topic words
        topic_words, _, _ = self.top2vec_model.get_topics(20, reduced=True)
        # Produce topic labels by concatenating top 5 words
        topic_labels = ["_".join(words[:5]) for words in topic_words]
        return topic_labels
    
    def init_model(self, docs=None):
        try:
            logger.info("Loading the Top2Vec model from disk.")
            self.top2vec_model = Top2Vec2.load(self.saved_top2vec_path)
            # Ensure the embedding model matches
            assert self.top2vec_model.embedding_model == self.embedding_model, \
                "The Top2Vec embedding model doesn't match the embedding model in the Retriever."
            # TODO: Ensure the umap_args and hdbscan_args match as well
        except Exception as e:
            logger.info(f"The Top2Vec model hasn't been trained or isn't valid: {e}")
            if self.document_store.get_document_count() > 1000:
                self.train()
            else:
                if docs is None:
                    raise RuntimeError("There isn't enough documents in the database for training the Top2Vec model.")
                else:
                    if len(docs) > 1000:
                        self.train(docs=list(map(lambda d: d.text, docs)))  # training the Top2Vec model with the uploaded documents
                    else:
                        raise RuntimeError("There isn't enough documents in the database or in the upload for training the Top2Vec model.")        

    def train(self, docs=None):
        if docs is None:
            # Get all documents from Document Store
            logger.info("Getting all documents from Document Store.")
            docs = self.document_store.get_all_documents(return_embedding=False)
            docs = list(map(lambda d: d.text, docs))
            logger.info(f"Beginning training of Top2Vec with {len(docs)} internal documents.")
        else:
            logger.info(f"Beginning training of Top2Vec with {len(docs)} external documents.")
        
        self.top2vec_model = Top2Vec2(
            docs,
            embedding_model=self.embedding_model,
            keep_documents=False,  # we don't need to keep the documents as the search isn't performed through top2vec
            workers=None,
            use_embedding_model_tokenizer=True,
            umap_args=self.umap_args,
            hdbscan_args=self.hdbscan_args
        )
        self.top2vec_model.hierarchical_topic_reduction(20)  # reduce the number of topics
        self.top2vec_model.save(self.saved_top2vec_path)

class CrossEncoderReRanker(BaseReader):
    """
    A re-ranker based on a BERT Cross-Encoder. The query and a candidate result are passed
    simoultaneously to the trasnformer network, which then output a single score between
    0 and 1 indicating how relevant the document is for the given query. Read the article
    in https://www.sbert.net/examples/applications/retrieve_rerank/README.html for further
    details.
    """

    def __init__(
        self,
        cross_encoder: str = "cross-encoder/ms-marco-TinyBERT-L-6",
        use_gpu: int = True,
        top_k: int = 10
    ):
        """
        :param cross_encoder: Local path or name of cross-encoder model in Hugging Face's model hub such as ``'cross-encoder/ms-marco-TinyBERT-L-6'``
        :param use_gpu: If < 0, then use cpu. If >= 0, this is the ordinal of the gpu to use
        :param top_k: The maximum number of answers to return
        """

        # # save init parameters to enable export of component config as YAML
        # self.set_config(
        #     cross_encoder=cross_encoder, use_gpu=use_gpu, top_k=top_k
        # )

        self.top_k = top_k

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("Can't find package `sentence-transformers` \n"
                              "You can install it via `pip install sentence-transformers` \n"
                              "For details see https://github.com/UKPLab/sentence-transformers ")
        # pretrained embedding models coming from: https://github.com/UKPLab/sentence-transformers#pretrained-models
        if use_gpu:
            device = "cuda"
        else:
            device = "cpu"
        self.cross_encoder = CrossEncoder(cross_encoder, device=device)

    def predict(self, query: str, documents: List[Document], top_k: Optional[int] = None):
        """
        Use the cross-encoder to find answers for a query in the supplied list of Document.
        Returns dictionaries containing answers sorted by (desc.) probability.
        Example:
         ```python
            |{
            |    'query': 'What is the capital of the United States?',
            |    'answers':[
            |                 {'answer': 'Washington, D.C. (also known as simply Washington or D.C., 
            |                  and officially as the District of Columbia) is the capital of 
            |                  the United States. It is a federal district. The President of 
            |                  the USA and many major national government offices are in the 
            |                  territory. This makes it the political center of the United 
            |                  States of America.',
            |                 'score': 0.717,
            |                 'document_id': 213
            |                 },...
            |              ]
            |}
         ```
        :param query: Query string
        :param documents: List of Document in which to search for the answer
        :param top_k: The maximum number of answers to return
        :return: Dict containing query and answers
        """
        if top_k is None:
            top_k = self.top_k

        # Score every document with the cross_encoder
        cross_inp = [[query, doc.text] for doc in documents]
        cross_scores = self.cross_encoder.predict(cross_inp)
        answers = [
            {
                'answer': documents[idx].text, 
                'score': cross_scores[idx],
                'document_id': documents[idx].id,
                'meta': documents[idx].meta
            }
            for idx in range(len(documents))
        ]

        # Sort answers by the cross-encoder scores and select top-k
        answers = sorted(
            answers, key=lambda k: k["score"], reverse=True
        )
        answers = answers[:top_k]

        results = {"query": query,
                   "answers": answers}

        return results

    def predict_batch(self, query_doc_list: List[dict], top_k: Optional[int] = None,  batch_size: Optional[int] = None):
        raise NotImplementedError("Batch prediction not yet available in CrossEncoderReRanker.")


class OpenDistroElasticsearchDocumentStore2(OpenDistroElasticsearchDocumentStore):
    def query_by_embedding(self,
                            query_emb: np.ndarray,
                            filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
                            top_k: int = 10,
                            index: Optional[str] = None,
                            return_embedding: Optional[bool] = None) -> List[Document]:
            """
            Find the document that is most similar to the provided `query_emb` by using a vector similarity metric.
            :param query_emb: Embedding of the query (e.g. gathered from DPR)
            :param filters: Optional filters to narrow down the search space. Follows Open Distro for 
            Elasticsearch syntax: https://opendistro.github.io/for-elasticsearch-docs/docs/elasticsearch/bool/. Example: 
                [
                    {
                        "terms": {
                            "author": [
                                "Alan Silva", 
                                "Mark Costa",
                            ]
                        }
                    },
                    {
                        "range": {
                            "timestamp": {
                                "gte": "01-01-2021",
                                "lt": "01-06-2021" 
                            }
                        }
                    }
                ]
            :param top_k: How many documents to return
            :param index: Index name for storing the docs and metadata
            :param return_embedding: To return document embedding
            :return:
            """
            if index is None:
                index = self.index

            if return_embedding is None:
                return_embedding = self.return_embedding

            if not self.embedding_field:
                raise RuntimeError("Please specify arg `embedding_field` in ElasticsearchDocumentStore()")
            else:
                # +1 in similarity to avoid negative numbers (for cosine sim)
                body = {
                    "size": top_k,
                    "query": {
                        "bool": {
                            "must": [
                                self._get_vector_similarity_query(query_emb, top_k)
                            ]
                        }
                    }
                }
                if filters:
                    body = self._filter_adapter(body, filters)

                excluded_meta_data: Optional[list] = None

                if self.excluded_meta_data:
                    excluded_meta_data = deepcopy(self.excluded_meta_data)

                    if return_embedding is True and self.embedding_field in excluded_meta_data:
                        excluded_meta_data.remove(self.embedding_field)
                    elif return_embedding is False and self.embedding_field not in excluded_meta_data:
                        excluded_meta_data.append(self.embedding_field)
                elif return_embedding is False:
                    excluded_meta_data = [self.embedding_field]

                if excluded_meta_data:
                    body["_source"] = {"excludes": excluded_meta_data}

                logger.debug(f"Retriever query: {body}")
                result = self.client.search(index=index, body=body, request_timeout=300)["hits"]["hits"]

                documents = [
                    self._convert_es_hit_to_document(hit, adapt_score_for_embedding=True, return_embedding=return_embedding)
                    for hit in result
                ]
                return documents
    
    def get_document_count(
        self, 
        filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
        index: Optional[str] = None,
        only_documents_without_embedding: bool = False
    ) -> int:
        """
        Return the number of documents in the document store.
        """
        index = index or self.index

        body: dict = {"query": {"bool": {}}}
        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": self.embedding_field}}]

        if filters:
            body = self._filter_adapter(body, filters)
        
        result = self.client.count(index=index, body=body)
        count = result["count"]
        return count

    def get_all_documents(
        self,
        index: Optional[str] = None,
        filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
        return_embedding: Optional[bool] = None,
        embedding_field: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        batch_size: int = 10_000,
    ) -> List[Document]:
        """
        Get documents from the document store.

        If only_documents_without_embedding=True, then it only retrieves the documents in the 
        index without a value in the embedding_field (defaults to self.embedding_field).

        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param embedding_field: field to consider when looking for document without embedding.
                                Defaults to self.embedding_field.
        :param only_documents_without_embedding: whether or not to return only documents without embedding in 
                                                 embedding_field.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """
        result = self.get_all_documents_generator(
            index=index, 
            filters=filters, 
            return_embedding=return_embedding,
            embedding_field=embedding_field,
            only_documents_without_embedding=only_documents_without_embedding,
            batch_size=batch_size
        )
        documents = list(result)
        return documents

    def get_all_documents_generator(
        self,
        index: Optional[str] = None,
        filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
        return_embedding: Optional[bool] = None,
        embedding_field: Optional[str] = None,
        only_documents_without_embedding: bool = False,
        batch_size: int = 10_000,
    ) -> Generator[Document, None, None]:
        """
        Get documents from the document store without an embedding. 
        
        Under-the-hood, documents are fetched in batches from the
        document store and yielded as individual documents. This method can be used to iteratively process
        a large number of documents without having to load all documents in memory.

        If only_documents_without_embedding=True, then it only retrieves the documents in the 
        index without a value in the embedding_field.
        
        :param index: Name of the index to get the documents from. If None, the
                      DocumentStore's default index (self.index) will be used.
        :param filters: Optional filters to narrow down the documents to return.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param return_embedding: Whether to return the document embeddings.
        :param embedding_field: field to consider when looking for document without embedding.
                                Defaults to self.embedding_field.
        :param only_documents_without_embedding: whether or not to return only documents without embedding in 
                                                 embedding_field.
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        """

        if index is None:
            index = self.index

        if return_embedding is None:
            return_embedding = self.return_embedding
        
        result = self._get_all_documents_in_index(
            index=index, 
            filters=filters, 
            batch_size=batch_size, 
            embedding_field=embedding_field,
            only_documents_without_embedding=only_documents_without_embedding
        )
        for hit in result:
            document = self._convert_es_hit_to_document(hit, return_embedding=return_embedding)
            yield document

    def _get_all_documents_in_index(
        self,
        index: str,
        filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
        batch_size: int = 10_000,
        only_documents_without_embedding: bool = False,
        embedding_field: Optional[str] = None
    ) -> Generator[dict, None, None]:
        """
        Return all documents in a specific index in the document store.
        If only_documents_without_embedding=True, then it only retrieves
        the documents in the index without a value in the embedding_field.
        """
        body: dict = {"query": {"bool": {}}}

        if embedding_field is None:
            embedding_field = self.embedding_field

        if filters:
            body = self._filter_adapter(body, filters)

        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": embedding_field}}]

        result = scan(self.client, query=body, index=index, size=batch_size, scroll="1d")
        yield from result

    def _filter_adapter(
        self,
        query_body: dict,
        filters: Optional[Union[List[dict], Dict[str, List[str]]]] = None,
    ) -> dict:
        # To not disrupt any of the code of Haystack we can accept both
        # the old filters format or the new format. The following if-else
        # clause deals with the operations for the right format.
        if isinstance(filters, dict):
            filter_clause = []
            for key, values in filters.items():
                if type(values) != list:
                    raise ValueError(
                        f'Wrong filter format for key "{key}": Please provide a list of allowed values for each key. '
                        'Example: {"name": ["some", "more"], "category": ["only_one"]} ')
                filter_clause.append(
                    {
                        "terms": {key: values}
                    }
                )
            query_body["query"]["bool"]["filter"] = filter_clause
        else:
            query_body["query"]["bool"]["filter"] = filters
        return query_body

    def update_embeddings(
        self,
        retriever,
        index: Optional[str] = None,
        filters: Optional[Dict[str, List[str]]] = None,
        update_existing_embeddings: bool = True,
        batch_size: int = 10_000
    ):
        """
        Updates the embeddings in the the document store using the encoding model specified in the retriever.
        This can be useful if want to add or change the embeddings for your documents (e.g. after changing the retriever config).
        :param retriever: Retriever to use to update the embeddings.
        :param index: Index name to update
        :param update_existing_embeddings: Whether to update existing embeddings of the documents. If set to False,
                                           only documents without embeddings are processed. This mode can be used for
                                           incremental updating of embeddings, wherein, only newly indexed documents
                                           get processed.
        :param filters: Optional filters to narrow down the documents for which embeddings are to be updated.
                        Example: {"name": ["some", "more"], "category": ["only_one"]}
        :param batch_size: When working with large number of documents, batching can help reduce memory footprint.
        :return: None
        """
        if index is None:
            index = self.index

        if self.refresh_type == 'false':
            self.client.indices.refresh(index=index)

        if not self.embedding_field:
            raise RuntimeError("Specify the arg `embedding_field` when initializing ElasticsearchDocumentStore()")

        if update_existing_embeddings:
            document_count = self.get_document_count(index=index)
            logger.info(f"Updating embeddings for all {document_count} docs ...")
        else:
            document_count = self.get_document_count(index=index, filters=filters,
                                                     only_documents_without_embedding=True)
            logger.info(f"Updating embeddings for {document_count} docs without embeddings ...")

        result = self._get_all_documents_in_index(
            index=index,
            filters=filters,
            batch_size=batch_size,
            only_documents_without_embedding=not update_existing_embeddings
        )

        logging.getLogger("elasticsearch").setLevel(logging.CRITICAL)

        with tqdm(total=document_count, position=0, unit=" Docs", desc="Updating embeddings") as progress_bar:
            for result_batch in get_batches_from_generator(result, batch_size):
                document_batch = [self._convert_es_hit_to_document(hit, return_embedding=False) for hit in result_batch]
                embeddings, umap_embeddings, topic_numbers, topic_labels = retriever.embed_passages(document_batch)  # type: ignore
                assert len(document_batch) == len(embeddings)

                if embeddings[0].shape[0] != self.embedding_dim:
                    raise RuntimeError(f"Embedding dim. of model ({embeddings[0].shape[0]})"
                                       f" doesn't match embedding dim. in DocumentStore ({self.embedding_dim})."
                                       "Specify the arg `embedding_dim` when initializing ElasticsearchDocumentStore()")
                doc_updates = []
                for doc, emb, umap_emb, topn, topl in zip(document_batch, embeddings, umap_embeddings, topic_numbers, topic_labels):
                    update = {"_op_type": "update",
                              "_index": index,
                              "_id": doc.id,
                              "doc": {
                                  self.embedding_field: emb.tolist(),
                                  "umap_embeddings": umap_emb.tolist(),
                                  "topic_number": topn,
                                  "topic_label": topl
                                  },
                              }
                    doc_updates.append(update)

                bulk(self.client, doc_updates, request_timeout=300, refresh=self.refresh_type)
                progress_bar.update(batch_size)
