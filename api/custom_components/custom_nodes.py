"""
See https://github.com/deepset-ai/haystack/issues/955 for further context
"""
import logging
import numpy as np
from typing import List, Optional, Dict, Generator, Union
from copy import deepcopy
from haystack import Document
from haystack.reader.base import BaseReader
from haystack.document_store.elasticsearch import OpenDistroElasticsearchDocumentStore
from elasticsearch.helpers import scan

logger = logging.getLogger(__name__)


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
                            filters: Optional[List[dict]] = None,
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
                    body["query"]["bool"]["filter"] = filters

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
            # To not disrupt any of the code of Haystack we can accept both
            # the old filters format or the new format. The following if-else
            # clause deals with the operations for the right format.
            if isinstance(filters, dict):
                filter_clause = []
                for key, values in filters.items():
                    filter_clause.append(
                        {
                            "terms": {key: values}
                        }
                    )
                body["query"]["bool"]["filter"] = filter_clause
            else:
                body["query"]["bool"]["filter"] = filters

        if only_documents_without_embedding:
            body['query']['bool']['must_not'] = [{"exists": {"field": embedding_field}}]

        result = scan(self.client, query=body, index=index, size=batch_size, scroll="1d")
        yield from result