"""
See https://github.com/deepset-ai/haystack/issues/955 for further context
"""
import logging
import numpy as np
from typing import List, Optional, Dict
from copy import deepcopy
from haystack import Document
from haystack.reader.base import BaseReader
from haystack.document_store.elasticsearch import OpenDistroElasticsearchDocumentStore

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
                            match: Optional[List[dict]] = None,
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
            :param match: Optional matching criteria to return results that match a term in the specified field.
            A list of dictionaries where the key is a field of the database and the value is the term you want to match.
            Example:
                [
                    {
                        "text": "Movies" 
                    },
                    {   
                        "text": "Cinema"
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

                if match:
                    match_clause = [{"match": i} for i in match]
                    body["query"]["bool"]["should"] = match_clause

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