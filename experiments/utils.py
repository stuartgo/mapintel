import logging
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import umap
import hdbscan
import tempfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from top2vec import Top2Vec
from top2vec.Top2Vec import default_tokenizer
from octis.models.CTM import CTM
from octis.models.contextualized_topic_models.datasets import dataset

try:
    import hnswlib

    _HAVE_HNSWLIB = True
except ImportError:
    _HAVE_HNSWLIB = False

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    import tensorflow_text

    _HAVE_TENSORFLOW = True
except ImportError:
    _HAVE_TENSORFLOW = False

try:
    from sentence_transformers import SentenceTransformer

    _HAVE_TORCH = True
except ImportError:
    _HAVE_TORCH = False

logger = logging.getLogger('top2vec')
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


class Top2Vec(Top2Vec):
    def __init__(self,
                 documents,
                 min_count=50,
                 embedding_model='doc2vec',
                 embedding_model_path=None,
                 speed='learn',
                 use_corpus_file=False,
                 document_ids=None,
                 keep_documents=True,
                 save_umap=True,
                 save_hdbscan=True,
                 workers=None,
                 tokenizer=None,
                 use_embedding_model_tokenizer=False,
                 umap_args=None,
                 hdbscan_args=None,
                 verbose=True
                 ):
        if verbose:
            logger.setLevel(logging.DEBUG)
            self.verbose = True
        else:
            logger.setLevel(logging.WARNING)
            self.verbose = False
        if tokenizer is None:
            tokenizer = default_tokenizer
        # validate documents
        if not (isinstance(documents, list) or isinstance(documents, np.ndarray)):
            raise ValueError("Documents need to be a list of strings")
        if not all((isinstance(doc, str) or isinstance(doc, np.str_)) for doc in documents):
            raise ValueError("Documents need to be a list of strings")
        if keep_documents:
            self.documents = np.array(documents, dtype="object")
        else:
            self.documents = None
        # validate document ids
        if document_ids is not None:
            if not (isinstance(document_ids, list) or isinstance(document_ids, np.ndarray)):
                raise ValueError("Documents ids need to be a list of str or int")
            if len(documents) != len(document_ids):
                raise ValueError("Document ids need to match number of documents")
            elif len(document_ids) != len(set(document_ids)):
                raise ValueError("Document ids need to be unique")
            if all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
                self.doc_id_type = np.str_
            elif all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
                self.doc_id_type = np.int_
            else:
                raise ValueError("Document ids need to be str or int")
            self.document_ids_provided = True
            self.document_ids = np.array(document_ids)
            self.doc_id2index = dict(zip(document_ids, list(range(0, len(document_ids)))))
        else:
            self.document_ids_provided = False
            self.document_ids = np.array(range(0, len(documents)))
            self.doc_id2index = dict(zip(self.document_ids, list(range(0, len(self.document_ids)))))
            self.doc_id_type = np.int_

        self.embedding_model_path = embedding_model_path
        if embedding_model == 'doc2vec':
            # validate training inputs
            if speed == "fast-learn":
                hs = 0
                negative = 5
                epochs = 40
            elif speed == "learn":
                hs = 1
                negative = 0
                epochs = 40
            elif speed == "deep-learn":
                hs = 1
                negative = 0
                epochs = 400
            elif speed == "test-learn":
                hs = 0
                negative = 5
                epochs = 1
            else:
                raise ValueError("speed parameter needs to be one of: fast-learn, learn or deep-learn")
            if workers is None:
                pass
            elif isinstance(workers, int):
                pass
            else:
                raise ValueError("workers needs to be an int")
            doc2vec_args = {"vector_size": 300,
                            "min_count": min_count,
                            "window": 15,
                            "sample": 1e-5,
                            "negative": negative,
                            "hs": hs,
                            "epochs": epochs,
                            "dm": 0,
                            "dbow_words": 1}
            if workers is not None:
                doc2vec_args["workers"] = workers
            logger.info('Pre-processing documents for training')
            if use_corpus_file:
                processed = [' '.join(tokenizer(doc)) for doc in documents]
                lines = "\n".join(processed)
                temp = tempfile.NamedTemporaryFile(mode='w+t')
                temp.write(lines)
                doc2vec_args["corpus_file"] = temp.name
            else:
                train_corpus = [TaggedDocument(tokenizer(doc), [i]) for i, doc in enumerate(documents)]
                doc2vec_args["documents"] = train_corpus
            logger.info('Creating joint document/word embedding')
            self.embedding_model = 'doc2vec'
            self.model = Doc2Vec(**doc2vec_args)
            if use_corpus_file:
                temp.close()
        else:  # allow use of unlimited models from tensorflow_hub or sentence-transformers
            self.embed = None
            self.embedding_model = embedding_model
            self._check_import_status()
            logger.info('Pre-processing documents for training')
            # preprocess documents (tokenizes and deaccents)
            tokenized_corpus = [tokenizer(doc) for doc in documents]  # the tokenizer is important here because we have to build a vocabulary
            def return_doc(doc):
                return doc
            # preprocess vocabulary
            vectorizer = CountVectorizer(tokenizer=return_doc, preprocessor=return_doc)
            doc_word_counts = vectorizer.fit_transform(tokenized_corpus)
            words = vectorizer.get_feature_names()
            word_counts = np.array(np.sum(doc_word_counts, axis=0).tolist()[0])
            vocab_inds = np.where(word_counts > min_count)[0]
            if len(vocab_inds) == 0:
                raise ValueError(f"A min_count of {min_count} results in "
                                 f"all words being ignored, choose a lower value.")
            self.vocab = [words[ind] for ind in vocab_inds]
            self._check_model_status()  # assigns self.embed
            logger.info('Creating joint document/word embedding')
            # embed words
            self.word_indexes = dict(zip(self.vocab, range(len(self.vocab))))
            self.word_vectors = self._l2_normalize(np.array(self.embed(self.vocab)))
            # embed documents
            if use_embedding_model_tokenizer:
                self.document_vectors = self._embed_documents(documents)
            else:
                train_corpus = [' '.join(tokens) for tokens in tokenized_corpus]
                self.document_vectors = self._embed_documents(train_corpus)

        # create 5D embeddings of documents
        logger.info('Creating lower dimension embedding of documents')
        if umap_args is None:
            umap_args = {'n_neighbors': 15,
                         'n_components': 5,
                         'metric': 'cosine'}
        umap_model = umap.UMAP(**umap_args).fit(self._get_document_vectors(norm=False))
        # find dense areas of document vectors
        logger.info('Finding dense areas of documents')
        if hdbscan_args is None:
            hdbscan_args = {'min_cluster_size': 15,
                             'metric': 'euclidean',
                             'cluster_selection_method': 'eom'}

        cluster = hdbscan.HDBSCAN(**hdbscan_args).fit(umap_model.embedding_)

        # save the UMAP and HDBSCAN model
        self.umap_model, self.cluster = None, None
        if save_umap:
            self.umap_model = umap_model
        if save_hdbscan:
            self.cluster = cluster

        # calculate topic vectors from dense areas of documents
        logger.info('Finding topics')
        # create topic vectors
        self._create_topic_vectors(cluster.labels_)
        # deduplicate topics
        self._deduplicate_topics()
        # find topic words and scores
        self.topic_words, self.topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.topic_vectors)
        # assign documents to topic
        self.doc_top, self.doc_dist = self._calculate_documents_topic(self.topic_vectors,
                                                                      self._get_document_vectors())
        # calculate topic sizes
        self.topic_sizes = self._calculate_topic_sizes(hierarchy=False)
        # re-order topics
        self._reorder_topics(hierarchy=False)
        # initialize variables for hierarchical topic reduction
        self.topic_vectors_reduced = None
        self.doc_top_reduced = None
        self.doc_dist_reduced = None
        self.topic_sizes_reduced = None
        self.topic_words_reduced = None
        self.topic_word_scores_reduced = None
        self.hierarchy = None
        # initialize document indexing variables
        self.document_index = None
        self.serialized_document_index = None
        self.documents_indexed = False
        self.index_id2doc_id = None
        self.doc_id2index_id = None
        # initialize word indexing variables
        self.word_index = None
        self.serialized_word_index = None
        self.words_indexed = False
        
    def get_umap(self):
        """
        Get the UMAP model when the model is built.
        Returns
        -------
        umap_model: umap
        """
        return self.umap_model

    def get_hdbscan_cluster(self):
        """
        Get the HDBSCAN model when the model is built.
        Returns
        -------
        cluster: HDBSCAN
        """
        return self.cluster
    
    def _get_document_vectors(self, norm=True):

        if self.embedding_model == 'doc2vec':

            if norm:
                self.model.dv.init_sims() 
                return self.model.dv.get_normed_vectors()  # gensim=>4.0.0
            else:
                return self.model.dv.vectors  # gensim=>4.0.0
        else:
            return self.document_vectors

    def _index2word(self, index):
        if self.embedding_model == 'doc2vec':
            return self.model.wv.index_to_key[index]  # gensim=>4.0.0
        else:
            return self.vocab[index]

    def _get_word_vectors(self):
        if self.embedding_model == 'doc2vec':
            self.model.wv.init_sims()
            return self.model.wv.get_normed_vectors()  # gensim=>4.0.0
        else:
            return self.word_vectors
    
    def _set_document_vectors(self, document_vectors):
        if self.embedding_model == 'doc2vec':
            self.model.dv.vectors = document_vectors
        else:
            self.document_vectors = document_vectors

    def _check_import_status(self):  # allow use of sentence-transformer model
        if self.embedding_model == "universal-sentence-encoder" or self.embedding_model == "universal-sentence-encoder-multilingual":
            if not _HAVE_TENSORFLOW:
                raise ImportError(f"{self.embedding_model} is not available.\n\n"
                                  "Try: pip install top2vec[sentence_encoders]\n\n"
                                  "Alternatively try: pip install tensorflow tensorflow_hub tensorflow_text")
        else:
            if not _HAVE_TORCH:
                raise ImportError(f"{self.embedding_model} is not available.\n\n"
                                  "Try: pip install top2vec[sentence_transformers]\n\n"
                                  "Alternatively try: pip install torch sentence_transformers")

    def _check_model_status(self):  # allow use of sentence-transformer model
        if self.embed is None:
            if self.verbose is False:
                logger.setLevel(logging.DEBUG)
            
            if self.embedding_model == "universal-sentence-encoder" or self.embedding_model == "universal-sentence-encoder-multilingual":
                if self.embedding_model_path is None:
                    logger.info(f'Downloading {self.embedding_model} model')
                    if self.embedding_model == "universal-sentence-encoder-multilingual":
                        module = "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
                    else:
                        module = "https://tfhub.dev/google/universal-sentence-encoder/4"
                else:
                    logger.info(f'Loading {self.embedding_model} model at {self.embedding_model_path}')
                    module = self.embedding_model_path
                self.embed = hub.load(module)

            else:
                if self.embedding_model_path is None:
                    logger.info(f'Downloading {self.embedding_model} model')
                    module = self.embedding_model
                else:
                    logger.info(f'Loading {self.embedding_model} model at {self.embedding_model_path}')
                    module = self.embedding_model_path
                model = SentenceTransformer(module)
                self.embed = model.encode

        if self.verbose is False:
            logger.setLevel(logging.WARNING)
    
    def add_documents(self, documents, doc_ids=None, tokenizer=None, use_embedding_model_tokenizer=False):
        """
        Update the model with new documents.

        The documents will be added to the current model without changing
        existing document, word and topic vectors. Topic sizes will be updated.

        If adding a large quantity of documents relative to the current model
        size, or documents containing a largely new vocabulary, a new model
        should be trained for best results.

        Parameters
        ----------
        documents: List of str

        doc_ids: List of str, int (Optional)
            Only required when doc_ids were given to the original model.

            A unique value per document that will be used for referring to
            documents in search results.

        tokenizer: callable (Optional, default None)
            Override the default tokenization method. If None then
            gensim.utils.simple_preprocess will be used.

        use_embedding_model_tokenizer: bool (Optional, default False)
            If using an embedding model other than doc2vec, use the model's
            tokenizer for document embedding.
        """
        # if tokenizer is not passed use default
        if tokenizer is None:
            tokenizer = default_tokenizer

        # add documents
        self._validate_documents(documents)
        if self.documents is not None:
            self.documents = np.append(self.documents, documents)

        # add document ids
        if self.document_ids_provided is True:
            self._validate_document_ids_add_doc(documents, doc_ids)
            doc_ids_len = len(self.document_ids)
            self.document_ids = np.append(self.document_ids, doc_ids)
            self.doc_id2index.update(dict(zip(doc_ids, list(range(doc_ids_len, doc_ids_len + len(doc_ids))))))

        elif doc_ids is None:
            num_docs = len(documents)
            start_id = max(self.document_ids) + 1
            doc_ids = list(range(start_id, start_id + num_docs))
            doc_ids_len = len(self.document_ids)
            self.document_ids = np.append(self.document_ids, doc_ids)
            self.doc_id2index.update(dict(zip(doc_ids, list(range(doc_ids_len, doc_ids_len + len(doc_ids))))))
        else:
            raise ValueError("doc_ids cannot be used because they were not provided to model during training.")

        if self.embedding_model == "doc2vec":
            docs_processed = [tokenizer(doc) for doc in documents]
            document_vectors = np.vstack([self.model.infer_vector(doc_words=doc,
                                                                  alpha=0.025,
                                                                  min_alpha=0.01,
                                                                  epochs=100) for doc in docs_processed])

            self._set_document_vectors(np.vstack([self._get_document_vectors(norm=False), document_vectors]))
            self.model.dv.init_sims()

        else:
            if use_embedding_model_tokenizer:
                docs_training = documents
            else:
                docs_processed = [tokenizer(doc) for doc in documents]
                docs_training = [' '.join(doc) for doc in docs_processed]
            document_vectors = self._embed_documents(docs_training)
            self._set_document_vectors(np.vstack([self._get_document_vectors(), document_vectors]))

        # update index
        if self.documents_indexed:
            # update capacity of index
            current_max = self.documents_index.get_max_elements()
            updated_max = current_max + len(documents)
            self.documents_index.resize_index(updated_max)

            # update index_id and doc_ids
            start_index_id = max(self.index_id2doc_id.keys()) + 1
            new_index_ids = list(range(start_index_id, start_index_id + len(doc_ids)))
            self.index_id2doc_id.update(dict(zip(new_index_ids, doc_ids)))
            self.doc_id2index_id.update(dict(zip(doc_ids, new_index_ids)))
            self.documents_index.add_items(document_vectors, new_index_ids)

        # update topics
        self._assign_documents_to_topic(document_vectors, hierarchy=False)

        if self.hierarchy is not None:
            self._assign_documents_to_topic(document_vectors, hierarchy=True)


class CTM(CTM):
    @staticmethod
    def preprocess(vocab, train, bert_model, test=None, validation=None,
                   bert_train_path=None, bert_test_path=None, bert_val_path=None):
        vocab2id = {w: i for i, w in enumerate(vocab)}
        vec = CountVectorizer(
            vocabulary=vocab2id, token_pattern=r'(?u)\b[\w+|\-]+\b')
        entire_dataset = train.copy()
        if test is not None:
            entire_dataset.extend(test)
        if validation is not None:
            entire_dataset.extend(validation)

        vec.fit(entire_dataset)
        idx2token = {v: k for (k, v) in vec.vocabulary_.items()}

        x_train = vec.transform(train)
        b_train = CTM.load_bert_data(bert_train_path, train, bert_model)

        train_data = dataset.CTMDataset(x_train, b_train, idx2token)
        input_size = len(idx2token.keys())

        if test is not None and validation is not None:
            x_test = vec.transform(test)
            b_test = CTM.load_bert_data(bert_test_path, test, bert_model)
            test_data = dataset.CTMDataset(x_test, b_test, idx2token)

            x_valid = vec.transform(validation)
            b_val = CTM.load_bert_data(bert_val_path, validation, bert_model)
            valid_data = dataset.CTMDataset(x_valid, b_val, idx2token)
            return train_data, test_data, valid_data, input_size
        if test is None and validation is not None:
            x_valid = vec.transform(validation)
            b_val = CTM.load_bert_data(bert_val_path, validation, bert_model)
            valid_data = dataset.CTMDataset(x_valid, b_val, idx2token)
            return train_data, valid_data, input_size
        if test is not None and validation is None:
            x_test = vec.transform(test)
            b_test = CTM.load_bert_data(bert_test_path, test, bert_model)
            test_data = dataset.CTMDataset(x_test, b_test, idx2token)
            return train_data, test_data, input_size
        if test is None and validation is None:
            return train_data, input_size