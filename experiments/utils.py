import logging
import numpy as np
from multiprocessing import cpu_count

from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
# from top2vec import Top2Vec
# from top2vec.Top2Vec import default_tokenizer
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from sentence_transformers import SentenceTransformer
from torch import nn
from typing import Iterable
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation


class LatentDirichletAllocation(LatentDirichletAllocation):
    def fit(self, X, embeddings=None, y=None):
        self.cv = CountVectorizer()
        doc_word_train = self.cv.fit_transform(X)
        super().fit(doc_word_train)

        return self

    def transform(self, X, embeddings=None):
        doc_word_test = self.cv.transform(X)
        doc_topic_dist = super().transform(doc_word_test)
        top_features_ind = self.components_.argsort(axis=1)[:, :-9:-1]  # test this
        feature_names = self.cv.get_feature_names()
        self.full_output = {
            'topics': [feature_names[i] for i in top_features_ind],
            'topic-word-matrix': self.components_ / self.components_.sum(axis=1)[:, np.newaxis],
            'topic-document-matrix': doc_topic_dist.T,
        }
        return doc_topic_dist


class BERTopic(BERTopic):
    def fit_transform(self, documents, embeddings, y=None):
        train_doc_topics, _ = super().fit_transform(documents, embeddings, y)
        self.full_output = {
                'topics': [[word[0] for word in values] for _, values in self.topics.items()],
                'topic-word-matrix': self.c_tf_idf,
                'topic-document-matrix': np.array(train_doc_topics),  # in BERTopic a document only belongs to a topic
            }
        return train_doc_topics


class CTMScikit(TransformerMixin, BaseEstimator):
    def __init__(self, contextual_size, inference_type="combined", n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False,
                 num_data_loader_workers=cpu_count() - 1, label_size=0, loss_weights=None, n_samples=20):

        self.ctm_model = None
        self.model_output = None
        self.contextual_size = contextual_size
        self.inference_type = inference_type
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.label_size = label_size
        self.loss_weights = loss_weights
        self.n_samples = n_samples

    def fit(self, X, embeddings, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {iterable of str}
            A collection of documents used for training the model.
        
        embeddings: {numpy array}
            A collection of vectorized documents used for training the model.
        """
        self.vectorizer = CountVectorizer()

        # BOW vectorize the corpus
        bow_embeddings = self.vectorizer.fit_transform(X)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        # Get training dataset
        training_dataset = CTMDataset(embeddings, bow_embeddings, self.id2token)

        if self.inference_type == 'combined':
            self.ctm_model = CombinedTM(
                bow_size=self.vectorizer.vocabulary_, contextual_size=self.contextual_size, n_components=self.n_components,
                model_type=self.model_type, hidden_sizes=self.hidden_sizes, activation=self.activation, dropout=self.dropout, 
                learn_priors=self.learn_priors, batch_size=self.batch_size, lr=self.lr, momentum=self.momentum, solver=self.solver, 
                num_epochs=self.num_epochs, reduce_on_plateau=self.reduce_on_plateau, num_data_loader_workers=self.num_data_loader_workers, 
                label_size=self.label_size, loss_weights=self.loss_weights
            )
        elif self.inference_type == 'zeroshot':
            self.ctm_model = ZeroShotTM(
                bow_size=self.vectorizer.vocabulary_, contextual_size=self.contextual_size, n_components=self.n_components,
                model_type=self.model_type, hidden_sizes=self.hidden_sizes, activation=self.activation, dropout=self.dropout, 
                learn_priors=self.learn_priors, batch_size=self.batch_size, lr=self.lr, momentum=self.momentum, solver=self.solver, 
                num_epochs=self.num_epochs, reduce_on_plateau=self.reduce_on_plateau, num_data_loader_workers=self.num_data_loader_workers, 
                label_size=self.label_size, loss_weights=self.loss_weights
            )
        else:
            raise ValueError("Argument 'inference_type' can only take values 'combined' or 'zeroshot'.")

        # Fit the CTM model
        self.ctm_model.fit(training_dataset)

        return self

    def transform(self, X, embeddings, y=None):
        """Infer the documents' topics for the input documents.
        Parameters
        ----------
        X : {iterable of str}
            Input document or sequence of documents.
        
        embeddings: {numpy array}
            Input document or sequence of documents vectorized.
        """
        if self.ctm_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        if self.inference_type == 'combined':
            # BOW vectorize the corpus
            bow_embeddings = self.vectorizer.transform(X)
        else:
            # dummy matrix
            bow_embeddings = csr_matrix(np.zeros((len(X), 1)))

        # Get testing dataset
        testing_dataset = CTMDataset(embeddings, bow_embeddings, self.id2token)

        # Get document topic distributions
        doc_topic_dist = self.ctm_model.get_doc_topic_distribution(testing_dataset, n_samples=self.n_samples)

        self.full_output = {
            'topics': self.ctm_model.get_topic_lists(),
            'topic-word-matrix': self.ctm_model.get_topic_word_distribution(),
            'topic-document-matrix': doc_topic_dist.T,
        }
        
        return np.argmax(doc_topic_dist, axis=0)  # get the most prominent topic for each document
    
    def fit_transform(self, X, embeddings, y=None, **fit_params):
        return self.fit(X, embeddings, **fit_params).transform(X, embeddings)


# # TODO: Finish Top2VecScikit
# class Top2VecScikit(TransformerMixin, BaseEstimator, Top2Vec):
#     def __init__(self, 
#                  nr_topics: int = None,
#                  embedding_model: str = None,
#                  umap_model: umap.UMAP = None,
#                  hdbscan_model: hdbscan.HDBSCAN = None,
#                  verbose: bool = False,
#                 ):
                 
#         # Topic-based parameters
#         self.nr_topics = nr_topics
#         self.verbose = verbose

#         # Embedding model
#         self.embedding_model = embedding_model

#         # UMAP
#         self.umap_model = umap_model or umap.UMAP(n_neighbors=15,
#                                                 n_components=5,
#                                                 min_dist=0.0,
#                                                 metric='cosine')

#         # HDBSCAN
#         self.hdbscan_model = hdbscan_model or hdbscan.HDBSCAN(min_cluster_size=10,
#                                                               metric='euclidean',
#                                                               cluster_selection_method='eom',
#                                                               prediction_data=True)

#         # TODO: We need to create self.word_vectors that correspond to the word vectors of the corpus vocab
#         self.word_vectors = None
        
#         # Document ids variables
#         self.document_ids_provided = False
#         self.document_ids = None
#         self.doc_id2index = None
#         self.doc_id_type = np.int_

#         # Initialize variables for hierarchical topic reduction
#         self.document_vectors = None
#         self.topic_vectors_reduced = None
#         self.doc_top_reduced = None
#         self.doc_dist_reduced = None
#         self.topic_sizes_reduced = None
#         self.topic_words_reduced = None
#         self.topic_word_scores_reduced = None
#         self.hierarchy = None

#         # Initialize document indexing variables
#         self.document_index = None
#         self.serialized_document_index = None
#         self.documents_indexed = False
#         self.index_id2doc_id = None
#         self.doc_id2index_id = None

#         # Initialize word indexing variables
#         self.word_index = None
#         self.serialized_word_index = None
#         self.words_indexed = False

#     def fit(self, X, embeddings, y=None):
#         """Fit the model according to the given training data.
#         Parameters
#         ----------
#         X : {iterable of str}
#             A collection of documents used for training the model.
        
#         embeddings: {numpy array}
#             A collection of vectorized documents used for training the model.
#         """
#         self.fit_transform(X, embeddings, y)
#         return self

#     def fit_transform(self, X, embeddings, y=None, ):
#         # Document ids variables
#         self.document_ids = np.array(range(0, len(X)))
#         self.doc_id2index = dict(zip(self.document_ids, list(range(0, len(self.document_ids)))))

#         # create 5D embeddings of documents
#         print('Creating lower dimension embedding of documents')
#         umap_emb = self.umap_model.fit_transform(embeddings)

#         # find dense areas of document vectors
#         print('Finding dense areas of documents')
#         cluster_labels = self.hdbscan_model.fit_predict(umap_emb)

#         # calculate topic vectors from dense areas of documents
#         print('Finding topics')

#         # create topic vectors
#         unique_labels = set(cluster_labels)
#         if -1 in unique_labels:
#             unique_labels.remove(-1)
#         self.topic_vectors = self._l2_normalize(
#             np.vstack([embeddings[np.where(cluster_labels == label)[0]].mean(axis=0) for label in unique_labels])
#             )

#         # deduplicate topics
#         self._deduplicate_topics()

#         # find topic words and scores
#         self.topic_words, self.topic_word_scores = self._find_topic_words_and_scores(topic_vectors=self.topic_vectors)

#         # assign documents to topic
#         self.doc_top, self.doc_dist = self._calculate_documents_topic(self.topic_vectors, embeddings)

#         # calculate topic sizes
#         self.topic_sizes = self._calculate_topic_sizes(hierarchy=False)

#         # re-order topics
#         self._reorder_topics(hierarchy=False)
        
#         # reduce the number of topics hierarchically
#         if len(self.topic_vectors) > self.nr_topics:
#             self.hierarchical_topic_reduction(self.nr_topics)
#             return self.doc_top_reduced

#         elif len(self.topic_vectors) == self.nr_topics:
#             print(f"No topic reduction necessary as model already has {self.nr_topics} topics.")

#         else:
#             print(f"Can't reduce number of topics to {self.nr_topics}. Model will have {len(self.topic_vectors)} topics.")
        
#         return self.doc_top

#         # if reduced:
#         #     result = {
#         #         'topics': [words[:10].tolist() for words in model.get_topics(num_topics, reduced=True)[0]],
#         #         'topic-word-matrix': np.inner(model.topic_vectors_reduced, model._get_word_vectors()),
#         #         'topic-document-matrix': model.doc_top_reduced,  # in Top2Vec a document only belongs to a topic
#         #     }
#         # else:
#         #     result = {
#         #         'topics': [words[:10].tolist() for words in model.get_topics(len(model.topic_vectors), reduced=False)[0]],
#         #         'topic-word-matrix': np.inner(model.topic_vectors, model._get_word_vectors()),
#         #         'topic-document-matrix': model.doc_top,  # in Top2Vec a document only belongs to a topic
#         #     }

#     def transform(self, X, embeddings, y=None):
#         """Infer the documents' topics for the input documents.
#         Parameters
#         ----------
#         X : {iterable of str}
#             Input document or sequence of documents.
        
#         embeddings: {numpy array}
#             Input document or sequence of documents vectorized.
#         """
#         # Document ids variables
#         start_id = max(self.document_ids) + 1
#         doc_ids = list(range(start_id, start_id + len(X)))
#         doc_ids_len = len(self.document_ids)
#         self.document_ids = np.append(self.document_ids, doc_ids)
#         self.doc_id2index.update(dict(zip(doc_ids, list(range(doc_ids_len, doc_ids_len + len(doc_ids))))))

#         # Find nearest topic vector to each of the test document embeddings
#         if self.hierarchy is not None:
#             doc_top = self._calculate_documents_topic(self.topic_vectors_reduced, embeddings, dist=False)
#         else:
#             doc_top = self._calculate_documents_topic(self.topic_vectors, embeddings, dist=False)
        
#         return doc_top

    
#     # def _get_document_vectors(self, norm=True):
#     #     if self.embedding_model == 'doc2vec':
#     #         if norm:
#     #             self.model.dv.init_sims() 
#     #             return self.model.dv.get_normed_vectors()  # gensim=>4.0.0
#     #         else:
#     #             return self.model.dv.vectors  # gensim=>4.0.0
#     #     else:
#     #         return self.document_vectors

#     # def _index2word(self, index):
#     #     if self.embedding_model == 'doc2vec':
#     #         return self.model.wv.index_to_key[index]  # gensim=>4.0.0
#     #     else:
#     #         return self.vocab[index]

#     # def _get_word_vectors(self):
#     #     if self.embedding_model == 'doc2vec':
#     #         self.model.wv.init_sims()
#     #         return self.model.wv.get_normed_vectors()  # gensim=>4.0.0
#     #     else:
#     #         return self.word_vectors
    
#     # def _set_document_vectors(self, document_vectors):
#     #     if self.embedding_model == 'doc2vec':
#     #         self.model.dv.vectors = document_vectors  # gensim=>4.0.0
#     #     else:
#     #         self.document_vectors = document_vectors


class SentenceTransformerScikit(TransformerMixin, BaseEstimator):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: str = None,
                 batch_size: int = 32, show_progress_bar: bool = None, output_value: str = 'sentence_embedding',
                 convert_to_numpy: bool = True, convert_to_tensor: bool = False, normalize_embeddings: bool = False
                ):

        self.sent_transformer = None
        self.model_name_or_path = model_name_or_path
        self.modules = modules
        self.device = device
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.output_value = output_value
        self.convert_to_numpy = convert_to_numpy
        self.convert_to_tensor = convert_to_tensor
        self.normalize_embeddings = normalize_embeddings

    def fit(self, X=None, y=None):
        """Initialize the pre-trained Sentence Transformer model.
        Parameters
        ----------
        X : {iterable of str}
            A list of documents used for training the model.
        """
        
        self.sent_transformer = SentenceTransformer(
            model_name_or_path=self.model_name_or_path,
            modules=self.modules,
            device=self.device
        )
        return self

    def transform(self, X, y=None):
        """Infer the vector representations for the input documents.
        Parameters
        ----------
        X : {iterable of str}
            Input document or sequence of documents.
        """
        if self.sent_transformer is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        return self.sent_transformer.encode(
            sentences=X, 
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            output_value=self.output_value,
            convert_to_numpy=self.convert_to_numpy,
            convert_to_tensor=self.convert_to_tensor,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings
        )


class Doc2VecScikit(TransformerMixin, BaseEstimator):
    """Base Doc2Vec module, wraps :class:`~gensim.models.doc2vec.Doc2Vec`.
    This model based on `Quoc Le, Tomas Mikolov: "Distributed Representations of Sentences and Documents"
    <https://cs.stanford.edu/~quocle/paragraph_vector.pdf>`_.
    """
    def __init__(self, dm_mean=None, dm=1, dbow_words=0, dm_concat=0, dm_tag_count=1, dv=None, 
                dv_mapfile=None, comment=None, trim_rule=None, vector_size=100, callbacks=(), alpha=0.025, 
                window=5, shrink_windows=True, min_count=5, max_vocab_size=None, sample=1e-3, seed=1, 
                workers=3, min_alpha=0.0001, hs=0, negative=5, cbow_mean=1, hashfxn=hash, epochs=10, 
                sorted_vocab=1, batch_words=10000):     

        self.d2v_model = None
        self.dm_mean = dm_mean
        self.dm = dm
        self.dbow_words = dbow_words
        self.dm_concat = dm_concat
        self.dm_tag_count = dm_tag_count
        self.dv = dv
        self.dv_mapfile = dv_mapfile
        self.comment = comment
        self.trim_rule = trim_rule

        # attributes associated with gensim.models.Word2Vec
        self.vector_size = vector_size
        self.callbacks = callbacks
        self.alpha = alpha
        self.window = window
        self.shrink_windows = shrink_windows
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        self.sample = sample
        self.seed = seed
        self.workers = workers
        self.min_alpha = min_alpha
        self.hs = hs
        self.negative = negative
        self.cbow_mean = int(cbow_mean)
        self.hashfxn = hashfxn
        self.epochs = epochs
        self.sorted_vocab = sorted_vocab
        self.batch_words = batch_words

    def fit(self, X, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {iterable of :class:`~gensim.models.doc2vec.TaggedDocument`, iterable of list of str, iterable of str}
            A collection of tagged documents used for training the model.
        """
        if isinstance(X[0], TaggedDocument):
            d2v_sentences = X
        elif isinstance(X[0], str):
            d2v_sentences = [TaggedDocument(simple_preprocess(doc, deacc=True), [i]) for i, doc in enumerate(X)]
        elif isinstance(X[0], list):
            d2v_sentences = [TaggedDocument(doc, [i]) for i, doc in enumerate(X)]
        
        self.d2v_model = Doc2Vec(
            documents=d2v_sentences, dm_mean=self.dm_mean, dm=self.dm,
            dbow_words=self.dbow_words, dm_concat=self.dm_concat, dm_tag_count=self.dm_tag_count,
            dv=self.dv, dv_mapfile=self.dv_mapfile, comment=self.comment, trim_rule=self.trim_rule, 
            vector_size=self.vector_size, callbacks=self.callbacks, alpha=self.alpha, window=self.window,
            shrink_windows=self.shrink_windows, min_count=self.min_count, max_vocab_size=self.max_vocab_size, 
            sample=self.sample, seed=self.seed, workers=self.workers, min_alpha=self.min_alpha, hs=self.hs,
            negative=self.negative, cbow_mean=self.cbow_mean, hashfxn=self.hashfxn, epochs=self.epochs, 
            sorted_vocab=self.sorted_vocab, batch_words=self.batch_words
        )
        return self

    def transform(self, X, y=None):
        """Infer the vector representations for the input documents.
        Parameters
        ----------
        X : {iterable of list of str, iterable of str}
            Input document or sequence of documents.
        """
        if self.d2v_model is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this method."
            )

        # The input as array of array
        if isinstance(X[0], str):
            X = [simple_preprocess(doc, deacc=True) for doc in X]
        
        vectors = [self.d2v_model.infer_vector(doc) for doc in X]
        return np.reshape(np.array(vectors), (len(X), self.d2v_model.vector_size))
