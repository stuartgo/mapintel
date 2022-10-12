import numpy as np
from multiprocessing import cpu_count
from dataclasses import dataclass, field
import time
from typing import Iterable, ClassVar, Dict, Optional

from scipy.sparse import csr_matrix
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from sklearn.feature_extraction.text import CountVectorizer
from contextualized_topic_models.datasets.dataset import CTMDataset
from contextualized_topic_models.models.ctm import ZeroShotTM, CombinedTM
from sentence_transformers import SentenceTransformer
from torch import nn
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""


@dataclass
class Timer:
    """Reference: https://realpython.com/python-timer/#a-python-timer-class"""
    timers: ClassVar[Dict[str, list]] = dict()
    name: Optional[str] = None
    _start_time: Optional[float] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Add timer to dict of timers after initialization"""
        if self.name is not None:
            self.timers.setdefault(self.name, [])

    def start(self) -> None:
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self) -> float:
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        # Calculate elapsed time
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None

        if self.name:
            self.timers[self.name].append(elapsed_time)

        return elapsed_time


class LatentDirichletAllocation(LatentDirichletAllocation):
    def __str__(self):
        return "LatentDirichletAllocation"

    def fit(self, X, embeddings=None, y=None):
        self.fit_transform(X, embeddings, y)
        return self

    def transform(self, X, embeddings=None):
        doc_word_test = self.cv.transform(X)
        doc_topic_dist = super().transform(doc_word_test)
            
        return np.argmax(doc_topic_dist, axis=1)  # get the most prominent topic for each document
    
    def fit_transform(self, X, embeddings=None, y=None, **fit_params):
        # Get document topic distribution
        self.cv = CountVectorizer()
        doc_word_train = self.cv.fit_transform(X)
        super().fit(doc_word_train)
        doc_topic_dist = super().transform(doc_word_train)

        # Get self.full_output
        top_features_ind = self.components_.argsort(axis=1)[:, :-11:-1]
        feature_names = self.cv.get_feature_names()
        self.full_output = {
            'topics': [[feature_names[i] for i in top_inds] for top_inds in top_features_ind],
            'topic-word-matrix': self.components_ / self.components_.sum(axis=1)[:, np.newaxis],
            'topic-document-matrix': doc_topic_dist.T,
        }
        return np.argmax(doc_topic_dist, axis=1)  # get the most prominent topic for each document


class BERTopic(BERTopic):
    def __str__(self):
        return "BERTopic"

    def fit(self, documents, embeddings, y=None):
        self.fit_transform(documents, embeddings, y)
        return self

    def fit_transform(self, documents, embeddings, y=None):
        # Get document topic distribution
        if type(documents) == np.ndarray:
            documents = documents.tolist()
        train_doc_topics, _ = super().fit_transform(documents, embeddings, y)

        # Get self.full_output
        self.full_output = {
                'topics': [[word[0] for word in values] for _, values in self.topics.items()],
                'topic-word-matrix': self.c_tf_idf,
                'topic-document-matrix': np.array(train_doc_topics),  # in BERTopic a document only belongs to a topic
            }
        return np.array(train_doc_topics)
    
    def transform(self, documents, embeddings):
        test_doc_topics, _ = super().transform(documents, embeddings)
        return np.array(test_doc_topics)


class CTMScikit(TransformerMixin, BaseEstimator):
    def __init__(self, inference_type="combined", n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64,
                 lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False,
                 num_data_loader_workers=cpu_count() - 1, label_size=0, loss_weights=None, n_samples=20):

        self.ctm_model = None
        self.model_output = None
        self.contextual_size = None
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
    
    def __str__(self):
        return "CTM"

    def fit(self, X, embeddings, y=None):
        """Fit the model according to the given training data.
        Parameters
        ----------
        X : {iterable of str}
            A collection of documents used for training the model.
        
        embeddings: {numpy array}
            A collection of vectorized documents used for training the model.
        """
        self.fit_transform(X, embeddings, y)
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
        
        return np.argmax(doc_topic_dist, axis=1)  # get the most prominent topic for each document
    
    def fit_transform(self, X, embeddings, y=None, **fit_params):

        self.contextual_size = embeddings.shape[1]  # Get dimension of input from embeddings
        self.vectorizer = CountVectorizer()

        # BOW vectorize the corpus
        bow_embeddings = self.vectorizer.fit_transform(X)
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(range(0, len(self.vocab)), self.vocab)}

        # Get training dataset
        training_dataset = CTMDataset(embeddings, bow_embeddings, self.id2token)

        if self.inference_type == 'combined':
            self.ctm_model = CombinedTM(
                bow_size=len(self.vocab), contextual_size=self.contextual_size, n_components=self.n_components,
                model_type=self.model_type, hidden_sizes=self.hidden_sizes, activation=self.activation, dropout=self.dropout, 
                learn_priors=self.learn_priors, batch_size=self.batch_size, lr=self.lr, momentum=self.momentum, solver=self.solver, 
                num_epochs=self.num_epochs, reduce_on_plateau=self.reduce_on_plateau, num_data_loader_workers=self.num_data_loader_workers, 
                label_size=self.label_size, loss_weights=self.loss_weights
            )
        elif self.inference_type == 'zeroshot':
            self.ctm_model = ZeroShotTM(
                bow_size=len(self.vocab), contextual_size=self.contextual_size, n_components=self.n_components,
                model_type=self.model_type, hidden_sizes=self.hidden_sizes, activation=self.activation, dropout=self.dropout, 
                learn_priors=self.learn_priors, batch_size=self.batch_size, lr=self.lr, momentum=self.momentum, solver=self.solver, 
                num_epochs=self.num_epochs, reduce_on_plateau=self.reduce_on_plateau, num_data_loader_workers=self.num_data_loader_workers, 
                label_size=self.label_size, loss_weights=self.loss_weights
            )
        else:
            raise ValueError("Argument 'inference_type' can only take values 'combined' or 'zeroshot'.")

        # Fit the CTM model
        self.ctm_model.fit(training_dataset)

        # Get document topic distributions
        doc_topic_dist = self.ctm_model.get_doc_topic_distribution(training_dataset, n_samples=self.n_samples)

        # Get self.full_output
        self.full_output = {
            'topics': self.ctm_model.get_topic_lists(),
            'topic-word-matrix': self.ctm_model.get_topic_word_distribution(),
            'topic-document-matrix': doc_topic_dist.T,
        }
        
        return np.argmax(doc_topic_dist, axis=1)  # get the most prominent topic for each document


class SentenceTransformerScikit(TransformerMixin, BaseEstimator):
    def __init__(self, model_name_or_path: str = None, modules: Iterable[nn.Module] = None, device: Optional[str] = None, 
                 cache_folder: Optional[str] = None, batch_size: int = 32, show_progress_bar: bool = None, 
                 output_value: str = 'sentence_embedding', convert_to_numpy: bool = True, convert_to_tensor: bool = False,
                 normalize_embeddings: bool = False
                ):

        self.sent_transformer = None
        self.model_name_or_path = model_name_or_path
        self.modules = modules
        self.device = device
        self.cache_folder = cache_folder
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.output_value = output_value
        self.convert_to_numpy = convert_to_numpy
        self.convert_to_tensor = convert_to_tensor
        self.normalize_embeddings = normalize_embeddings
    
    def __str__(self):
        return "SentenceTransformer"

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
            device=self.device,
            cache_folder=self.cache_folder
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

    def __str__(self):
        return "Doc2Vec"

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
