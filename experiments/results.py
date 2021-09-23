import os
import sys
import json
import csv
import logging
from functools import singledispatch
from random import seed, shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer

from octis.dataset.dataset import Dataset
from octis.models.model import AbstractModel
from octis.evaluation_metrics.metrics import AbstractMetric
from octis.models.CTM import CTM
from octis.models.LDA import LDA
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence, WECoherencePairwise, _load_default_texts
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../"))  # Necessary so we can import custom modules from api. See: https://realpython.com/lessons/module-search-path/

from experiments.utils import Top2Vec

logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
logger = logging.getLogger(__name__)


@singledispatch
def to_serializable(val):
    """Used to serilize values by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used to serilize values if *val* is an instance of numpy.float32."""
    return np.float64(val)


def np_to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: np_to_native(v) for k, v in obj.items()}
    else:
        return obj


def parse_kwargs(model, kwargs, valid_components):
    for k, v in kwargs.items():
        v = np_to_native(v)  # converting numpy data types to native data types
        if "__" in k:
            comp, param = k.split("__")
            if comp in valid_components:  # deal with "__" hyperparameters
                model.hyperparameters[comp][param] = v
            else:
                print(f"{comp} is not a valid component. Valid components should be passed in 'valid_components'.")
        else:
            if k in valid_components:  # deal with component hyperparameters
                model.hyperparameters[k].update(v)
            elif k in model.hyperparameters:  # deal with ordinary hyperparameters
                model.hyperparameters[k] = v
            else:
                print(f"{k} is not a valid key for kwargs. Valid keys should be in the format [subcomponent]__[hyperparameter] or be in 'valid_components' or be an existing hyperparameter.")
                continue            


class BERTopic_octis(AbstractModel):
    def __init__(
        self,
        num_topics=100, 
        embedding_model="sentence-transformers/msmarco-distilbert-base-v4", 
        use_partitions=False,
        **kwargs
    ):
            super().__init__()
            self.valid_components = ['umap_args', 'hdbscan_args', 'vectorizer_args']
            self.hyperparameters = dict()
            self.hyperparameters["num_topics"] = num_topics
            self.hyperparameters["embedding_model"] = embedding_model
            self.hyperparameters["umap_args"] = {}
            self.hyperparameters["hdbscan_args"] = {}
            self.hyperparameters["vectorizer_args"] = {}
            self.use_partitions = use_partitions
            # Parsing kwargs
            parse_kwargs(self, kwargs, self.valid_components)

    def train_model(self, dataset, hyperparams={}, top_words=10):
        # Parsing and updating hyperparameters
        parse_kwargs(self, hyperparams, self.valid_components)

        # Setting hyperparameters
        if len(self.hyperparameters["umap_args"]) > 0:
            umap_model = UMAP(**self.hyperparameters["umap_args"])
        else:
            umap_model = UMAP(
                n_neighbors=30, 
                n_components=2,
                min_dist=0.0,
                metric='cosine',
                random_state=1
        )
        if len(self.hyperparameters["hdbscan_args"]) > 0:
            hdbscan_model = HDBSCAN(**self.hyperparameters["hdbscan_args"])
        else:
            hdbscan_model = HDBSCAN(
                min_cluster_size=50, 
                metric='euclidean',
                prediction_data=True
            )
        if len(self.hyperparameters["vectorizer_args"]) > 0:
            vectorizer_model = CountVectorizer(**self.hyperparameters["vectorizer_args"])
            n_gram_range = self.hyperparameters["vectorizer_args"].get(['ngram_range'], (1,1))
        else:
            vectorizer_model = CountVectorizer(
                ngram_range=(1, 1),
                stop_words="english"
            )
            n_gram_range = (1, 1)
        
        model = BERTopic(
            n_gram_range=n_gram_range,
            top_n_words=top_words,  # defines number of words per topic
            nr_topics=self.hyperparameters["num_topics"],
            embedding_model=self.hyperparameters["embedding_model"],
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )
        
        # Train the BERTopic model
        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(use_validation=False)
            train_corpus = list(map(lambda x: " ".join(x), train_corpus))
            test_corpus = list(map(lambda x: " ".join(x), test_corpus))
            train_doc_topics, _ = model.fit_transform(train_corpus)
            test_doc_topics, _ = model.transform(test_corpus)

            result = {
                'topics': [[word[0] for word in values] for _, values in model.topics.items()],
                'topic-word-matrix': model.c_tf_idf,
                'topic-document-matrix': np.array(train_doc_topics),  # in BERTopic a document only belongs to a topic
                'test-topic-document-matrix': np.array(test_doc_topics)
            }
            
        else:
            data_corpus = list(map(lambda x: " ".join(x), dataset.get_corpus()))
            train_doc_topics, _ = model.fit_transform(data_corpus)

            result = {
                'topics': [[word[0] for word in values] for _, values in model.topics.items()],
                'topic-word-matrix': model.c_tf_idf,
                'topic-document-matrix': np.array(train_doc_topics),  # in BERTopic a document only belongs to a topic
            }
            
        return result


class Top2Vec_octis(AbstractModel):
    def __init__(
        self,
        num_topics=100, 
        embedding_model="doc2vec", 
        use_partitions=False,
        **kwargs
    ):
            super().__init__()
            self.valid_components = ['umap_args', 'hdbscan_args']
            self.hyperparameters = dict()
            self.hyperparameters["num_topics"] = num_topics
            self.hyperparameters["embedding_model"] = embedding_model
            self.hyperparameters["umap_args"] = {}
            self.hyperparameters["hdbscan_args"] = {}
            self.use_partitions = use_partitions
            # Parsing kwargs
            parse_kwargs(self, kwargs, self.valid_components)
    
    def train_model(self, dataset, hyperparams={}, top_words=10):
        # Parsing and updating hyperparameters
        parse_kwargs(self, hyperparams, self.valid_components)

        if len(self.hyperparameters["umap_args"]) == 0:
            self.hyperparameters["umap_args"] = None
        
        if len(self.hyperparameters["hdbscan_args"]) == 0:
            self.hyperparameters["hdbscan_args"] = None

        num_topics = self.hyperparameters["num_topics"]
        
        # Train the Top2Vec model
        if self.use_partitions:
            train_corpus, test_corpus = dataset.get_partitioned_corpus(use_validation=False)
            train_corpus = list(map(lambda x: " ".join(x), train_corpus))
            test_corpus = list(map(lambda x: " ".join(x), test_corpus))
            model = Top2Vec(
                documents=train_corpus,
                embedding_model=self.hyperparameters["embedding_model"],
                umap_args=self.hyperparameters["umap_args"],
                hdbscan_args=self.hyperparameters["hdbscan_args"],
                keep_documents=False,
                use_corpus_file=True,
                save_umap=False,
                save_hdbscan=False
            )
            reduced = True
            try:
                model.hierarchical_topic_reduction(num_topics)  # Reduce the number of topics discovered by Top2Vec
            except ValueError:
                print(f"Can't reduce number of topics to {num_topics}. Model will have {len(model.topic_vectors)} topics.")
                reduced = False
            model.add_documents(documents=test_corpus)  # Add test corpus - new document topics are appended to doc_top_reduced 

            if reduced:
                result = {
                    'topics': [words[:top_words].tolist() for words in model.get_topics(num_topics, reduced=True)[0]],
                    'topic-word-matrix': np.inner(model.topic_vectors_reduced, model._get_word_vectors()),
                    'topic-document-matrix': model.doc_top_reduced[:len(train_corpus)],  # in Top2Vec a document only belongs to a topic
                    'test-topic-document-matrix': model.doc_top_reduced[len(train_corpus):]
                }
            else:
                result = {
                    'topics': [words[:top_words].tolist() for words in model.get_topics(num_topics, reduced=False)[0]],
                    'topic-word-matrix': np.inner(model.topic_vectors, model._get_word_vectors()),
                    'topic-document-matrix': model.doc_top[:len(train_corpus)],  # in Top2Vec a document only belongs to a topic
                    'test-topic-document-matrix': model.doc_top[len(train_corpus):]
                }
        else:
            data_corpus = list(map(lambda x: " ".join(x), dataset.get_corpus()))
            model = Top2Vec(
                documents=data_corpus,
                embedding_model=self.hyperparameters["embedding_model"],
                umap_args=self.hyperparameters["umap_args"],
                hdbscan_args=self.hyperparameters["hdbscan_args"],
                keep_documents=False,
                use_corpus_file=True,
                save_umap=False,
                save_hdbscan=False
            )
            reduced = True
            try:
                model.hierarchical_topic_reduction(num_topics)  # Reduce the number of topics discovered by Top2Vec
            except ValueError:
                print(f"Can't reduce number of topics to {num_topics}. Model will have {len(model.topic_vectors)} topics.")
                reduced = False
            
            if reduced:
                result = {
                    'topics': [words[:top_words].tolist() for words in model.get_topics(num_topics, reduced=True)[0]],
                    'topic-word-matrix': np.inner(model.topic_vectors_reduced, model._get_word_vectors()),
                    'topic-document-matrix': model.doc_top_reduced,  # in Top2Vec a document only belongs to a topic
                }
            else:
                result = {
                    'topics': [words[:top_words].tolist() for words in model.get_topics(num_topics, reduced=False)[0]],
                    'topic-word-matrix': np.inner(model.topic_vectors, model._get_word_vectors()),
                    'topic-document-matrix': model.doc_top,  # in Top2Vec a document only belongs to a topic
                }
            
        return result


class OptMetric(AbstractMetric):
    def __init__(self, texts=None, topk=10):
        super().__init__()
        if texts is None:
            self._texts = _load_default_texts()
        else:
            self._texts = texts
        self.topk = topk

    def score(self, model_output):
        m1 = Coherence(texts=self._texts, topk=self.topk, measure='c_v')
        m2 = TopicDiversity(topk=self.topk)
        return m1.score(model_output) * m2.score(model_output)


def create_octis_files(backup_file, data_dir, partition=False):
    # Open json backup file
    with open(backup_file, "r") as file:
        jsondata = json.load(file)

    # Remove #SEPTAG# and get categ
    texts = list(map(lambda x: x['text'].replace("#SEPTAG#", " "), jsondata))
    categs = list(map(lambda x: x['meta']['category'], jsondata))

    # Write vocabulary file
    with open(os.path.join(data_dir, "vocabulary.txt"), "w", newline='') as file:
        vectorizer = CountVectorizer(
            ngram_range=(1, 1),
            stop_words="english"
        )
        vectorizer.fit(texts)
        for i in vectorizer.vocabulary_.keys():
            file.writelines(i+'\n')

    # Write corpus file
    seed(1)
    rand_ix = list(range(len(texts)))
    shuffle(rand_ix)
    with open(os.path.join(data_dir, "corpus.tsv"), "w", newline='') as file:
        csv_writer = csv.writer(file, delimiter="\t")
        for i, ix in enumerate(rand_ix):
            if partition:
                # 0.75 - 0.15 - 0.1 | train - val - test split
                if i < round(len(rand_ix)*0.75):
                    split = "train"
                elif i < round(len(rand_ix)*0.9):
                    split = "val"
                else:
                    split = "test"
            else:
                split = "train"
            csv_writer.writerow([texts[ix], split, categs[ix]])


def topic_evaluation(model_output, texts):
    """
    Provide a topic model output and get the score on several metrics.
    """
    output = {}

    # Topic Diversity
    metric = TopicDiversity(topk=10)
    output['topic_diversity'] = metric.score(model_output)

    # Topic Diversity - inverted RBO
    metric = InvertedRBO(topk=10)
    output['inverted_rbo'] = metric.score(model_output)

    # Topic Coherence - internal
    metric = Coherence(texts=texts, topk=10, measure='c_v')
    output['topic_coherence_c_v'] = metric.score(model_output)

    # Topic Coherence - external
    metric = WECoherencePairwise(topk=10)
    output['topic_coherence_we_coherence_pairwise'] = metric.score(model_output)

    return output


if __name__ == "__main__":    
    out_path = os.path.join(dirname, "../outputs/experiments/")
    
    # Define path to sbert model
    sbert_dir = os.path.join(dirname, "../data/experiments/msmarco-distilbert-base-v4")
    if not os.path.isdir(sbert_dir):
        sbert_dir = "sentence-transformers/msmarco-distilbert-base-v4"

    # Creating the OCTIS files if necessary
    if not os.path.exists(os.path.join(dirname, "../data/experiments/corpus.tsv")):
        logger.info("Creating the OCTIS files.")
        create_octis_files(
            backup_file=os.path.join(dirname, "../data/backups/mongodb_cleaned_docs.json"),
            data_dir=os.path.join(dirname, "../data/experiments"),
            partition=False
        )

    # Loading the dataset
    logger.info("Loading the dataset.")
    dataset = Dataset()
    dataset.load_custom_dataset_from_folder(os.path.join(dirname, "../data/experiments/"))
    docs = list(map(lambda x: " ".join(x), dataset.get_corpus()))

    # Obtaining the reduced space
    logger.info("Obtaining the reduced space for projecting the topic embeddings.")
    embeddings = SentenceTransformer(sbert_dir).encode(docs, show_progress_bar=True)
    umap_emb = UMAP(
        n_neighbors=30, 
        n_components=2,
        min_dist=0.0,
        metric='cosine',
        random_state=1
    ).fit_transform(embeddings)

    # Preprocess the docs to match the topic words (necessary to evaluate Coherence)
    prep = CountVectorizer(lowercase=True).build_preprocessor()
    prep_docs = list(map(lambda x: prep(x).split(" "), docs))
    
    # Define models
    models = {
        "CTM": CTM(
            num_topics=20, 
            bert_model=sbert_dir,
            use_partitions=False,
            bert_path=out_path
        ),
        "Top2Vec": Top2Vec_octis(
            num_topics=20,
            umap_args={
                "min_dist": 0.0,  # accentuates clusters in low dim space
                "random_state":1
            },
            hdbscan_args={
                "min_samples": 10,  # we fix min_samples and explore min_cluster_size
                "memory": out_path,  # we cache the hard computation and recompute only the relatively cheap flat cluster extraction
                "metric": 'euclidean',
                "prediction_data": True
            }
        ),
        "BERTopic": BERTopic_octis(
            num_topics=20,
            embedding_model=sbert_dir,
            umap_args={
                "min_dist": 0.0,  # accentuates clusters in low dim space
                "random_state":1
            },
            hdbscan_args={
                "min_samples": 10,  # we fix min_samples and explore min_cluster_size
                "memory": out_path,  # we cache the hard computation and recompute only the relatively cheap flat cluster extraction
                "metric": 'euclidean',
                "prediction_data": True
            }
        ),
        "LDA": LDA(
            num_topics=20,
            random_state=0
        )
    }

    # Define the search space
    search_space = {
        "CTM": {
            "model_type": Categorical({'prodLDA', 'LDA'}),
            "activation": Categorical({'sigmoid', 'relu', 'softplus'}), 
            "num_layers": Categorical({1, 3, 5, 10}), 
            "num_epochs": Categorical({33, 66, 100}),
            "dropout": Real(0.0, 0.95),
            "lr": Real(2e-3, 2e-1),
            "num_neurons": Categorical({100, 300, 500}),
            "inference_type": Categorical({'zeroshot', 'combined'})
        },
        "Top2Vec": {
            "embedding_model": Categorical({"doc2vec", sbert_dir}),
            "umap_args__n_neighbors": Integer(10, 50),
            "umap_args__n_components": Categorical({2, 5, 10, 25, 50}),
            "umap_args__metric": Categorical({'cosine', 'euclidean'}),
            "hdbscan_args__min_cluster_size": Integer(10, 60),
            "hdbscan_args__cluster_selection_epsilon": Real(0.1, 1.0),
            "hdbscan_args__cluster_selection_method": Categorical({'eom', 'leaf'})
        },
        "BERTopic": {
            "min_topic_size": Integer(10, 60),
            "umap_args__n_neighbors": Integer(10, 50),
            "umap_args__n_components": Categorical({2, 5, 10, 25, 50}),
            "umap_args__metric": Categorical({'cosine', 'euclidean'}),
            "hdbscan_args__min_cluster_size": Integer(10, 60),
            "hdbscan_args__cluster_selection_epsilon": Real(0.1, 1.0),
            "hdbscan_args__cluster_selection_method": Categorical({'eom', 'leaf'})
        },
        "LDA": {
            "alpha": Categorical({"asymmetric", "auto"}),
            "iterations": Integer(50, 200),
            "decay": Real(0.5, 1.0),
            "passes": Categorical({1, 2})
        }
    }

    # Define evaluation metric - multiplication of coherence and diversity
    eval_metric = OptMetric(texts=prep_docs, topk=10)

    # Hyperparameter search with Bayesian optimization
    assert search_space.keys() == models.keys(), "search_space and models keys should match!"
    for i in search_space.keys():
        # Initialize an optimizer object and start the optimization
        logger.info(f"Start Bayesian optimization of {i} model.")
        model = models[i]
        optimizer = Optimizer()
        results = optimizer.optimize(
            model, 
            dataset,
            eval_metric, 
            search_space[i], 
            save_path=out_path, # path to store the results
            save_name=f"{i}_results",
            save_models=False,
            number_of_call=30, # number of optimization iterations (only explore points with most potential)
            model_runs=1, # number of different evaluation of the function in the same point and with the same hyperparameters
            random_state=0
        )

        # Get best config
        best_ix = np.argmax(results.func_vals)
        best_config = dict(map(lambda x: (x[0], x[1][best_ix]), results.x_iters.items()))
        best_config_eval = results.func_vals[best_ix]

        # Evaluate extra metrics on best model config
        logger.info(f"Evaluate extra metrics on {i} best config model.")
        model.set_hyperparameters(**best_config)
        model_output = model.train_model(dataset)
        extra_results = topic_evaluation(model_output, prep_docs)
        extra_results['optimization_metric'] = best_config_eval

        # Plot the 2D UMAP projection with the topic labels
        logger.info(f"Obtain UMAP plot of {i} best config model.")
        topic_labels = list(map(lambda x: "_".join(x[:5]), model_output['topics']))
        if len(model_output['topic-document-matrix'].shape) == 1:
            doc_topics = model_output['topic-document-matrix']
        else:
            doc_topics = np.argmax(model_output['topic-document-matrix'], axis=0)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            umap_emb[:, 0], 
            umap_emb[:, 1], 
            s=4, 
            c=doc_topics, 
            cmap='tab20'
        )
        ax.set_title(i)
        legend = ax.legend(scatter.legend_elements(num=len(topic_labels))[0], topic_labels, bbox_to_anchor=(1,1), loc="upper left", title="Topics")
        fig.savefig(os.path.join(out_path, f"{i}_umap_topics.png"), bbox_extra_artists=(legend,), bbox_inches='tight')

        # # Evaluate 40, 60, 80, 100 topics on best model config
        # extra_results["20_topics"] = best_config_eval
        # for j in [40, 60, 80, 100]:
        #     logger.info(f"Evaluate {j} topics on {i} best config model.")
        #     model.set_hyperparameters(num_topics=j)
        #     model_output = model.train_model(dataset)
        #     extra_results[f"{j}_topics"] = eval_metric.score(model_output)
        
        # Save extra_results
        logger.info(f"Saving extra results of {i} best config model.")
        with open(os.path.join(out_path, f"{i}_extra_results.json"), "w") as file:
            json.dump(extra_results, file, default=to_serializable)