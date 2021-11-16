import os
import sys
import re
import string
from itertools import compress
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from multiprocessing import cpu_count

import mlflow
import optuna
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from octis.evaluation_metrics.diversity_metrics import TopicDiversity, InvertedRBO
from octis.evaluation_metrics.coherence_metrics import Coherence, WECoherencePairwise

dirname = os.path.dirname(__file__)
outputs_dir = os.path.join(dirname, "../outputs/experiments")
sys.path.append(os.path.join(dirname, "../"))  # Necessary so we can import custom modules from api. See: https://realpython.com/lessons/module-search-path/

from experiments.utils import Doc2VecScikit, SentenceTransformerScikit, CTMScikit, BERTopic, LatentDirichletAllocation

VALID_EMBEDDINGS_MODELS = ['doc2vec', 'sentence-transformers/msmarco-distilbert-base-v4']
VALID_TOPIC_MODELS = ['BERTopic', 'CTM', 'LDA']

# TODO:
# - In the SentenceTransformer can we avoid fitting in each fold since there's no actual fit?
# - Define which metric(s) to optimize on hyperparameter searching
# - Print information on parameters selected
# - Use other datasets for validating the methodology
# - Set the sampler and pruner for the optuna optimizatinon process
# - Log the training and inference time (average across folds) as a metric
# - Define MLflow project file
# - Create test set for obtainining unbiased evaluations
# - Log topic word labels and pass them to the UMAP plot legend


def clean_text(text):
    re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text) 

    return text


def prepare_20newsgroups(dataset_file):
    print("Load and clean the dataset.")
    # Check whether there is a saved dataset in disk
    if os.path.isfile(dataset_file):
        # Load the data from disk
        df = pd.read_csv(dataset_file)
        X_clean, y_clean = df['X_clean'], df['y_clean']
    else:
        # Loading the data
        newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
        X, y = newsgroups_data.data, newsgroups_data.target

        # Clean text
        X_clean = list(map(clean_text, X))
        blank = np.array([len(doc) > 2 for doc in X_clean])  # Remove blank documents
        fourwords = np.array([len(doc.split(' ')) > 4 for doc in X_clean])  # Remove documents with 4 words or less
        outliers = np.array(['the most current orbital' not in doc for doc in X_clean])  # Remove outliers (in embedding space)
        X_clean = list(compress(X_clean, blank & fourwords & outliers))
        y_clean = list(compress(y, blank & fourwords & outliers))

        # Save the dataset to disk
        pd.DataFrame({'X_clean': X_clean, 'y_clean': y_clean}).to_csv(dataset_file, index=False)

    y_labels = [
        'alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 
        'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt',
        'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'
    ]
    return X_clean, y_clean, y_labels


def suggest_hyperparameters(trial):
    print('Suggest hyperparameters.')
    hyperparams = {}

    # Define embedding_model
    hyperparams['embedding_model'] = trial.suggest_categorical('embedding_model', VALID_EMBEDDINGS_MODELS)
    if hyperparams['embedding_model'] == 'doc2vec':
        hyperparams['dm'] = trial.suggest_categorical('dm', [0, 1])
    
    # Define UMAP hyperparameters
    hyperparams['umap_args__n_neighbors'] = trial.suggest_int('n_neighbors', 10, 50)
    hyperparams['umap_args__n_components'] = trial.suggest_categorical('n_components', [2, 5, 10, 25, 50])
    hyperparams['umap_args__metric'] = trial.suggest_categorical('metric', ['cosine', 'euclidean'])

    # Define topics_model
    hyperparams['topic_model'] = trial.suggest_categorical('topic_model', VALID_TOPIC_MODELS)
    if hyperparams['topic_model'] == 'BERTopic':
        # Setting hyperparameters for BERTopic
        hyperparams['min_topic_size'] = trial.suggest_int('min_topic_size', 10, 60)
        hyperparams['hdbscan_args__min_cluster_size'] = trial.suggest_int('min_cluster_size', 30, 150)
        hyperparams['hdbscan_args__cluster_selection_epsilon'] = trial.suggest_float('cluster_selection_epsilon', 0.01, 1.0, log=True)
        hyperparams['hdbscan_args__cluster_selection_method'] = trial.suggest_categorical('cluster_selection_method', ['eom', 'leaf'])

    elif hyperparams['topic_model'] == 'CTM':
        # Setting hyperparameters for CTM
        hyperparams['model_type'] = trial.suggest_categorical('model_type', ['prodLDA', 'LDA'])
        hyperparams['activation'] = trial.suggest_categorical('activation', ['relu', 'softplus'])
        hyperparams['hidden_sizes'] = trial.suggest_categorical('hidden_sizes', [(100,), (100, 100), (100, 100, 100), (300,), (300, 300), (300, 300, 300)])
        hyperparams['num_epochs'] = trial.suggest_categorical('num_epochs', [33, 66, 100])
        hyperparams['dropout'] = trial.suggest_float('dropout', 0.0, 0.4)
        hyperparams['lr'] = trial.suggest_float('lr', 2e-3, 2e-1)
        hyperparams['inference_type'] = trial.suggest_categorical('inference_type', ['zeroshot', 'combined'])
    
    elif hyperparams['topic_model'] == 'LDA':
        # Setting hyperparameters for LDA
        hyperparams['learning_decay'] = trial.suggest_float('learning_decay', 0.5, 1.0)
        hyperparams['max_iter'] = trial.suggest_int('max_iter', 2, 10)
        hyperparams['max_doc_update_iter'] = trial.suggest_int('max_doc_update_iter', 50, 200)

    else:
        raise ValueError(f"topic_model={hyperparams['topic_model']} is not defined!")
    
    return hyperparams


def define_embedding_model(hyperparams):
    print('Define the embedding model.')
    embedding_model = hyperparams['embedding_model']
    if embedding_model == 'doc2vec':
        # Based on hyperparameters used in Top2Vec
        model = Doc2VecScikit(
            dm=hyperparams['dm'],
            dbow_words=1,
            vector_size=300,
            min_count=50,
            window=15,
            sample=1e-5,
            negative=0,
            hs=1,
            epochs=40,
            seed=0,
            workers=cpu_count() - 1
        )

    elif embedding_model == 'sentence-transformers/msmarco-distilbert-base-v4':
        model = SentenceTransformerScikit(
            model_name_or_path=embedding_model,
            show_progress_bar=True
        )

    else:
        raise ValueError(f"embedding_model={embedding_model} is not defined!")
    
    return model


def define_topic_model(hyperparams):
    print('Define the topic model.')
    if hyperparams['topic_model'] == 'BERTopic':
        # Setting model components
        umap_model = UMAP(
            n_neighbors=hyperparams['umap_args__n_neighbors'], 
            n_components=hyperparams['umap_args__n_components'],
            min_dist=0.0,
            metric=hyperparams['umap_args__metric'],
            random_state=1
        )
        hdbscan_model = HDBSCAN(
            min_cluster_size=hyperparams['hdbscan_args__min_cluster_size'], 
            cluster_selection_epsilon=hyperparams['hdbscan_args__cluster_selection_epsilon'],
            cluster_selection_method=hyperparams['hdbscan_args__cluster_selection_method'],
            metric='euclidean',
            memory=outputs_dir,  # we cache the hard computation and recompute only the relatively cheap flat cluster extraction
            prediction_data=True
        )
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 1),
            stop_words="english"
        )
        n_gram_range = (1, 1)

        # Declaring the model
        model = BERTopic(
            n_gram_range=n_gram_range,
            nr_topics=20,
            min_topic_size=hyperparams['min_topic_size'],
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model
        )

    elif hyperparams['topic_model'] == 'CTM':
        # Declaring the model
        model = CTMScikit(
            n_components=20, 
            model_type = hyperparams['model_type'],
            activation = hyperparams['activation'],
            hidden_sizes = hyperparams['hidden_sizes'],
            num_epochs = hyperparams['num_epochs'],
            dropout = hyperparams['dropout'],
            lr = hyperparams['lr'],
            inference_type = hyperparams['inference_type'],
            num_data_loader_workers=cpu_count() - 1
        )

    elif hyperparams['topic_model'] == 'LDA':
        # Declaring the model
        model = LatentDirichletAllocation(
            learning_decay=hyperparams['learning_decay'],
            max_iter=hyperparams['max_iter'],
            max_doc_update_iter=hyperparams['max_doc_update_iter'],
            n_components=20, 
            random_state=0
        )
    
    else:
        raise ValueError(f"topic_model={hyperparams['topic_model']} is not defined!")
    
    return model


def umap_evaluation(umap_emb_train, umap_emb_test, y_train, y_test, k_range=[10, 20, 40, 80, 160]):
    accuracies_train = {}
    accuracies_test = {}
    for k in k_range:
        # Initialize the KNN classifier
        knn = KNeighborsClassifier(
            n_neighbors=k,
            weights='uniform',
            algorithm='brute',
            metric='cosine',
        )

        # Get KNN classifier predictions
        knn.fit(umap_emb_train, y_train)
        y_train_pred = knn.predict(umap_emb_train)
        y_test_pred = knn.predict(umap_emb_test)

        # Compute accuracies
        accuracies_train[f'umap_{k}nn_acc_train'] = accuracy_score(y_train, y_train_pred)
        accuracies_test[f'umap_{k}nn_acc_test'] = accuracy_score(y_test, y_test_pred)

    return accuracies_train, accuracies_test


def cluster_evaluation(topics, y, outlier_label=None):
    assert len(topics) == len(y), f'topics and y have different lengths ({len(topics)}, {len(y)}).'
    nmi = normalized_mutual_info_score(y, topics)
    if outlier_label:
        doc_ids = [top != outlier_label for top in topics]
        nmi_filtered = normalized_mutual_info_score(list(compress(y, doc_ids)), list(compress(topics, doc_ids)))
        return nmi, nmi_filtered
    else:
        return nmi, None


def topic_evaluation(model_output, texts):
    """
    Provide a topic model output and get the score on several metrics.
    """
    # Topic Diversity
    metric = TopicDiversity(topk=10)
    topic_diversity = metric.score(model_output)

    # Topic Diversity - inverted RBO
    metric = InvertedRBO(topk=10)
    inverted_rbo = metric.score(model_output)

    # Topic Coherence - internal
    metric = Coherence(texts=texts, topk=10, measure='c_v')
    topic_coherence_c_v = metric.score(model_output)

    # Topic Coherence - external
    metric = WECoherencePairwise(topk=10)
    topic_coherence_we_coherence_pairwise = metric.score(model_output)

    return topic_diversity, inverted_rbo, topic_coherence_c_v, topic_coherence_we_coherence_pairwise


def umap_plot_labels(umap_emb, labels, label_names, topics):
    # Plot the 2D UMAP projection with the topic labels vs original labels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 11))
    plt.subplots_adjust(wspace=0.5)

    # Set axis 1 - Original labels
    ax1.set_title("Original labels")
    scatter = ax1.scatter(umap_emb[:, 0], umap_emb[:, 1], s=4, c=labels, cmap='tab20')
    ax1.legend(scatter.legend_elements(num=len(label_names))[0], label_names, title="Original", bbox_to_anchor=(1,1), loc="upper left")

    # Set axis 2 - Topic labels
    ax2.set_title("Topic labels")
    scatter = ax2.scatter(umap_emb[:, 0], umap_emb[:, 1], s=4, c=topics, cmap='tab20')
    ax2.legend(*scatter.legend_elements(num=len(set(topics))), title="Topics", bbox_to_anchor=(1,1), loc="upper left")

    return fig


def train_infer_models(topic_model, umap_model, emb_model, X_train, X_test):
    infer = {}

    # Fit and transform the embedding model
    print(f"Fit and transform the {emb_model} embedding model.")
    emb_train = emb_model.fit_transform(X_train)
    emb_test = emb_model.transform(X_test)

    # Fit and transform the topic model
    print(f'Fit and transform the {topic_model} topic model.')
    infer['top_train'] = topic_model.fit_transform(X_train, embeddings=emb_train)
    infer['top_test'] = topic_model.transform(X_test, embeddings=emb_test)

    # Fit and transform the UMAP model on 2 components
    print(f"Reduce embeddings to 2 dimensions with UMAP.")
    infer['umap_emb_train'] = umap_model.fit_transform(emb_train)
    infer['umap_emb_test'] = umap_model.transform(emb_test)
    
    # Get full output dictionary from topic_model
    infer['tm_full_output'] = topic_model.full_output

    return infer


def evaluate_models(infer, y_train, y_test, X_train, y_labels=None, plot=False):
    artifacts = {}

    # Save the number of topics identified
    artifacts['ntopics'] = len(infer['tm_full_output']['topics'])

    # Evaluate the UMAP model on test split
    print("Evaluate UMAP on K-NN accuracy.")
    knn_accuracies_train, knn_accuracies_test = umap_evaluation(infer['umap_emb_train'], infer['umap_emb_test'], y_train, y_test)
    artifacts.update(knn_accuracies_train)
    artifacts.update(knn_accuracies_test)

    # Evaluate the clustering on agreement between true labels and topics
    # 0 value indicates two independent label assignments; 1 value indicates two agreeable label assignments
    print("Evaluate clustering on Mutual Information.")
    artifacts['nmi_train'], artifacts['nmi_filtered_train'] = cluster_evaluation(infer['top_train'], y_train, outlier_label=-1)
    artifacts['nmi_test'], artifacts['nmi_filtered_test'] = cluster_evaluation(infer['top_test'], y_test, outlier_label=-1)

    # Evaluate the Topic model on Coherence and Diversity metrics
    print("Evaluate topics on Diversity and Coherence metrics.")
    artifacts['topic_diversity'], artifacts['inverted_rbo'], artifacts['topic_coherence_c_v'], artifacts['topic_coherence_we_coherence'] = \
        topic_evaluation(infer['tm_full_output'], list(map(lambda x: x.split(' '), X_train)))

    if plot:
        # UMAP plot on last split of k-fold cross validation: Original labels VS Topics
        print("Produce UMAP plot: Original labels VS Topics.")
        artifacts['train_fig'] = umap_plot_labels(infer['umap_emb_train'], y_train, y_labels, infer['top_train'])
        artifacts['test_fig'] = umap_plot_labels(infer['umap_emb_test'], y_test, y_labels, infer['top_test'])

    return artifacts


def objective(trial):
    with mlflow.start_run():
        # Load dataset
        X_clean, y_clean, y_labels = prepare_20newsgroups(os.path.join(outputs_dir, '20newsgroups_prep.csv'))

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(trial)

        # Define embedding model
        emb_model = define_embedding_model(hyperparams)

        # Define topic model
        topic_model = define_topic_model(hyperparams)

        # Define UMAP model for projecting space to 2 dimensions
        umap_model = UMAP(
            n_neighbors=hyperparams['umap_args__n_neighbors'], 
            n_components=2,
            min_dist=0.0,
            metric=hyperparams['umap_args__metric'],
            random_state=1
        )

        # Apply Stratified K-fold
        split_metrics = defaultdict(list)
        n_splits = 10
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
        for n, (train_ix, test_ix) in enumerate(skf.split(X_clean, y_clean)):
            print(f'Iteration number {n + 1} out of {n_splits}.')

            # Get train and test samples
            X_train, X_test = np.array(X_clean)[train_ix], np.array(X_clean)[test_ix]
            y_train, y_test = np.array(y_clean)[train_ix], np.array(y_clean)[test_ix]
            
            # Train and infer with topic_model, umap_model and emb_model
            infer = train_infer_models(topic_model, umap_model, emb_model, X_train, X_test)

            # Evaluate the topic_model and umap_model
            if n == n_splits - 1:  # Get UMAP plot on last iteration only
                artifacts = evaluate_models(infer, y_train, y_test, X_train, y_labels, plot=True)
            else:
                artifacts = evaluate_models(infer, y_train, y_test, X_train)

            # Append split metrics
            for k, v in artifacts.items():
                if 'fig' in k:
                    continue
                split_metrics[k].append(v)

        print('Log artifacts.')
        # Log parameters with mlflow
        mlflow.log_param("cv-folds", 10)
        mlflow.log_params(trial.params)

        # Get averages and standard deviations of metrics
        agg_metrics = {}
        for k, v in split_metrics.items():
            agg_metrics[k + '_mean'] = np.mean(v)
            agg_metrics[k + '_std'] = np.std(v)
        
        # Log metrics with mlflow
        mlflow.log_metrics(agg_metrics)

        # Log figures with mlflow
        mlflow.log_figure(artifacts['train_fig'], 'umap_train_plot.png')
        mlflow.log_figure(artifacts['test_fig'], 'umap_test_plot.png')

        return agg_metrics


def log_best_model(best_trial):
    with mlflow.start_run(run_name='best-model'):
        # Load dataset
        X_clean, y_clean, y_labels = prepare_20newsgroups(os.path.join(outputs_dir, '20newsgroups_prep.csv'))

        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(best_trial)

        # Define embedding model
        emb_model = define_embedding_model(hyperparams)

        # Define topic model
        topic_model = define_topic_model(hyperparams)

        # Define UMAP model for projecting space to 2 dimensions
        umap_model = UMAP(
            n_neighbors=hyperparams['umap_args__n_neighbors'], 
            n_components=2,
            min_dist=0.0,
            metric=hyperparams['umap_args__metric'],
            random_state=1
        )

        # Get train and test samples
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=0, stratify=y_clean)
        
        # Train and infer with topic_model, umap_model and emb_model
        infer = train_infer_models(topic_model, umap_model, emb_model, X_train, X_test)

        # Evaluate the topic_model and umap_model
        artifacts = evaluate_models(infer, y_train, y_test, X_train, y_labels, plot=True)

        print('Log artifacts.')
        # Log parameters with mlflow
        mlflow.log_param("test_size", 0.2)
        mlflow.log_params(best_trial.params)

        # Log metrics with mlflow
        train_fig = artifacts.pop('train_fig')
        test_fig = artifacts.pop('test_fig')
        mlflow.log_metrics(artifacts)

        # Log figures with mlflow
        mlflow.log_figure(train_fig, 'umap_train_plot.png')
        mlflow.log_figure(test_fig, 'umap_test_plot.png')

        # Log best model TODO: set the path where the model is located.
        mlflow.log_artifact(topic_model)


if __name__ == "__main__":
    print("Performing Hyper-parameter tuning.")
    mlflow.set_tracking_uri(outputs_dir)
    mlflow.set_experiment("my-experiment")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1, n_jobs=1)

    # Print optuna study statistics
    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Trial number: ", best_trial.number)
    print("  Loss (trial value): ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    print("  Log best model: ")
    log_best_model(best_trial)