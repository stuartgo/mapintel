"""
Visualize the document embeddings produced by a given model using t-SNE.

Receives a path to an embedding model file as an argument, loads it and
gets the embeddings for the training and test set. If the embedding 
dimensionality is large, LSA is applied to suppress some noise and speed
up the computations required by t-SNE. 

Outputs a figure of the 2D embedding space for the passed model.

Examples
--------
>>> # Pass a .model file as a positional argument to the script
>>> python src/visualization/embedding_space.py models/saved_models/file.model
"""

import logging
import os
import re
from pathlib import Path

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import models
from joblib import load
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE

# TODO: to get the optimal t-SNE maps we can run t-sne 10 times and get the map with lowest KL
# "It is perfectly fine to run t-SNE ten times, and select the solution with the lowest KL divergence"

def read_data(data_file):
    """Read the preprocessed news corpus and splits it into training and test set.

    Args:
        data_file (str): path to csv file that holds the preprocessed data

    Returns:
        tuple: unique categories list, training documents and test documents
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(data_file, names=[
                     'id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data
    all_docs = df.loc[~df['prep_text'].isna()]
    train_docs = all_docs.loc[all_docs['split'] == 'train']
    test_docs = all_docs.loc[all_docs['split'] == 'test']
    unq_topics = df['category'].unique().tolist()
    logger.info(
        f'{train_docs.shape[0]} documents from train set out of {df.shape[0]} documents')
    del df, all_docs

    return unq_topics, train_docs, test_docs


def embedding_vectors(model_name, train_docs, test_docs):
    """Get embedding vectors from train_docs and test_docs using a given model.
    The function uses the model_name to extract the model type and then obtain
     the embeddings in the correct way for the given model.
    Applies LSA dimensionality reduction to BOW and TF-IDF vectors to reduce
     their dimensionality to 100 components.

    Args:
        model_name (str): Complete path to model file
        train_docs (array-like): Strings of each train document
        test_docs (array-like): Strings of each test document

    Raises:
        ValueError: When it isn't possible to infer model type off model_name

    Returns:
        tuple of ndarray or sparse matrix: embedding vectors of train documents and test documents
    """
    logger = logging.getLogger(__name__)
    # Extract model_type out model_tag
    if re.search(".*Vectorizer\.joblib$", os.path.basename(model_name)):
        model_type = "vectorizer"
    elif re.search("^doc2vec.*\.model$", os.path.basename(model_name)):
        model_type = "doc2vec"
    else:
        raise ValueError("Couldn't extract valid model_type from model_name.")

    if model_type == "doc2vec":
        # Loading fitted model
        model = models.doc2vec.Doc2Vec.load(model_name)
        # Obtain the vectorized corpus
        vect_train_corpus = np.vstack(
            [model.docvecs[i] for i in range(train_docs.shape[0])])
        vect_test_corpus = np.vstack(
            [model.infer_vector(i) for i in test_docs.str.split()])

        return vect_train_corpus, vect_test_corpus

    elif model_type == "vectorizer":
        # Loading fitted model
        model = load(model_name)
        # Obtain the vectorized corpus
        vect_train_corpus = model.transform(train_docs)
        vect_test_corpus = model.transform(test_docs)
        # LSA to reduce vector dimensionality
        logger.info("Applying LSA to reduce dimensionality...")
        lsa = TruncatedSVD(100, random_state=1)
        vect_train_corpus_red = lsa.fit_transform(vect_train_corpus)
        vect_test_corpus_red = lsa.transform(vect_test_corpus)
        explained_variance = lsa.explained_variance_ratio_.sum()
        logger.info("Explained variance of the LSA step: {}%".format(
            int(explained_variance * 100)))

        return vect_train_corpus_red, vect_test_corpus_red


@click.command()
@click.argument('model_name', type=click.Path(exists=True))
def main(model_name):
    logger = logging.getLogger(__name__)

    model_tag = os.path.splitext(os.path.basename(model_name))[0]

    logger.info('Reading data...')
    # Reading data into memory
    unq_topics, train_docs, test_docs = read_data(data_file)

    logger.info('Obtaining document embeddings...')
    # Obtain the vectorized corpus
    vect_train_corpus, vect_test_corpus = embedding_vectors(
        model_name, train_docs['prep_text'], test_docs['prep_text'])

    # Fit t-SNE model and obtain 2D projections
    logger.info('Fitting t-SNE model...')
    tsne_model = TSNE(**tsne_kwargs, n_jobs=-1, verbose=1)
    embedded_train_corpus = tsne_model.fit_transform(vect_train_corpus)
    train_corpus_KL = tsne_model.kl_divergence_
    embedded_test_corpus = tsne_model.fit_transform(vect_test_corpus)
    test_corpus_KL = tsne_model.kl_divergence_

    # Visualize a 2D map of the vectorized corpus
    logger.info('Plotting 2D embedding space...')
    categ_map = dict(
        zip(unq_topics, ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'brown']))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    iterator = zip(
        axes.flatten(),  # axes
        [embedded_train_corpus, embedded_test_corpus],  # corpora
        [train_docs['category'], test_docs['category']],  # topics
        [f'Train Corpus t-SNE Map - KL: {test_corpus_KL}', f'Test Corpus t-SNE Map - KL: {train_corpus_KL}']  # titles
    )

    for ax, X, top, tit in iterator:
        # Color for each point
        color_points = list(map(lambda x: categ_map[x], top))
        # Scatter plot
        ax.scatter(X[:, 0], X[:, 1], c=color_points)
        # Produce a legend with the unique colors from the scatter
        handles = [mpatches.Patch(color=c, label=l)
                   for l, c in categ_map.items()]
        ax.legend(handles=handles, loc="upper left",
                  title="Topics", bbox_to_anchor=(0., 0.6, 0.4, 0.4))
        # Set title
        ax.set_title(tit)

    # Plot and save figure
    plt.savefig(os.path.join(out_dir, model_tag + ".png"))
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    data_file = os.path.join(
        project_dir, "data", "processed", "newsapi_docs.csv")
    model_dir = os.path.join(project_dir, "models", "saved_models")
    out_dir = os.path.join(project_dir, "models", "figures")

    # t-SNE kwargs
    tsne_kwargs = dict(
        n_components=2, perplexity=30, learning_rate=200, n_iter=1000,
        n_iter_without_progress=300, metric='cosine', init='pca', random_state=1
    )

    main()
