"""
Visualize the document embeddings produced by a given model using t-SNE.

Receives a path to a document embedding file as an argument and loads it to
obtain the corpus embeddings. If the embedding dimensionality is large,
LSA is applied to suppress some noise and speed up the computations required
by t-SNE.

Outputs a figure of the 2D embedding space for the passed document embeddings.

Examples
--------
>>> # Pass a .model file as a positional argument to the script
>>> python src/visualization/embedding_space.py outputs/saved_embeddings/test_CountVectorizer.npz
"""

import logging
import os
import re
from pathlib import Path

import click
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from src import PROJECT_ROOT
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)

# TODO: to get the optimal t-SNE maps we can run t-sne 10 times and get the map with lowest KL
# "It is perfectly fine to run t-SNE ten times, and select the solution with the lowest KL divergence"


@click.command()
@click.argument('embeddings_file', type=click.Path(exists=True))
def main(embeddings_file):
    logger = logging.getLogger(__name__)

    split, model_tag = os.path.splitext(os.path.basename(embeddings_file))[0].split("_")

    logger.info('Reading data...')
    # Reading data into memory
    # Identify if we are working with train or test corpus
    if split == "train":
        unq_topics, docs, _ = read_data(data_file)
    elif split == "test":
        unq_topics, _, docs = read_data(data_file)
    else:
        raise ValueError("Could not identify whether the embedding vectors are from the \
        train or test corpus.")

    logger.info('Obtaining document embeddings...')
    # Obtain the vectorized corpus
    embedding_dict = format_embedding_files([embeddings_file])
    gen = embeddings_generator(embedding_dict)
    _, vect_corpus = list(gen)[0]

    # Apply TruncatedSVD (aka LSA) for large dimensionality embeddings
    if vect_corpus.shape[1] > max_dim:
        logger.info("Applying LSA to reduce dimensionality...")
        lsa = TruncatedSVD(100, random_state=1)
        vect_corpus = lsa.fit_transform(vect_corpus)
        explained_variance = lsa.explained_variance_ratio_.sum()
        logger.info("Explained variance of the LSA step: {}%".format(
            int(explained_variance * 100)))

    # Fit t-SNE model and obtain 2D projections
    logger.info('Fitting t-SNE model...')
    tsne_model = TSNE(**tsne_kwargs, n_jobs=-1, verbose=1)
    tsne_corpus = tsne_model.fit_transform(vect_corpus)
    corpus_kl = tsne_model.kl_divergence_

    # Visualize a 2D map of the vectorized corpus
    logger.info('Plotting 2D embedding space...')
    categ_map = dict(
        zip(unq_topics, ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'brown']))

    # Figure
    plt.figure(figsize=(11, 7))
    # Color for each point
    color_points = list(map(lambda x: categ_map[x], docs['category']))
    # Scatter plot
    plt.scatter(tsne_corpus[:, 0], tsne_corpus[:, 1], c=color_points)
    # Produce a legend with the unique colors from the scatter
    handles = [mpatches.Patch(color=c, label=l) for l, c in categ_map.items()]
    plt.legend(handles=handles, loc="upper left",
               title="Topics", bbox_to_anchor=(0., 0.6, 0.4, 0.4))
    # Set title
    plt.title(f'{split.capitalize()} Corpus t-SNE Map - KL: {corpus_kl}')

    # Plot and save figure
    plt.savefig(os.path.join(out_dir, model_tag + ".png"))
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")

    # Set maximum dimensionality to apply direct t-SNE
    max_dim = 100

    # t-SNE kwargs
    tsne_kwargs = dict(
        n_components=2, perplexity=30, learning_rate=200, n_iter=1000,
        n_iter_without_progress=300, metric='cosine', init='pca', random_state=1
    )

    main()
