"""
Performs Hyperparameter tuning of the t-SNE model.

Receives a path to a test corpus embeddings file (doc2vecdbowd100n5mc2*
model by default) and loads it. Plots the t-SNE space for various
hyperparameter configurations of the embeddings of that model.

Outputs a figure of the 2D embedding space for each hyperparameter
setting.
"""
import glob
import logging
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from src import PROJECT_ROOT
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)
from src.visualization.embedding_space import embedding_vectors, read_data


def main(embeddings_file):
    logger = logging.getLogger(__name__)

    split, model_tag = os.path.splitext(os.path.basename(embeddings_file))[0].split("_")
    if split != 'test':
        raise ValueError("Perform the hyperparameter tuning on the test set.")

    logger.info('Reading data...')
    # Reading data into memory
    unq_topics, _, docs = read_data(data_file)

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
    for params in param_grid:
        logger.info(f'Fitting t-SNE model with {params} ...')
        tsne_model = TSNE(**params, n_jobs=-1, verbose=1, random_state=1)
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
        plt.title(f'Test Corpus t-SNE Map - KL: {corpus_kl}\n{params}')

        # Save figure
        plt.savefig(os.path.join(
            out_dir, model_tag + "_" + str(params) + ".png"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    embeddings_dir = os.path.join(PROJECT_ROOT, "outputs", "saved_embeddings")
    out_dir = os.path.join(PROJECT_ROOT, "outputs", "figures", "tse_tuning")

    # Check if out_dir exists. If it doesn't then create the directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Default test embeddings to use for hyperparameter tuning of t-SNE
    # Using glob because the filename depends on the number of threads used to
    # train the model and therefore we set it as a pattern
    default_embeddings = glob.glob(os.path.join(embeddings_dir, "test_doc2vecdmcd100n5w5mc2*.npy"))
    if len(default_embeddings) > 1:
        raise ValueError("There is more than one file with the same pattern as the default document.")
    else:
        default_embeddings = default_embeddings[0]

    # Set maximum dimensionality to apply direct t-SNE
    max_dim = 100

    # t-SNE ParameterGrid
    param_grid = ParameterGrid(
        {
            'perplexity': [5, 10, 40],
            'learning_rate': [200],
            'n_iter': [1000, 2000],
            'init': ['pca'],
            'metric': ['cosine']
        }
    )

    main(default_embeddings)
