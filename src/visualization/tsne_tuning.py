"""
Performs Hyperparameter tuning of the t-SNE model.

Receives a path to an embedding model file (doc2vecdbowd100n5mc2t4.model 
by default), loads it and gets the embeddings for the test set.
Plots the t-SNE space for various hyperparameter configurations.

Outputs a figure of the 2D embedding space for each hyperparameter
setting.
"""
import logging
import os
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import ParameterGrid
from src.visualization.embedding_space import read_data, embedding_vectors


def main(model_name):
    logger = logging.getLogger(__name__)

    model_tag = os.path.splitext(os.path.basename(model_name))[0]

    logger.info('Reading data...')
    # Reading data into memory
    unq_topics, train_docs, test_docs = read_data(data_file)

    logger.info('Obtaining document embeddings...')
    # Obtain the vectorized corpus
    _, vect_test_corpus = embedding_vectors(
        model_name, train_docs['prep_text'], test_docs['prep_text'])

    # Fit t-SNE model and obtain 2D projections
    for params in param_grid:
        logger.info(f'Fitting t-SNE model with {params} ...')
        tsne_model = TSNE(**params, n_jobs=-1, verbose=1, random_state=1)
        embedded_test_corpus = tsne_model.fit_transform(vect_test_corpus)

        # Visualize a 2D map of the vectorized corpus
        logger.info('Plotting 2D embedding space...')
        categ_map = dict(
            zip(unq_topics, ['red', 'blue', 'green', 'yellow', 'orange', 'black', 'brown']))

        # Figure
        plt.figure(figsize=(13, 7))
        # Color for each point
        color_points = list(map(lambda x: categ_map[x], test_docs['category']))
        # Scatter plot
        plt.scatter(embedded_test_corpus[:, 0],
                    embedded_test_corpus[:, 1], c=color_points)
        # Produce a legend with the unique colors from the scatter
        handles = [mpatches.Patch(color=c, label=l)
                   for l, c in categ_map.items()]
        plt.legend(handles=handles, loc="upper left",
                   title="Topics", bbox_to_anchor=(0., 0.6, 0.4, 0.4))
        # Set title
        plt.title(f'Test Corpus Embedding Space\n{params}')

        # Save figure
        plt.savefig(os.path.join(
            out_dir, model_tag + "_" + str(params) + ".png"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    data_file = os.path.join(
        project_dir, "data", "processed", "newsapi_docs.csv")
    model_dir = os.path.join(project_dir, "models", "saved_models")
    out_dir = os.path.join(project_dir, "models", "figures", "tse_tuning")

    # Check if out_dir exists. If it doesn't then create the directory
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Default model instance to use for hyperparameter tuning of t-SNE
    default_model = os.path.join(model_dir, "doc2vecdbowd100n5mc2t4.model")

    # t-SNE ParameterGrid
    param_grid = ParameterGrid(
        {
            'perplexity': [15, 30, 45],
            'learning_rate': [200, 400],
            'n_iter': [1000, 2000],
            'init': ['random', 'pca'],
            'metric': ['cosine', 'euclidean']
        }
    )

    main(default_model)
