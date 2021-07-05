"""
Self-Organizing Map

"""

import logging
import os

import click
import joblib
import numpy as np
import sompy
from matplotlib.pyplot import show
from src import PROJECT_ROOT
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)

def save_som(som, out_path):
    """Save the SOM into a .som file

    Args:
            som (object): fitted som object
            out_path (str): output path where the som object will 
                be saved
    """
    with open(out_path, 'wb') as outfile:
        joblib.dump(som, outfile)


def load_som(inp_path):
    """Load the SOM from a .som file

    Args:
            inp_path (str): input path from where the som object 
                will be loaded

    Returns:
            [obect]: fitted som object
    """
    with open(inp_path, 'rb') as infile:
        som = joblib.load(infile)
    return som

def check_som_exists(fname, type, dir):
	"""Check if SOM object/visualization with same fname 
        exists before fitting it.

	Args:
		fname (str): filename to check
		type ('som'; 'viz'): whether to check for SOM 
            object or visualization

	Returns:
		[bool]: whether check returns True or False
	"""
	if type == "som":
		fpath = os.path.join(dir, fname + '.som')
	elif type == "viz":
		fpath = os.path.join(dir, fname + '.png')
	else:
		raise ValueError("type can only take 'viz' or 'som' values.")

	# Check path existence
	if os.path.exists(fpath):
		return True
	else:
		return False


@click.command()
@click.argument('embeddings_file', type=click.Path(exists=True))
def main(embeddings_file):
    logger = logging.getLogger(__name__)

    split, model_tag = os.path.splitext(os.path.basename(embeddings_file))[0].split("_")

    logger.info('Reading data...')
    # Reading data into memory
    # Assert we are working with train corpus
    if split == "train":
        _, train_docs, _ = read_data(data_file)
    else:
        raise ValueError("For clustering, use the train set embedding vectors.")

    train_labels = train_docs['category']

    logger.info('Obtaining document embeddings...')
    # Obtain the vectorized corpus
    embedding_dict = format_embedding_files([embeddings_file])
    gen = embeddings_generator(embedding_dict)
    _, vect_train_corpus = list(gen)[0]

    # Setting random seed
    np.random.seed(42)

    som = sompy.SOMFactory().build(
        vect_train_corpus,
        **som_build_kwargs
    ) 
    som.train(n_job=-1, 
        verbose='info',
        **som_train_kwargs
    )

    u = sompy.umatrix.UMatrixView(
        width=7,
        height=7,
        text_size=6,
        title="U-Matrix",
    ) 
    _, umat = u.show(
        som,
        distance=2, 
        contour=True,
        blob=False
    )
    show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(PROJECT_ROOT, "data",
                             "processed", "newsapi_docs.csv")
    model_dir = os.path.join(PROJECT_ROOT, "outputs", "saved_models")
    fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures", "som")

    # SOM kwargs
    som_build_kwargs = dict(
        mapsize=(50, 50),
        initialization='pca',
        neighborhood='gaussian',
        training='batch',
        lattice='hexa'
    )
    som_train_kwargs = dict(
        train_rough_len=5, 
        train_finetune_len=5
    )

    main()
