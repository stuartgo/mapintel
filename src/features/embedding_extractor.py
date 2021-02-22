"""
Provides document embedding saving and loading functions for use
by each document embedding creation pipeline.
"""
import logging
import os

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import issparse, save_npz, load_npz


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
    all_docs = df.loc[~df['prep_text'].isna()]  # get only documents with prep_text
    train_docs = all_docs.loc[all_docs['split'] == 'train']
    test_docs = all_docs.loc[all_docs['split'] == 'test']
    unq_topics = all_docs['category'].unique().tolist()
    logger.info(
        f'{train_docs.shape[0]} documents from train set out of {df.shape[0]} documents')

    return unq_topics, train_docs, test_docs


def format_embedding_files(embedding_files):
    """Formats the embedding_files list into a 2-level dictionary with
    the model names and the respective train and test embedding files

    :param embedding_files: list of npy/npz files holding the embeddings
    :return: 2-level dictionary with the model names and the respective
    train and test embedding files
    """
    # Forming embeddings dictionary
    embeddings = defaultdict(lambda: defaultdict(str))
    for file in embedding_files:
        split, model = os.path.splitext(os.path.basename(file))[0].split("_")
        embeddings[model][split] = file
    return embeddings


def embeddings_generator(embeddings):
    """Generator that loads a set of npy files corresponding to the train
    and/or test embeddings of a specific model

    :param embeddings: 2-level dictionary with model names and the respective
    train and/or test embedding files
    :return: yields the model name, the train and/or test corpus embeddings for
    a given model in embeddings
    """
    logger = logging.getLogger(__name__)
    # Generator that load the embedding vectors for each model
    for model in embeddings:
        logger.info(f'Loading embeddings of {model}...')
        # Check whether the file holds a sparse matrix
        if os.path.splitext(list(embeddings[model].values())[0])[1] == ".npz":
            if not embeddings[model]['train']:
                yield model, load_npz(embeddings[model]['test'])
            elif not embeddings[model]['test']:
                yield model, load_npz(embeddings[model]['train'])
            else:
                yield model, load_npz(embeddings[model]['train']), load_npz(embeddings[model]['test'])
        else:
            if not embeddings[model]['train']:
                yield model, np.load(embeddings[model]['test'])
            elif not embeddings[model]['test']:
                yield model, np.load(embeddings[model]['train'])
            else:
                yield model, np.load(embeddings[model]['train']), np.load(embeddings[model]['test'])


def save_embeddings(output_file, vectors):
    """Saves embedding vectors to output_dir

    :param output_file: string path of the file that will hold the embeddings
    :param vectors: embedding vectors to save to disk
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Saving embeddings to {output_file}...')
    if os.path.exists(output_file):
        os.remove(output_file)
    if issparse(vectors):
        save_npz(output_file, vectors)
    else:
        np.save(output_file, vectors)
