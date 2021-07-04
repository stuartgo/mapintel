"""
Fits a set of Doc2vec models to the preprocessed data in 
"data/processed/newsapi_docs.csv" and saves the fitted models for 
posterior use in "outputs/saved_models and the embedding vectors in
"outputs/saved_embeddings"
"""
import collections
import logging
import multiprocessing
import os
from string import punctuation

import numpy as np
from gensim import models
from src import PROJECT_ROOT
from src.features.embedding_extractor import read_data, save_embeddings

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# https://radimrehurek.com/gensim/models/doc2cvec.html
# https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    _, train_docs, test_docs = read_data(data_file)

    # Transforming DataFrame to list of namedtuples
    train_docs = [
        NewsDocument([tag], row['id'], row['col'],
                     row['category'], row['prep_text'].split())
        for tag, (_, row) in enumerate(train_docs.iterrows())
    ]

    # Doc2Vec models
    assert models.doc2vec.FAST_VERSION > -1,\
        "Gensim won't use a C compiler, which will severely increase running time."
    for model in simple_models:
        # get vocabulary of training data as weight matrix has |V| rows
        model.build_vocab(train_docs)
        logger.info("%s vocabulary scanned & state initialized" % model)

    for model in simple_models:
        logger.info("Training %s..." % model)
        # Training the model
        model.train(train_docs, total_examples=len(
            train_docs), epochs=model.epochs)

        logger.info("Saving %s..." % model)
        # Saving the fitted model
        model_name = str(model).lower().translate(
            str.maketrans('', '', punctuation))
        model.save(os.path.join(output_dir_models, f"{model_name}.model"))

        logger.info("Saving document embeddings...")
        # Saving the embeddings from the train docs
        vect_train_corpus = np.vstack(
            [model.docvecs[i] for i in range(len(train_docs))])
        save_embeddings(
            os.path.join(output_dir_embeddings, f"train_{model_name}.npy"), vect_train_corpus)

        # Saving the embeddings from the test docs
        vect_test_corpus = np.vstack(
            [model.infer_vector(i) for i in test_docs['prep_text'].str.split()])
        save_embeddings(
            os.path.join(output_dir_embeddings, f"test_{model_name}.npy"), vect_test_corpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    output_dir_models = os.path.join(PROJECT_ROOT, "outputs", "saved_models")
    output_dir_embeddings = os.path.join(PROJECT_ROOT, "outputs", "saved_embeddings")


    # Hyperparameter setting
    common_kwargs = dict(
        vector_size=100, epochs=20, min_count=2,
        sample=0, workers=multiprocessing.cpu_count(), negative=5, hs=0,
    )

    # Models to use
    simple_models = [
        # PV-DBOW plain
        models.doc2vec.Doc2Vec(dm=0, **common_kwargs),
        # PV-DM w/ default averaging; a higher starting alpha may improve CBOW/PV-DM modes
        models.doc2vec.Doc2Vec(dm=1, window=10, alpha=0.05,
                               comment='alpha=0.05', **common_kwargs),
        # PV-DM w/ concatenation - big, slow, experimental mode
        # window=5 (both sides) approximates paper's apparent 10-word total window size
        models.doc2vec.Doc2Vec(dm=1, dm_concat=1, window=5, **common_kwargs),
    ]

    # Data structure for holding data for each document
    NewsDocument = collections.namedtuple(
        'NewsDocument', ['tags', 'id', 'col', 'category', 'words'])

    main()
