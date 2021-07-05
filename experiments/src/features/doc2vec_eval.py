"""
Evaluates the doc2vec embeddings in "outputs/saved_embeddings" and
outputs/appends the predictive scores to "outputs/embedding_predictive_scores.csv"
"""
import logging
import os
from collections import defaultdict, namedtuple
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from src import PROJECT_ROOT
from src.features.embedding_eval import (export_results, log_loss_score,
                                         predictive_model_score)
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)

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
        NewsDocument([tag], row['id'], row['col'], row['category'],
                     row['prep_text'].split(), row['split'], row['text'])
        for tag, (_, row) in enumerate(train_docs.iterrows())]
    test_docs = [
        NewsDocument([tag], row['id'], row['col'], row['category'],
                     row['prep_text'].split(), row['split'], row['text'])
        for tag, (_, row) in enumerate(test_docs.iterrows())]

    # Create embeddings generator
    embedding_dict = format_embedding_files(embedding_files)
    gen = embeddings_generator(embedding_dict)

    # Creating objects to store data inside loop
    models_out = defaultdict(lambda: [])
    # Creating constants (invariable across loop iterations)
    train_targets = [doc.category for doc in train_docs]
    test_targets = [doc.category for doc in test_docs]

    # Evaluating embeddings
    for modelname, train_vecs, test_vecs in gen:
        logger.info(f'Evaluating embeddings of {modelname}...')
        # Predictive downstream task (i.e. classifying news topics)
        test_scores, _, _ = predictive_model_score(
            train_vecs, train_targets, test_vecs, test_targets)
        models_out[modelname].append(test_scores)
        print("Model %s predictive score: %f\n" % (modelname, test_scores))

        # Log-loss of predicting whether pairs of observations belong to the same category
        cost = log_loss_score(test_vecs, test_targets)
        models_out[modelname].append(cost)
        print("Model %s log-loss: %f\n" % (modelname, cost))
        print("-----------------------------------------------------------------------------------------")

    # Concatenating doc2vec dm and dbow models
    logger.info('Concatenating PV-DM and PV-DBOW models...')
    modelname_dbow = list(filter(lambda x: "dbow" in x, embedding_dict.keys()))
    modelname_dm = list(filter(lambda x: "dm" in x, embedding_dict.keys()))
    modelname_concat = product(modelname_dbow, modelname_dm)

    for pairs in modelname_concat:
        modelname = "+".join(name for name in pairs)
        logger.info(f'Evaluating concatenated {modelname} model...')

        # Get document vectors and targets
        train_vecs = np.hstack([np.load(embedding_dict[key]['train']) for key in pairs])
        test_vecs = np.hstack([np.load(embedding_dict[key]['test']) for key in pairs])

        # Predictive downstream task (i.e. classifying news topics)
        test_scores, _, _ = predictive_model_score(
            train_vecs, train_targets, test_vecs, test_targets)
        models_out[modelname].append(test_scores)
        logger.info("Model %s predictive score: %f\n" % (modelname, test_scores))

        # Log-loss of predicting whether pairs of observations belong to the same category
        cost = log_loss_score(test_vecs, test_targets)
        models_out[modelname].append(cost)
        logger.info("Model %s log-loss: %f\n" % (modelname, cost))
        print("-----------------------------------------------------------------------------------------")

    # Exporting results
    logger.info(f'Exporting results...')
    models_output = pd.DataFrame(
        models_out, index=["Mean_accuracy", "Log_loss"]).T
    export_results(models_output, out_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    embedding_dir = os.path.join(PROJECT_ROOT, "outputs", "saved_embeddings")
    embedding_files = [os.path.join(embedding_dir, f) for f in os.listdir(
        embedding_dir) if "doc2vec" in f]
    out_path = os.path.join(PROJECT_ROOT, "outputs",
        "embedding_predictive_scores.csv")

    # Data structure for holding data for each document
    NewsDocument = namedtuple(
        'NewsDocument', ['tags', 'id', 'col', 'category', 'words', 'split', 'original'])

    main()
