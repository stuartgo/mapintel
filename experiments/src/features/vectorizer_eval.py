"""
Evaluates the BOW and TF-IDF embeddings in "outputs/saved_models" and outputs/
appends the predictive scores to "outputs/embedding_predictive_scores.csv"
"""
import logging
import os
import re
from collections import defaultdict

import pandas as pd
from src import PROJECT_ROOT
from src.features.embedding_eval import (export_results, log_loss_score,
                                         predictive_model_score)
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    _, train_docs, test_docs = read_data(data_file)

    # Create embeddings generator
    embedding_dict = format_embedding_files(embedding_files)
    gen = embeddings_generator(embedding_dict)

    # Creating objects to store data inside loop
    models_out = defaultdict(lambda: [])
    # Creating constants (invariable across loop iterations)
    train_targets = train_docs['category']
    test_targets = test_docs['category']

    # Evaluating embeddings
    for modelname, train_vecs, test_vecs in gen:
        logger.info(f'Evaluating embeddings of {modelname} model...')
        # Predictive downstream task (i.e. classifying news topics)
        test_scores, _, _ = predictive_model_score(
            train_vecs, train_targets, test_vecs, test_targets)
        models_out[modelname].append(test_scores)
        logger.info("Model %s predictive score: %f\n" % (modelname, test_scores))

        # Log-loss of predicting whether pairs of observations belong to the same category
        cost = log_loss_score(test_vecs, test_targets.to_list())
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
        embedding_dir) if "Vectorizer" in f]
    out_path = os.path.join(PROJECT_ROOT, "outputs",
        "embedding_predictive_scores.csv")

    main()
