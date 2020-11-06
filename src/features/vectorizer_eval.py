"""
Evaluates the BOW and TF-IDF embeddings in "models/saved_models" and outputs/
appends the predictive scores to "models/embedding_predictive_scores.csv"
"""
from collections import defaultdict
import logging
import os
import re
from pathlib import Path

import pandas as pd
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from src.features.embedding_eval import (compare_documents,
                                         export_results,
                                         predictive_model_score,
                                         log_loss_score)


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(data_file, names=[
                     'id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data
    all_docs = df.loc[~df['prep_text'].isna()]
    train_docs = all_docs.loc[all_docs['split'] == 'train']
    test_docs = all_docs.loc[all_docs['split'] == 'test']
    logger.info(
        f'{train_docs.shape[0]} documents from train set out of {df.shape[0]} documents')
    del df

    # Loading fitted models
    logger.info('Loading fitted models...')
    model_instances = [load(file) for file in model_files]

    # Creating objects to store data inside loop
    models_out = defaultdict(lambda: [])

    # Creating constants (invariable across loop iterations)
    # random test doc to evaluate distances
    test_doc_eval = test_docs.sample(1)

    # Evaluating fitted models
    for model in model_instances:
        modelname = re.sub(' ', '', str(model))
        logger.info(f'Evaluating fitted {modelname} model...')

        # Get document vectors
        train_vecs = model.transform(train_docs['prep_text'])
        test_vecs = model.transform(test_docs['prep_text'])

        # Predictive downstream task (i.e. classifying news topics)
        test_scores, _, _ = predictive_model_score(train_vecs,
                                                   train_docs['category'], test_vecs, test_docs['category'])
        models_out[modelname].append(test_scores)
        print("Model %s predictive score: %f\n" % (modelname, test_scores))

        # Log-loss of predicting whether pairs of observations belong to the same category
        cost = log_loss_score(test_vecs, test_docs['category'].tolist())
        models_out[modelname].append(cost)
        print("Model %s log-loss: %f\n" % (modelname, cost))

        # Get cosine similarity between random test doc and train docs
        train_vectors = model.transform(train_docs['prep_text'])
        inferred_unknown_vector = model.transform(test_doc_eval['prep_text'])
        sims = cosine_similarity(inferred_unknown_vector, train_vectors)[0]
        sims = sorted(list(zip(train_docs.index, sims)),
                      reverse=True, key=lambda x: x[1])

        # Do close documents seem more related than distant ones?
        print("Do close documents seem more related than distant ones?")
        compare_documents(
            test_doc_eval.index[0], test_doc_eval['text'].iloc[0], sims, train_docs['text'])
        print("-----------------------------------------------------------------------------------------")

    # Exporting results
    logger.info(f'Exporting results...')
    models_output = pd.DataFrame(
        models_out, index=["Mean_accuracy", "Log_loss"]).T
    export_results(models_output, out_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    data_file = os.path.join(
        project_dir, "data", "processed", "newsapi_docs.csv")
    model_dir = os.path.join(project_dir, "models", "saved_models")
    model_files = [os.path.join(model_dir, f) for f in os.listdir(
        model_dir) if re.search(".*Vectorizer\.joblib$", f)]
    out_path = os.path.join(project_dir, "models",
                            "embedding_predictive_scores.csv")

    main()
