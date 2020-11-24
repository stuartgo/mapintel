"""
Evaluates the doc2vec embeddings in "models/saved_models" and outputs/appends
the predictive scores to "models/embedding_predictive_scores.csv"
"""
import logging
import os
import re
from collections import defaultdict, namedtuple
from itertools import product
from random import choice, sample

import pandas as pd
from gensim import models
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from src import PROJECT_ROOT
from src.features.embedding_eval import (compare_documents,
                                         evaluate_inferred_vectors,
                                         export_results, log_loss_score,
                                         predictive_model_score)

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# https://radimrehurek.com/gensim/models/doc2cvec.html
# https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(data_file, names=[
                     'id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data from DataFrame to Named Tuple for doc2vec training
    all_docs = [
        NewsDocument([tag], row['id'], row['col'], row['category'], row['prep_text'].split(),
                     row['split'], row['text']) for tag, (_, row) in enumerate(df.iterrows())
        if row['prep_text'] is not None
    ]
    train_docs = [doc for doc in all_docs if doc.split == 'train']
    test_docs = [doc for doc in all_docs if doc.split == 'test']
    logger.info(
        f'{len(train_docs)} documents from train set out of {df.shape[0]} documents')
    del df

    # Loading fitted models
    logger.info('Loading fitted models...')
    model_instances = [models.doc2vec.Doc2Vec.load(
        file) for file in model_files]

    # Creating objects to store data inside loop
    models_out = defaultdict(lambda: [])

    # Creating constants (invariable across loop iterations)
    # sample train docs to evaluate inferred vs learned vectors
    train_samples = sample(train_docs, k=1000)
    test_doc_eval = choice(test_docs)  # random test doc to evaluate distances
    train_targets = [doc.category for doc in train_docs]
    test_targets = [doc.category for doc in test_docs]

    # Evaluating fitted models
    for model in model_instances:
        modelname = str(model)
        logger.info(f'Evaluating fitted {modelname} model...')

        # Get document vectors and targets
        train_vecs = [model.docvecs[doc.tags[0]] for doc in train_docs]
        test_vecs = [model.infer_vector(doc.words) for doc in test_docs]

        # Predictive downstream task (i.e. classifying news topics)
        test_scores, _, _ = predictive_model_score(
            train_vecs, train_targets, test_vecs, test_targets)
        models_out[modelname].append(test_scores)
        print("Model %s predictive score: %f\n" % (modelname, test_scores))

        # Log-loss of predicting whether pairs of observations belong to the same category
        cost = log_loss_score(test_vecs, test_targets)
        models_out[modelname].append(cost)
        print("Model %s log-loss: %f\n" % (modelname, cost))

        # Are inferred vectors close to the precalculated ones?
        top10_distribution = evaluate_inferred_vectors(model, train_samples)
        # models_out[modelname].append(top10_distribution[0] / 1000)
        print('Are inferred vectors close to the precalculated ones?')
        # We want documents to be the most similar with themselves (i.e. rank 0)
        print(top10_distribution, "\n")

        # Get cosine similarity between random test doc and train docs
        inferred_unknown_vector = model.infer_vector(test_doc_eval.words)
        sims = model.docvecs.most_similar(
            [inferred_unknown_vector], topn=model.docvecs.count)

        # Do close documents seem more related than distant ones?
        print("Do close documents seem more related than distant ones?")
        compare_out = compare_documents(test_doc_eval.tags[0], test_doc_eval.original, sims,
                                        list(map(lambda x: x.original, train_docs)))
        # models_out[modelname].append(compare_out)
        print("-----------------------------------------------------------------------------------------")

    # Concatenating doc2vec dm and dbow models
    logger.info('Concatenating PV-DM and PV-DBOW models...')
    model_instances_dbow = list(filter(lambda x: x.dm == 0, model_instances))
    model_instances_dm = list(filter(lambda x: x.dm == 1, model_instances))
    model_instances_concat = [ConcatenatedDoc2Vec(
        pair) for pair in product(model_instances_dbow, model_instances_dm)]

    for model in model_instances_concat:
        modelname = str(model)
        logger.info(f'Evaluating concatenated {modelname} model...')

        # Get document vectors and targets
        train_vecs = [model.docvecs[doc.tags[0]] for doc in train_docs]
        test_vecs = [model.infer_vector(doc.words) for doc in test_docs]

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

    # Exporting results
    logger.info(f'Exporting results...')
    models_output = pd.DataFrame(
        models_out, index=["Mean_accuracy", "Log_loss"]).T
    export_results(models_output, out_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.WARNING, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
    model_files = [os.path.join(model_dir, f) for f in os.listdir(
        model_dir) if re.search("^doc2vec.*\.model$", f)]
    out_path = os.path.join(PROJECT_ROOT, "models",
                            "embedding_predictive_scores.csv")

    # Data structure for holding data for each document
    NewsDocument = namedtuple(
        'NewsDocument', ['tags', 'id', 'col', 'category', 'words', 'split', 'original'])

    main()
