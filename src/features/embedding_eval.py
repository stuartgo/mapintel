"""
Provides document embedding evaluation functions for use by each document
embedding evaluation pipeline.
"""
import logging
from collections import Counter
import os
from functools import reduce

import pandas as pd
from sklearn.linear_model import LogisticRegression
# TODO: Think about which classification evaluation measure(s) to use

logger = logging.getLogger(__name__)


def predictive_model_score(X_train, y_train, X_test, y_test):
    """Evaluates the document embeddings by applying a LogisticRegression 
     classifier on top and predicting the documents' categories.

    Args:
        X_train (array-like, sparse matrix): Training data embeddings
        y_train (array-like): Training data labels
        X_test (array-like, sparse matrix): Test data embeddings
        y_test (array-like): Test data labels

    Returns:
        tuple: outputs test scores (subset accuracy), test predictions and the 
        fitted model instance
    """
    # Train Logistic Regression
    logit = LogisticRegression(
        multi_class='multinomial', max_iter=200, n_jobs=-1)
    logit.fit(X_train, y_train)

    # Predict & evaluate
    test_predictions = logit.predict(X_test)
    test_scores = logit.score(X_test, y_test)
    return (test_scores, test_predictions, logit)


def evaluate_inferred_vectors(model, train_docs):
    """Evaluates the inferred vectors by comparing with the fitted ones. 
    For each element in train_docs, infer the embedding vector, compute the
     cosine similarity with the fitted embedding vectors and extract 
     self-similarity rank (i.e. position at which fitted vector is the most 
     similar to inferred vector). Obtains the distribution of self-similiraty 
     rank. Optimally we want as much documents to be the most similar with
     themselves.

    NOTE: AS IT IS RIGHT NOW, THIS FUNCTION ONLY WORKS IN DOC2VEC PIPELINE

    Args:
        model (Doc2Vec): fitted Doc2Vec model instance
        train_docs (list of namedtuple instances): namedtuple formatted
         training data (must have "tags" and "words" field)

    Returns:
        list: distribution of self-similiraty rank (up to 10th rank)
    """
    ranks = []
    for doc in train_docs:
        tags, words = doc.tags, doc.words
        inferred_vector = model.infer_vector(words)
        sims = model.docvecs.most_similar(
            [inferred_vector], topn=model.docvecs.count)
        rank = [tag_sim for tag_sim, _ in sims].index(tags)
        ranks.append(rank)
    top10_distribution = list(
        map(lambda x: x[1], sorted(Counter(ranks).items())[:10]))
    top10_count = reduce(lambda a, b: a + b, top10_distribution)
    if top10_count < 1000:
        top10_distribution[-1] += (1000 - top10_count)
    return top10_distribution


def compare_documents(base_doc_id, base_doc_rep, sims, compare_corpus):
    """Compare a base document with the most similar, second most similar, 
     median and least similar document from a corpus of documents.

    Args:
        base_doc_id (int): id of the base document
        base_doc_rep (string): base document representation (raw text)
        sims (list of tuples (doc_id, distance)): similarities of base 
         document with compare_corpus
        compare_corpus (array-like): corpus used to compute sims list 
        (raw text)

    Returns:
        list: output list holds base_doc_rep, distance and compare_doc_rep 
         for each compare document
    """
    output = [base_doc_rep]
    print('TARGET (%d): «%s»\n' % (base_doc_id, base_doc_rep))
    print('SIMILAR/DISSIMILAR DOCS ACCORDING TO DOC2VEC:')
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2),
                         ('LEAST', len(sims) - 1)]:
        output.extend([sims[index][1], compare_corpus[sims[index][0]]])
        print('%s %s: «%s»\n' %
              (label, sims[index], compare_corpus[sims[index][0]]))
    return output


def export_results(predictive_scores, out_path, overwrite=False):
    """Exports predictive scores of each embedding model to out_path.
    If overwrite=True it will overwrite the file, otherwise it will 
     append results if the file exists and the models don't yet have
     any predictive score registered.

    Args:
        predictive_scores (Pandas Series): Series with model name in index
         and the corresponding predictive score values
        out_path (string): output path to csv file where to store results
        overwrite (bool, optional): If True, overwrites the out_path file.
         Defaults to False.
    """
    if overwrite:
        predictive_scores.to_csv(out_path, index_label='Model')
    else:
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            existing_models = df['Model'].unique()
            non_existing_scores = predictive_scores.loc[~predictive_scores.index.isin(
                existing_models)]
            if non_existing_scores.empty:
                logger.warning(f'predictive_scores already in out_path file.')
                return
            non_existing_scores.to_csv(out_path, mode='a', header=False)
        else:
            predictive_scores.to_csv(out_path, index_label='Model')
