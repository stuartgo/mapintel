"""
Provides document embedding evaluation functions for use by each document
embedding evaluation pipeline.
"""
import logging
from collections import Counter
import os
from functools import reduce

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def log_loss_score(X, y):
    """Given a dataset of N documents and their categories, evaluates the document embeddings 
    vectors by classifying whether each of the N(N−1)/2 pairs of documents belongs to the
    same category or not.
    Applies min-max scaling over the cosine similarity of each pair of documents and 
    then uses it as the predicted value for computing the binary cross-entropy cost across
    all N(N−1)/2 pairs of documents.

    Args:
        X (array-like, sparse matrix): Documents' embedding vectors
        y (array-like): Documents' category labels 

    Returns:
        float: Binary cross-entropy cost of pairs of documents
    """
    # Cosine similarity of test set instances
    sim_matrix = cosine_similarity(X)
    # Returns tuple with two arrays, each with the indices along one dimension
    tri_idx = np.triu_indices(len(y), 1)
    # Pairs row and col indices and checks if corresponding observations have the same label
    y_labels = np.array([1 if y[i] == y[j] else 0 for i, j in zip(*tri_idx)])
    # Get unique similarities (upper triangle)
    sim_unique = np.expand_dims(sim_matrix[tri_idx], axis=1)
    # Get probability array of unique upper triangle using MinMaxScaler [0.001, 0.999] to avoid np.log(0)
    y_pred = MinMaxScaler((0.001, 0.999)).fit_transform(sim_unique)[:, 0]
    # Binary cross-entropy
    log_loss = y_labels * np.log(y_pred) + (1 - y_labels) * np.log(1 - y_pred)
    cost = -1 * np.mean(log_loss)
    return cost


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
        multi_class='multinomial', max_iter=200, random_state=0, n_jobs=-1)
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
    print('SIMILAR/DISSIMILAR DOCS:')
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims) // 2),
                         ('LEAST', len(sims) - 1)]:
        output.extend([sims[index][1], compare_corpus[sims[index][0]]])
        print('%s %s: «%s»\n' %
              (label, sims[index], compare_corpus[sims[index][0]]))
    return output


def export_results(model_outputs, out_path):
    """Exports the model_outputs of each embedding model to out_path.
    If the output file already exists, it will preserve model outputs that only
    exist there and concatenate model_outputs, otherwise it will create
    the output file with just model_outputs.

    Args:
        model_outputs (Pandas DataFrame): DataFrame with model name in index
         and the corresponding outputs as values (1 column for each output)
        out_path (string): output path to csv file where to store results
    """
    if os.path.exists(out_path):
        df = pd.read_csv(out_path)
        # Get Models that only exist in the csv file
        set_difference = df.loc[~df['Model'].isin(model_outputs.index)]
        # Concatenate model_outputs with the set_difference
        pd.DataFrame(
            np.concatenate(
                [set_difference.values, model_outputs.reset_index().values], axis=0),
            columns=set_difference.columns
        ).to_csv(out_path, index=False)  # Export to csv
    else:
        model_outputs.reset_index().to_csv(
            out_path, header=['Model', "Mean_accuracy", "Log_loss"], index=False)
