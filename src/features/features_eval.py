import collections
import logging
import multiprocessing
import os
from pathlib import Path
from string import punctuation
import re
from random import sample, choice

import numpy as np
import pandas as pd
from gensim import models
# from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from sklearn.linear_model import LogisticRegression

# https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html
# https://radimrehurek.com/gensim/models/doc2cvec.html
# https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html


def predictive_model_score(model, train_docs, test_docs):
    """Report error rate on test_doc sentiments, using supplied model and train_docs"""

    train_targets = [doc.category for doc in train_docs]
    train_regressors = [model.docvecs[doc.tags[0]] for doc in train_docs]
    test_targets = [doc.category for doc in test_docs]
    test_regressors = [model.infer_vector(doc.words) for doc in test_docs]

    # Train Logistic Regression
    logit = LogisticRegression(multi_class='multinomial', n_jobs=-1)
    logit.fit(train_regressors, train_targets)

    # Predict & evaluate
    test_predictions = logit.predict(test_regressors)
    test_scores = logit.score(test_regressors, test_targets)
    return (test_scores, test_predictions, logit)


def compare_documents(base_doc_id, base_doc_rep, similar, compare_corpus):
    """
    Compare a given document with the most similar, second most similar, median and least similar document
    from a corpus of documents
    :param base_doc_id: id of the base document
    :param base_corpus: corpus to base the comparison (unprocessed). Should contain the base document.
    :param similar: similarity list of the base document
    :param compare_corpus: corpus to compare to (unprocessed)
    :return: None
    """
    print(u'TARGET (%d): «%s»\n' % (base_doc_id, base_doc_rep))
    print(u'SIMILAR/DISSIMILAR DOCS ACCORDING TO DOC2VEC:')
    for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(similar) // 2), 
    ('LEAST', len(similar) - 1)]:
        print(u'%s %s: «%s»\n' % (label, similar[index], compare_corpus[similar[index][0]].original))


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(data_file, names=['id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data
    all_docs = [
        NewsDocument([tag], row['id'], row['col'], row['category'], row['prep_text'].split(),
        row['split'], row['text']) for tag, (_, row) in enumerate(df.iterrows())
        if row['prep_text'] is not None
    ]
    train_docs = [doc for doc in all_docs if doc.split == 'train']
    test_docs = [doc for doc in all_docs if doc.split == 'test']
    logger.info('Percentage of documents from test set: {0:.2f}%'.format((len(test_docs)/df.shape[0])*100))
    del df

    # Creating objects to make and store evaluations
    model_scores = collections.defaultdict(lambda: 1.0)
    test_doc_eval = choice(test_docs)
    
    # Evaluating fitted models
    for file in model_files:
        modelname = os.path.splitext(os.path.basename(file))[0]
        logger.info(f'Evaluating fitted {modelname} model...')

        # Loading model
        model = models.doc2vec.Doc2Vec.load(file)

        # Predictive downstream task (i.e. classifying news topics)
        test_scores, test_predictions, logit = predictive_model_score(model, train_docs, test_docs)
        model_scores[modelname] = test_scores
        print("Model %s score: %f " % (model, test_scores))

        # Are inferred vectors close to the precalculated ones?
        ranks = []
        train_sample = sample(train_docs, k=1000)
        for doc in train_sample:
            tags, words = doc.tags, doc.words
            inferred_vector = model.infer_vector(words)
            sims = model.docvecs.most_similar([inferred_vector], topn=model.docvecs.count)
            rank = [tag_sim for tag_sim, _ in sims].index(tags)
            ranks.append(rank)
        # Optimally we want as much documents to be the most similar with themselves (i.e. rank 0)
        print('Are inferred vectors close to the precalculated ones?')
        print(collections.OrderedDict(sorted(collections.Counter(ranks).items())[:10]))

        # Do close documents seem more related than distant ones?
        inferred_unknown_vector = model.infer_vector(test_doc_eval.words)
        sims = model.docvecs.most_similar([inferred_unknown_vector], topn=model.docvecs.count)
        print("Do close documents seem more related than distant ones?")
        compare_documents(test_doc_eval.tags[0], test_doc_eval.original, sims, train_docs)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    data_file = os.path.join(project_dir, "data", "processed", "newsapi_docs.csv")
    model_dir = os.path.join(project_dir, "models", "saved_models")
    model_files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if re.search("^doc2vec.*\.model$", f)]

    # Data structure for holding data for each document
    NewsDocument = collections.namedtuple('NewsDocument', ['tags', 'id', 'col', 'category', 'words', 'split', 'original'])

    main()
