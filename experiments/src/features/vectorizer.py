"""
Fits a BOW and TF-IDF model to the preprocessed data in 
"data/processed/newsapi_docs.csv" and saves the fitted models for 
posterior use in "outputs/saved_models" and the embedding vectors in
"outputs/saved_embeddings"
"""
import logging
import os

from joblib import dump
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src import PROJECT_ROOT
from src.features.embedding_extractor import read_data, save_embeddings


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    _, train_docs, test_docs = read_data(data_file)

    logger.info('Fitting CountVectorizer...')
    # Fitting BOW representation
    cv = CountVectorizer(**vect_kwargs)
    vect_train_corpus = cv.fit_transform(train_docs['prep_text'])
    vect_test_corpus = cv.transform(test_docs['prep_text'])
    logger.info('Saving fitted CountVectorizer...')
    # Saving fitted model
    dump(cv, os.path.join(output_dir_models, "CountVectorizer.joblib"))
    logger.info("Saving CountVectorizer embeddings...")
    # Saving the embeddings from the train docs
    save_embeddings(
        os.path.join(output_dir_embeddings, "train_CountVectorizer"), vect_train_corpus)
    # Saving the embeddings from the test docs
    save_embeddings(
        os.path.join(output_dir_embeddings, "test_CountVectorizer"), vect_test_corpus)

    logger.info('Fitting TfidfVectorizer...')
    # Fitting BOW representation
    tfidf = TfidfVectorizer(**vect_kwargs)
    vect_train_corpus = tfidf.fit_transform(train_docs['prep_text'])
    vect_test_corpus = tfidf.transform(test_docs['prep_text'])
    logger.info('Saving fitted TfidfVectorizer...')
    # Saving fitted model
    dump(tfidf, os.path.join(output_dir_models, "TfidfVectorizer.joblib"))
    logger.info("Saving TfidfVectorizer embeddings...")
    # Saving the embeddings from the train docs
    save_embeddings(
        os.path.join(output_dir_embeddings, "train_TfidfVectorizer"), vect_train_corpus)
    # Saving the embeddings from the test docs
    save_embeddings(
        os.path.join(output_dir_embeddings, "test_TfidfVectorizer"), vect_test_corpus)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(
        PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
    output_dir_models = os.path.join(
        PROJECT_ROOT, "outputs", "saved_models")
    output_dir_embeddings = os.path.join(
        PROJECT_ROOT, "outputs", "saved_embeddings")

    # Hyperparameter setting
    vect_kwargs = dict(max_features=10000, ngram_range=(1, 3))

    main()
