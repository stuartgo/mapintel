"""
Fits a BOW and TF-IDF model to the preprocessed data in 
"data/processed/newsapi_docs.csv" and saves the fitted models for 
posterior use in "models/saved_models"
"""
import logging
import os
from pathlib import Path

import pandas as pd
from joblib import dump, load
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def main():
    logger = logging.getLogger(__name__)

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(data_file, names=[
                     'id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data
    train_docs = df.loc[(df['split'] == 'train') &
                        (df['prep_text'] is not None)]
    logger.info('Percentage of documents from train set: {0:.2f}%'.format(
        (train_docs.shape[0]/df.shape[0])*100))
    del df

    logger.info('Fitting CountVectorizer...')
    # Fitting BOW representation
    cv = CountVectorizer(**vect_kwargs)
    cv.fit(train_docs['prep_text'])

    logger.info('Saving fitted CountVectorizer...')
    # Saving fitted model
    dump(cv, os.path.join(output_path, "CountVectorizer.joblib"))

    logger.info('Fitting TfidfVectorizer...')
    # Fitting BOW representation
    tfidf = TfidfVectorizer(**vect_kwargs)
    tfidf.fit(train_docs['prep_text'])

    logger.info('Saving fitted TfidfVectorizer...')
    # Saving fitted model
    dump(tfidf, os.path.join(output_path, "TfidfVectorizer.joblib"))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    data_file = os.path.join(
        project_dir, "data", "processed", "newsapi_docs.csv")
    output_path = os.path.join(
        project_dir, "models", "saved_models")

    # Hyperparameter setting
    vect_kwargs = dict(max_features=10000, ngram_range=(1, 3))

    main()
