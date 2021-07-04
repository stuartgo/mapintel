# -*- coding: utf-8 -*-~
"""
Builds the processed (model ready) csv file with preprocessed documents.
Reads "data/interim/newsapi_docs.csv" file with cleaned mongodb data and 
outputs the same preprocessed documents to "data/processed/newsapi_docs.csv"
"""
import logging
import os
from string import punctuation

import pandas as pd
from joblib import dump
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from src.data.text_preprocessing import CorpusPreprocess
from src import PROJECT_ROOT


def main():
    logger = logging.getLogger(__name__)
    logger.info(f'Making final data set from raw data.')

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(inp_file, names=[
                     'id', 'col', 'category', 'text', 'split'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')
    X_train, X_test = df.loc[df['split'] == 'train',
                             'text'], df.loc[df['split'] == 'test', 'text']

    logger.info('Preprocessing data...')
    # Preprocessing text
    prep = CorpusPreprocess(stop_words=stopwords.words('english'), lowercase=True, strip_accents=True,
                            strip_punctuation=punctuation, stemmer=PorterStemmer(), max_df=0.9, min_df=2)
    df.loc[df['split'] == 'train', 'processed_text'] = prep.fit_transform(
        X_train, tokenize=False)
    df.loc[df['split'] == 'test', 'processed_text'] = prep.transform(
        X_test, tokenize=False)

    logger.info('Saving fitted transformer...')
    # Saving fitted transformer for applying preprocessing on new samples
    dump(prep, out_fitted)

    logger.info(f'Saving {df.shape[0]} processed documents as csv...')
    # Saving cleaned results as csv
    df.to_csv(out_file, index=False, header=False)
    logger.info(f'File saved in {out_file}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    inp_file = os.path.join(PROJECT_ROOT, "data", "interim", 'newsapi_docs.csv')
    out_file = os.path.join(PROJECT_ROOT, "data",
                            "processed", 'newsapi_docs.csv')
    out_fitted = os.path.join(PROJECT_ROOT, "outputs",
                              "saved_models", "CorpusPreprocess.joblib")

    main()
