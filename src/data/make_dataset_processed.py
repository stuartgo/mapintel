# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
from pathlib import Path
from string import punctuation

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from joblib import dump
from nltk.corpus import stopwords
from src.data.text_preprocessing import CorpusPreprocess


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Making final data set from raw data.')

    logger.info('Reading data...')
    # Reading data into memory
    df = pd.read_csv(inp_file, names=['id', 'col', 'category', 'text', 'split'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')
    X_train, X_test = df.loc[df['split'] == 'train', 'text'], df.loc[df['split']=='test', 'text']

    logger.info('Preprocessing data...')
    # Preprocessing text
    prep = CorpusPreprocess(stop_words=stopwords.words('english'), lowercase=True, strip_accents=True,
                            strip_punctuation=punctuation, stemmer=True, max_df=0.9, min_df=2)
    df.loc[df['split']=='train', 'processed_text'] = prep.fit_transform(X_train, tokenize=False)
    df.loc[df['split']=='test', 'processed_text'] = prep.transform(X_test, tokenize=False)

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

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    inp_file = os.path.join(project_dir, "data", "interim", 'newsapi_docs.csv')
    out_file = os.path.join(project_dir, "data", "processed", 'newsapi_docs.csv')
    out_fitted = os.path.join(project_dir, "models", "saved_models", "corpus_preprocess.pkl")

    main()
