# -*- coding: utf-8 -*-
import json
import logging
import os
import sys
from pathlib import Path

import click
import pandas as pd
from bson import ObjectId
from dotenv import find_dotenv, load_dotenv
from pymongo import MongoClient
from src.data.text_preprocessing import (join_results, results_cleaner)


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


@click.command()
@click.argument('test_size', type=click.FLOAT, default=0.2)
def main(test_size):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Making final data set from raw data. Test size: {test_size}.')

    collection_list = db.list_collection_names()
    for col in collection_list:
        logger.info(f"Collection {col} contains {db[col].count_documents({})} documents")

    # Getting text from each article
    pipeline = [
        {  # project fields
            '$project': {
                '_id': 1,
                'category': 1,
                'text': {
                    '$concat': [
                        {'$ifNull': ['$title', '']},
                        ' ',
                        {'$ifNull': ['$description', '']},
                        ' ',
                        {'$ifNull': ['$content', '']},
                    ]
                }
            }
        }
    ]
    # Saving aggregation pipeline results into files
    prep_results = []
    for col in collection_list:
        logger.info(f'Cleaning articles from collection {col}...')
        results = list(db[col].aggregate(pipeline))  # materializing query
        for r in results:
            r.setdefault('col', col)
        clean_results = results_cleaner(results)
        prep_results += clean_results
    logger.info('Joining documents from both collections...')
    output_results_train, output_results_test = join_results(prep_results, test_size)
    
    logger.info(f'Saving {len(output_results_train) + len(output_results_test)} documents as csv...')
    # Saving cleaned results as csv
    df_train = pd.DataFrame(output_results_train); df_train['split'] = 'train'
    df_test = pd.DataFrame(output_results_test); df_test['split'] = 'test'
    pd.concat([df_train, df_test]).to_csv(out_file, index=False, header=False)
    logger.info(f'File saved in {out_file}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Finding project_dir
    project_dir = Path(__file__).resolve().parents[2]
    out_file = os.path.join(project_dir, "data", "interim", "newsapi_docs.csv")

    # Loading environmental variables
    MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
    MONGOPASSWORD = os.environ.get("MONGOPASSWORD")
    MONGODB = os.environ.get("MONGODB")

    # Connecting to mongodb
    db_client = MongoClient(f"mongodb+srv://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb.e2na5.mongodb.net/{MONGODB}?retryWrites=true&w=majority")

    # Database object
    db = db_client.news

    main()
