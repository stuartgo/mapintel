# -*- coding: utf-8 -*-
from src.data.text_preprocessing import results_cleaner, join_results
import click
import sys
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import json
from pymongo import MongoClient
from bson import ObjectId
import pandas as pd


# Finding project_dir
project_dir = Path(__file__).resolve().parents[2]


class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)


@click.command()
@click.argument('test_size', type=click.FLOAT, default=0.2)
@click.argument('out_path', type=click.Path(), default=os.path.join(project_dir, "data", "processed"))
def main(test_size, out_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info(f'Making final data set from raw data. Output path: {out_path}. Test size: {test_size}.')

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
    
    logger.info(f'Saving {len(output_results_train) + len(output_results_test)} documents as json...')
    # Saving cleaned results as csv
    df_train = pd.DataFrame(output_results_train); df_train['split'] = 'train'
    df_test = pd.DataFrame(output_results_test); df_test['split'] = 'test'
    pd.concat([df_train, df_test]).to_csv(os.path.join(out_path, "newsapi_docs.csv"), index=False, header=False)
    logger.info(f'Files saved in {out_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # Loading environmental variables
    MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
    MONGOPASSWORD = os.environ.get("MONGOPASSWORD")
    MONGODB = os.environ.get("MONGODB")

    # Connecting to mongodb
    db_client = MongoClient(f"mongodb+srv://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb.e2na5.mongodb.net/{MONGODB}?retryWrites=true&w=majority")

    # Database object
    db = db_client.news

    main()
