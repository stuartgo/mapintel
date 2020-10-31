"""
AWS lambda function. Script that will be executed by AWS lambda service.
Requests documents from NewsAPI, removes documents without content and 
description and inserts the documents into the MongoDB database.
"""
import json
import logging
import os
from datetime import datetime, timedelta

import newsapi
from pymongo import MongoClient
from pymongo.errors import BulkWriteError
# import sys
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv())

# TODO: set up a VPC peering connection between your AWS VPC and MongoDB Atlas (we want to deploy a production grade setup.
#  This means we won’t connect over the open internet);
# TODO: Try to re-use the database connection (https://www.mongodb.com/blog/post/optimizing-aws-lambda-performance-with-mongodb-atlas-and-nodejs)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

# Getting environmental variables
NEWSAPIKEY = os.environ.get("NEWSAPIKEY")
MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
MONGOPASSWORD = os.environ.get("MONGOPASSWORD")
MONGODB = os.environ.get("MONGODB")

# Establish connections to NewsAPI and MongoDB
news_client = newsapi.NewsApiClient(api_key=NEWSAPIKEY)
# db_client = MongoClient(f"mongodb://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb-shard-00-00.e2na5.mongodb.net:27017,newsapi-mongodb-shard-00-01.e2na5.mongodb.net:27017,newsapi-mongodb-shard-00-02.e2na5.mongodb.net:27017/{MONGODB}?ssl=true&replicaSet=atlas-pwkj8y-shard-0&authSource=admin&retryWrites=true&w=majority")
db_client = MongoClient(
    f"mongodb+srv://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb.e2na5.mongodb.net/{MONGODB}?retryWrites=true&w=majority")


def lambda_handler(event, context):
    logger = logging.getLogger(__name__)

    # Database object
    db = db_client.news

    # Get top_headlines
    logger.info("Getting top_headlines results...")
    top_headlines_results = 0
    top_headlines = []
    for categ in newsapi.const.categories:
        request = news_client.get_top_headlines(
            category=categ, language="en", country="us", page_size=100)
        top_headlines_results += request["totalResults"]
        # adding category field and updating source field to every document in articles
        list(map(lambda x: x.update(
            {"category": categ, "source": x["source"]["id"]}), request["articles"]))
        # getting source id
        top_headlines += (request["articles"])

    # Get everything using source categories
    logger.info("Getting everything results...")
    everything_results = 0
    everything_sources = set()
    everything = []
    for categ in newsapi.const.categories:
        sources = set(map(lambda x: x["id"], news_client.get_sources(
            category=categ, language="en")["sources"]))
        request = news_client.get_everything(sources=",".join(sources), from_param=datetime.today() - timedelta(1),
                                             to=datetime.today(), page_size=100)
        everything_results += request["totalResults"]
        # Guaranteeing requests only come from requested sources
        request_sources = set(
            map(lambda x: x["source"]['id'], request["articles"]))
        diff_sources = request_sources.difference(sources)
        # tracking which sources used
        everything_sources.update(request_sources)
        request["articles"] = list(
            filter(lambda x: x["source"]["id"] not in diff_sources, request["articles"]))
        # Adding category field and updating source field to every document in articles
        list(map(lambda x: x.update(
            {"category": categ, "source": x["source"]["id"]}), request["articles"]))
        everything += request["articles"]
    count_sources = len(everything_sources)

    # Removing documents without content and description
    logger.info("Removing documents without content and description...")

    def filter_fun(x):
        return (x["content"] != None and x["description"] != None) and \
            (x["content"] != "" and x["description"] != "") and \
            (x["content"] != "" and x["description"] != None) and \
            (x["content"] != None and x["description"] != "")

    top_headlines = list(filter(filter_fun, top_headlines))
    everything = list(filter(filter_fun, everything))

    # Insert documents into top_headlines collection
    logger.info("Inserting documents into top_headlines collection...")
    init_count = db.top_headlines.count_documents({})
    logger.info(
        f"Current number of documents in top_headlines collection: {init_count} ")
    try:
        db.top_headlines.insert_many(top_headlines, ordered=False)
    except BulkWriteError as e:
        logger.warning(
            f"Some documents weren't inserted because of unique index. Error message:\n {str(e)}")
    final_count = db.top_headlines.count_documents({})
    inserted_count_top_headlines = final_count - init_count
    logger.info(
        f"Number of documents in top_headlines collection after insertion: {final_count}")
    logger.info(
        f"Number of top_headlines results: {top_headlines_results}, Number of documents inserted into top_headlines: {inserted_count_top_headlines}")

    # Insert documents into everything collection
    logger.info("Inserting documents into everything collection...")
    init_count = db.everything.count_documents({})
    logger.info(
        f"Current number of documents in everything collection: {init_count} ")
    try:
        db.everything.insert_many(everything, ordered=False)
    except BulkWriteError as e:
        logger.warning(
            f"Some documents weren't inserted because of unique index. Error message:\n {str(e)}")
    final_count = db.everything.count_documents({})
    inserted_count_everything = final_count - init_count
    logger.info(
        f"Number of documents in everything collection after insertion: {final_count}")
    logger.info(
        f"Number of everything results: {everything_results}, Number of documents inserted into top_headlines: {inserted_count_everything}, Number of sources used: {count_sources}")

    return {
        'top_headlines_info': json.dumps({"top_headlines_results": top_headlines_results,
                                          "inserted_count_top_headlines": inserted_count_top_headlines}),
        'everything_info': json.dumps({"everything_results": everything_results,
                                       "inserted_count_everything": inserted_count_everything,
                                       "count_sources": count_sources}),
        'statusCode': 200,
        'body': json.dumps('Task executed successfully!')
    }
