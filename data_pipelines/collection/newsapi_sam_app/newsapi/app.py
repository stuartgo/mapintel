"""
AWS lambda function. Script that will be executed by AWS lambda service.
Requests documents from NewsAPI, removes documents without content and 
description and inserts the documents into the S3 bucket.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from io import BytesIO

import boto3
import newsapi

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

# Getting environmental variables
NEWSAPIKEY = os.environ.get("NEWSAPIKEY")
BUCKET = os.environ.get("BUCKET")

# Establish connections to NewsAPI and mapinte-news bucket
news_client = newsapi.NewsApiClient(api_key=NEWSAPIKEY)
s3_client = boto3.resource("s3")


def lambda_handler(event, context):
    logger = logging.getLogger(__name__)
    now = datetime.today()
    folder_name = now.strftime("%d-%m-%Y-%H:%M")

    # Bucket object
    bucket = s3_client.Bucket(BUCKET)

    # Get top_headlines
    logger.info("Getting top_headlines results...")
    top_headlines_results = 0
    top_headlines = []
    for categ in newsapi.const.categories:
        request = news_client.get_top_headlines(
            category=categ, language="en", country="us", page_size=100
        )
        top_headlines_results += request["totalResults"]
        # adding category field and updating source field to every document in articles
        list(
            map(
                lambda x: x.update({"category": categ, "source": x["source"]["id"]}),
                request["articles"],
            )
        )
        # getting source id
        top_headlines += request["articles"]

    # Get everything using source categories
    logger.info("Getting everything results...")
    everything_results = 0
    everything_sources = set()
    everything = []
    for categ in newsapi.const.categories:
        sources = set(
            map(
                lambda x: x["id"],
                news_client.get_sources(category=categ, language="en")["sources"],
            )
        )
        request = news_client.get_everything(
            sources=",".join(sources),
            from_param=now - timedelta(1),
            to=now,
            page_size=100,
        )
        everything_results += request["totalResults"]
        # Guaranteeing requests only come from requested sources
        request_sources = set(map(lambda x: x["source"]["id"], request["articles"]))
        diff_sources = request_sources.difference(sources)
        # tracking which sources used
        everything_sources.update(request_sources)
        request["articles"] = list(
            filter(lambda x: x["source"]["id"] not in diff_sources, request["articles"])
        )
        # Adding category field and updating source field to every document in articles
        list(
            map(
                lambda x: x.update({"category": categ, "source": x["source"]["id"]}),
                request["articles"],
            )
        )
        everything += request["articles"]
    count_sources = len(everything_sources)

    # Removing documents without content and description
    logger.info("Removing documents without content and description...")

    def filter_fun(x):
        return (
            (x["content"] != None and x["description"] != None)
            and (x["content"] != "" and x["description"] != "")
            and (x["content"] != "" and x["description"] != None)
            and (x["content"] != None and x["description"] != "")
        )

    top_headlines = list(filter(filter_fun, top_headlines))
    everything = list(filter(filter_fun, everything))

    # Upload top_headlines into bucket
    logger.info("Uploading top_headlines into bucket...")
    top_headlines_obj = bucket.Object(f"{folder_name}/top_headlines.json")
    f = BytesIO(json.dumps(top_headlines).encode("utf-8"))
    top_headlines_obj.upload_fileobj(f)

    # Upload everything into bucket
    logger.info("Uploading everything into bucket...")
    everything_obj = bucket.Object(f"{folder_name}/everything.json")
    f = BytesIO(json.dumps(everything).encode("utf-8"))
    everything_obj.upload_fileobj(f)

    return {
        "top_headlines_info": json.dumps(
            {
                "top_headlines_results": top_headlines_results,
            }
        ),
        "everything_info": json.dumps(
            {
                "everything_results": everything_results,
                "count_sources": count_sources,
            }
        ),
        "statusCode": 200,
        "body": json.dumps("Task executed successfully!"),
    }
