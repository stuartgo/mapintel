from dotenv import load_dotenv, find_dotenv
import os
from datetime import datetime, timedelta
import newsapi
from pymongo import MongoClient
import dns  # required for connecting with SRV

# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)

NEWSAPIKEY = os.environ.get("NEWSAPIKEY")
MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
MONGOPASSWORD = os.environ.get("MONGOPASSWORD")


class Connect(object):
    @staticmethod
    def get_connection():
        # return MongoClient(f"mongodb+srv://{username}:{password}@cluster0.e2na5.gcp.mongodb.net/{dbname}?retryWrites=true&w=majority")
        return MongoClient(f"mongodb://{MONGOUSERNAME}:{MONGOPASSWORD}@cluster0-shard-00-00.e2na5.gcp.mongodb.net:27017,cluster0-shard-00-01.e2na5.gcp.mongodb.net:27017,cluster0-shard-00-02.e2na5.gcp.mongodb.net:27017/news?ssl=true&replicaSet=atlas-umdth0-shard-0&authSource=admin&retryWrites=true&w=majority")


# Establish connection to NewsAPI and MongoDB
news_client = newsapi.NewsApiClient(api_key=NEWSAPIKEY)
db_client = Connect().get_connection()
db = db_client.news

# Get top_headlines
top_headlines_results = 0
top_headlines = []
for categ in newsapi.const.categories:
    request = news_client.get_top_headlines(category=categ, language="en", country="us", page_size=100)
    top_headlines_results += request["totalResults"]
    # adding category field and updating source field to every document in articles
    list(map(lambda x: x.update({"category": categ, "source": x["source"]["id"]}), request["articles"]))
    # getting source id
    top_headlines += (request["articles"])

# Get everything using source categories
everything_results = 0
everything_sources = set()
everything = []
for categ in newsapi.const.categories:
    sources = set(map(lambda x: x["id"], news_client.get_sources(category=categ, language="en")["sources"]))
    request = news_client.get_everything(sources=",".join(sources), from_param=datetime.today() - timedelta(1),
                                         to=datetime.today(), page_size=100)
    everything_results += request["totalResults"]
    # Guaranteeing requests only come from requested sources
    request_sources = set(map(lambda x: x["source"]['id'], request["articles"]))
    diff_sources = request_sources.difference(sources)
    everything_sources.update(request_sources)  # tracking which sources used
    request["articles"] = list(filter(lambda x: x["source"]["id"] not in diff_sources, request["articles"]))
    # Adding category field and updating source field to every document in articles
    list(map(lambda x: x.update({"category": categ, "source": x["source"]["id"]}), request["articles"]))
    everything += request["articles"]

# Insert documents into top_headlines collection
init_count = db.top_headlines.count_documents({})
print("\nCurrent number of documents in top_headlines collection: ", init_count)
db.top_headlines.insert_many(top_headlines)
final_count = db.top_headlines.count_documents({})
print("Number of documents in top_headlines collection after insertion: ", final_count)
print("Number of top_headlines results: {}, Number of documents inserted into top_headlines: {}".
      format(top_headlines_results, final_count-init_count))

print("\n-------------------------------------------------------------------------------------------------------------")

# Insert documents into everything collection
init_count = db.everything.count_documents({})
print("\nCurrent number of documents in everything collection: ", init_count)
db.everything.insert_many(everything)
final_count = db.everything.count_documents({})
print("Number of documents in everything collection after insertion: ", final_count)
print("Number of everything results: {}, Number of documents inserted into everything: {}, Number of sources used: {}".
      format(everything_results, final_count-init_count, len(everything_sources)))
