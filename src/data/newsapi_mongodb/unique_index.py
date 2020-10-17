import os
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

MONGOUSERNAME = os.environ.get("MONGOUSERNAME")
MONGOPASSWORD = os.environ.get("MONGOPASSWORD")
MONGODB = os.environ.get("MONGODB")

# Establish connections to MongoDB
# db_client = MongoClient(f"mongodb://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb-shard-00-00.e2na5.mongodb.net:27017,newsapi-mongodb-shard-00-01.e2na5.mongodb.net:27017,newsapi-mongodb-shard-00-02.e2na5.mongodb.net:27017/{MONGODB}?ssl=true&replicaSet=atlas-pwkj8y-shard-0&authSource=admin&retryWrites=true&w=majority")
db_client = MongoClient(
    f"mongodb+srv://{MONGOUSERNAME}:{MONGOPASSWORD}@newsapi-mongodb.e2na5.mongodb.net/{MONGODB}?retryWrites=true&w=majority")

# Database object
db = db_client.news

# Creating unique index
collection_list = db.list_collection_names()
for col in collection_list: 
    db[col].create_index(
        [
            ("description", 1),
            ("content", 1)
        ],
        unique = True
    )

# print(db.top_headlines.count_documents({}))
# db.top_headlines.delete_one({'_id': ObjectId("5f8a14f4601f54a54657a1ee")})
# print(db.top_headlines.count_documents({}))
# sys.exit(0)

# Testing whether we can insert already existing document
# duplicated_doc = {
#     "source": None,
#     "author": "Brandon Lee Gowto",
#     "title": "Eagles to start Jalen Mills at cornerback, Marcus Epps at safety again...",
#     "description": "Secondary change up.",
#     "url": "https://www.bleedinggreennation.com/2020/10/4/21501492/jalen-mills-eag...",
#     "urlToImage": "https://cdn.vox-cdn.com/thumbor/zAtOYRtDGrSfDrlfk1gh2VHHAjQ=/0x167:188...",
#     "publishedAt": "2020-10-04T21:52:27Z",
#     "content": "The Philadelphia Eagles will be starting Jalen Mills at cornerback and Marcus Epps at safety in their Week 4 game against the San Francisco 49ers, according to one report:#Eagles lineup changes, pe… [+1127 chars]",
#     "category": "sports"
# }

# another_doc = {
#     "source": None,
#     "author": "test author",
#     "title": "test title",
#     "description": "test description 2",
#     "url": "test url",
#     "urlToImage": "test urlToImage",
#     "publishedAt": "test publishedAt",
#     "content": "test content 2",
#     "category": "test category"
# }

# result = db.top_headlines.insert_many([duplicated_doc, another_doc], ordered=False)
# print(db.top_headlines.count_documents({}))
# print(f"ID inserted: {result.inserted_ids}")
# print(db.top_headlines.count_documents({}))


# Seeing how many duplicates we have in each collection
pipeline = [
    {
        "$group": {
            "_id": {'description': '$description', 'content': '$content'},
            "_idsNeedsToBeDeleted": {"$push": "$$ROOT._id"} # push all `_id`'s to an array
        }
    },
    # Remove first element - which is removing a doc
    {
        "$project": {
            "_id": 0,
            "_idsNeedsToBeDeleted": {  
                "$slice": [
                    "$_idsNeedsToBeDeleted", 1, {"$size": "$_idsNeedsToBeDeleted"}
                ]
            }
        }
    },
    {
        "$unwind": "$_idsNeedsToBeDeleted" # Unwind `_idsNeedsToBeDeleted`
    },
    # Group without a condition & push all `_idsNeedsToBeDeleted` fields to an array
    {
        "$group": { "_id": "", "_idsNeedsToBeDeleted": { "$push": "$_idsNeedsToBeDeleted" } }
    },
    { 
        "$project" : { "_id" : 0 }  # Optional stage
    }
    # At the end you'll have an [{ _idsNeedsToBeDeleted: [_ids] }] or []
]
    
collection_list = db.list_collection_names()
for col in collection_list:
    try:
        idsList = list(db[col].aggregate(pipeline))[0]["_idsNeedsToBeDeleted"]
        print(f"{len(idsList)} instances of documents with duplicated content and description in {col}\n")
    except:
        print(f"0 instances of documents with duplicated content and description in {col}\n")
