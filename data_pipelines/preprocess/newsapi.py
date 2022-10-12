import json
from datetime import datetime

import requests


def main():
    # Get documents
    docs = requests.get(
        "https://awsmisc.s3.eu-west-2.amazonaws.com/backups/mongodb_cleaned_docs.json"
    ).json()

    # Preprocess
    docs_final = []
    for doc in docs:
        new_doc = {
            "text": doc["text"].replace("#SEPTAG#", " "),
            "image_url": doc["meta"]["urltoimage"],
            "url": doc["meta"]["url"],
            "title": doc["text"].split("#SEPTAG#")[0],
            "snippet": " ".join(doc["text"].split("#SEPTAG#")[1:]),
            "timestamp": datetime.strptime(
                doc["meta"]["publishedat"], "%Y-%m-%dT%H:%M:%SZ"
            ).strftime("%d-%m-%Y %H:%M:%S"),
        }
        docs_final.append(new_doc)

    return docs_final


if __name__ == "__main__":
    docs = main()
    with open("artifacts/newsapi.json", "w") as file:
        file.write(json.dumps(docs))
