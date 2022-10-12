import json

import bson


def main():
    # Get documents
    with open("artifacts/final_news.bson", "rb") as f:
        docs = bson.decode_all(f.read())

    # TODO: Preprocess text - use text_cleaner.py file in mapintel_project
    # Preprocess
    docs_final = []
    for doc in docs:
        new_doc = {
            "text": doc["text"],
            "image_url": doc["image"],
            "url": doc["url"],
            "title": doc["title"],
            "timestamp": doc["timestamp"].strftime("%d-%m-%Y %H:%M:%S"),
        }
        docs_final.append(new_doc)

    return docs_final


if __name__ == "__main__":
    docs = main()
    with open("artifacts/arquivo.json", "w") as file:
        file.write(json.dumps(docs))
