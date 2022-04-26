import json
import logging
import os
import sys
from os import path

import requests
from haystack.utils import get_batches_from_generator
from tqdm.auto import tqdm

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../"))

from api.utils import load_pipeline_from_yaml

logger = logging.getLogger(__file__)
indexing_pipeline = load_pipeline_from_yaml("indexing")
DEBUG = os.getenv("DEBUG", 0)


def download_dump_data(dump):
    docs = requests.get(
        f"https://awsmisc.s3.eu-west-2.amazonaws.com/backups/{dump}.json"
    ).json()
    return docs


def read_dump_data(dump_path):
    with open(dump_path, "r") as file:
        docs = json.load(file)
    return docs


def write_dump_data(dump_path, data):
    with open(dump_path, "w") as file:
        json.dump(data, file)


def extract_texts_from_dump(data):
    texts = list(map(lambda x: x["text"], data))
    return texts


def get_dump_data(dump):
    dump_path = path.join(dirname, f"../artifacts/{dump}.json")
    if path.exists(dump_path):
        logger.info(f"Reading the {dump} data dump from disk.")
        docs = read_dump_data(dump_path)
    else:
        logger.info(f"Downloading the {dump} data dump.")
        docs = download_dump_data(dump)
        write_dump_data(dump_path, docs)
    if DEBUG:
        docs = docs[:100]
    texts = extract_texts_from_dump(docs)

    return docs, texts


def load_index_docs(dump):
    # Check there isn't documents already in the Document Store
    doc_count = indexing_pipeline.get_node("DocumentStore").get_document_count()
    logger.info(f"Document Store contains {doc_count} documents.")
    if doc_count > 0:
        logger.info(
            "Document Store already contains documents. Skipping loading of documents."
        )
        return None
    else:
        # Load the data dump
        documents, texts = get_dump_data(dump)
        if DEBUG:
            logger.info(f"Loading the {dump} data dump in DEBUG mode.")
        else:
            logger.info(f"Loading the {dump} data dump.")

        # Training the Retriever
        try:
            if len(texts) > 50000:
                # RANDOM SAMPLE TO FIT IN MEMORY
                from random import sample, seed

                seed(10)
                texts = sample(texts, 50000)
            logger.info(
                "Training the Retriever component. This might take some time..."
            )
            indexing_pipeline.get_node("Retriever").train(texts)

        except Exception as e:
            logger.warning(e)

        # Embeds the documents in dicts and writes them to the document store
        logger.info("Running indexing pipeline. This might take some time...")
        batch_size = 30000  # Inserting in batches to not run out of memory
        with tqdm(
            total=len(documents), position=0, unit="Docs", desc="Indexing documents"
        ) as progress_bar:
            for batch in get_batches_from_generator(documents, batch_size):
                indexing_pipeline.run(documents=batch)
                progress_bar.update(batch_size)


if __name__ == "__main__":
    # Receive dump name from cli arguments
    dump = sys.argv[1]

    # Populate database
    load_index_docs(dump)
