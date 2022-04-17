import logging
import os
import sys
from pathlib import Path

import requests
from haystack.utils import get_batches_from_generator
from tqdm.auto import tqdm

dirname = os.path.dirname(__file__)
sys.path.append(
    os.path.join(dirname, "../")
)  # Necessary so we can import custom modules from api. See: https://realpython.com/lessons/module-search-path/

from api.config import INDEXING_NU_PIPELINE_NAME, PIPELINE_YAML_PATH
from api.custom_components.custom_pipe import CustomPipeline

logger = logging.getLogger(__file__)

try:
    logger.info("Loading Indexing Pipeline from yaml file.")
    INDEXING_PIPELINE = CustomPipeline.load_from_yaml(
        Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME
    )
except KeyError:
    logger.info(
        "Indexing Pipeline not found in the YAML configuration. News Upload API will not be available."
    )
    raise KeyError


def get_dump_data(dump):
    logger.info(f"Loading the {dump} data dump.")
    docs = requests.get(
        f"https://awsmisc.s3.eu-west-2.amazonaws.com/backups/{dump}.json"
    ).json()
    texts = list(map(lambda x: x["text"], docs))

    return docs, texts


def load_index_docs(dump):
    # Check there isn't documents already in the Document Store
    doc_count = INDEXING_PIPELINE.get_node("DocumentStore").get_document_count()
    logger.info(f"Document Store contains {doc_count} documents.")
    if doc_count > 0:
        logger.info(
            "Document Store already contains documents. Backup loading should be the first doucument insertion in the Document Store."
        )
        return None
    else:
        # Load the data dump
        documents, texts = get_dump_data(dump)

        # Training the Retriever
        try:
            if len(texts) > 50000:
                # RANDOM SAMPLE TO FIT IN MEMORY
                from random import sample, seed

                seed(10)
                texts = sample(texts, 50000)
            INDEXING_PIPELINE.get_node("Retriever").train(texts)

        except Exception as e:
            logger.warning(e)

        # Embeds the documents in dicts and writes them to the document store
        logger.info("Running indexing pipeline.")
        batch_size = 30000  # Inserting in batches to not run out of memory
        with tqdm(
            total=len(documents), position=0, unit="Docs", desc="Indexing documents"
        ) as progress_bar:
            for batch in get_batches_from_generator(documents, batch_size):
                INDEXING_PIPELINE.run(documents=batch)
                progress_bar.update(batch_size)


if __name__ == "__main__":
    # Receive dump name from cli arguments
    dump = sys.argv[1]

    # Populate database
    load_index_docs(dump)
