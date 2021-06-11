import json
import os
import logging
from pathlib import Path
from tqdm.auto import tqdm
from haystack.utils import get_batches_from_generator

from custom_components.custom_pipe import CustomPipeline
from custom_components.text_cleaner import documents_cleaner, main
from config import PIPELINE_YAML_PATH, INDEXING_NU_PIPELINE_NAME

logger = logging.getLogger(__name__)

try:
    INDEXING_PIPELINE = CustomPipeline.load_from_yaml(Path(PIPELINE_YAML_PATH), pipeline_name=INDEXING_NU_PIPELINE_NAME)
except KeyError:
    logger.info("Indexing Pipeline not found in the YAML configuration. News Upload API will not be available.")
    raise KeyError

clean_backup = './api/backups/mongodb_cleaned_docs.json'


def backups_load():
    # Check there isn't documents already in the Document Store
    doc_count = INDEXING_PIPELINE.get_node("DocumentStore").get_document_count()
    logger.info(f"Document Store contains {doc_count} documents.")
    if doc_count > 0:
        logger.info("Document Store already contains documents. Backup loading should be the first doucument insertion in the Document Store.")
        return None
    
    # Check if cleaned backup exists
    if not os.path.isfile(clean_backup):
        logger.info("Cleaned backups don't exist.")
        main()

    # Load each JSON in backups and add everything to documents
    logger.info("Loading the cleaned backup.")
    with open(clean_backup, 'r') as file:
        documents = json.load(file)

    # Embeds the documents in dicts and writes them to the document store
    logger.info("Running indexing pipeline.")
    batch_size = 30000  # Inserting in batches to not run out of memory
    with tqdm(total=len(documents), position=0, unit="Docs", desc="Indexing documents") as progress_bar:
        for batch in get_batches_from_generator(documents, batch_size):
            INDEXING_PIPELINE.run(documents=batch)
            progress_bar.update(batch_size)

if __name__ == '__main__':
    backups_load()