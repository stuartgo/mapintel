import re
from bs4 import BeautifulSoup
from langdetect import detect
import logging
import os
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _text_cleaner(text):
    """Text cleaning function. Removes HTML tags, escaped characters (e.g. \n)
    and removes NewsAPI text patterns and URLs

    Args:
        text (string): text of a news article

    Returns:
        string: cleaned text
    """
    # Removes HTML tags
    text = BeautifulSoup(text, features="lxml").get_text()
    # Remove escaped characters
    escapes = ''.join([chr(char) for char in range(1, 32)])
    text = text.translate(str.maketrans('', '', escapes))
    # Remove patterns
    expressions = ['… [+ [0-9]*chars]$', '…', 'https?://\S+']
    for i in expressions:
        text = re.sub(i, '', text)
    return text


def _detect_non_english(text):
    """Function that detects if there's non-english characters in text

    Args:
        text (string): text of a news article

    Returns:
        boolean: True if there's non-english characters exist
    """
    # korean
    if re.search("[\uac00-\ud7a3]", text):
        return True
    # japanese
    if re.search("[\u3040-\u30ff]", text):
        return True
    # chinese
    if re.search("[\u4e00-\u9FFF]", text):
        return True
    # arabic
    if re.search("[\u0600-\u06FF]", text):
        return True
    # devanagari (hindi)
    if re.search("[\u0900-\u097F]", text):
        return True
    return False


def string_cleaner(text):
    """Cleans the text by applying _detect_non_english and _text_cleaner

    Args:
        text (string): string of text

    Returns:
        tuple: string: cleaned text, bool: whether to include the document
    """
    if detect(text) != 'en' or _detect_non_english(text):
        return None, False
    else:
        return _text_cleaner(text), True


def documents_cleaner(documents):
    # Cleaning the documents
    logger.info("Cleaning the documents.")
    dicts=[]
    for doc in documents:
        # TODO: make the for loop faster (parallelization?)
        if (doc['description']=='' or doc['description']==None) and (doc['content']=='' or doc['content']==None):
            # Don't include documents without content and description
            continue
        concat_text = " ".join(
            map(lambda x: '' if x is None else x, [doc['title'], doc['description'], doc['content']])
        )
        concat_text, include = string_cleaner(concat_text)
        if include:
            dicts.append(
                {
                    'text': concat_text, 
                    'meta': {
                        'source': doc['source'], 
                        'author': doc['author'], 
                        'publishedat': doc['publishedAt'],
                        'url': doc['url'],
                        'urltoimage': doc['urlToImage'],
                        'category': doc['category']
                    }
                }
            )
    
    # Go over results_list and remove duplicates based on 'text'f
    logger.info("Removing the duplicated documents.")
    holder = {}
    for d in dicts:
        holder.setdefault(d['text'], d['meta'])  # each key is unique
    # Reformat into list of dictionaries
    dicts = [{'text': k, 'meta': v} for k, v in holder.items()]

    return dicts


def clean_backups(backups_dir):
    # Check there is documents to load
    backups = os.listdir(backups_dir)
    if len(list(backups)) == 0:
        logger.info("No backup files to load into the document store.")
    
    else:
        # Load each JSON in backups and add everything to documents
        logger.info(f"Loading the raw backups: {list(backups)}.")
        documents = []
        for bak in backups:
            with open(os.path.join(backups_dir, bak), 'r') as file:
                for line in file:
                    documents.append(json.loads(line))
        
        # Cleaning the documents
        dicts = documents_cleaner(documents)

        # Exporting the cleaned documents to a json
        logger.info("Exporting the cleaned documents.")
        with open(os.path.join(backups_dir, 'mongodb_cleaned_docs.json'), 'w') as f:
            f.write(json.dumps(dicts))


if __name__ == "__main__":
    backups_dir = '../data/backups'
    clean_backups(backups_dir)
