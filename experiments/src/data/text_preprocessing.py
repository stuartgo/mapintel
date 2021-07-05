"""
Provides text preprocessing functions and the CorpusPreprocessing
class for use by the data preparation scripts.
"""
import re
import unicodedata
from collections import defaultdict

from bs4 import BeautifulSoup
from langdetect import detect
from nltk import word_tokenize
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


# TODO: coordinate roles of results_cleaner and CorpusPreprocess (remove overlap)
# TODO: tokenize numbers and other word types to reduce vocabulary


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

def results_cleaner(agg_results):
    """Cleans the results of the aggregation pipeline mongodb query by
     applying _detect_non_english and _text_cleaner

    Args:
        agg_results (list of dicts): output of aggregation pipeline query

    Returns:
        list of dicts: cleaned aggregation pipeline results
    """
    remove_idx = []
    # Iterate over results: apply _detect_non_english and _text_cleaner
    for i, r in enumerate(agg_results):
        if detect(r['text']) != 'en' or _detect_non_english(r['text']):
            remove_idx.append(i)
            continue
        r['text'] = _text_cleaner(r['text'])
    # Remove non-english articles or articles with cjk characters 
    for ix in sorted(remove_idx, reverse=True):
        del agg_results[ix]
    return agg_results


def join_results(results_list):
    """Join cleaned results of aggregation pipeline on different
    collections by removing duplicated documents across collections

    Args:
        results_list (list of dicts): aggregation pipeline results 
        of multiple collection

    Returns:
        list of dicts: aggregation pipeline results of multiple 
        collection without duplicates
    """
    # Go over results_list and remove duplicates based on 'text'
    holder = {}
    for result in results_list:
        value = holder.setdefault(result['text'], [])  # each key is unique
        value.append(result['_id'])
        value.append(result['col'])
        value.append(result['insert_date'])
        value.append(result['category'])
    # Reformat into list of dictionaries
    join_results_list = [{'id': v[0], 'col': v[1], 'insert_date': v[2],
                        'category': v[3], 'text': k} for k, v in holder.items()]
    return join_results_list


def _remove_html_tags(text):
    return BeautifulSoup(text, features="lxml").get_text()


def _remove_accents(text):
    return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('UTF8')


def _remove_punctuation(text, punctuation_list):
    return text.translate(str.maketrans('', '', punctuation_list))


class CorpusPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, language='english', stop_words=None, lowercase=True, strip_accents=False,
                 strip_punctuation=None, stemmer=None, max_df=1.0, min_df=1):
        """Scikit-learn like Transformer for Corpus preprocessing.
        Preprocesses text by applying multiple tasks (e.g. lowecasing, stemming, etc).
        Fits the data for obtaining vocabulary_ (mapping of terms to document frequencies)
         and stop_words_ (terms that were ignored because of either 'max_df', 'min_df' or 'stop_words').

        Args:
            language (str, optional): language of the input documents. Defaults to 'english'.
            stop_words (list, optional): list of stop words to be removed. Defaults to None.
            lowercase (bool, optional): lowercases text if True. Defaults to True.
            strip_accents (bool, optional): strips accents from text if True. Defaults to False.
            strip_punctuation (iterable, optional): strips provided punctuation from text if not None.
             Defaults to None.
            stemmer (Stemmer instance, optional): applies the provided Stemmer's stem method to text.
             Defaults to None.
            max_df (float in range [0.0, 1.0] or int, optional): ignore terms with a document frequency higher 
             than the given threshold. If float, the parameter represents a proportion of documents, integer 
             absolute counts. Defaults to 1.0.
            min_df (float in range [0.0, 1.0] or int, optional): ignore terms with a document frequency lower 
             than the given threshold. If float, the parameter represents a proportion of documents, integer 
             absolute counts. Defaults to 1.

        Raises:
            ValueError: max_df and min_df are bounded to range [0.0, 1.0]
        """
        self.language = language
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.strip_accents = strip_accents
        self.strip_punctuation = strip_punctuation
        self.stemmer = stemmer
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")

    def fit(self, X, y=None):
        # Building vocabulary_ and stop_words_
        self.fit_transform(X)

        return self

    def fit_transform(self, X, y=None, tokenize=True):
        # Preprocess and tokenize corpus
        corpus = self._word_tokenizer(X)

        # Build vocabulary document frequencies
        vocab_df = defaultdict(int)
        for doc in corpus:
            for unique in set(doc):
                vocab_df[unique] += 1

        # Find stop_words_ based on max_df and min_df
        if self.stop_words is None:
            self.stop_words_ = set()
        else:
            self.stop_words_ = set(self.stop_words)

        if self.max_df is not None:
            if isinstance(self.max_df, float):
                vocab_rel_df = {k: v / len(X) for k, v in vocab_df.items()}
                self.stop_words_.update(
                    {k for k, v in vocab_rel_df.items() if v > self.max_df})
            else:
                self.stop_words_.update(
                    {k for k, v in vocab_df.items() if v > self.max_df})

        if self.min_df is not None:
            if isinstance(self.min_df, float):
                vocab_rel_df = {k: v / len(X) for k, v in vocab_df.items()}
                self.stop_words_.update(
                    {k for k, v in vocab_rel_df.items() if v < self.min_df})
            else:
                self.stop_words_.update(
                    {k for k, v in vocab_df.items() if v < self.min_df})

        # Remove stop_words_ from vocabulary
        for k in self.stop_words_:
            vocab_df.pop(k, None)

        # Set vocabulary_
        self.vocabulary_ = vocab_df

        # Remove stop_words from corpus
        if self.stop_words is not None:
            corpus = [[token for token in doc if token not in self.stop_words]
                      for doc in corpus]

        # Split vs merged
        if not tokenize:
            corpus = [" ".join(doc) for doc in corpus]

        return corpus

    def transform(self, X, y=None, tokenize=True):
        # Check if fit has been called
        check_is_fitted(self)

        # Preprocess and tokenize corpus
        corpus = self._word_tokenizer(X)

        # Remove stop_words from corpus
        corpus = [[token for token in doc if token not in self.stop_words_]
                  for doc in corpus]

        # Split vs merged
        if not tokenize:
            corpus = [" ".join(doc) for doc in corpus]

        return corpus

    def _word_tokenizer(self, X):
        """Preprocesses and tokenizes each document by applying a
         preprocessing function.
        Args:
            X (iterable): documents to preprocess
        Returns:
            list: preprocessed and tokenized documents
        """

        # Map all transformations specified
        docs = map(_remove_html_tags, X)
        if self.lowercase:
            docs = map(str.lower, docs)
        if self.strip_accents:
            docs = map(_remove_accents, docs)
        if self.strip_punctuation is not None:
            docs = [_remove_punctuation(doc, self.strip_punctuation) for doc in docs]

        # Ensure 'punkt' tokenizer is installed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Word tokenizer
        corpus = [word_tokenize(doc, language=self.language) for doc in docs]

        if self.stemmer is not None:
            corpus = [[self.stemmer.stem(token)
                       for token in doc] for doc in corpus]

        return corpus
