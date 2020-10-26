import random
import unicodedata
import os
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from pandas import DataFrame


class CorpusPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, stop_words=None, lowercase=True, strip_accents=False,
                 strip_punctuation=None, stemmer=True, max_df=1.0, min_df=1):
        """
        Scikit-learn like Transformer for Corpus preprocessing
        :param stop_words: examples
        :param lowercase:
        :param strip_accents:
        :param strip_punctuation:
        :param stemmer: Applies PorterStemmer which is known for its simplicity and speed. It is commonly useful in
        Information Retrieval Environments (IR Environments) for fast recall and fetching of search queries
        :param max_df:
        :param min_df:

        :attr vocabulary_: dict
            A mapping of terms to document frequencies.
        :attr stop_words_ : set
            Terms that were ignored because they either:
              - occurred in too many documents (`max_df`)
              - occurred in too few documents (`min_df`)
              - also contains the same terms as stop_words
        """
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
                self.stop_words_.update({k for k, v in vocab_rel_df.items() if v > self.max_df})
            elif isinstance(self.max_df, int):
                self.stop_words_.update({k for k, v in vocab_df.items() if v > self.max_df})
            else:
                raise ValueError("max_df parameter should be int or float")

        if self.min_df is not None:
            if isinstance(self.min_df, float):
                vocab_rel_df = {k: v / len(X) for k, v in vocab_df.items()}
                self.stop_words_.update({k for k, v in vocab_rel_df.items() if v < self.min_df})
            elif isinstance(self.min_df, int):
                self.stop_words_.update({k for k, v in vocab_df.items() if v < self.min_df})
            else:
                raise ValueError("min_df parameter should be int or float")

        # Remove stop_words_ from vocabulary
        for k in self.stop_words_:
            vocab_df.pop(k, None)

        # Set vocabulary_
        self.vocabulary_ = vocab_df

        # Remove stop_words from corpus
        if self.stop_words is not None:
            corpus = [[token for token in doc if token not in self.stop_words] for doc in corpus]

        # Split vs merged
        if not tokenize:
            corpus = [" ".join(doc) for doc in corpus]

        return corpus

    def transform(self, X, y=None, tokenize=True):
        # Check if fit had been called
        check_is_fitted(self)

        # Preprocess and tokenize corpus
        corpus = self._word_tokenizer(X)

        # Remove stop_words from corpus
        corpus = [[token for token in doc if token not in self.stop_words_] for doc in corpus]

        # Split vs merged
        if not tokenize:
            corpus = [" ".join(doc) for doc in corpus]

        return corpus

    def _word_tokenizer(self, X):
        """
        Preprocesses and tokenizes documents
        :param X: list of documents
        :return: list of preprocessed and tokenized documents
        """
        # Define function conditionally so we only need to evaluate the condition once instead at every document
        if self.strip_accents and self.lowercase and self.strip_punctuation is not None:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Lowercase
                doc = doc.lower()
                # Remove accentuation
                doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('ASCII')
                # Remove punctuation
                doc = doc.translate(str.maketrans('', '', self.strip_punctuation))
                return doc
        elif self.strip_accents and self.lowercase:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Lowercase
                doc = doc.lower()
                # Remove accentuation
                doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('ASCII')
                return doc
        elif self.strip_accents and self.strip_punctuation is not None:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Remove accentuation
                doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('ASCII')
                # Remove punctuation
                doc = doc.translate(str.maketrans('', '', self.strip_punctuation))
                return doc
        elif self.lowercase and self.strip_punctuation is not None:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Lowercase
                doc = doc.lower()
                # Remove punctuation
                doc = doc.translate(str.maketrans('', '', self.strip_punctuation))
                return doc
        elif self.strip_accents:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Remove accentuation
                doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('ASCII')
                return doc
        elif self.lowercase:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Lowercase
                doc = doc.lower()
                return doc
        else:
            def doc_preprocessing(doc):
                # Removes HTML tags
                doc = BeautifulSoup(doc, features="lxml").get_text()
                # Remove punctuation
                doc = doc.translate(str.maketrans('', '', self.strip_punctuation))
                return doc

        # Apply cleaning function over X
        corpus = map(doc_preprocessing, X)

        # Word tokenizer
        corpus = [word_tokenize(doc) for doc in corpus]

        if self.stemmer:
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(token) for token in doc] for doc in corpus]

        return corpus



def check_random_doc_similarity(doc2vec_model, train_corpus, test_corpus=None):
    """
    Function that randomly chooses a document from either the train or test corpus and compares it with the
    documents from the train corpus. Enables model assessment and testing.
    :param doc2vec_model: gensim.models.doc2vec.Doc2Vec fitted instance
    :param train_corpus: list of gensim.models.doc2vec.TaggedDocument formatted documents
    :param test_corpus: list of preprocessed (same way as train) test documents
    :return: ID of the random chosen document and list of similarities of documents of train corpus with
    selected document
    """
    if test_corpus is not None:
        doc_id = random.randint(0, len(test_corpus) - 1)
        sims = similarity_query(doc2vec_model, test_corpus[doc_id])
        return doc_id, sims
    else:
        doc_id = random.randint(0, len(train_corpus) - 1)
        sims = similarity_query(doc2vec_model, train_corpus[doc_id].words)
        return doc_id, sims


def similarity_query(doc2vec_model, unknown_doc):
    """
    Performs a similarity query of an unknown document against known documents of the train corpus
    :param doc2vec_model: gensim.models.doc2vec.Doc2Vec fitted instance
    :param unknown_doc: preprocessed unknown document
    :return: list of similarities of documents of train corpus with unknown document
    """
    inferred_unknown_vector = doc2vec_model.infer_vector(unknown_doc)
    sims = doc2vec_model.docvecs.most_similar([inferred_unknown_vector], topn=len(doc2vec_model.docvecs))
    return sims


def export_test_results(doc2vec_model, prep_base_corpus, raw_base_corpus, raw_compare_corpus,
                        out_path=os.path.join(".", "outputs", "test_doc2vec.xlsx")):
    """
    Exports the similarity query results of every document in prep_base_corpus to an excel notebook
    :param doc2vec_model: gensim.models.doc2vec.Doc2Vec fitted instance
    :param prep_base_corpus: corpus to base the comparison (processed)
    :param raw_base_corpus: corpus to base the comparison (unprocessed)
    :param raw_compare_corpus: corpus to compare to (unprocessed)
    :param out_path: path of the exported excel file
    :return: None
    """
    docs_list = []
    for doc_id in range(len(prep_base_corpus) - 1):
        # Initialize line_list
        line_list = [doc_id, raw_base_corpus[doc_id]]
        # Get similarities to base doc
        sims = similarity_query(doc2vec_model, prep_base_corpus[doc_id])
        # Append compare docs
        for index in [0, 1, len(sims) // 2, len(sims) - 1]:
            # Append compare doc id
            line_list.append(sims[index][0])
            line_list.append(sims[index][1])
            line_list.append(raw_compare_corpus[sims[index][0]])
        # Append line_list to docs_list
        docs_list.append(line_list)

    print('Exporting to test_doc2vec.xlsx...')
    cols = ['BASE_DOC_ID', 'BASE_DOC', 'MOST_SIMILAR_ID', 'MOST_SIMILAR_DIST', 'MOST_SIMILAR',
            'SECOND_MOST_SIMILAR_ID', 'SECOND_MOST_SIMILAR_DIST', 'SECOND_MOST_SIMILAR',
            'MEDIAN_SIMILAR_ID', 'MEDIAN_SIMILAR_DIST', 'MEDIAN_SIMILAR',
            'LEAST_SIMILAR_ID', 'LEAST_SIMILAR_DIST', 'LEAST_SIMILAR']
    DataFrame(docs_list, columns=cols).to_excel(out_path, index=False)
