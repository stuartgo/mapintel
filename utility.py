import random
import unicodedata
from collections import defaultdict
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from gensim import models


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

    def fit_transform(self, X, y=None):
        # Preprocess and tokenize corpus
        corpus = self._word_tokenizer(X)

        # Build vocabulary document frequencies
        vocab_df = defaultdict(int)
        for doc in corpus:
            for unique in set(doc):
                vocab_df[unique] += 1

        # Find stop_words_ based on max_df and min_df
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
        corpus = [[token for token in doc if token not in self.stop_words_] for doc in corpus]

        return corpus

    def transform(self, X, y=None):
        # Check if fit had been called
        check_is_fitted(self)

        # Preprocess and tokenize corpus
        corpus = self._word_tokenizer(X)

        # Remove stop_words from corpus
        corpus = [[token for token in doc if token not in self.stop_words_] for doc in corpus]

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

        # Stemmer
        if self.stemmer:
            stemmer = PorterStemmer()
            corpus = [[stemmer.stem(token) for token in doc] for doc in corpus]

        return corpus


def compare_documents(base_doc, similar, compare_corpus, base_doc_id=None):
    """
    Compare a given document with the most similar, second most similar, median and least similar document
    from a corpus of documents
    :param base_doc_id: id of the base document
    :param base_doc: tokenized base document
    :param similar: similarity list of the base document
    :param compare_corpus: corpus to compare the base document to
    :return: None
    """
    if base_doc_id is None:
        base_doc_id = "unknown"
    if isinstance(compare_corpus[0], models.doc2vec.TaggedDocument):
        print('Document ({}): «{}»\n'.format(base_doc_id, ' '.join(base_doc)))
        print(u'SIMILAR/DISSIMILAR DOCS ACCORDING TO DOC2VEC:')
        for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(similar) // 2), ('LEAST', len(similar) - 1)]:
            print(u'%s %s: «%s»\n' % (label, similar[index], ' '.join(compare_corpus[similar[index][0]].words)))
    else:
        print('Document ({}): «{}»\n'.format(base_doc_id, ' '.join(base_doc)))
        print(u'SIMILAR/DISSIMILAR DOCS ACCORDING TO DOC2VEC:')
        for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(similar) // 2),
                             ('LEAST', len(similar) - 1)]:
            print(u'%s %s: «%s»\n' % (label, similar[index], ' '.join(compare_corpus[similar[index][0]])))


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
    if test_corpus:
        doc_id = random.randint(0, len(test_corpus) - 1)
        inferred_vector = doc2vec_model.infer_vector(test_corpus[doc_id])
        sims = doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(doc2vec_model.docvecs))
        return doc_id, sims
    else:
        doc_id = random.randint(0, len(train_corpus) - 1)
        inferred_vector = doc2vec_model.infer_vector(train_corpus[doc_id].words)
        sims = doc2vec_model.docvecs.most_similar([inferred_vector], topn=len(doc2vec_model.docvecs))
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
    # inferred_unknown_vectors = np.vstack([doc2vec_model.infer_vector(doc) for doc in unknown_docs])
    # if against == "known":
    #     return [doc2vec_model.docvecs.most_similar([vec], topn=len(doc2vec_model.docvecs)) for vec in inferred_unknown_vectors]
    # elif against == "unknown":
    #     return squareform(pdist(inferred_unknown_vectors, "cosine"))
