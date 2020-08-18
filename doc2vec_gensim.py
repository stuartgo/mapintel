from datetime import datetime, timedelta
import random
import collections
from collections import defaultdict
from nltk.corpus import stopwords
from newsapi import NewsApiClient
# import numpy as np
# from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from gensim import models


# TODO: add features to CorpusPreprocess,
#       build mongodb backend to store api requests,
#       webscrape full content from urls provided by api,
#       host project on github

class CorpusPreprocess(BaseEstimator, TransformerMixin):
    def __init__(self, freq_train_only=False, min_freq=1, stopwords=stopwords.words('english')):
        """
        Scikit-learn like Transformer for Corpus preprocessing
        :param freq_train_only: whether or not to find word frequency only on train or also in test
        :param min_freq: minimum word frequency to enter in corpus
        :param stopwords: set of frequent words to exclude from corpus
        """
        self.freq_train_only = freq_train_only
        self.min_freq = min_freq
        self.stopwords = stopwords

    def fit(self, X, y=None):
        # Lowercase each document, split it by white space
        texts = self.simple_tokenizer(X)

        # Count word frequencies
        self.frequency_ = defaultdict(int)
        for text in texts:
            for token in text:
                self.frequency_[token] += 1

        # Register the training set so we don't double count frequencies by transforming it
        self.X_ = X

        return self

    def transform(self, X, y=None):
        # Check if fit had been called
        check_is_fitted(self)

        # Lowercase each document, split it by white space and filter out self.stopwords
        texts = self.simple_tokenizer(X, self.stopwords)

        if not self.freq_train_only and X != self.X_:
            # Update word frequencies
            for text in texts:
                for token in text:
                    self.frequency_[token] += 1

        # Only keep words that appear more than self.min_freq
        texts = [[token for token in text if self.frequency_[token] > self.min_freq] for text in texts]

        return texts

    @staticmethod
    def simple_tokenizer(X, stopwords=None):
        """
        Static method that lowers the case and splits the words (i.e. tokenizes) in each document. Optionally removes
        stopwords.
        :param X: list of documents
        :param stopwords: stopwords to remove from tokenized corpus
        :return: list of tokenized documents
        """
        if stopwords:
            return [[word for word in document.lower().split() if word not in stopwords] for document in X]
        else:
            return [[word for word in document.lower().split()] for document in X]


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


# Init
newsapi = NewsApiClient(api_key='982796d7dec8411d9ec9d8f09d20666c')

# Get news articles
articles = newsapi.get_everything(language='en',
                                  domains='bbc.co.uk',
                                  from_param=datetime.today() - timedelta(30),
                                  to=datetime.today(),
                                  page_size=100)

corpus = list(set([c['content'] for c in articles['articles'] if c['content']]))

# Train/ test split
test_idx = random.sample(range(len(corpus)), int(len(corpus) * 0.1))
test_corpus = [corpus[i] for i in test_idx]
train_corpus = list(set(corpus).difference(set(test_corpus)))

# Preprocessing
prep = CorpusPreprocess()
processed_train_corpus = prep.fit_transform(train_corpus)
processed_test_corpus = prep.transform(test_corpus)

# TaggedDocument format (input to doc2vec)
tagged_corpus = [models.doc2vec.TaggedDocument(text, [i]) for i, text in enumerate(processed_train_corpus)]

# Doc2Vec model
model = models.doc2vec.Doc2Vec(vector_size=40, min_count=2, epochs=200)
model.build_vocab(tagged_corpus)
# model.wv.vocab['later'].count  # this accesses the count of a word in the vocabulary
model.train(tagged_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# Assessing Doc2Vec model
ranks = []
for doc_id in range(len(tagged_corpus)):
    inferred_vector = model.infer_vector(tagged_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank = [docid for docid, sim in sims].index(doc_id)
    ranks.append(rank)

print(collections.OrderedDict(sorted(collections.Counter(ranks).items())))

# Pick a random document from the train corpus, infer its vector and check similarity with other documents
doc_id, sims = check_random_doc_similarity(model, tagged_corpus)
compare_documents(tagged_corpus[doc_id].words, sims, tagged_corpus, doc_id)

# Pick a random document from the test corpus, infer its vector and check similarity with other documents
doc_id, sims = check_random_doc_similarity(model, tagged_corpus, processed_test_corpus)
compare_documents(processed_test_corpus[doc_id], sims, tagged_corpus, doc_id)

# Get new news articles
new_articles = newsapi.get_everything(language='en',
                                      domains='bbc.co.uk',
                                      from_param=datetime.today() - timedelta(30),
                                      to=datetime.today() - timedelta(20),
                                      page_size=10)

new_corpus = list(set([c['content'] for c in new_articles['articles'] if c['content']]))

# Apply preprocessing
new_processed_corpus = prep.transform(new_corpus)

# Similarity query
unkwnown_doc = new_processed_corpus[2]
sims = similarity_query(model, unkwnown_doc)
compare_documents(unkwnown_doc, sims, tagged_corpus)
