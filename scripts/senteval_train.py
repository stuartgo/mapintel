import unicodedata
import os
import json
from string import punctuation
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load


# Define preprocessor: takes a doc and transforms it preserving the tokenizing and n-grams generation steps
def _preprocessor(doc):
    # Removes HTML tags
    doc = BeautifulSoup(doc, features="lxml").get_text()
    # Lowercase
    doc = doc.lower()
    # Remove accentuation
    doc = unicodedata.normalize('NFKD', doc).encode('ASCII', 'ignore').decode('ASCII')
    # Remove punctuation
    doc = doc.translate(str.maketrans('', '', punctuation))
    return doc


# Define tokenizer: takes a doc and tokenizes it preserving the preprocessing and n-grams generation steps
def _tokenizer(doc, stopword_prep=False):
    # Word tokenizer
    doc = word_tokenize(doc)
    # Apply Stemmer
    doc = [PorterStemmer().stem(token) for token in doc]
    if stopword_prep:
        doc = doc[0]
    return doc


_consistent_stop_word = {_tokenizer(_preprocessor(word), stopword_prep=True) for word in stopwords.words('english')}


class SentevalTrain:
    def load_data(self, data_path):
        # Reading files into memory
        all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn][1:]
        self.corpus = []
        for file in all_files:
            with open(file, 'r') as f:
                self.corpus.append(f.read())

    def train(self):
        # Setting count_vect without fixed vocabulary
        self.count_vect = CountVectorizer(preprocessor=_preprocessor,
                                          tokenizer=_tokenizer,
                                          stop_words=_consistent_stop_word,
                                          ngram_range=(1, 1),
                                          max_df=0.8,
                                          min_df=3,
                                          max_features=None,
                                          vocabulary=None)

        # Fitting the count_vect object
        self.count_vect.fit(self.corpus)

        # Analyzing vocabulary
        print("Vocabulary length is {} and number of words excluded is {}".format(
            len(self.count_vect.vocabulary_), len(self.count_vect.stop_words_)))

        # Fitting the tfidf_vect object
        # Applying TfidfVectorizer is equivalent to applying CountVectorizer followed by TfidfTransformer
        self.tfidf_vect = Pipeline([
            ('count', CountVectorizer(preprocessor=_preprocessor,
                                      tokenizer=_tokenizer,
                                      stop_words=_consistent_stop_word,
                                      ngram_range=(1, 1),
                                      max_df=0.8,
                                      min_df=3,
                                      max_features=None,
                                      vocabulary=self.count_vect.vocabulary_)),
            ('tfid', TfidfTransformer())
        ]).fit(self.corpus)

        return self.count_vect, self.tfidf_vect

    def dump(self, vocab_file, tfidf_file):
        # Writing vocabulary to json
        print("Writing vocabulary to {}".format(vocab_file))
        with open(vocab_file, mode="w") as file:
            file.write(json.dumps(self.count_vect.vocabulary_))

        # Saving a pickle of the tfidf_vect fitted model
        print("Writing tfidf_vect model to {}".format(tfidf_file))
        dump(self.tfidf_vect, tfidf_file)

    @staticmethod
    def load(vocab_file, tfidf_file):
        # Reading vocabulary from .json
        try:
            with open(vocab_file, mode="r") as file:
                vocab = json.load(file)
        except FileNotFoundError:
            print("No file {} found. You may need to fit the CountVectorizer first.".format(vocab_file))

        # Reading tfidf_vect from .joblib
        try:
            tfidf_vect = load(tfidf_file)
        except FileNotFoundError:
            print("No file {} found. You may need to fit the TfidfTransformer first.".format(tfidf_file))

        # Setting count_vect with fixed vocabulary
        count_vect = CountVectorizer(preprocessor=_preprocessor,
                                     tokenizer=_tokenizer,
                                     stop_words=_consistent_stop_word,
                                     ngram_range=(1, 1),
                                     max_df=0.8,
                                     min_df=3,
                                     max_features=None,
                                     vocabulary=vocab)

        return count_vect, tfidf_vect
