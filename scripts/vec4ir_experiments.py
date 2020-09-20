from scripts.utility import CorpusPreprocess
import os
import pandas as pd
from nltk.corpus import stopwords
from string import punctuation
from gensim.models import Word2Vec, Doc2Vec
from sklearn.model_selection import train_test_split
from vec4ir.doc2vec import Doc2VecInference
from vec4ir.core import Retrieval
from vec4ir.base import Tfidf, Matching
from vec4ir.word2vec import WordCentroidDistance

data_path = os.path.join(".", "data")
output_path = os.path.join(".", "outputs")

# Reading files into memory
all_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_path)) for f in fn][1:]
corpus = []
for file in all_files:
    with open(file, 'r') as f:
        corpus.append(f.read())

# Saving topics from each article
topics = [path.split("\\")[3] for path in all_files]

df = pd.read_csv(os.path.join(data_path, "sts-train.csv"))

# Train/ test split
train_corpus, test_corpus, train_topics, test_topics = train_test_split(corpus, topics, test_size=0.1, random_state=0)

# Preprocessing
prep = CorpusPreprocess(stop_words=stopwords.words('english'), lowercase=True, strip_accents=True,
                        strip_punctuation=punctuation, stemmer=True, max_df=0.5, min_df=3)
processed_train_corpus = prep.fit_transform(train_corpus, tokenize=False)
processed_test_corpus = prep.transform(test_corpus, tokenize=False)

# Setting 1 - Default Matching | tfidf model | No query expansion
match_op = Matching()
tfidf = Tfidf()
retrieval = Retrieval(retrieval_model=tfidf, matching=match_op)
retrieval.fit(processed_train_corpus)
idx = retrieval.query(prep.transform(["American elections republicans"], tokenize=False)[0], k=3)
[train_corpus[i] for i in idx.tolist()]

# TODO: Use pre-trained word-embeddings: https://radimrehurek.com/gensim/auto_examples/howtos/run_downloader_api.html
# Setting 2 - Default Matching | WordCentroid model | No query expansion
match_op = Matching()
model = Word2Vec(processed_train_corpus, min_count=1)
wcd = WordCentroidDistance(model.wv)
retrieval = Retrieval(retrieval_model=wcd, matching=match_op)
retrieval.fit(processed_train_corpus)
idx = retrieval.query(prep.transform(["American elections republicans"], tokenize=False)[0], k=3)
[train_corpus[i] for i in idx.tolist()]

# Setting 3 - Default Matching | Doc2vec model | No query expansion
match_op = Matching()
model = Doc2Vec(vector_size=40, min_count=2, epochs=200)
model = Doc2Vec.load(os.path.join(output_path, "doc2vec_model"))  # loading trained embeddings
doc2vec = Doc2VecInference(model=model, analyzer=lambda x: x.split())
retrieval = Retrieval(retrieval_model=doc2vec, matching=match_op)
retrieval.fit(processed_train_corpus)
idx = retrieval.query(prep.transform(["American elections republicans"], tokenize=False)[0], k=3)
[train_corpus[i] for i in idx.tolist()]

