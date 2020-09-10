from scripts.utility import CorpusPreprocess, check_random_doc_similarity, compare_documents, similarity_query
from datetime import datetime, timedelta
import random
import collections
from newsapi import NewsApiClient
from string import punctuation
from nltk.corpus import stopwords
from gensim import models
# import numpy as np
# from scipy.spatial.distance import pdist, squareform

# TODO: add date, price, weekday, ... token to CorpusPreprocess
#       webscrape full content from urls provided by api

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
prep = CorpusPreprocess(stop_words=stopwords.words('english'), lowercase=True, strip_accents=True,
                        strip_punctuation=punctuation, stemmer=True, max_df=0.2, min_df=2)
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

# Optimally we want as much documents to be the most similar with themselves (i.e. rank 0)
print(collections.OrderedDict(sorted(collections.Counter(ranks).items())))

# Pick a random document from the train corpus, infer its vector and check similarity with other documents
doc_id, sims = check_random_doc_similarity(model, tagged_corpus)
compare_documents(doc_id, train_corpus, sims, train_corpus)

# Pick a random document from the test corpus, infer its vector and check similarity with other documents
doc_id, sims = check_random_doc_similarity(model, tagged_corpus, processed_test_corpus)
compare_documents(doc_id, test_corpus, sims, train_corpus)

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
doc_id = random.randint(0, len(test_corpus) - 1)
unkwnown_doc = new_processed_corpus[doc_id]
sims = similarity_query(model, unkwnown_doc)
compare_documents(doc_id, new_corpus, sims, train_corpus)

