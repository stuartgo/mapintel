# Evaluation methods:
# https://radimrehurek.com/gensim/auto_examples/howtos/run_doc2vec_imdb.html

# Predictive downstream task (i.e. classifying news topics)

# Are inferred vectors close to the precalculated ones?
# ranks = []
# for tagged_doc in tagged_corpus:
#     tags, words = tagged_doc.tags, tagged_doc.words
#     inferred_vector = model.infer_vector(words)
#     sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
#     rank = [tag_sim for tag_sim, _ in sims].index(tags)
#     ranks.append(rank)

# # Optimally we want as much documents to be the most similar with themselves (i.e. rank 0)
# print(collections.OrderedDict(sorted(collections.Counter(ranks).items())))

# Do close documents seem more related than distant ones?
# # Pick a random document from the train corpus, infer its vector and check similarity with other documents
# doc_ix = randint(0, len(X_train) - 1)
# doc_id, base_doc_rep = X_train.reset_index().loc[doc_ix].values.flatten()
# doc_rep = processed_train_corpus[doc_ix]
# inferred_unknown_vector = model.infer_vector(doc_rep)
# sims = model.docvecs.most_similar([inferred_unknown_vector], topn=len(model.docvecs))
# compare_documents(doc_id, base_doc_rep, sims, X_train.to_list())

# print("---------------------------------------------------------------------------------------------------------------------------------\n")
# # Pick a random document from the test corpus, infer its vector and check similarity with other documents
# doc_ix = randint(0, len(X_test) - 1)
# doc_id, base_doc_rep = X_test.reset_index().loc[doc_ix].values.flatten()
# doc_rep = processed_test_corpus[doc_ix]
# inferred_unknown_vector = model.infer_vector(doc_rep)
# sims = model.docvecs.most_similar([inferred_unknown_vector], topn=len(model.docvecs))
# compare_documents(doc_id, base_doc_rep, sims, X_train.to_list())

# Do the word vectors show useful similarities?

# Are the word vectors from this dataset any good at analogies?
