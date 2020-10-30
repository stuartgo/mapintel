"""
Visualize embedding space. INCOMPLETE!
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE

# TODO: complete!
train_corpus = None  # define unprocessed train_corpus
processed_test_corpus = None  # define processed test_corpus
doc2vec_model = None  # define the doc2vec fitted model
topics = None  # define the possible label values

# Obtain the vectorized corpus
vect_train_corpus = np.vstack([doc2vec_model.docvecs[i] for i in range(len(train_corpus))])
vect_test_corpus = np.vstack([doc2vec_model.infer_vector(i) for i in processed_test_corpus])
# vect_mix_corpus = np.vstack([vect_train_corpus, vect_test_corpus])

# Visualize a 2D map of the vectorized corpus
tsne_model = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=5000,
                  n_iter_without_progress=300, metric='cosine', verbose=1)

embedded_train_corpus = tsne_model.fit_transform(vect_train_corpus)
embedded_test_corpus = tsne_model.fit_transform(vect_test_corpus)
# embedded_mix_corpus = np.vstack([embedded_train_corpus, embedded_test_corpus])

categ_map = dict(zip(set(topics), ['red', 'blue', 'green', 'yellow', 'orange']))
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, X, top, tit in zip(axes.flatten(), [embedded_train_corpus, embedded_test_corpus], [train_topics, test_topics],
                           ['Train Corpus Embedding Space', 'Test Corpus Embedding Space']):
    # Color for each point
    color_points = list(map(lambda x: categ_map[x], top))
    # Scatter plot
    ax.scatter(X[:, 0], X[:, 1], c=color_points)
    # Produce a legend with the unique colors from the scatter
    handles = [mpatches.Patch(color=c, label=l) for l, c in categ_map.items()]
    ax.legend(handles=handles, loc="upper left", title="Topics", bbox_to_anchor=(0., 0.6, 0.4, 0.4))
    # Set title
    ax.set_title(tit)
# plt.savefig(os.path.join(models_path, "tsne_news_embeddings.png"))  # exports png to current directory
plt.show()