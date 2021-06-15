from os.path import join, abspath, pardir, dirname
import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from gensim.parsing.preprocessing import preprocess_documents
from gensim.sklearn_api.d2vmodel import D2VTransformer
import sompy
from umap import UMAP

DATA_PATH = join(
    dirname(abspath(__file__)), pardir, 'data', 'news2.csv'
)
SAVE_PATH = join(
    dirname(abspath(__file__)), pardir, 'outputs'
)

df = pd.read_csv(DATA_PATH)
df.drop(columns='dataset', inplace=True)

preprocess = FunctionTransformer(preprocess_documents)

# These are just mock params
encoder = Pipeline([
    ('preprocess', preprocess),
    ('doc2vec', D2VTransformer(
        size=25,
        iter=50,
        min_count=20,
        seed=42,
        workers=8
    )),
    ('stdscaler', StandardScaler()),
])

X = encoder.fit_transform(df.text)

# Save encoder
pickle.dump(encoder, open(join(SAVE_PATH, 'encoder2.pkl'), 'wb'))

# Found a som implementation in pytorch, I'll adapt it later for our needs
# https://github.com/theblackfly/fast-som
# https://github.com/giannisnik/som
# https://github.com/bougui505/quicksom

np.random.seed(42)
som = sompy.SOMFactory.build(
    X, (20, 20),
    mapshape='planar', lattice='hexa',
    normalization='var', initialization='pca',
    neighborhood='gaussian', training='batch',
)
som.train(train_rough_len=50, train_finetune_len=100, n_job=8, verbose='info')

u = sompy.umatrix.UMatrixView(
    9, 9, 'U-Matrix', show_axis=True, text_size=8, show_text=True
)

umatrix = u.build_u_matrix(som)

# Get umap projections
projections = UMAP(random_state=42).fit_transform(X)

# Preprocess data

# assign a unit ID to each document
df['docs_bmu'] = som.find_bmu(X)[0]

# som units values in the input and grid spaces
codebook = som.codebook.matrix
bmus = pd.DataFrame(codebook)
indices = np.indices(umatrix.shape)
bmus['row'] = indices[0].flatten()
bmus['col'] = indices[1].flatten()
bmus['umat_val'] = umatrix.flatten()

categories = pd.get_dummies(df.category)
categories['docs_bmu'] = df['docs_bmu']
categories = categories.groupby('docs_bmu').mean()
bmus['dominant_cat'] = categories.idxmax(1)
bmus['dominant_cat_perc'] = categories.max(1)
bmus = bmus.join(categories)
bmus = bmus.join(
    df.groupby('docs_bmu').size().rename('docs_count')
)

# cluster som units
# number of clusters was based on a dendrogram, intencionally went with a
# relatively large number of clusters

clusterer = AgglomerativeClustering(n_clusters=16)
bmus['cluster_label'] = clusterer.fit_predict(bmus[list(range(25))])

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['docs_bmu', 'timestamp', 'text', 'category']]

output = {
    'df': df,
    'projections': projections,
    'doc_embeddings': X,
    'bmus': bmus
}
pickle.dump(output, open(join(SAVE_PATH, 'ml_engine2.pkl'), 'wb'))
