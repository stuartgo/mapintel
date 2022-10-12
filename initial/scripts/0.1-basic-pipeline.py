from os.path import join, abspath, pardir, dirname
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from gensim.parsing.preprocessing import preprocess_documents
from gensim.sklearn_api.d2vmodel import D2VTransformer
import sompy

DATA_PATH = join(dirname(abspath(__file__)), pardir, 'data', 'news.csv')

df = pd.read_csv(DATA_PATH)
# mask = df.dataset == 'train'
df.drop(columns='dataset', inplace=True)

# df_train, df_test = df[mask], df[~mask]

@FunctionTransformer
def preprocess(text):
    """Preprocess corpus"""
    return preprocess_documents(text)

# These are just mock params
encoder = Pipeline([
    ('preprocess', preprocess),
    ('doc2vec', D2VTransformer(size=25, iter=50, min_count=20, seed=42, workers=8)),
    ('stdscaler', StandardScaler()),
])

X = encoder.fit_transform(df.text)

# --------------------------------------------------------------------------------------
# SOMPY
# --------------------------------------------------------------------------------------
"""
This som is not good, but perhaps we can take it to make a decent version of it.
I couldn't get the som-learn option to work properly.
"""

som = sompy.SOMFactory.build(
    X, (50,50),
    mapshape='planar', lattice='hexa',
    normalization='var', initialization='pca',
    neighborhood='gaussian', training='batch',
)
som.train(train_rough_len=10, train_finetune_len=10, n_job=8, verbose='info')

u = sompy.umatrix.UMatrixView(9, 9, 'U-Matrix', show_axis=True, text_size=8, show_text=True)

_, umat = u.show(
    som,
    distance=2,
    contour=True,
    blob=False
)

# Extract results
codebook = som.codebook.matrix
bmu_indices = som.find_bmu(codebook)[0].astype(int)
bmu_indices = bmu_indices.reshape(*som.codebook.mapsize)
projected = codebook.reshape(*som.codebook.mapsize,codebook.shape[-1])
docs_bmu = som.find_bmu(X)[0]

# Store data to feed to streamlit
# NOTE: I was not able to store the encoder pipeline, must check this later.
output = {
    'som': som,
    'umatrix': umat,
    'embeddings': X,
    'docs_bmu': docs_bmu,
    'codebook': codebook,
    'bmu_indices_map': bmu_indices,
    'bmu': projected
}
pickle.dump(output, open('../outputs/ml_engine.pkl', 'wb'))
