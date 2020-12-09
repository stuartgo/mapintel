'''
Self-Organising Maps

Fit a SOM for a given model.

Produces a .som file (saved in models/saved_models/)
and generates a U-matrix plot (saved in src/cluster/).

TODO:
- implement component planes

Examples
--------
>>> # Fit a map for a given model, and save resulting SOM model
>>> # in same directory, with same filename, but .som extension
>>> python som.py ../../models/saved_models/TfidfVectorizer.joblib
'''


import logging
import os
import re
from sys import exit

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim import models
from minisom import MiniSom
from pylab import axis, bone, colorbar, pcolor, plot, show
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import ParameterGrid
from src import PROJECT_ROOT

# from src.visualization.embedding_space import embedding_vectors, read_data

'''
def load_embeddings():

	return embeddings
'''


def fit_som(som_file_name, vect_train_corpus, som_params):
	''' Fit a SOM

	Parameters
	----------
	model_name:        (str) model name
	vect_train_corpus: dataset
	sigma:             (float) neighbourhood function spread
	lr:                (int) learning rate
	n_iter:            (int) number of iterations

	Returns
	-------
	som:               SOM object
	som_dim:           (int) dimensionality of the SOM (N, where SOM is NxN)
	'''

	logging.info("SOM does not yet exist. Fitting SOM... {}".format(som_file_name))

	# Separating n_iter to pass to train method
	n_iter = som_params['n_iter']
	params = som_params.copy()
	params.pop('n_iter')

	# Initialise a 6x6 SOM
	som = MiniSom(**params, random_seed=0)
	
	# Train the SOM
	# NB. sigma and learning_rate are both reduced over the course of training
	som.train(vect_train_corpus, n_iter, verbose=True)

	# Save the SOM
	save_som(som, som_file_name)

	return som


# def winning_neuron_per_doc(som, vect_train_corpus):
# 	''' Build a dict with the winning neuron coordinates for each document 

# 	NB. this is an experimental function, not used in the pipeline right now
# 	'''
# 	n_docs = np.shape(vect_train_corpus)[0]
# 	winner = {}
# 	winner = {n: som.winner(vect_train_corpus[n]) for n in range(n_docs)}
# 	return winner


def som_fname(model_name, som_params):
	''' Compile the SOM filename based on its hyper-parameters
	'''
	# Get filename based on som_params
	params_name = "".join([k+str(v) for k, v in som_params.items()])
	fname = model_name + params_name

	return fname


def viz_som(som, data, labels, som_file_name, som_params, subplot_idx=None):
	''' Plot the U-matrix
	'''

	logging.info("Plotting SOM U-matrix: {}".format(som_file_name))
	
	bone()

	# Distance map aka U-matrix
	pcolor(som.distance_map().T)
	# plt.colorbar()

	# Loading labels
	n_labels = len(labels.unique())
	target = range(n_labels)
	labs = {}
	labels_int = []
	for i,l in enumerate(labels.unique()):
		labs[l] = i
	target = labs
	
	# Need the labels list to be numbers and not strings
	# in order to index the markers
	labels_int = [labs[n] for n in labels]
	target = labels_int

	markers = ['o','s','D','x','o','s','D']
	colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

	for cnt, xx in enumerate(data):
		# For sample xx...
		# Get winning neuron
		w = som.winner(xx)

		# Place a marker on the winning position for this sample...
		# Index for marker colour and shape
		mrk_idx = target[cnt]
		# Offset the markers by an amount between 0.2-0.8 to avoid overlap
		offset = 0.2 + 0.05*mrk_idx
		# Marker coordinates
		x = w[0] + offset
		y = w[1] + offset

		plot(x, y,
			markers[target[cnt]-1],
			markerfacecolor='None',
			markeredgecolor=colors[target[cnt]-1],
			markersize=5,
			markeredgewidth=2,
			alpha=0.1)
	#axis([0,som.weights.shape[0],0,som.weights.shape[1]])

	plt.title(som_file_name)
	plt.savefig(os.path.join(fig_dir, som_file_name + '.png'))


def check_som_exists(fname, type):
	'''Check if SOM exists before fitting it'''
	
	if type == "som":
		fpath = os.path.join(model_dir, fname + '.som')
	elif type == "viz":
		fpath = os.path.join(fig_dir, fname + '.png')
	else:
		raise ValueError("type can only take 'viz' or 'som' values.")

	# Check path existence
	if os.path.exists(fpath):
		return True
	else:
		return False


def save_som(som, model_name):
	"""Save the SOM into a .som file

	Args:
		som (object): MiniSom class instance, fitted to the embeddings
		model_name (str): name of the embedding model, used to name the SOM
	Returns:
		None
	"""

	fname = model_name + '.som'
	fpath = os.path.join(model_dir, fname)

	logging.info("Saving SOM: {}".format(fpath))

	with open(fpath, 'wb') as outfile:
		joblib.dump(som, outfile)


def load_som(fname):
	'''Load SOM from a .som file'''

	fname = fname + '.som'
	fpath = os.path.join(model_dir, fname)

	logging.info("Loading SOM: {}".format(fpath))

	with open(fpath, 'wb') as infile:
		som = joblib.load(infile)
	return som


def read_data(data_file):
    """Read the preprocessed news corpus and splits it into training and test set.
    Args:
        data_file (str): path to csv file that holds the preprocessed data
    Returns:
        tuple: unique categories list, training documents and test documents
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(data_file, names=[
                     'id', 'col', 'category', 'text', 'split', 'prep_text'])
    logger.info(f'Read data has a size of {df.memory_usage().sum()//1000}Kb')

    logger.info('Formatting data...')
    # Formatting data
    all_docs = df.loc[~df['prep_text'].isna()]
    train_docs = all_docs.loc[all_docs['split'] == 'train']
    test_docs = all_docs.loc[all_docs['split'] == 'test']

    train_labels = train_docs['category']
    test_labels = test_docs['category']

    unq_topics = df['category'].unique().tolist()
    logger.info(
        f'{train_docs.shape[0]} documents from train set out of {df.shape[0]} documents')
    del df, all_docs

    return unq_topics, train_docs, test_docs, train_labels, test_labels


def embedding_vectors(model_path, train_docs, test_docs):
    """Get embedding vectors from train_docs and test_docs using a given model.
    The function uses the model_name to extract the model type and then obtain
     the embeddings in the correct way for the given model.
    Applies LSA dimensionality reduction to BOW and TF-IDF vectors to reduce
     their dimensionality to 100 components.
    Args:
        model_name (str): Complete path to model file
        train_docs (array-like): Strings of each train document
        test_docs (array-like): Strings of each test document
    Raises:
        ValueError: When it isn't possible to infer model type off model_name
    Returns:
        tuple of ndarray or sparse matrix: embedding vectors of train documents and test documents
    """
    logger = logging.getLogger(__name__)
    # Extract model_type out model_tag
    if re.search(".*Vectorizer\.joblib$", os.path.basename(model_path)):
        model_type = "vectorizer"
    elif re.search("^doc2vec.*\.model$", os.path.basename(model_path)):
        model_type = "doc2vec"
    else:
        raise ValueError("Couldn't extract valid model_type from model_name.")

    if model_type == "doc2vec":
        # Loading fitted model
        model = models.doc2vec.Doc2Vec.load(model_path)
        # Obtain the vectorized corpus
        vect_train_corpus = np.vstack(
            [model.docvecs[i] for i in range(train_docs.shape[0])])
        vect_test_corpus = np.vstack(
            [model.infer_vector(i) for i in test_docs.str.split()])

        return vect_train_corpus, vect_test_corpus

    elif model_type == "vectorizer":
        # Loading fitted model
        model = joblib.load(model_path)
        # Obtain the vectorized corpus
        vect_train_corpus = model.transform(train_docs)
        vect_test_corpus = model.transform(test_docs)
        # LSA to reduce vector dimensionality
        logger.info("Applying LSA to reduce dimensionality...")
        lsa = TruncatedSVD(100, random_state=1)
        vect_train_corpus_red = lsa.fit_transform(vect_train_corpus)
        vect_test_corpus_red = lsa.transform(vect_test_corpus)
        explained_variance = lsa.explained_variance_ratio_.sum()
        logger.info("Explained variance of the LSA step: {}%".format(
            int(explained_variance * 100)))

        return vect_train_corpus_red, vect_test_corpus_red


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def main(model_path):
	logger = logging.getLogger(__name__)

	model_tag = os.path.splitext(os.path.basename(model_path))[0]

	# Reading data into memory
	logger.info('Reading data...')
	unq_topics, train_docs, test_docs, train_labels, test_labels = read_data(data_file)

	# Obtain the vectorized corpus
	logger.info('Obtaining document embeddings...')
	vect_train_corpus, vect_test_corpus = embedding_vectors(
		model_path, train_docs['prep_text'], test_docs['prep_text'])

	# Fit SOM
	idx_mdl = 0
	for params in param_grid:
		logger.info("Fitting SOM #{}".format(idx_mdl))
		# Add input_len to dictionary
		params.update({'input_len': np.shape(vect_train_corpus)[1]})
		logger.info("Fitting SOM with hyper-parameters: {}".format(params))

		# Get output file name and check if it already exists
		som_file_name = som_fname(model_tag, params)

		if check_som_exists(som_file_name, "som"):
			logging.info("SOM already exists. Skip model fitting for {}".format(som_file_name))
			if check_som_exists(som_file_name, "viz"):
				logging.info("SOM plot already exists: {}".format(som_file_name))
			else:
				som = load_som(som_file_name)
				viz_som(som, vect_train_corpus, train_labels, som_file_name, params)
		else:
			som = fit_som(som_file_name, vect_train_corpus, params)
			viz_som(som, vect_train_corpus, train_labels, som_file_name, params)

		idx_mdl += 1


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# Defining Paths
	data_file = os.path.join(PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
	model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
	fig_dir = os.path.join(PROJECT_ROOT, "models", "figures", "som")

	# Hyperparameter grid
	param_grid = ParameterGrid(
        {	
			'x': [16],
			'y': [16],
            'sigma': [0.5, 1],
            'learning_rate': [0.5, 1],
			'neighborhood_function': ['gaussian'],
            'n_iter': [100, 500, 1000]
        }
    )

	main()
