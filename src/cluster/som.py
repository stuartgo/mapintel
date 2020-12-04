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


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sys import exit
from minisom import MiniSom
import click
import logging
import os
import re
import joblib
from src import PROJECT_ROOT

from sklearn.decomposition import TruncatedSVD
from gensim import models
#from src.visualization.embedding_space import read_data, embedding_vectors


'''
def load_embeddings():

	return embeddings
'''


def tune_som(vect_train_corpus, train_labels, model_name):
	''' Tune & plot SOMs for different hyperparameters

	Parameters
	----------
	vect_train_corpus: (float) neighbourhood function spread
	train_labels:      labels for the dataset
	model_name:        (str) model name

	Returns
	-------
	som:               SOM object
	som_dim:           (int) dimensionality of the SOM (N, where SOM is NxN)
	'''
	
	logger = logging.getLogger(__name__)

	som_hyperparams = {}
	som_hyperparams['neighbourhood_function_spread'] = [0.5, 1]
	som_hyperparams['learning_rate'] = [0.5, 1]
	som_hyperparams['n_iter'] = [100, 500, 1000]
	
	idx_mdl = 0
	models = {}

	for n_iter in som_hyperparams['n_iter']:
		for lr in som_hyperparams['learning_rate']:
			for sigma in som_hyperparams['neighbourhood_function_spread']:
				
				logger.info("Fitting SOM #{}".format(idx_mdl))
				logger.info("Fitting SOM with hyper-parameters:  \
				          \nn_iter: {}\nlearning_rate: {}\nsigma: {} \
				          ".format(n_iter, lr, sigma))

				som, som_dim = fit_som(model_name, vect_train_corpus, sigma, lr, n_iter)
				models[idx_mdl] = som
				idx_mdl += 1

				# main call
				viz_som(som, som_dim, vect_train_corpus, train_labels, model_name, n_iter, lr, sigma)

	return som, som_dim


def fit_som(model_name, vect_train_corpus, sigma, lr, n_iter):
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
	
	som_exists = check_som_exists(model_name)

	if som_exists is True:
		logging.info("SOM already exists. Skip model fitting for {}".format(model_name))
		som = load_som(model_name)
		return som, None

	elif som_exists is False:
		logging.info("SOM does not yet exist. Fitting SOM... {}".format(model_name))

		embed_dimensionality = np.shape(vect_train_corpus)[1]
		som_dim = 16

		# Initialise a 6x6 SOM
		som = MiniSom(som_dim, som_dim,
					  embed_dimensionality,
					  sigma=sigma,
					  learning_rate=lr,
					  neighborhood_function='gaussian',
					  random_seed=0)
		
		# Train the SOM
		# NB. sigma and learning_rate are both reduced over the course of training
		som.train(vect_train_corpus, n_iter)

		# Save the SOM
		som_model_name = som_fname(model_name, som_dim, n_iter, lr, sigma)
		save_som(som, som_model_name)


		return som, som_dim


def winning_neuron_per_doc(som, vect_train_corpus)
	''' Build a dict with the winning neuron coordinates for each document 

	NB. this is an experimental function, not used in the pipeline right now
	'''
	n_docs = np.shape(vect_train_corpus)[0]
	winner = {}
	winner = {n: som.winner(vect_train_corpus[n]) for n in range(n_docs)}
	return winner


def som_fname(model_name, som_dim, n_iter, lr, sigma):
	''' Compile the SOM filename based on its hyper-parameters
	'''

	fname = model_name + "__" + str(som_dim) + "x" + str(som_dim) + \
	          "_n_iter" + str(n_iter) + "lr" + str(lr) + "sigma" + str(sigma)

	return fname


def viz_fname(model_name, som_dim, n_iter, lr, sigma):
	''' Compile the U-matrix filename based on the SOM's hyper-parameters
	'''

	fname = model_name + "__" + str(som_dim) + "x" + str(som_dim) + \
	          "_n_iter" + str(n_iter) + "lr" + str(lr) + "sigma" + str(sigma) + \
	          ".png"

	if os.path.exists(fname):
		som_plot_exists = True
	else:
		som_plot_exists = False

	return fname, som_plot_exists


def viz_som(som, som_dim, data, labels, model_name, n_iter, lr, sigma, subplot_idx=None):
	''' Plot the U-matrix
	'''

	from pylab import plot, axis, show, pcolor, colorbar, bone
	
	fname, som_plot_exists = viz_fname(model_name, som_dim, n_iter, lr, sigma)
	if som_plot_exists is True:
		logging.info("SOM plot already exists: {}".format(fname))

	elif som_plot_exists is False:
		logging.info("Plotting SOM U-matrix: {}".format(fname))

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

		title = str(som_dim) + "x" + str(som_dim) + " " + model_name
		plt.title(title)

		plt.savefig(fname)



def check_som_exists(model_name):
	'''Check if SOM exists before fitting it'''
	fname = model_name.split('.')[0] + '.som'
	fpath = os.path.join(model_dir, fname)
	if os.path.exists(fpath):
		return True
	else:
		return False



#def save_som(som, model_name):
def save_som(som, som_model_name):
	"""Save the SOM into a .som file

	Args:
		som (object): MiniSom class instance, fitted to the embeddings
		model_name (str): name of the embedding model, used to name the SOM
	Returns:
		None
	"""

	#fname = model_name.split('.')[0] + '.som'
	fname = som_model_name + '.som'
	fpath = os.path.join(model_dir, fname)

	logging.info("Saving SOM: {}".format(fpath))

	with open(fpath, 'wb') as outfile:
		joblib.dump(som, outfile)
	return


def load_som(model_name):
	'''Load SOM from a .som file'''

	fname = model_name.split('.')[0] + '.som'
	with open(os.path.join(model_dir, fname), 'wb') as infile:
		print('loading...?', os.path.join(model_dir, fname))
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
	som, som_dim  = tune_som(vect_train_corpus, train_labels, model_tag)


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# Defining Paths
	data_file = os.path.join(PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
	model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
	out_dir = os.path.join(PROJECT_ROOT, "models", "figures")

	main()
