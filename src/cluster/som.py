'''
Self-Organising Maps

Fit a SOM for a given model.

Produces a .som file (saved in models/saved_models/)
and generates a U-matrix plot (saved in src/cluster/).

TODO:
- implement component planes
- improve u-matrix
- update docstrings

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
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import ParameterGrid
from src import PROJECT_ROOT


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
	params_name = "".join(sorted([k+str(v) for k, v in som_params.items()]))
	fname = model_name + "__" + params_name

	return fname


def viz_som(som, data, labels, unq_topics, som_file_name):
	"""Plot the U-matrix

	Args:
		som ([type]): [description]
		data ([type]): [description]
		labels ([type]): [description]
		unq_topics ([type]): [description]
		som_file_name ([type]): [description]
	"""
	logging.info("Plotting SOM U-matrix: {}".format(som_file_name))

	# Position of the neurons on an euclidean plane that reflects the chosen topology
	xx, yy = som.get_euclidean_coordinates()
	# Distance map of the weights. Each cell is the normalised sum of the distances between its neighbours
	umatrix = som.distance_map()
	# Weights of the neural network
	weights = som.get_weights()
	# Quantization error
	qterror = som.quantization_error(data)

	# Set figure and parameters
	f = plt.figure(figsize=(10,10))
	ax = f.add_subplot(111)
	ax.set_aspect('equal')
	colormap = plt.get_cmap('Blues')
	markers = dict(zip(unq_topics, ['o', '+', 'x', 'v', '^', 's', '*']))
	colors = dict(zip(unq_topics, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']))

	# Iteratively add hexagons
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
			hex = RegularPolygon((xx[(i, j)], wy), 
								numVertices=6, 
								radius=.95 / np.sqrt(3),
								facecolor=colormap(umatrix[i, j]), 
								alpha=.4, 
								edgecolor='gray')
			ax.add_patch(hex)  # Add the hexagonal patch to the axis 

	# Labelling each unit in the output map
	for cnt, x in enumerate(data):
		# getting the winner
		w = som.winner(x)
		# place a marker on the winning position for the sample xx
		wx, wy = som.convert_map_to_euclidean(w) 
		wy = wy * 2 / np.sqrt(3) * 3 / 4
		plt.plot(wx, wy, 
				markers[labels[cnt]], 
				markerfacecolor='None',
				markeredgecolor=colors[labels[cnt]], 
				markersize=12, 
				markeredgewidth=2)

	xrange = np.arange(weights.shape[0])
	yrange = np.arange(weights.shape[1])
	plt.xticks(xrange-.5, xrange)
	plt.yticks(yrange * 2 / np.sqrt(3) * 3 / 4, yrange)

	# Add distance colorbar to figure
	divider = make_axes_locatable(plt.gca())
	ax_cb = divider.new_horizontal(size="5%", pad=0.05)  
	cb1 = plt.colorbar(cm.ScalarMappable(cmap=colormap), 
					cax=ax_cb, orientation='vertical', alpha=.4)
	cb1.ax.get_yaxis().labelpad = 16
	cb1.ax.set_ylabel('Distance from neurons in the neighbourhood',
					rotation=270, fontsize=14)
	plt.gcf().add_axes(ax_cb)

	# Add legend to figure
	legend_elements = [Line2D([0], [0], marker=markers[topic], color=colors[topic], label=topic,
	 	markerfacecolor='w', markersize=14, linestyle='None', markeredgewidth=2) for topic in unq_topics]
	ax.legend(handles=legend_elements, bbox_to_anchor=(0, 1.1), loc='upper left', 
			borderaxespad=0., ncol=4, fontsize=14)

	ax.set_title(f"Embeddings: {som_file_name.split('__')[0]}, Quantization error: {np.round(qterror, 2)}",
		loc='center', pad=55, fontsize=16)
	logging.info("Saving SOM U-matrix: {}".format(som_file_name))
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

	with open(fpath, 'rb') as infile:
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
				viz_som(som, vect_train_corpus, train_labels, unq_topics, som_file_name)
		else:
			som = fit_som(som_file_name, vect_train_corpus, params)
			viz_som(som, vect_train_corpus, train_labels, unq_topics, som_file_name)

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
			'x': [30],
			'y': [30],
            'sigma': [1, 0.5],
            'learning_rate': [1, 0.5],
			'neighborhood_function': ['gaussian'],
            'n_iter': [1000, 500, 100],
			'topology': ['hexagonal']
        }
    )

	main()
