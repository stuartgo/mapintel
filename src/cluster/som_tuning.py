'''
Self-Organising Maps

Fit a SOM for a given model.

Produces a .som file (saved in models/saved_models/)
and generates a U-matrix plot (saved in src/cluster/).

TODO:
- improve u-matrix
- update docstrings
- add multiprocessing option to MiniSom2
- change decay_function function parameter
- implement the unfolding and fine-tuning training phases (maybe as a decay_function)

Examples
--------
>>> # Fit a map for a given model, and save resulting SOM model
>>> # in same directory, with same filename, but .som extension
>>> python som.py ../../models/saved_models/TfidfVectorizer.joblib
'''

import logging
import os

import click
from sklearn.model_selection import ParameterGrid
from src import PROJECT_ROOT
from src.cluster.som_clustering import (MiniSom2, check_som_exists, load_som,
                                        save_som)
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)


def check_som_exists(fname, type):
	"""Check if SOM exists before fitting it

	Args:
		fname ([type]): [description]
		type ([type]): [description]

	Raises:
		ValueError: [description]

	Returns:
		[type]: [description]
	"""
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


@click.command()
@click.argument('embeddings_file', type=click.Path(exists=True))
def main(embeddings_file):
	logger = logging.getLogger(__name__)

	split, model_tag = os.path.splitext(os.path.basename(embeddings_file))[0].split("_")

	logger.info('Reading data...')
	# Reading data into memory
	# Assert we are working with train corpus
	if split == "train":
		_, train_docs, _ = read_data(data_file)
	else:
		raise ValueError("For clustering, use the train set embedding vectors.")

	train_labels = train_docs['category']

	logger.info('Obtaining document embeddings...')
	# Obtain the vectorized corpus
	embedding_dict = format_embedding_files([embeddings_file])
	gen = embeddings_generator(embedding_dict)
	_, vect_train_corpus = list(gen)[0]
	input_len = vect_train_corpus.shape[1]

	# Iterate over grid
	for idx_mdl, params in enumerate(param_grid):
		logger.info(f"Fitting SOM #{idx_mdl}")
		logger.info("Fitting SOM with hyper-parameters: {}".format(params))

		# Define SOM
		som = MiniSom2(random_seed=0, verbose=True, input_len=input_len, **params)

		if check_som_exists(str(som), "som"):
			logger.info("SOM model {} already exists in disk...".format(str(som)))
			if check_som_exists(str(som), "viz"):
				logger.info("SOM figure {} already exists in disk...".format(str(som)))
			else:
				# Loading the SOM model
				logger.info('Loading SOM model...')
				som = load_som(os.path.join(model_dir, str(som) + '.som'))
				# Plot and save the U-matrix
				logger.info('Plotting and saving the U-matrix...')
				som.u_matrix(vect_train_corpus, train_labels, fig_dir)
		else:
			# Fit SOM
			logger.info('Fitting SOM model...')
			som.fit(vect_train_corpus, train_labels)
			# Plot and save the U-matrix
			logger.info('Plotting and saving the U-matrix...')
			som.u_matrix(vect_train_corpus, train_labels, fig_dir)
			# Save the SOM fitted model
			logger.info('Saving SOM model...')
			outpath = os.path.join(model_dir, str(som) + '.som')
			save_som(som, outpath)


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
