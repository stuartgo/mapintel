'''
Self-Organising Maps

Fit a SOM for a given embedding vector space.

Produces a .som file and generates a U-matrix plot.

Examples
--------
>>> # Fit a map for a given model, and save resulting SOM model
>>> # in same directory, with same filename, but .som extension
>>> python src/som/som_tuning.py outputs/saved_embeddings/train_doc2vecdmcd100n5w5mc2t12.npy
'''

import logging
import os

import click
import numpy as np
import sompy
from sklearn.model_selection import ParameterGrid
from src import PROJECT_ROOT
from src.som.som_clustering import (check_som_exists, 
									load_som,
									save_som)
from src.features.embedding_extractor import (embeddings_generator,
                                              format_embedding_files,
                                              read_data)


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

	# Iterate over grid
	for idx_mdl, params in enumerate(param_grid):
		logger.info(f"Fitting SOM #{idx_mdl}")
		logger.info("Fitting SOM with hyper-parameters: {}".format(params))

		train_keys = ['train_rough_len', 'train_finetune_len']
		som_params = {k: v for k, v in params.items() if k not in train_keys}
		train_params = {k: v for k, v in params.items() if k in train_keys}

		# output file name
		outname = "som" + "".join(map(str, params.values()))
		outname = outname.translate(str.maketrans('', '', ' ,()'))
		outfigpath = os.path.join(fig_dir, outname + '.png')
		outsompath = os.path.join(model_dir, outname + '.som')

		# Setting random seed
		np.random.seed(42)

		# Define SOM
		som = sompy.SOMFactory().build(
        vect_train_corpus,
        **som_params
		)

		if check_som_exists(outname, "som", model_dir):
			logger.info("SOM model {} already exists in disk...".format(outname))
			if check_som_exists(outname, "viz", fig_dir):
				logger.info("SOM figure {} already exists in disk...".format(outname))
			else:
				# Loading the SOM model
				logger.info('Loading SOM model...')
				som = load_som(os.path.join(model_dir, outname + '.som'))
				# Plot and save the U-matrix
				logger.info('Plotting and saving the U-matrix...')
				u = sompy.umatrix.UMatrixView(
					width=7,
        			height=7,
        			text_size=6,
					title="U-Matrix",
				)
				u.show(
					som,
					distance=2, 
				)
				u.save(outfigpath)
		else:
			# Fit SOM
			logger.info('Fitting SOM model...')
			som.train(n_job=-1, 
				verbose='info',
				**train_params
			)
			# Plot and save the U-matrix
			logger.info('Plotting and saving the U-matrix...')
			u = sompy.umatrix.UMatrixView(
				width=7,
				height=7,
				text_size=6,
				title="U-Matrix",
			)
			u.show(
				som,
				distance=2, 
			)
			u.save(outfigpath)
			# Save the SOM fitted model
			logger.info('Saving SOM model...')
			save_som(som, outsompath)


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# Defining Paths
	data_file = os.path.join(PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
	model_dir = os.path.join(PROJECT_ROOT, "outputs", "saved_models")
	fig_dir = os.path.join(PROJECT_ROOT, "outputs", "figures", "som")

	# Hyperparameter grid
	param_grid = ParameterGrid(
        {	
			'mapsize': [(50,50)],
            'initialization': ['random','pca'],
			'neighborhood': ['gaussian'],
            'training': ['batch'],
			'lattice': ['hexa'],
			'train_rough_len': [50, 100],
			'train_finetune_len': [50, 100]
        }
    )

	main()
