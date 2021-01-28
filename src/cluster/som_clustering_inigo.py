"""
Self-Organizing Maps (insert docstring here)

# TO-DO:
Improve decay_function parameter of MiniSom as it receives a function 
right now which cannot be saved in the file name and makes it unpickable.
My suggestion would be to create some functions and let the parameter
which pre-defined function to use.
"""

import logging
import os
from sys import exit

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
from minisom import MiniSom
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import RegularPolygon
from matplotlib.lines import Line2D
from src import PROJECT_ROOT
from src.visualization.embedding_space import embedding_vectors, read_data


class MiniSom2(MiniSom):
	def __init__(self, n_iter, verbose=False, **minisom_kwargs):	
		super().__init__(**minisom_kwargs)
		self._n_iter = n_iter
		self._verbose = verbose
		self._dist = minisom_kwargs.get('activation_distance', 'euclidean')
		self._neig = minisom_kwargs.get('neighborhood_function', 'gaussian')
	
	# Inherit docstring
	__doc__ = MiniSom.__doc__

	def __str__(self):
		hyperparams = {
			'x': len(self._neigx),
			'y': len(self._neigy),
			'inplen': self._input_len,
			'sigma': self._sigma,
			'lr': self._learning_rate,
			'neig': self._neig,
			'topol': self.topology,
			'dist': self._dist,
			'niter': self._n_iter
		}
		return 'som_'+ ''.join(sorted([k+str(v) for k, v in hyperparams.items()]))
	
	def fit(self, X, y=None):
		"""[summary]

		Args:
			X ([type]): [description]
			y ([type], optional): [description]. Defaults to None.
		"""
		self.train(X, self._n_iter, True, self._verbose)
	

	def u_matrix(self, X, y=None, out_dir=None):
		"""[summary]

		Args:
			X ([type]): [description]
			y ([type], optional): [description]. Defaults to None.
		"""
		# Position of the neurons on an euclidean plane that reflects the chosen topology
		xx, yy = self.get_euclidean_coordinates()
		# Distance map of the weights. Each cell is the normalised sum of the distances between its neighbours
		umatrix = self.distance_map()
		# Weights of the neural network
		weights = self.get_weights()
		# Quantization error
		qterror = self.quantization_error(X)

		# Set figure and parameters
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)
		ax.set_aspect('equal')
		colormap = plt.get_cmap('Blues')
		unq_topics = np.unique(y)
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
		for cnt, x in enumerate(X):
			# getting the winner
			w = self.winner(x)
			# place a marker on the winning position for the sample xx
			wx, wy = self.convert_map_to_euclidean(w) 
			wy = wy * 2 / np.sqrt(3) * 3 / 4
			plt.plot(wx, wy, 
					markers[y[cnt]], 
					markerfacecolor='None',
					markeredgecolor=colors[y[cnt]], 
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

		ax.set_title(f"Embeddings: {self.__str__()}, Quantization error: {np.round(qterror, 2)}",
			loc='center', pad=55, fontsize=16)

		'''
		if out_dir:
			plt.savefig(os.path.join(out_dir, self.__str__() + '.png'))
		else:
			plt.savefig(self.__str__() + '.png')
		'''
		plt.show()
	

	def u_matrix_interactive(self, X, y=None, out_dir=None):
		"""[summary]

		Args:
			X ([type]): [description]
			y ([t	ype], optional): [description]. Defaults to None.
		"""

		### IMR
		from matplotlib import cm, colorbar

		from bokeh.colors import RGB
		from bokeh.io import curdoc, show, output_notebook
		from bokeh.transform import factor_mark, factor_cmap
		from bokeh.models import ColumnDataSource, HoverTool
		from bokeh.plotting import figure, output_file
		
		from sys import exit
		## /IMR

		# Position of the neurons on an euclidean plane that reflects the chosen topology
		xx, yy = self.get_euclidean_coordinates()
		# Distance map of the weights. Each cell is the normalised sum of the distances between its neighbours
		umatrix = self.distance_map()
		# Weights of the neural network
		weights = self.get_weights()
		# Quantization error
		qterror = self.quantization_error(X)

		# Set figure and parameters
		fig = plt.figure(figsize=(10,10))
		ax = fig.add_subplot(111)
		ax.set_aspect('equal')
		colormap = plt.get_cmap('Blues')
		unq_topics = np.unique(y)
		markers = dict(zip(unq_topics, ['o', '+', 'x', 'v', '^', 's', '*']))
		colors = dict(zip(unq_topics, ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']))

		###
		hex_centre_col, hex_centre_row = [], []
		hex_colour = []
		label = []
		SPECIES = unq_topics
		###

		# Iteratively add hexagons
		for i in range(weights.shape[0]):
			for j in range(weights.shape[1]):
				wy = yy[(i, j)] * 2 / np.sqrt(3) * 3 / 4
				'''
				hex = RegularPolygon((xx[(i, j)], wy), 
									numVertices=6, 
									radius=.95 / np.sqrt(3),
									facecolor=colormap(umatrix[i, j]), 
									alpha=.4, 
									edgecolor='gray')
				ax.add_patch(hex)  # Add the hexagonal patch to the axis 
				'''
				###
				hex_centre_col.append(xx[(i, j)])
				hex_centre_row.append(wy)
				hex_colour.append(cm.Blues(umatrix[i, j]))
				###

		###
		# Loading labels	
		labels = y
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
		t = target

		weight_x, weight_y = [], []
		###

		# Labelling each unit in the output map
		for cnt, x in enumerate(X):
			# getting the winner
			w = self.winner(x)
			# place a marker on the winning position for the sample xx
			wx, wy = self.convert_map_to_euclidean(w) 
			wy = wy * 2 / np.sqrt(3) * 3 / 4
			'''
			plt.plot(wx, wy, 
					markers[y[cnt]], 
					markerfacecolor='None',
					markeredgecolor=colors[y[cnt]], 
					markersize=12, 
					markeredgewidth=2)
			'''
			weight_x.append(wx)
			weight_y.append(wy)
			label.append(SPECIES[t[cnt]-1])

		# convert matplotlib colour palette to bokeh colour palette
		hex_plt = [(255 * np.array(i)).astype(int) for i in hex_colour]
		hex_bokeh = [RGB(*tuple(rgb)).to_hex() for rgb in hex_plt]


		output_file("resulting_images/som_seed_hex.html")

		# initialise figure/plot
		fig = figure(title="SOM: Hexagonal Topology",
					 plot_height=800, plot_width=800,
					 match_aspect=True,
					 tools="wheel_zoom,save,reset")

		# create data stream for plotting
		source_hex = ColumnDataSource(
			data = dict(
				x=hex_centre_col,
				y=hex_centre_row,
				c=hex_bokeh
			)
		)

		source_pages = ColumnDataSource(
			data=dict(
				wx=weight_x,
				wy=weight_y,
				species=label
			)
		)


		MARKERS = ['circle', 'diamond', 'cross', 'inverted_triangle', 'triangle', 'square', 'asterisk']


		# add shapes to plot
		fig.hex(x='y', y='x', source=source_hex,
				#size=100 * (.95 / np.sqrt(3)),
				size=50 * (.95 / np.sqrt(3)),
				alpha=.5,
				line_color='gray',
				fill_color='c')

		fig.scatter(x='wy', y='wx', source=source_pages,
					legend_field='species',
					#size=20,
					size=10,
					alpha=0.2,
					marker=factor_mark(field_name='species', markers=MARKERS, factors=SPECIES),
					color=factor_cmap(field_name='species', palette='Category10_7', factors=SPECIES))

		# add hover-over tooltip
		fig.add_tools(HoverTool(
			tooltips=[
				#("label", '@species'),
				#("(x,y)", '($x, $y)')],
				("label", '@species')],
			mode="mouse",
			point_policy="follow_mouse"
		))

		show(fig)
	
		"""
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

		ax.set_title(f"Embeddings: {self.__str__()}, Quantization error: {np.round(qterror, 2)}",
			loc='center', pad=55, fontsize=16)

		'''
		if out_dir:
			plt.savefig(os.path.join(out_dir, self.__str__() + '.png'))
		else:
			plt.savefig(self.__str__() + '.png')
		'''
		plt.show()
		"""

# def winning_neuron_per_doc(som, vect_train_corpus):
# 	''' Build a dict with the winning neuron coordinates for each document 

# 	NB. this is an experimental function, not used in the pipeline right now
# 	'''
# 	n_docs = np.shape(vect_train_corpus)[0]
# 	winner = {}
# 	winner = {n: som.winner(vect_train_corpus[n]) for n in range(n_docs)}
# 	return winner


def save_som(som, out_path):
	"""Save the SOM into a .som file

	Args:
		som ([type]): [description]
		out_path ([type]): [description]
	"""
	with open(out_path, 'wb') as outfile:
		joblib.dump(som, outfile)


def load_som(inp_path):
	"""Load SOM from a .som file


	Args:
		inp_path ([type]): [description]

	Returns:
		[type]: [description]
	"""
	with open(inp_path, 'rb') as infile:
		som = joblib.load(infile)
	return som


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
def main(model_path):
	logger = logging.getLogger(__name__)
	model_tag = os.path.splitext(os.path.basename(model_path))[0]

	logger.info('Reading data...')
    # Reading data into memory
	_, train_docs, test_docs = read_data(data_file)
	train_labels = train_docs['category']

	logger.info('Obtaining document embeddings...')
    # Obtain the vectorized corpus
	vect_train_corpus, _ = embedding_vectors(
        model_path, train_docs['prep_text'], test_docs['prep_text'])

	# Define SOM
	input_len = vect_train_corpus.shape[1]
	som = MiniSom2(random_seed=0, verbose=True, input_len=input_len, **som_kwargs)
	
	# Fit SOM
	logger.info('Fitting SOM model...')
	som.fit(vect_train_corpus, train_labels)

	# Plot and save the U-matrix
	logger.info('Plotting and saving the U-matrix...')
	#som.u_matrix(vect_train_corpus, train_labels, fig_dir)
	
	# BOKEH
	som.u_matrix_interactive(vect_train_corpus, train_labels, fig_dir)

	# Save the SOM fitted model
	print("TODO: check if SOM is already saved before overwriting it... also check before retraining!")
	#logger.info('Saving SOM model...')
	#outpath = os.path.join(model_dir, str(som) + '.som')
	#save_som(som, outpath)


if __name__ == '__main__':
	log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
	logging.basicConfig(level=logging.INFO, format=log_fmt)

	# Defining Paths
	data_file = os.path.join(PROJECT_ROOT, "data", "processed", "newsapi_docs.csv")
	model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
	fig_dir = os.path.join(PROJECT_ROOT, "models", "figures", "som")

	# SOM kwards
	som_kwargs = dict(
        x=30, y=30, n_iter=200, learning_rate=1, sigma=1,
		neighborhood_function='gaussian', topology='hexagonal' 
    )

	main()
