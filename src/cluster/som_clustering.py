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
from collections import defaultdict
import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.spatial import distance_matrix
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sompy.visualization.plot_tools import plot_hex_map
from sompy.visualization.umatrix import UMatrixView
from skimage.color import rgb2gray
from skimage.feature import blob_log
import sompy
from src import PROJECT_ROOT
from src.visualization.embedding_space import embedding_vectors, read_data


class UMatrix(UMatrixView):
    def __init__(self, som, distance2=1, row_normalized=False, show_data=True,
                 contoor=True, blob=False, labels=False, **MatplotView_kwargs):
        super().__init__(**MatplotView_kwargs)
        self.som = som
        self.distance2 = distance2
        self.row_normalized = row_normalized
        self.show_data = show_data
        self.contoor = contoor
        self.blob = blob
        self.labels = labels

    def _build_u_matrix(self):
        return super().build_u_matrix(self.som, distance=self.distance2, row_normalized=self.row_normalized)

    def show(self):
        # Builds the U-Matrix - each cell is the average distance of a
        # given unit to its neighbors, where neighbors are units with
        # distance less than self.distance2
        umat = self._build_u_matrix()
        # The mapsize of the codebook (tuple)
        msz = self.som.codebook.mapsize
        # Gets the BMU id for each data sample
        proj = self.som.project_data(self.som.data_raw)
        # Gets the x and y coordinates of each sample's BMU
        coord = self.som.bmu_ind_to_xy(proj)[:, :2]

        # Prepare figure object (self._fig)
        self.prepare()
        # set fontsize of the figure title
        plt.rc('figure', titlesize=self.text_size)
        colormap = plt.get_cmap('RdYlBu_r')

        if self.som.codebook.lattice == "rect":
            ax = self._fig.add_subplot(111)
            ax.imshow(umat, cmap=colormap, alpha=1)

        elif self.som.codebook.lattice == "hexa":
            # Why hex_shrink=0.5 works? Otherwise, the polygons overlap...
            ax, cents = plot_hex_map(
                np.flip(umat, axis=1), fig=self._fig, colormap=colormap, colorbar=False, hex_shrink=0.5)

        else:
            raise ValueError(
                'lattice argument of SOM object should take either "rect" or "hexa".')

        # Add distance colorbar to figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(cm.ScalarMappable(cmap=colormap), cax=cax,
                            orientation='vertical')

        # TODO: The options below are not working for the hexagonal grid
        if self.contoor:
            mn = np.min(umat.flatten())
            md = np.median(umat.flatten())
            ax.contour(umat, np.linspace(mn, md, 15), linewidths=0.7,
                       cmap=plt.cm.get_cmap('Blues'))

        if self.show_data:
            ax.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray',
                       marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
            ax.axis('off')

        if self.labels:
            if self.labels is True:
                labels = self.som.build_data_labels()
            for label, x, y in zip(self.labels, coord[:, 1], coord[:, 0]):
                ax.annotate(str(label), xy=(x, y),
                            horizontalalignment='center',
                            verticalalignment='center')

        # Adjust image size
        ratio = float(msz[0])/(msz[0]+msz[1])
        self._fig.set_size_inches((1-ratio)*15, ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.00, wspace=.000)

        sel_points = list()
        if self.blob:
            image = 1 / umat
            rgb2gray(image)

            # 'Laplacian of Gaussian'
            blobs = blob_log(image, max_sigma=5, num_sigma=4, threshold=.152)
            blobs[:, 2] = blobs[:, 2] * np.sqrt(2)
            plt.imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)
            sel_points = list()

            for blob in blobs:
                row, col, r = blob
                c = plt.Circle((col, row), r, color='red', linewidth=2,
                               fill=False)
                ax.add_patch(c)
                dist = distance_matrix(
                    coord, np.array([row, col])[np.newaxis, :])
                sel_point = dist <= r
                plt.plot(coord[:, 1][sel_point[:, 0]],
                         coord[:, 0][sel_point[:, 0]], '.r')
                sel_points.append(sel_point[:, 0])

        plt.show()
        return sel_points, umat


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

    # This som implementation does not have a random seed parameter
    # We're going to set it up ourselves
    np.random.seed(42)

    som = sompy.SOMFactory().build(
        vect_train_corpus,
        mapsize=(50, 50),
        initialization='random',
        neighborhood='gaussian',
        training='batch',
        lattice='hexa'
    )
    som.train(n_job=-1, verbose='info',
              train_rough_len=10, train_finetune_len=10)

    # U-matrix of the 50x50 grid
    _, umat = UMatrix(
        width=12,
        height=12,
        text_size=6,
        title="U-Matrix",
        som=som,
        distance2=1,
        row_normalized=False,
        show_data=False,
        contoor=True
    ).show()
    # u = sompy.umatrix.UMatrixView(
    #     12, 12, 'umatrix', show_axis=True, text_size=8, show_text=True)

    # # Plotting training error history - TODO
    # plt.plot(np.arange(som._n_iter), history['quantization_error'], label='quantization error')
    # # plt.plot(np.arange(som._n_iter), history['topographic_error'], label='topographic error')
    # plt.ylabel('error')
    # plt.xlabel('iteration index')
    # plt.legend()
    # plt.show()

    # # Plot and save the U-matrix - TODO
    # logger.info('Plotting and saving the U-matrix...')
    # som.u_matrix(vect_train_corpus, train_labels, fig_dir)

    # # Save the SOM fitted model
    # logger.info('Saving SOM model...')
    # outpath = os.path.join(model_dir, str(som) + '.som')
    # save_som(som, outpath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Defining Paths
    data_file = os.path.join(PROJECT_ROOT, "data",
                             "processed", "newsapi_docs.csv")
    model_dir = os.path.join(PROJECT_ROOT, "models", "saved_models")
    fig_dir = os.path.join(PROJECT_ROOT, "models", "figures", "som")

    # # SOM kwards - TODO
    # som_kwargs = dict(
    #     x=50, y=50, n_iter=100000, learning_rate=10, sigma=10,
    # 	neighborhood_function='gaussian', topology='hexagonal'
    # )

    main()
