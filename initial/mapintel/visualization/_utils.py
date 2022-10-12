import numpy as np
from bokeh.util.hex import cartesian_to_axial, axial_to_cartesian
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable


def get_hex_grid(x, y):
    """Converts the regular coordinates to fit an hex grid."""
    xx, yy = np.indices((x, y)).astype(float)

    w, h = np.sqrt(3), 2
    xx_offset = np.zeros_like(xx)
    xx_offset[:, 1::2] += 0.5 * w
    hex_col = np.rot90(
        np.flip((xx * w - xx_offset), axis=0)
    ).flatten()
    hex_row = np.rot90(
        np.flip((yy * h * 3 / 4), axis=0)
    ).flatten()

    # get each unit attributes
    hex_values = {}

    hex_values['hex_col'], hex_values['hex_row'] = cartesian_to_axial(
        hex_col,
        hex_row,
        size=1,
        orientation='pointytop'
    )

    return hex_values


def get_hex_overlay_grid(hex_col, hex_row):
    """
    Converts axial coordinates to allow overlaying a scatter plot into the
    umatrix (allows the selection of certain regions of the plot).
    """
    return axial_to_cartesian(
        hex_col,
        hex_row,
        size=1,
        orientation='pointytop'
    )


def val_to_color(col, cmap='RdYlBu_r'):
    """Converts a column of values to hex-type colors"""
    norm = Normalize(vmin=col.min(), vmax=col.max(), clip=True)
    mapper = ScalarMappable(norm=norm, cmap=cmap)
    rgba = mapper.to_rgba(col)

    return np.apply_along_axis(rgb2hex, 1, rgba)
