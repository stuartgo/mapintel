# NOTE: Bokeh might accidentally be an exceptional tool to complement
# some streamlit shortcomings...
# https://discuss.streamlit.io/t/bokeh-can-provide-layouts-tabs-advanced-tables-and-js-callbacks-in-streamlit/1108

import pickle
from os.path import join, abspath, pardir, dirname
import streamlit as st
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, LassoSelectTool
from bokeh.plotting import figure
from bokeh.util.hex import cartesian_to_axial, axial_to_cartesian
from bokeh.transform import linear_cmap

from matplotlib.pyplot import get_cmap
from matplotlib.colors import rgb2hex, Normalize
from matplotlib.cm import ScalarMappable

from streamlit_bokeh_events import streamlit_bokeh_events
from datetime import datetime
import numpy as np
import pandas as pd

DATAPATH = join(dirname(abspath(__file__)), pardir, 'data', 'news.csv')
FILEPATH = join(dirname(abspath(__file__)), pardir, 'outputs', 'ml_engine.pkl')

df = pd.read_csv(DATAPATH)
ml_outputs = pickle.load(open(FILEPATH, 'rb'))

umat = ml_outputs['umatrix']
bmu_id_mapper = ml_outputs['bmu_indices_map']
df['docs_bmu'] = ml_outputs['docs_bmu']


# get necessary data

categories = pd.get_dummies(df.category)
categories['docs_bmu'] = df['docs_bmu']
categories = categories.groupby('docs_bmu').mean()


indices = np.indices(bmu_id_mapper.shape)
row = indices[0].flatten()
col = indices[1].flatten()
umat_val = umat.flatten()
bmu_id = bmu_id_mapper.flatten()
cat = categories.idxmax(1)
cat_perc = categories.max(1)

bmus = pd.DataFrame(index=bmu_id)
bmus['row'] = row
bmus['col'] = col
bmus['umat_val'] = umat_val
bmus['cat'] = cat
bmus['cat_perc'] = cat_perc

def get_hex_grid(x,y):
    """Converts the regular coordinates to fit an hex grid."""
    xx, yy = np.indices((x,y)).astype(float)

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


hex_coords = get_hex_grid(*umat.shape)
bmus['hex_col'], bmus['hex_row'] = hex_coords.values()
bmus['overlay_col'], bmus['overlay_row'] = get_hex_overlay_grid(
    bmus.hex_col,
    bmus.hex_row
)
bmus['umat_colors'] = val_to_color(bmus['umat_val'])

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('docs_bmu')[['timestamp', 'text', 'category']]

######################################################################
# The app actually starts here
######################################################################

# Configure sidebar
st.sidebar.title('Query Selected Documents')
keyword = st.sidebar.text_input('Keyword').lower()
start_date = st.sidebar.date_input('Start date', value=datetime(2020,10,5))
end_date = st.sidebar.date_input('End date')

# Just a simple header for future reference
st.subheader("Select Points From Map")

# create data objects
bmu_src = ColumnDataSource(bmus)

# U-matrix visualization
p = figure(tools="box_select,lasso_select,tap,reset", toolbar_location='below')
p.toolbar.logo = None
p.hex_tile(
    "hex_col",
    "hex_row",
    size=1,
    fill_color='umat_colors',
    line_color='white',
    source=bmu_src
)
p.scatter(
    "overlay_col",
    "overlay_row",
    fill_color='umat_colors',
    line_color='umat_colors',
    source=bmu_src
)
p.add_tools(HoverTool(
    tooltips=[
        ('','@cat: @cat_perc')
    ],
    mode="mouse",
    point_policy="follow_mouse",
    )
)

p.select_one(LassoSelectTool).overlay.fill_alpha = 0
p.select_one(LassoSelectTool).overlay.line_alpha = 1
p.xgrid.visible = False
p.ygrid.visible = False
p.axis.visible = False
p.outline_line_color = 'white'

# define events
bmu_src.selected.js_on_change(
    "indices",
    CustomJS(
        args=dict(source=bmu_src),
        code="""
        document.dispatchEvent(
            new CustomEvent("TestSelectEvent", {detail: {indices: cb_obj.indices}})
        )
    """,
    ),
)

event_result = streamlit_bokeh_events(
    events="TestSelectEvent",
    bokeh_plot=p,
    key="foo",
    debounce_time=100,
    refresh_on_update=False
)

st.subheader("Selected Documents")

# Any other selection method can be added here
if event_result and "TestSelectEvent" in event_result:
    indices = event_result["TestSelectEvent"].get("indices", [])
    df_query = df.loc[indices]
    df_query = df_query[
        df_query.timestamp.apply(lambda x: x >= start_date and x <= end_date)
        &
        df_query.text.str.lower().apply(lambda x: keyword in x)
    ]
    st.table(df_query)
elif keyword:
    df_query = df[
        df_query.timestamp >= str(start_date)
        &
        df_query.timestamp <= str(end_date)
        &
        df.text.str.lower().apply(lambda x: keyword in x)
    ]
    st.table(df_query)
else:
    st.table()
