# Utilities
import pickle
from os.path import join, abspath, pardir, dirname
from datetime import datetime

# Data handling
import numpy as np

# Visualization
import streamlit as st
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, HoverTool, LassoSelectTool, Select
from streamlit_bokeh_events import streamlit_bokeh_events

# Mapintel functions
from mapintel.visualization import (
    get_hex_grid,
    val_to_color,
    get_hex_overlay_grid
)

DATA_PATH = join(
    dirname(abspath(__file__)), pardir, 'outputs', 'ml_engine.pkl'
)

# Read data
ml_engine = pickle.load(open(DATA_PATH, 'rb'))
df = ml_engine['df']
bmus = ml_engine['bmus']

hex_coords = get_hex_grid(bmus.row.max()+1, bmus.col.max()+1)

bmus['hex_col'], bmus['hex_row'] = hex_coords.values()
bmus['overlay_col'], bmus['overlay_row'] = get_hex_overlay_grid(
    bmus.hex_col,
    bmus.hex_row
)

color_vars = [
    c
    for c in bmus.columns[(bmus.dtypes == float) | (bmus.dtypes == int)]
    if type(c) == str
]

######################################################################
# The app actually starts here
######################################################################

# Configure sidebar
st.sidebar.title('Query Selected Documents')
keyword = st.sidebar.text_input('Keyword').lower()
start_date = st.sidebar.date_input('Start date', value=datetime(2020, 10, 5))
end_date = st.sidebar.date_input('End date')
umat_color_var = st.sidebar.selectbox(
    'Variable to visualize',
    color_vars,
    color_vars.index('umat_val')
)

# Just a simple header for future reference
st.subheader("Select Points From Map")

# Configure U-matrix colors
bmus['umat_colors'] = val_to_color(bmus[umat_color_var])

# create data objects
bmu_src = ColumnDataSource(bmus.drop(columns=[
    c for c in bmus.columns if type(c) != str
]))

# U-matrix visualization
p = figure(tools="box_select,lasso_select,tap,reset", toolbar_location='below')
p.toolbar.logo = None
hex_tile = p.hex_tile(
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
        ('', '@dominant_cat: @dominant_cat_perc'),
        ('', '@docs_count docs')
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

# add drop down to select color values
select = Select(
    title='Variable to visualize',
    value='umat_val',
    options=color_vars
)
select.js_link('value', hex_tile.glyph, 'fill_color')

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
    bokeh_plot=column(p, select),
    key="foo",
    debounce_time=1000,
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
        df.timestamp.apply(lambda x: x >= start_date and x <= end_date)
        &
        df.text.str.lower().apply(lambda x: keyword in x)
    ]
    st.table(df_query)
else:
    st.table()
