import numpy as np
import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events
from vis_components import umatrix_cluster_plot, umatrix_plot


def umatrix_page(bmus, df, color_vars):
    """Setup U-matrix visualization page."""

    color_var_labels = list(color_vars.keys())

    color_var = st.selectbox(
        'Variable to visualize',
        color_var_labels
    )
    color_var = color_vars[color_var]

    # U-matrix visualization
    if color_var == 'cluster_label':
        p = umatrix_plot(
            bmus.row,
            bmus.col,
            bmus.cluster_label,
            bmus.docs_count,
            bmus.dominant_cat,
            bmus.dominant_cat_perc
        )
    else:
        p = umatrix_cluster_plot(
            bmus.row,
            bmus.col,
            bmus[color_var],
            bmus.cluster_label,
            bmus.docs_count,
            bmus.dominant_cat,
            bmus.dominant_cat_perc
        )

    event_result = streamlit_bokeh_events(
        events="TestSelectEvent",
        bokeh_plot=p,
        key=color_var,
        debounce_time=1000,
        refresh_on_update=False
    )

    # Get mask of selected documents
    if event_result and "TestSelectEvent" in event_result:
        indices = event_result["TestSelectEvent"].get("indices", [])
        mask = df.docs_bmu.isin(indices).values
    else:
        mask = np.zeros(df.shape[0]).astype(bool)

    return mask
