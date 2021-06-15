import numpy as np
# from datetime import datetime
import streamlit as st
from streamlit_bokeh_events import streamlit_bokeh_events
from vis_components import umap_plot


def umap_page(df, projections):
    # This line is incorrect if document search is used
    p = umap_plot(
        index=df.index,
        x=projections[:, 0],
        y=projections[:, 1],
        text=df.text,
        categories=df.category,
    )

    event_result = streamlit_bokeh_events(
        events="TestSelectEvent",
        bokeh_plot=p,
        key=f'{df.size}_{df.category.index[0]}',
        debounce_time=1000,
        refresh_on_update=False
    )

    # Get mask of selected documents
    if event_result and "TestSelectEvent" in event_result:
        indices = event_result["TestSelectEvent"].get("indices", [])
        mask = df.docs_bmu.isin(indices).values
    else:
        mask = np.zeros(df.shape[0]).astype(bool)

    return mask, event_result


# TODO: apply boolean filters to the query (prior to semantic search)
# def apply_query(
#     df, projections, distances, umatrix_mask, keyword, start_date, end_date
# ):
#     """Apply query to documents' dataframe and xy projections."""

#     # Apply umatrix and document similarity queries
#     df.drop(columns='docs_bmu', inplace=True)
#     df[['x_proj', 'y_proj']] = projections
#     if distances is not None:
#         df['distance'] = distances

#         df_query = (
#             df.iloc[umatrix_mask].sort_values('distance')
#             if umatrix_mask.any()
#             else df.sort_values('distance')
#         )
#     elif umatrix_mask.any():
#         df_query = df.iloc[umatrix_mask]
#     else:
#         df_query = df.copy()

#     # Apply sidebar query
#     if keyword:
#         keyword_mask = df_query.text.str.lower()\
#             .apply(lambda x: keyword in x).values
#     else:
#         keyword_mask = np.ones(df_query.shape[0]).astype(bool)

#     if (
#         start_date != datetime(2021, 3, 1).date() or
#         end_date != datetime.now().date()
#     ):
#         date_mask = df_query.timestamp.apply(
#             lambda x: x >= start_date and x <= end_date
#         ).values
#     else:
#         date_mask = np.ones(df_query.shape[0]).astype(bool)

#     sidebar_mask = keyword_mask & date_mask

#     df_query = df_query.iloc[sidebar_mask]
#     projections = df_query[['x_proj', 'y_proj']].values
#     df_query.drop(columns=['x_proj', 'y_proj'], inplace=True)

#     applied_query = not sidebar_mask.all() or umatrix_mask.any()

#     return df_query, projections, applied_query
