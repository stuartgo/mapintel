import numpy as np
from pandas import DataFrame
from streamlit_bokeh_events import streamlit_bokeh_events
from vis_components import umap_plot



def umap_page(documents: DataFrame, query: dict):
    if query:
        documents = documents.append(
            {
                "answer": query['query_text'],
                "umap_embeddings_x": query['query_umap'][0],
                "umap_embeddings_y": query['query_umap'][1],
                "topic": "Query"
            },
            ignore_index=True
        )

    p = umap_plot(
        index=documents.index,
        x=documents['umap_embeddings_x'],
        y=documents['umap_embeddings_y'],
        text=documents['answer'],
        categories=documents['topic'].str.capitalize()
    )
    
    query_text = documents.loc[documents['topic'] == "Query", "answer"]
    event_result = streamlit_bokeh_events(
        events="TestSelectEvent",
        bokeh_plot=p,
        key=f'umap_plot_{documents.shape[0]}_{query_text}',
        debounce_time=1000,
        refresh_on_update=False
    )

    # Get mask of selected documents
    if event_result and "TestSelectEvent" in event_result:
        indices = event_result["TestSelectEvent"].get("indices", [])
        mask = documents.loc[indices, 'document_id'].tolist()
    else:
        mask = None

    return mask
