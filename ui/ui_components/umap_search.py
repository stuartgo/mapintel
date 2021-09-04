from pandas import DataFrame
from ui.vis_components.umap import umap_plot


def umap_page(documents: DataFrame, query: dict, unique_topics: list):
    if query:
        query_label = query['query_text']
        if len(query_label) > 40:
            query_label = query_label[:40] + "..."
        documents = documents.append(
            {
                "answer": query['query_text'],
                "umap_embeddings_x": query['query_umap'][0],
                "umap_embeddings_y": query['query_umap'][1],
                "topic": query_label
            },
            ignore_index=True
        )
    else:
        query_label = "Query"
    
    p = umap_plot(
        index=documents.index,
        x=documents['umap_embeddings_x'],
        y=documents['umap_embeddings_y'],
        text=documents['answer'],
        topics=documents['topic'],
        unique_topics=[query_label] + unique_topics
    )
    
    return p