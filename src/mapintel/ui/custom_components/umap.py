from textwrap import wrap

import plotly.graph_objects as go
from pandas import DataFrame

from ui.custom_components.utils import cat_to_color


def umap_page(documents: DataFrame, query: dict, unique_topics: list):
    # Set query row
    if query:
        query_label = query["query_text"]
        if len(query_label) > 40:
            query_label = query_label[:40] + "..."
        documents = documents.append(
            {
                "answer": query["query_text"],
                "umap_embeddings_x": query["query_umap"][0],
                "umap_embeddings_y": query["query_umap"][1],
                "topic_label": query_label,
            },
            ignore_index=True,
        )
    else:
        query_label = "Query"
    # Produce umap plot and get configurations
    p, config = umap_plot(
        documents=documents,
        unique_topics=unique_topics,
        query_label=query_label,
    )

    return p, config


def umap_plot(documents, unique_topics, query_label):
    # Initialize the figure
    fig = go.Figure(
        layout=dict(
            height=850,
            width=1500,
            legend=dict(
                title_text="Topics (select legend entries to hide points)\n",
                title_font_size=13,
                itemsizing="constant",
            ),
        )
    )

    # Add traces for each topic
    for c, topic in zip(cat_to_color(unique_topics), unique_topics):
        # Get data for the selected topic
        topic_mask = documents["topic_label"] == topic
        data = documents.loc[topic_mask]
        hover_data = data[["title", "snippet"]].applymap(
            lambda x: "<br>".join(wrap(x, width=100)) if x else ""
        )

        # Add topics traces
        fig.add_trace(
            go.Scattergl(
                mode="markers",
                x=data["umap_embeddings_x"],
                y=data["umap_embeddings_y"],
                text=data["topic_label"],
                customdata=hover_data,
                opacity=0.5,  # trace opacity
                marker=dict(
                    size=4, color=c, opacity=0.5, symbol="circle"  # marker opacity
                ),
                name=topic,
                hovertemplate="<b>%{customdata[0]}</b><br><br>"
                + "<b>Topic</b>: %{text}<br>"
                + "<b>Content</b>: %{customdata[1]}"
                + "<extra></extra>",
            )
        )

    # Add query trace (last so the marker is in front)
    query_mask = documents["topic_label"] == query_label
    data = documents.loc[query_mask]
    fig.add_trace(
        go.Scattergl(
            mode="markers",
            x=data["umap_embeddings_x"],
            y=data["umap_embeddings_y"],
            text=data["topic_label"],
            opacity=0.5,  # trace opacity
            marker=dict(
                size=20,
                color=cat_to_color([query_label]),
                opacity=1,  # marker opacity
                symbol="x-dot",
                line=dict(color="black", width=1),
            ),
            name=query_label,
            hovertemplate="<b>Query Marker</b><br><br>"
            + "<b>Query</b>: %{text}<br>"
            + "<extra></extra>",
        )
    )

    # Define plot configurations options
    config = {
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "select2d",
            "lasso2d",
            "autoScale2d",
            "toggleSpikelines",
            "hoverCompareCartesian",
        ],
    }

    return fig, config