from bokeh.plotting import figure
from bokeh.models import CDSView, ColumnDataSource, GroupFilter
from ui.vis_components.utils import cat_to_color


def umap_plot(index, x, y, text, topics, unique_topics, size=4):
    """Plots a scatter plot with umap projections."""
    
    # Creating the ColumnDataSource
    topic_label = unique_topics[0]
    text = list(map(lambda x: x.split("#SEPTAG#"), text))
    source = ColumnDataSource(dict(
        index=index,
        x=x,
        y=y,
        categories=topics,
        title=list(map(lambda x: x[0], text)),
        body=list(map(lambda x: "\n".join(x[1:]) if len(x)>1 else "", text)),
        marker=["circle" if categ != topic_label else "x" for categ in topics],
        size=[size if categ != topic_label else size*5 for categ in topics],
        alpha=[0.5 if categ != topic_label else 1 for categ in topics],
        color=cat_to_color(topics)
    ))

    # Defining the Tooltip format
    TOOLTIPS = """
        <div>
            <div>
                <span style="font-size: 15px; font-weight: bold;">@title{safe}</span>
            </div>
            <div>
                <span style="color: @color;">@categories</span>
            </div>
            <div>
                <span>@body{safe}</span>
            </div>
        </div>
    """

    # Initialize the figure
    p = figure(
        tooltips=TOOLTIPS,
        tools='pan,wheel_zoom,lasso_select,box_zoom,reset',
        active_drag="pan",
        active_scroll="wheel_zoom",
        plot_width=1000,
        plot_height=700
    )

    # Plot the scatter plots for each topic label
    for topic in unique_topics:
        view = CDSView(
            source=source, 
            filters=[GroupFilter(column_name="categories", group=topic)]
        )
        p.scatter(
            x='x',
            y='y',
            size='size',
            color='color',
            marker='marker',
            alpha='alpha',
            legend_label=topic,
            source=source,
            view=view
        )
    
    # Set figure attributes
    ## General plot configs
    p.toolbar.logo = None
    p.toolbar_location = 'below'
    p.toolbar.autohide = True
    p.axis.visible = True
    p.outline_line_color = '#ffffff'

    ## Legend position and interactivity configs
    p.add_layout(p.legend[0], 'right')
    p.legend.click_policy = "hide"
    p.legend.inactive_fill_alpha = 0.5
    p.legend.inactive_fill_color = "darkgray"

    # Legend title configs
    p.legend.title = "Topics (select legend entries to hide points)"
    p.legend.title_text_font_style = "normal"
    p.legend.title_text_alpha = 1
    p.legend.title_text_color = "whitesmoke"

    ## Legend label configs
    p.legend.label_text_alpha = 1
    p.legend.label_text_font_size = "9pt"

    return p
