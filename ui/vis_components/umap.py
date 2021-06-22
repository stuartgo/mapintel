from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    CustomJS
)
from .utils import cat_to_color


def umap_plot(index, x, y, text, categories, size=3):
    """Plots a scatter plot with umap projections."""
    query_ix = categories.index[categories == "Query"]

    source = ColumnDataSource(dict(
        index=index,
        x=x,
        y=y,
        texts=text,
        categories=categories,
        color=[i if j != query_ix else "red" for j, i in enumerate(cat_to_color(categories))],
        marker=["circle" if i != "Query" else "x" for i in categories],
        size=[size if i != "Query" else size*3 for i in categories],
        alpha=[0.5 if i != "Query" else 1 for i in categories]
    ))

    TOOLTIPS = """
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">
                    @categories
                </span>
            </div>
            <div>
                <span>@texts{safe}</span>
            </div>
        </div>
    """

    p = figure(
        tooltips=TOOLTIPS,
        tools='pan,wheel_zoom,lasso_select,box_zoom,reset',
        active_drag="pan",
        active_scroll="wheel_zoom",
        plot_width=698
    )

    p.scatter(
        x='x',
        y='y',
        size='size',
        color='color',
        marker='marker',
        alpha='alpha',
        # legend_field performs better but messes up the ordering
        legend_group='categories',
        source=source
    )

    p.scatter()
    
    p.toolbar.logo = None
    p.toolbar_location = 'below'
    p.toolbar.autohide = True
    p.axis.visible = False
    p.outline_line_color = '#ffffff'
    # p.legend.title= "Topics"
    # p.legend.title_text_font_style = "normal"
    # p.legend.title_text_alpha = 1
    p.legend.label_text_alpha = 1
    # p.add_layout(p.legend[0], 'right')

    # define events
    source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=source),
            code="""
                document.dispatchEvent(
                    new CustomEvent(
                        "TestSelectEvent",
                        {detail: {indices: cb_obj.indices}}
                    )
                )
            """
        )
    )

    return p
