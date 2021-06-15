from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    CustomJS
)
from .utils import cat_to_color


def umap_plot(index, x, y, text, categories, size=3):
    """Plots a scatter plot with umap projections."""

    source = ColumnDataSource(dict(
        index=index,
        x=x,
        y=y,
        texts=text,
        categories=categories,
        color=cat_to_color(categories)
    ))

    TOOLTIPS = """
        <div>
            <div>
                <span style="font-size: 17px; font-weight: bold;">
                    @categories
                </span>
                <span style="font-size: 15px; color: #966;">
                    @index
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
    )

    p.scatter(
        x='x',
        y='y',
        size=size,
        color='color',
        alpha=0.5,
        # legend_field performs better but messes up the ordering
        legend_group='categories',
        source=source
    )
    
    p.toolbar.logo = None
    p.toolbar_location = 'below'
    p.toolbar.autohide = True
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.axis.visible = False
    p.outline_line_color = '#ffffff'
    p.legend.title= "Topics"
    p.legend.title_text_font_style = "normal"
    p.legend.title_text_alpha = 1
    p.legend.label_text_alpha = 1
    
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
        """,
        ),
    )

    return p
