import pandas as pd
from bokeh.plotting import figure
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    HoverTool,
    LassoSelectTool
)
from .utils import (
    get_hex_grid,
    get_hex_overlay_grid,
    val_to_color
)


def umatrix_plot(x, y, values, counts, categories, cat_perc, cmap='RdYlBu_r'):
    """Create U-matrix Bokeh figure."""

    # Get hex coordinates
    source = {
        'counts': counts,
        'category': categories,
        'cat_perc': cat_perc
    }
    source['hex_col'], source['hex_row'] = get_hex_grid(
        x.max()+1, y.max()+1
    )

    source['overlay_col'], source['overlay_row'] = get_hex_overlay_grid(
        source['hex_col'],
        source['hex_row']
    )
    source['fill_color'] = val_to_color(values, cmap=cmap)

    source = ColumnDataSource(pd.DataFrame(source))

    # Make umatrix
    p = figure(
        tools="box_select,lasso_select,tap,reset", toolbar_location='below'
    )
    p.toolbar.logo = None
    p.toolbar.autohide = True
    p.hex_tile(
        'hex_col',
        'hex_row',
        size=1,
        fill_color='fill_color',
        line_color='white',
        source=source
    )

    # hex objects cannot be selected, since the behavior for selecting objects
    # is not yet defined. Overlaying a scatter plot to allow data selection.
    p.scatter(
        'overlay_col',
        'overlay_row',
        fill_color='fill_color',
        line_color='fill_color',
        source=source
    )

    # Format figure
    p.add_tools(HoverTool(
        tooltips=[
            ('', '@category: @cat_perc'),
            ('', '@counts docs')
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


def umatrix_cluster_plot(
    x, y, values, clusters, counts, categories, cat_perc, cmap='RdYlBu_r'
):
    """Create U-matrix Bokeh figure."""

    # Get hex coordinates
    source = {
        'counts': counts,
        'category': categories,
        'cat_perc': cat_perc,
        'clusters': clusters
    }
    source['hex_col'], source['hex_row'] = get_hex_grid(
        x.max()+1, y.max()+1
    )

    source['overlay_col'], source['overlay_row'] = get_hex_overlay_grid(
        source['hex_col'],
        source['hex_row']
    )
    source['fill_color'] = val_to_color(values, cmap='Greys')
    source['cluster_color'] = val_to_color(clusters, cmap=cmap)
    source['alpha'] = ((values - values.min()) / (values - values.min()).max())

    source = ColumnDataSource(pd.DataFrame(source))

    # Make umatrix
    p = figure(
        tools="box_select,lasso_select,tap,reset", toolbar_location='below'
    )
    p.toolbar.logo = None
    p.toolbar.autohide = True
    p.hex_tile(
        'hex_col',
        'hex_row',
        size=1,
        alpha=.20,
        fill_color='cluster_color',
        line_color='cluster_color',
        source=source
    )

    # overlay class labels
    p.hex_tile(
        'hex_col',
        'hex_row',
        size=1,
        alpha='alpha',
        fill_color='grey',
        line_color='cluster_color',
        source=source
    )

    # hex objects cannot be selected, since the behavior for selecting objects
    # is not yet defined. Overlaying a scatter plot to allow data selection.
    p.scatter(
        'overlay_col',
        'overlay_row',
        alpha=0,
        size=0,
        fill_color='cluster_color',
        line_color='cluster_color',
        source=source
    )

    # Format figure
    p.add_tools(HoverTool(
        tooltips=[
            ('', '@category: @cat_perc'),
            ('', '@counts docs')
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
