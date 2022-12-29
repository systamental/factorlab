import pandas as pd
from typing import Optional, List
import plotly.graph_objects as go


def publish_table(df,
                  reset_index: Optional[bool] = False,
                  decimals: Optional[int] = None,
                  title: str = '',
                  width: Optional[int] = None,
                  height: Optional[int] = None,
                  col_width: List[int] = [200, 100],
                  ) -> None:

    """
    Computes key performance metrics for asset or strategy returns and creates publication quality table with
    Plotly.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe/table for which to convert to Plotly format.
    reset_index: bool, optional, default False
        Resets the dataframe/table index to avoid being cut off in Plotly.
    decimals: int, optional, defaul None
    title: str
        Table title.
    width: int, optional, default None
        Table width. Computed from dataframe shape otherwise.
    height: int, optional, default None
        Table height. Computed from dataframe shape otherwise.
    col_width: list, optional, default [200, 100]
        First column and table column widths.
    """
    # type conversion
    table = df.copy().astype(float)

    # decimals
    if decimals is not None:
        table = table.round(decimals=decimals)

    # reset index
    if reset_index:
        table = table.reset_index()
        cols = table.columns.to_list()
        cols[0] = ''
        table.columns = cols

    # create table, headers
    fig = go.Figure(data=[go.Table(
        columnwidth=col_width,

        # format headers
        header=dict(
            values=list('<b>' + table.columns + '</b>'),
            font=dict(color='black', family='Georgia', size=12),
            align=['left', 'center'],
            fill=dict(color='whitesmoke')
        ),

        # rows
        cells=dict(
            values=[table[col] for col in table.columns],
            font=dict(color=['black'], family='Georgia'),
            align=['left', 'center'],
            height=25,
            fill=dict(color=['white'])
        ))])

    # set table format
    if width is None:
        width = len(table.columns) * 100
    if height is None:
        height = len(table) * 30 + 30

    fig.update_layout(title='<b>' + title + '</b>',
                      title_font_color='black',
                      title_font_family='Georgia',
                      title_font_size=14,
                      title_x=0,
                      margin=dict(l=0, r=0, t=30, b=0), width=width, height=height)
    fig.show()
