import pandas as pd
from typing import Optional, List
from importlib import resources
from PIL import Image
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_series(df, y_label=None, title=None, subtitle=None, colors=None, logo=None, source=None):
    # line plot in Systamental style
    # plot size
    fig, ax = plt.subplots(figsize=(15, 7))

    # line colors
    if colors == 'dark':
        colors = ['#00588D', '#A81829', '#005F73', '#005F52', '#714C00', '#4C5900', '#78405F', '#674E1F', '#3F5661']
    elif colors == 'light':
        colors = ['#5DA4DF', '#FF6B6C', '#25ADC2', '#4DAD9E', '#C89608', '#9DA521', '#C98CAC', '#B99966', '#89A2AE']
    else:
        colors = ['#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F', '#D1B07C', '#758D99']

    # plot
    df.dropna().plot(color=colors, linewidth=2, rot=0, ax=ax)

    # font
    plt.rcParams['font.family'] = 'georgia'

    # legend
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # grid
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.6, zorder=1)
    ax.set_facecolor("whitesmoke")

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format x-axis
    ax.set_xlim(df.index.get_level_values('date')[0], df.index.get_level_values('date')[-1])

    # Reformat y-axis tick labels
    ax.set_ylabel(y_label)
    ax.yaxis.tick_right()

    # add systamental logo
    if logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Add in title and subtitle
    if subtitle is None:
        y = 0.89
    else:
        y = 0.92
    ax.text(x=0.13, y=y, s=f"{title}", transform=fig.transFigure, ha='left', fontsize=14,
            weight='bold', alpha=.8, fontdict=None)
    ax.text(x=0.13, y=.89, s=subtitle, transform=fig.transFigure, ha='left', fontsize=12, alpha=.8,
            fontdict=None)

    # Set source text
    if source is not None:
        ax.text(x=0.13, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=None)


def plot_bar(df, x_label=None, y_label=None, title=None, subtitle=None, color: int = 0, logo=None, source=None):
    # bar plot in Systamental style
    # plot size
    fig, ax = plt.subplots(figsize=(15, 7))

    # line colors
    colors = ['#98DAFF', '#FFA39F', '#6FE4FB', '#86E5D4', '#FFCB4D', '#D7DB5A', '#FFC2E3', '#F2CF9A', '#BFD8E5']

    # plot
    df.plot(kind='bar', color=colors[color], legend=False, rot=0, ax=ax)

    # font
    plt.rcParams['font.family'] = 'georgia'

    # legend
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    # grid
    ax.set_axisbelow(True)
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.6)
    ax.set_facecolor("whitesmoke")

    # y-axis
    ax.set_xlabel(x_label)

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # Reformat y-axis tick labels
    ax.set_ylabel(y_label)
    ax.yaxis.tick_right()

    # add systamental logo
    if logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Add in title and subtitle
    if subtitle is None:
        y = 0.89
    else:
        y = 0.92
    ax.text(x=0.13, y=y, s=f"{title}", transform=fig.transFigure, ha='left',
            fontsize=14, weight='bold', alpha=.8, fontdict=None)
    ax.text(x=0.13, y=.89, s=f"{subtitle}", transform=fig.transFigure, ha='left',
            fontsize=12, alpha=.8, fontdict=None)

    # Set source text
    if source is not None:
        ax.text(x=0.13, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=None)


def plot_heatmap():

    pass


def plot_table(df,
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
