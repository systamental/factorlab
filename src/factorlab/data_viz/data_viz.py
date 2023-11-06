import pandas as pd
from math import ceil
from typing import Optional, Union
from importlib import resources
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


def add_fonts():

    # Get the path to the directory where this Python script is located
    current_dir = Path(__file__).resolve().parent
    # Construct the relative path to the 'fonts' directory
    fonts_dir = current_dir / 'conf' / 'fonts'
    # Use Path.glob() to find all .ttf font files within the 'fonts' directory
    font_files = list(fonts_dir.glob('*.ttf'))
    # add to matplotlib fonts
    for font_file in font_files:
        fm.fontManager.addfont(font_file)


def plot_series(data: Union[pd.Series, pd.DataFrame],
                fig_size: tuple = (15, 7),
                color_lightness: Optional[str] = None,
                font: str = 'Lato',
                y_label: Optional[str] = None,
                title: Optional[str] = None,
                subtitle: Optional[str] = None,
                title_font: Optional[str] = None,
                add_line: bool = False,
                line_color: Optional[int] = None,
                add_logo: bool = False,
                source: Optional[str] = None,
                source_font: Optional[str] = None
                ):
    """
    Creates a time series plot from a data object (dataframe or series) with the x-axis labeled date.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
        Data object from which to create time series plot.
    fig_size: tuple
        Tuple (width, height) of figure object.
    color_lightness: str, {'dark', 'light', 'medium'}, optional
        Color lightness for the time series plot.
    font: str, optional, default None
        Font used for all text in the plot.
    y_label: str, optional, default None
        Text describing the units of measurement for the y-axis.
    title: str, optional, default None
        Title to use for the plot.
    subtitle: str, optional, default None
        Subtitle to use for the plot.
    title_font: str, optional, default None
        Font used for the text in the title and subtitle.
    add_line: bool, default False
        Adds a horizontal line running from the top left to top right of the plot.
    line_color: int, optional, default None
        Color of added line.
    add_logo: bool, default False
        Adds the Systamental logo to the bottom left corner of the plot.
    source: str, optional, default None
        Adds text for the source of the data in the plot.
    source_font: str, optional, default None
        Font used for the source text.

    """
    # plot size
    fig, ax = plt.subplots(figsize=fig_size)

    # colors
    if color_lightness == 'dark':
        colors = ['#00588D', '#A81829', '#005F73', '#005F52', '#714C00', '#4C5900', '#78405F', '#674E1F', '#3F5661']
    elif color_lightness == 'light':
        colors = ['#5DA4DF', '#FF6B6C', '#25ADC2', '#4DAD9E', '#C89608', '#9DA521', '#C98CAC', '#B99966', '#89A2AE']
    else:
        colors = ['#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F', '#D1B07C', '#758D99']

    # plot
    data.dropna().plot(color=colors, linewidth=2, rot=0, ax=ax)

    # font
    add_fonts()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font

    # grid
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.6, zorder=1)
    ax.set_facecolor("whitesmoke")

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format x-axis
    ax.set_xlim(data.index.get_level_values('date')[0], data.index.get_level_values('date')[-1])

    # format y-axis
    ax.set_ylabel(y_label)
    ax.yaxis.tick_right()
    ax.ticklabel_format(style='plain', axis='y')

    # legend
    if isinstance(data, pd.Series) or data.shape[1] == 1:
        ncols = 1
        height = 0.93
    else:
        ncols = ceil(data.shape[1] / 2)
        height = 0.96
    ax.legend(
        loc="upper right",
        ncol=ncols,
        bbox_to_anchor=(0.9, height),
        bbox_transform=fig.transFigure,
        facecolor='white',
        edgecolor='white'
    )

    # add in title and subtitle
    if subtitle is None:
        ax.text(x=0.125, y=0.89, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=16, weight='bold', alpha=.8, fontdict=title_font)
    else:
        ax.text(x=0.125, y=0.93, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=16, weight='bold', alpha=.8, fontdict=title_font)
        ax.text(x=0.125, y=.89, s=f"{subtitle}", transform=fig.transFigure, ha='left',
                fontsize=14, alpha=.8, fontdict=title_font)

    # add in line and tag
    line_colors = ['black', '#758D99', '#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F',
                   '#D1B07C', '#758D99']
    if add_line:
        if line_color is not None:
            line_color = line_colors[line_color]
        else:
            line_color = line_colors[0]
        # create line
        ax.plot([0.12, .9],  # set line width
                [.98, .98],  # set line height
                transform=fig.transFigure,  # set location relative to plot
                clip_on=False,
                color=line_color,
                linewidth=.6)

    # add systamental logo
    if add_logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # set source text
    if source is not None:
        ax.text(x=0.125, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)


def plot_bar(data: Union[pd.Series, pd.DataFrame],
             fig_size: tuple = (15, 7),
             color_lightness: Optional[str] = None,
             color: Optional[str] = None,
             font: Optional[str] = 'Lato',
             x_label: Optional[str] = None,
             y_label: Optional[str] = None,
             title: Optional[str] = None,
             subtitle: Optional[str] = None,
             title_font: Optional[str] = None,
             add_line: bool = False,
             line_color: Optional[int] = 0,
             add_logo: bool = False,
             source: Optional[str] = None,
             source_font: Optional[str] = None
             ):
    """
    Creates a bar plot from a data object (dataframe or series).

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
        Data object from which to create bar plot.
    fig_size: tuple
        Tuple (width, height) of figure object.
    color_lightness: str, {'dark', 'light', 'medium'}, optional
        Color lightness for the bar plot.
    color: int, optional, default None
        Color represented by the index of the color codes in the colors variable.
    font: str, optional, default None
        Font used for all text in the plot.
    x_label: str, optional, default None
        Text describing the units of measurement for the x-axis.
    y_label: str, optional, default None
        Text describing the units of measurement for the y-axis.
    title: str, optional, default None
        Title to use for the plot.
    subtitle: str, optional, default None
        Subtitle to use for the plot.
    title_font: str, optional, default None
        Font used for the text in the title and subtitle.
    add_line: bool, default False
        Adds a horizontal line running from the top left to top right of the plot.
    line_color: int, optional, default None
        Color of added line.
    add_logo: bool, default False
        Adds the Systamental logo to the bottom left corner of the plot.
    source: str, optional, default None
        Adds text for the source of the data in the plot.
    source_font: str, optional, default None
        Font used for the source text.

    """
    # plot size
    fig, ax = plt.subplots(figsize=fig_size)

    # color lightness
    if color_lightness == 'dark':
        colors = ['#00588D', '#A81829', '#005F73', '#005F52', '#714C00', '#4C5900', '#78405F', '#674E1F', '#3F5661']
    elif color_lightness == 'light':
        colors = ['#5DA4DF', '#FF6B6C', '#25ADC2', '#4DAD9E', '#C89608', '#9DA521', '#C98CAC', '#B99966', '#89A2AE']
    elif color_lightness == 'medium':
        colors = ['#98DAFF', '#FFA39F', '#6FE4FB', '#86E5D4', '#FFCB4D', '#D7DB5A', '#FFC2E3', '#F2CF9A', '#BFD8E5']
    else:
        colors = ['#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F', '#D1B07C', '#758D99']

    # color
    if color is not None:
        color = colors[color]
    else:
        color = colors[0]

    # plot
    data.plot(kind='bar', color=color, legend=False, rot=0, ax=ax)

    # font
    add_fonts()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font

    # grid
    ax.set_axisbelow(True)
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.6)
    ax.set_facecolor("whitesmoke")

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format x-axis
    ax.set_xlabel(x_label)

    # format y-axis
    ax.set_ylabel(y_label)
    ax.yaxis.tick_right()
    ax.ticklabel_format(style='plain', axis='y')

    # legend
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.9, 0.94),
        bbox_transform=fig.transFigure,
        facecolor='white',
        edgecolor='white'
    )

    # add in title and subtitle
    if subtitle is None:
        ax.text(x=0.125, y=0.89, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=16, weight='bold', alpha=.8, fontdict=title_font)
    else:
        ax.text(x=0.125, y=0.93, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=16, weight='bold', alpha=.8, fontdict=title_font)
        ax.text(x=0.125, y=.89, s=f"{subtitle}", transform=fig.transFigure, ha='left',
                fontsize=14, alpha=.8, fontdict=title_font)

    # add in line and tag
    line_colors = ['black', '#758D99', '#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F',
                   '#D1B07C', '#758D99']
    if add_line:
        if line_color is not None:
            line_color = line_colors[line_color]
        else:
            line_color = line_colors[0]
        # create line
        ax.plot([0.12, .9],  # set line width
                [.98, .98],  # set line height
                transform=fig.transFigure,  # set location relative to plot
                clip_on=False,
                color=line_color,
                linewidth=.6)

    # add systamental logo
    if add_logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Set source text
    if source is not None:
        ax.text(x=0.125, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)


def plot_table(data: pd.DataFrame,
               fig_size: Optional[tuple] = None,
               font: str = 'Lato',
               title: Optional[str] = None,
               subtitle: Optional[str] = None,
               title_font: Optional[str] = None,
               add_logo: bool = False,
               source: Optional[str] = None,
               source_font: Optional[str] = None
               ):
    """
    Create a table from a data object.

    Parameters
    ----------
    data: pd.DataFrame
        Data object from which to create time series plot.
    fig_size: tuple, optional, default None
        Tuple (width, height) of figure object.
    font: str, optional, default None
        Font used for all text in the plot.
    title: str, optional, default None
        Title to use for the plot.
    subtitle: str, optional, default None
        Subtitle to use for the plot.
    title_font: str, optional, default None
        Font used for the text in the title and subtitle.
    add_logo: bool, default False
        Adds the Systamental logo to the bottom left corner of the plot.
    source: str, optional, default None
        Adds text for the source of the data in the plot.
    source_font: str, optional, default None
        Font used for the source text.
    """
    # number of rows and cols
    n_rows, n_cols = data.shape[0], data.shape[1]

    # plot size
    if fig_size is None:
        fig_size = (n_cols * 2, (n_rows // 2))
    fig = plt.figure(figsize=fig_size)
    ax = plt.subplot(111)

    # font
    add_fonts()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font

    # set ax
    ax.set_xlim(0, n_cols + 1)
    ax.set_ylim(0, n_rows + 2)
    ax.set_axis_off()

    # reset index
    if data.index.name is not None:
        data = data.reset_index().iloc[::-1].copy()
    else:
        data = data.iloc[::-1].copy()

    # add cols and rows
    for col in range(0, n_cols + 1):
        for row in range(0, n_rows):
            ax.annotate(
                xy=(col + 0.5, row + 0.25),
                text=data.iloc[row, col],
                ha='center'
            )

    # add headers
    for col in range(0, n_cols + 1):
        ax.annotate(
            xy=(col + 0.5, n_rows + 0.25),
            text=data.columns[col],
            weight='bold',
            ha='center'
        )

    # Add dividing lines
    # top and bottom lines
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [n_rows + 1, n_rows + 1], lw=1.5, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [n_rows, n_rows], lw=1, color='black', marker='', zorder=4)
    ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [0, 0], lw=1, color='black', marker='', zorder=4)
    # center lines
    for x in range(1, n_rows):
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [x, x], lw=0.5, color='gray', alpha=0.1, zorder=3, marker='')

    # add in title and subtitle
    if subtitle is None:
        ax.text(x=0.125, y=0.83, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=14, weight='bold', alpha=.8, fontdict=title_font)
    else:
        ax.text(x=0.125, y=0.9, s=f"{title}", transform=fig.transFigure, ha='left',
                fontsize=14, weight='bold', alpha=.8, fontdict=title_font)
        ax.text(x=0.125, y=.83, s=f"{subtitle}", transform=fig.transFigure, ha='left',
                fontsize=12, alpha=.8, fontdict=title_font)

    # add systamental logo
    if add_logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Set source text
    if source is not None:
        ax.text(x=0.125, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)

    plt.show()


def plot_heatmap():

    pass
