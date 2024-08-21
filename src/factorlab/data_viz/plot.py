import pandas as pd
import numpy as np
from typing import Optional, Union
from importlib import resources
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import seaborn as sns


def add_fonts():
    # Get the directory of the current script or module
    current_dir = Path(__file__).resolve().parent
    # Navigate to the parent directory
    parent_dir = current_dir.parent
    # Navigate to the child directory (replace 'child_directory' with your actual directory)
    child_dir = parent_dir / 'conf' / 'fonts'
    # list of font files
    font_files = list(child_dir.glob('*.ttf'))
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
    # plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font

    # grid
    ax.grid(which="major", axis='y', color='#758D99', alpha=0.6, zorder=1)
    ax.set_facecolor("whitesmoke")

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format y-axis
    ax.set_ylabel(y_label)
    ax.yaxis.tick_right()
    ax.ticklabel_format(style='plain', axis='y')

    # legend
    if isinstance(data, pd.Series) or data.shape[1] <= 15:

        # cols and location
        ncols = 5
        location = "upper right"

        if isinstance(data, pd.Series) or data.shape[1] <= 5:
            height = 0.93
        elif data.shape[1] <= 10:
            height = 0.96
        else:
            height = 0.98

        ax.legend(
            loc=location,
            ncol=ncols,
            bbox_to_anchor=(0.9, height),
            bbox_transform=fig.transFigure,
            facecolor='white',
            edgecolor='white'
        )

    else:
        ax.get_legend().set_visible(False)

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
        logo = Image.open(img_path)
        # plt.figimage(logo, origin='upper')
        # ax.imshow(logo, origin='upper')
        fig.figimage(logo, 0, -1, origin='upper')

    # set source text
    if source is not None:
        ax.text(x=0.135, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)


def plot_bar(data: Union[pd.Series, pd.DataFrame],
             axis: str = 'vertical',
             fig_size: Union[tuple, str] = 'auto',
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
    axis: str, {'vertical', 'horizontal'}, default 'vertical'
        Axis along which to plot the bar plot.
    fig_size: tuple or str, default 'auto'
        Tuple (width, height) of figure object, defaults to 'auto'
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
    # axis
    if fig_size == 'auto':
        if axis == 'vertical':
            width, height = data.shape[0], data.shape[0] / 10 + 5
        else:
            width, height = 7, data.shape[0] / 4 + 3
        # set figsize
        fig_size = (width, height)

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
    if axis == 'horizontal':
        data.plot(kind='barh', color=color, legend=False, rot=0, ax=ax)
    else:
        data.plot(kind='bar', color=color, legend=False, rot=0, ax=ax)

    # font
    add_fonts()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams['font.sans-serif'] = font

    # grid
    ax.set_axisbelow(True)
    ax.set_facecolor("whitesmoke")
    if axis == 'horizontal':
        ax.grid(which="major", axis='x', color='#758D99', alpha=0.6)
    else:
        ax.grid(which="major", axis='y', color='#758D99', alpha=0.6)

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format x-axis
    ax.set_xlabel(x_label)
    ax.margins(x=0)

    # format y-axis
    if axis == 'vertical':
        ax.yaxis.tick_right()  # side
        ax.set_ylabel(y_label)  # label
        ax.ticklabel_format(style='plain', axis='y')
        ax.tick_params(right=False)
    else:
        ax.yaxis.tick_left()  # side
        ax.set_ylabel(y_label)  # label
        ax.tick_params(left=False)
        ax.margins(y=0)

    # add in line and tag
    line_colors = ['black', '#758D99', '#006BA2', '#DB444B', '#3EBCD2', '#379A8B', '#EBB434', '#B4BA39', '#9A607F',
                   '#D1B07C', '#758D99']
    if add_line:
        if line_color is not None:
            line_color = line_colors[line_color]
        else:
            line_color = line_colors[0]

        # create line
        ax.plot([0, 1],  # set line width
                [1.05, 1.05],  # set line height
                transform=ax.transAxes,  # set location relative to axis
                clip_on=False,
                color=line_color,
                linewidth=.6)

    # add in title and subtitle
    if subtitle is None:
        ax.set_title(title, loc='left', pad=None, fontsize=16, weight='bold', alpha=.8, fontdict=title_font)

    else:
        ax.set_title(title, loc='left', pad=25, fontsize=16, weight='bold', alpha=.8, fontdict=title_font)
        ax.text(x=0, y=1.01, s=f"{subtitle}", transform=ax.transAxes, ha='left',
                fontsize=14, alpha=.8, fontdict=title_font)

    # add systamental logo
    if add_logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Set source text
    if source is not None:
        ax.text(x=0, y=-0.2, s=f"""Source: {source}""", transform=ax.transAxes, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)

    plt.show()


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


def plot_scatter(data: pd.DataFrame,
                 x: str,
                 y: str,
                 hue: Optional[str] = None,
                 fit: bool = False,
                 fit_method: str = 'linear',
                 fig_size: tuple = (15, 7),
                 font: str = 'Lato',
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
    Creates a scatter plot from a data object (dataframe or series) with the x and y-axis labeled.

    Parameters
    ----------
    data: pd.Series or pd.DataFrame
        Data object from which to create scatter plot.
    x: str
        Column name for the x-axis.
    y: str
        Column name for the y-axis.
    hue: str, optional, default None
        Column name for the hue.
    fit: bool, default False
        Adds a line of best fit to the scatter plot.
    fit_method: str, {'linear', 'lowess', 'logistic'}, default 'linear'
        Method used to fit the line to the scatter plot.
    fig_size: tuple
        Tuple (width, height) of figure object.
    font: str, optional, default None
        Font used for all text in the plot.
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

    # plot
    if fit:
        if fit_method == 'linear':
            sns.regplot(data=data, x=x, y=y, fit_reg=True, ax=ax,
                        scatter_kws={'color': '#006BA2'}, line_kws={'color': '#FF6B6C'})
        elif fit_method == 'logistic':
            sns.regplot(data=data, x=x, y=y, logistic=True, ax=ax,
                        scatter_kws={'color': '#006BA2'}, line_kws={'color': '#FF6B6C'})
        elif fit_method == 'lowess':
            sns.regplot(data=data, x=x, y=y, lowess=True, ax=ax,
                        scatter_kws={'color': '#006BA2'}, line_kws={'color': '#FF6B6C'})
        elif fit_method == 'robust':
            sns.regplot(data=data, x=x, y=y, robust=True, ax=ax,
                       scatter_kws={'color': '#006BA2'}, line_kws={'color': '#FF6B6C'})
    else:
        sns.scatterplot(data=data, x=x, y=y, hue=hue, legend=False, ax=ax)

    # font
    add_fonts()
    plt.rcParams['font.sans-serif'] = font

    # grid
    ax.grid(which="major", axis='x', color='#758D99', alpha=0.6, zorder=1)
    ax.set_facecolor("whitesmoke")

    # remove splines
    ax.spines[['top', 'right', 'left']].set_visible(False)

    # format y-axis
    ax.yaxis.tick_right()
    ax.ticklabel_format(style='plain', axis='y')

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
        logo = Image.open(img_path)
        fig.figimage(logo, 0, -1, origin='upper')

    # set source text
    if source is not None:
        ax.text(x=0.135, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                alpha=.8, fontdict=source_font)


def plot_heatmap(data: pd.DataFrame,
                 fig_size: Union[str, tuple] = 'auto',
                 font: str = 'Lato',
                 title: Optional[str] = None,
                 subtitle: Optional[str] = None,
                 title_font: Optional[str] = None,
                 x_label: Optional[str] = None,
                 y_label: Optional[str] = None,
                 add_logo: bool = False,
                 source: Optional[str] = None,
                 source_font: Optional[str] = None
                 ):
        """
        Creates a heatmap from a data object (dataframe or series).

        Parameters
        ----------
        data: pd.DataFrame
            Data object from which to create heatmap.
        fig_size: tuple or str, default 'auto'
            Tuple (width, height) of figure object, defaults to 'auto'
        font: str, default 'Lato'
            Font used for all text in the plot.
        title: str, default None
            Title to use for the plot.
        subtitle: str, default None
            Subtitle to use for the plot.
        title_font: str, default None
            Font used for the text in the title and subtitle.
        x_label: str, default None
            Text describing the units of measurement for the x-axis.
        y_label: str, default None
            Text describing the units of measurement for the y-axis.
        add_logo: bool, default False
            Adds the Systamental logo to the bottom left corner of the plot.
        source: str, default None
            Adds text for the source of the data in the plot.
        source_font: str, default None
            Font used for the source text.
        """
        # font
        add_fonts()
        plt.rcParams['font.sans-serif'] = font

        # plot size
        if fig_size == 'auto':
            fig_size = (data.shape[0] * 2, data.shape[1] * 2)
        fig, ax = plt.subplots(figsize=fig_size)

        # plot heatmap
        sns.heatmap(data.round(2), cmap="vlag_r", center=0, cbar=False, annot=True, annot_kws={"fontsize": 12},
                    square=True)
        sns.set(font_scale=1)

        # add title
        ax.set_title(title, loc='left', weight='bold', pad=20, fontsize=18, family=title_font)
        ax.text(x=0, y=-0.02, s=subtitle, ha='left', fontsize=15, alpha=.8, fontdict=title_font)

        # add x & y labels
        ax.set_xlabel(x_label, weight='bold')
        ax.set_ylabel(y_label, weight='bold')

        # add systamental logo
        if add_logo:
            with resources.path("factorlab", "systamental_logo.png") as f:
                img_path = f
            logo = Image.open(img_path)
            fig.figimage(logo, 0, -1, origin='upper')

        # set source text
        if source is not None:
            ax.text(x=0.135, y=0.05, s=f"""Source: {source}""", transform=fig.transFigure, ha='left', fontsize=10,
                    alpha=.8, fontdict=source_font)


def monthly_returns_heatmap(
        returns: Union[pd.Series, pd.DataFrame],
        series: str = None,
        ret_type: str = 'log',
        logo: bool = True
) -> None:
    """
    Creates a heatmap of monthly and yearly returns.

    Parameters
    ----------
    returns: pd.Series or pd.DataFrame
        Dataframe or series with DatetimeIndex and returns (cols).
    series: str, default None
        Name of the col/series to compute monthly returns.
    ret_type: str, {'log', 'simple'}, default 'log'
        Type of returns.
    logo: bool, default True
        Adds systamental logo to plot.
    """
    if series is None:
        raise ValueError("Please provide a series to compute monthly returns.")

    # plot size
    fig, ax = plt.subplots(figsize=(15, 15))

    # returns, %
    if ret_type == 'simple':
        ret_df = np.log(returns + 1) * 100
    else:
        ret_df = returns * 100

    # reset index
    ret_df.reset_index(inplace=True)
    # get year and month
    ret_df["year"] = ret_df.date.apply(lambda x: x.year)
    ret_df["month"] = ret_df.date.apply(lambda x: x.strftime("%B"))

    # create table
    table = ret_df.pivot_table(index="year", columns="month", values=series, aggfunc="sum").fillna(0)
    # rename cols, index
    table.columns.name, table.index.name = '', ''
    cols = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December"]
    table = table.reindex(columns=cols)
    table.columns = [col[:3] for col in cols]
    # compute yearly return
    table.loc[:, 'Year'] = table.sum(axis=1)
    table = table.round(decimals=2)  # round

    # plot heatmap
    sns.heatmap(table, annot=True, cmap='RdYlGn', center=0, square=True, cbar=False, fmt='g')
    plt.yticks(rotation=0)  # rotate y-ticks

    # add systamental logo
    if logo:
        with resources.path("factorlab", "systamental_logo.png") as f:
            img_path = f
        img = Image.open(img_path)
        plt.figimage(img, origin='upper')

    # Adding title
    ax.set_title('Monthly Returns (%)', loc='left', weight='bold', pad=20, fontsize=14, family='georgia')
    ax.text(x=0, y=-0.05, s=f"{series.title()}", ha='left', fontsize=12, alpha=.8, fontdict=None, family='georgia')
