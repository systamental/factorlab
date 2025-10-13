import pandas as pd
from typing import Callable, Dict, Union, Iterable, Any
from functools import partial
from itertools import product
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from importlib import resources

from factorlab.signal_generation.signal import Signal
from factorlab.strategy_backtesting.performance import Performance


def feat_partial(feature: Callable,
                 df: pd.DataFrame,
                 **kwargs: Dict,
                 ) -> Callable:
    """
    Creates a partial function for a factor (feature) by fixing specified arguments.

    Parameters
    ----------
    feature: callable
        Alpha or risk factor to construct, e.g. Trend, Value, Carry, etc.
    df: pd.DataFrame - pd.MultiIndex
        DataFrame with DatetimeIndex (level 0), tickers (level 1) and raw data (cols).
    kwargs: dict
        Keyword arguments for the factor (feature).

    Returns
    -------
    feat_part: callable
        Partial function for factor.
    """
    feat_part = partial(feature, df, **kwargs)()

    return feat_part


def algo_partial(feat_part: Callable,
                 algo: str,
                 **kwargs: Dict
                 ) -> Union[pd.Series, pd.DataFrame]:
    """
    Calls the method/algorithm for the specified factor, e.g. for the Trend factor
    we can call the 'price_mom' method/algorithm.

    Parameters
    ----------
    feat_part: callable
        Partial function created for factor (feat_partial).
    algo: str
        Method/algorithm to use with the factor (feat_part).
    kwargs: dict
        Keyword arguments for the method/algorithm.

    Returns
    -------
    feature: pd.Series or pd.DataFrame - pd.MultiIndex
        Series or Dataframe with DatetimeIndex (level 0), ticker (level 1), and factor values (cols).
    """
    feature = getattr(feat_part, algo)(**kwargs)

    return feature


def feat(df: pd.DataFrame,
         feature: Callable,
         algo: str,
         feat_args: Dict,
         algo_args: Dict
         ) -> Union[pd.Series, pd.DataFrame]:
    """
    Constructs an alpha/risk factor (feature) from the specified parameters.

    Parameters
    ----------
    df: pd.DataFrame - pd.MultiIndex
        DataFrame with DatetimeIndex (level 0), tickers (level 1) and raw data (cols).
    feature: callable
        Alpha or risk factor to construct, e.g. Trend, Value, Carry, etc.
    algo: str
        Method/algorithm to use with the factor (feature).
    feat_args: dict
        Keyword arguments for the factor (feature).
    algo_args: dict
        Keyword arguments for the method/algorithm.

    Returns
    -------
    feature: pd.Series or pd.DataFrame - pd.MultiIndex
        Series or Dataframe with DatetimeIndex (level 0), ticker (level 1), and factor values (cols).
    """
    # compute partial fcn
    feat_part = feat_partial(feature, df, **feat_args)
    # compute factor with specified algo/method
    feature = algo_partial(feat_part, algo, **algo_args)

    return feature


def factor_partial(*args: Union[pd.Series, pd.DataFrame],
                   **kwargs: Dict
                   ) -> Callable:
    """
    Creates a partial function/callable for a factor strategy while specified arguments.

    Parameters
    ----------
    args: pd.Series or pd.DataFrame
        Factor dataframe and forward return series.
    kwargs: dict
        Keyword arguments for the factor strategy.

    Returns
    -------
    fact_part: callable
        Partial function for factor strategy.
    """
    factor_part = partial(Factor, *args)
    fact_part = factor_part(**kwargs)

    return fact_part


def ret_partial(factor_part: Callable,
                **kwargs: Dict
                ) -> Union[pd.Series, pd.DataFrame]:
    """
    Calls the return method/algorithm for the specified factor strategy, e.g. for a long/short factor strategy
    we can call the 'returns' method/algorithm to get factor returns.

    Parameters
    ----------
    factor_part: callable
        Partial function/callable created for factor strategy (factor_partial)
    kwargs: any
        Keyword arguments for the returns' method/algorithm.

    Returns
    -------
    ret: pd.Series or pd.DataFrame - pd.MultiIndex
        Factor returns series or Dataframe with DatetimeIndex (level 0), ticker (level 1),
        and factor returns (cols).
    """
    ret = getattr(factor_part, 'returns')(**kwargs)

    return ret


def factor_ret(factors: Union[pd.Series, pd.DataFrame],
               ret: Union[pd.Series, pd.DataFrame],
               factor_args: Dict,
               ret_args: Dict
               ) -> Union[pd.Series, pd.DataFrame]:

    """
    Computes factor returns with factor and return arguments.

    Parameters
    ----------
    factors: pd.Series or pd.DataFrame
        Series or dataframe with DatetimeIndex (level 0), ticker (level 1) and factor values (cols).
    ret: pd.Series or pd.DataFrame
        Series or dataframe with DatetimeIndex (level 0), ticker (level 1) and returns (cols).
    factor_args: dict
        Keyword arguments for computation of factor-based strategy.
    ret_args: dict
        Keyword arguments for computation of factor-based returns.

    Returns
    -------
    ret: pd.Series or pd.DataFrame
        Factor returns series or dataframe with DatetimeIndex (level 0), ticker (level 1) and factor returns (cols).
    """
    # get factor partial fcn/callable
    factor_part = factor_partial(factors, ret, **factor_args)
    # compute factor returns
    ret = ret_partial(factor_part, **ret_args)

    return ret


def grid_parameters(parameters: dict[str, Iterable[Any]]) -> Iterable[dict[str, Any]]:
    """
    Yields all parameter combinations/pairs.

    Parameters
    ----------
    parameters: dict
        Parameter space.

    Yields
    -------
    Iterable
        Parameter pairs.
    """
    for params in product(*parameters.values()):
        yield dict(zip(parameters.keys(), params))


def compute_metric(
        factor: Union[pd.Series, pd.DataFrame],
        ret: Union[pd.Series, pd.DataFrame],
        factor_args: Dict,
        ret_args: Dict,
        metric: str
):
    """
    Compute performance metric to be used in grid search.

    Parameters
    ----------
    factor: pd.Series or pd.DataFrame
        Factor to compute factor returns.
    ret: pd.Series or DataFrame
        Returns.
    factor_args: dict
        Keyword arguments for computation of factor-based strategy.
    ret_args: dict
        Keyword arguments for computation of factor-based returns.
    metric: str, {'cumulative_ret', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio'},
            default sharpe_ratio
        Performance metric to compute.

    Returns
    -------
    val: pd.Series
        Performance metric value.
    """
    val = getattr(Performance(factor_ret(factor, ret, factor_args, ret_args)), metric)()

    return val


def factor_param_grid_search(
        df: Union[pd.Series, pd.DataFrame],
        ret: Union[pd.Series, pd.DataFrame],
        feature: Callable,
        algo: str,
        feat_args: Dict,
        algo_args: Dict,
        factor_args: Dict,
        ret_args: Dict,
        metric: str = 'sharpe_ratio'
):
    """
    Factor parameter grid search. Computes metric across all pairs/combinations 
    of parameter values for factor (feature) and method/algorithms.
    
    Parameters
    ----------
    df: pd.DataFrame - pd.MultiIndex
        DataFrame with DatetimeIndex (level 0), tickers (level 1) and raw data (cols).
    ret: pd.Series or DataFrame
        Returns.
    feature: callable
        Alpha or risk factor to construct, e.g. Trend, Value, Carry, etc.
    algo: str
        Method/algorithm to use with the factor (feature).
    feat_args: dict
        Keyword arguments for the factor (feature).
    algo_args: dict
        Keyword arguments for the method/algorithm.
    factor_args: dict
        Keyword arguments for computation of factor-based strategy.
    ret_args: dict
        Keyword arguments for computation of factor-based returns.
    metric: str, {'cumulative_ret', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio'},
            default sharpe_ratio
        Performance metric to compute.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with parameter and metric values for all alpha/risk factor parameter combinations.
    """
    # loop through params and compute metrics
    metrics = Parallel(n_jobs=8)(delayed(compute_metric)(feat(df, feature, algo, feat_param, algo_param),
                                                         ret,
                                                         factor_args,
                                                         ret_args,
                                                         metric)
                                 for feat_param in grid_parameters(feat_args) for algo_param in
                                 grid_parameters(algo_args))

    # convert vals to list
    metrics = [i[0] for i in metrics]

    # create param values
    params = [(feat_param | algo_param) for feat_param in grid_parameters(feat_args) for algo_param in
              grid_parameters(algo_args)]
    df = pd.DataFrame(params, metrics).reset_index().rename(columns={'index': metric}).sort_values(by=metric,
                                                                                                   ascending=False)

    return df


def strategy_param_grid_search(
        factor: Union[pd.Series, pd.DataFrame],
        ret: Union[pd.Series, pd.DataFrame],
        factor_args: Dict,
        ret_args: Dict,
        metric: str = 'sharpe_ratio',
):
    """
    Factor strategy parameter grid search. Computes metric across all pairs/combinations
    of parameter values for strategy and returns method.

    Parameters
    ----------
    factor: pd.Series or pd.DataFrame
        Factor to compute factor returns.
    ret: pd.Series or DataFrame
        Returns.
    factor_args: dict
        Keyword arguments for computation of factor-based strategy.
    ret_args: dict
        Keyword arguments for computation of factor-based returns.
    metric: str, {'cumulative_ret', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'omega_ratio'},
            default sharpe_ratio
        Performance metric to compute.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with parameter and metric values for all factor strategy parameter combinations.

    """
    if not isinstance(factor, pd.Series):
        raise TypeError("Factor must be a series for parameter grid search.")

    metrics = Parallel(n_jobs=8)(delayed(compute_metric)(factor, ret, factor_param, ret_param, metric)
                                 for factor_param in grid_parameters(factor_args) for ret_param in
                                 grid_parameters(ret_args))
    metrics = [i[0] for i in metrics]

    params = [(factor_param | ret_param) for factor_param in grid_parameters(factor_args) for ret_param in
              grid_parameters(ret_args)]
    df = pd.DataFrame(params, metrics).reset_index().rename(columns={'index': metric}).sort_values(by=metric,
                                                                                                   ascending=False)

    return df


def param_heatmap(
        param_df: pd.DataFrame,
        metric: str,
        fixed_params: Dict,
        plot_params: Dict
):
    """
    Creates a parameter heatmap for the performance metric across all combinantions/pairs of parameter values
    computed in the parameter grid search.

    Parameters
    ----------
    param_df: pd.DataFrame
        Dataframe with parameter and metric values for all factor strategy parameter combinations.
    metric: str
        Performance metric.
    fixed_params: dict
        Dictionary with parameters to keep fixed.
    plot_params: dict
        Dictionary with parameters to visualize.

    """
    # more than 3 params
    if param_df.shape[1] > 3:

        # fix params
        for param, val in fixed_params.items():
            param_df = param_df[getattr(param_df, param) == val]

    # compare params
    plot_params.append(metric)
    param_df = param_df[plot_params]

    # drop duplicates
    param_df = param_df[~ param_df.duplicated()]
    # create heatmap
    param_matrix = param_df.pivot(*tuple(plot_params))

    # plot heatmap
    plt.style.use('seaborn')
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(param_matrix.shape[0] * 5, param_matrix.shape[1] * 5))
    sns.set(font_scale=min(param_matrix.shape[0], param_matrix.shape[1]) / 3)
    # plot heatmap
    sns.heatmap(param_matrix, cmap="vlag_r", center=0, cbar=False, annot=True,
                annot_kws={"fontsize": min(param_matrix.shape[0], param_matrix.shape[1]) * 5}, square=True)

    # add systamental logo
    with resources.path("factorlab", "systamental_logo.png") as f:
        img_path = f
    img = Image.open(img_path)
    plt.figimage(img, origin='upper')

    # Adding title
    ax.set_title('Parameter Heatmap', loc='left', weight='bold', pad=20, fontsize=14, family='georgia')
    ax.text(x=0, y=-0.05, s=f"{metric.title()}", ha='left', fontsize=12, alpha=.8, fontdict=None, family='georgia')
