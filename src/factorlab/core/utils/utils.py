import pandas as pd
import numpy as np
from pandas.core.groupby import DataFrameGroupBy
from typing import Union, Optional


def to_dataframe(data: Union[pd.Series, pd.DataFrame], name: Optional[str] = None) -> pd.DataFrame:
    """
    Converts input to DataFrame and ensures float64 dtype.
    This method is used to standardize input for transformations and features.
    It handles both Series and DataFrame inputs, converting Series to a DataFrame
    with a single column.

    Parameters
    ----------
    data: Union[pd.Series, pd.DataFrame]
        Input data to be converted. If a Series, it will be converted to a DataFrame.
    name: str
        The name of the column if the input is a Series. This will be used as the column name
        in the resulting DataFrame.

    Returns
    -------
    pd.DataFrame
        A DataFrame with float64 dtype. If input is a Series, it will be converted to a DataFrame.
        If input is already a DataFrame, it will be cast to float64.
    """
    if isinstance(data, pd.Series):
        if name is not None:
            return data.to_frame(name=name).astype("float64")
        else:
            return data.to_frame().astype("float64")
    elif isinstance(data, pd.DataFrame):
        return data.astype("float64")
    else:
        raise TypeError("Input must be a pandas Series or DataFrame")


def grouped(df: pd.DataFrame, axis: str = 'ts') -> Union[pd.DataFrame, pd.core.groupby.generic.DataFrameGroupBy]:
    """
    Returns either a grouped object or the original DataFrame depending on the index type.

    This method abstracts away the logic of checking whether a DataFrame is indexed by
    multiple levels (e.g., datetime + ticker). It returns a groupby object if a MultiIndex
    is detected (grouped by the second level), otherwise it returns the original DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with either single-level or MultiIndex.
    axis : str, {'ts', 'cs'}, default 'ts'
        Axis along which to group. If 'ts', groups by level=1 (e.g., ticker);
        if 'cs', groups by level=0 (e.g., date).

    Returns
    -------
    Union[pd.DataFrame, DataFrameGroupBy]
        If MultiIndex, returns df grouped by level=1 (e.g., ticker);
        else, returns the original DataFrame unchanged.
    """
    if isinstance(df.index, pd.MultiIndex):
        return df.groupby(level=1, group_keys=False) if axis == 'ts' else df.groupby(level=0, group_keys=False)
    return df


def maybe_droplevel(result: Union[pd.Series, pd.DataFrame], level: int = 0) -> pd.DataFrame:
    """
    Drops a level from the index if the original DataFrame has a MultiIndex.

    Parameters
    ----------
    result : pd.DataFrame
        The resulting DataFrame after a group operation.
    level : int
        The index level to drop.

    Returns
    -------
    pd.DataFrame
        Possibly modified DataFrame with one index level dropped.
    """
    if result.index.nlevels > 2:
        return result.droplevel(level).sort_index()
    return result


def safe_divide(numer: pd.DataFrame, denom: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
    """
    Safely divides a DataFrame or Series by another, handling shape and index alignment.

    Parameters
    ----------
    numer : pd.DataFrame
        Numerator data (e.g., centered values).
    denom : pd.DataFrame or pd.Series
        Denominator data (e.g., std dev, MAD).

    Returns
    -------
    pd.DataFrame
    """
    # Ensure division by zero is handled
    denom = denom.replace(0, np.nan)

    # Handle cases where numerator and denominator have different shapes or indices
    if numer.index.equals(denom.index) and numer.shape != denom.shape:
        return numer / denom.values

    # multi + single index, same tickers/level 1, different columns
    elif numer.index.nlevels > 1 and set(numer.index.get_level_values(1).unique()) == set(denom.index) and \
            not numer.columns.equals(denom.columns):
        denom_reidx = denom.reindex(numer.index.get_level_values(level=1)).set_index(numer.index)
        df1 = pd.concat([numer, denom_reidx], axis=1)
        return df1.div(df1.iloc[:, -1], axis=0).iloc[:, :-1]

    # multi index, same index, same cols
    elif numer.index.nlevels > 1 and numer.index.equals(denom.index) and numer.columns.equals(denom.columns):
        return numer / denom

    # multi index, same index, same shape, different cols
    elif numer.index.nlevels > 1 and numer.index.equals(denom.index) and numer.shape == denom.shape:
        return numer / denom.values

    # multi index, diff index, diff single cols
    elif (numer.index.nlevels > 1 and not numer.index.equals(denom.index) and numer.shape[1] == 1
          and denom.shape[1] == 1):
        denom_reidx = denom.reindex(numer.index)
        return numer / denom_reidx.values

    else:
        return numer / denom.squeeze()


