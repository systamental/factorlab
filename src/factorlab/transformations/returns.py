import pandas as pd
import numpy as np
from typing import Union, Optional, List

from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe, grouped


class Returns(BaseTransform):
    """
    Computes different types of returns: 'diff', 'log', 'pct', 'cum'.

    Acts as a factory delegating computation to specific return transforms.
    Fully pipeline-compatible with fit/transform/fit_transform methods.

    Parameters
    ----------
    method : str, default 'pct'
        Type of return to compute. Must be one of {'diff', 'log', 'pct', 'cum'}.
    **kwargs : dict
        Additional keyword arguments passed to the specific return transform.
    """

    def __init__(self, method: str = "pct", **kwargs):
        super().__init__(name="Returns", description="Factory for various return transformations.")

        self.method = method
        self.kwargs = kwargs

        self._method_map = {
            'diff': Difference,
            'log': LogReturn,
            'pct': PctChange,
            'cum': CumulativeReturn
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid method '{self.method}', must be one of {list(self._method_map.keys())}")

        self._transformer = self._method_map[self.method](**self.kwargs)

    @property
    def inputs(self) -> list[str]:
        """Required input columns for this transform, delegated to the specific return transformer."""
        return self._transformer.inputs

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'Returns':
        """Fit the delegated return transformer. For stateless transforms, marks as fitted."""
        self._transformer.fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply the delegated return transformation. The delegated transformer now
        correctly implements the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError("Returns transformer must be fitted before transform()")
        return self._transformer.transform(X)


class Difference(BaseTransform):
    """
    Computes arithmetic difference: p_t - p_{t-lag}.

    Implements the Phase 1: Accumulation Contract.

    Parameters
    ----------
    input_col : str, default 'close'
        Column name of the price series to compute returns on.
    output_col: str, default 'diff'
        Column name for the computed output.
    lags : int, default 1
        Number of periods to lag.
    """
    def __init__(self, input_col: str = 'close', output_col: str = 'diff', lags: int = 1):
        super().__init__(name="Difference")

        self.input_col = input_col
        self.output_col = output_col
        self.lags = lags

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.input_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'Difference':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for calculating arithmetic difference and accumulating the result.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'diff' column appended.
        """
        # Calculate the difference
        result = grouped(df_input[self.input_col].to_frame()).diff(self.lags).squeeze()

        df_input[self.output_col] = result
        return df_input

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to compute the difference, implementing the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)


class LogReturn(BaseTransform):
    """
    Computes log returns: log(p_t / p_{t-lag}).

    Implements the Phase 1: Accumulation Contract.

    Parameters
    ----------
    input_col : str, default 'close'
        Column name of the price series to compute returns on.
    output_col: str, default 'ret'
        Column name for the computed output.
    lags : int, default 1
        Number of periods to lag.
    """
    def __init__(self, input_col: str = 'close', output_col: str = 'ret', lags: int = 1):
        super().__init__(name="LogReturn")

        self.input_col = input_col
        self.output_col = output_col
        self.lags = lags

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.input_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'LogReturn':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for calculating log returns and accumulating the result.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'ret' column appended.
        """
        series = df_input[self.input_col]

        # Handle non-positive prices by converting to NaN, then taking log
        series = series.where(series > 0, np.nan)
        log_series = np.log(series)

        # Calculate the log difference (log(p_t) - log(p_{t-lag}) = log(p_t / p_{t-lag}))
        log_diff = grouped(log_series).diff(self.lags)

        result = log_diff.replace([np.inf, -np.inf], np.nan)

        df_input[self.output_col] = result
        return df_input

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to compute log returns, implementing the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)


class PctChange(BaseTransform):
    """
    Computes arithmetic percent change: (p_t / p_{t-lag}) - 1.

    Implements the Phase 1: Accumulation Contract.

    Parameters
    ----------
    input_col : str, default 'close'
        Column name of the price series to compute returns on.
    output_col: str, default 'ret'
        Column name for the computed output.
    lags : int, default 1
        Number of periods to lag.
    """
    def __init__(self, input_col: str = 'close', output_col: str = 'ret', lags: int = 1):
        super().__init__(name="PctChange")

        self.input_col = input_col
        self.output_col = output_col
        self.lags = lags

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.input_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'PctChange':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for calculating percent change and accumulating the result.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'ret' column appended.
        """
        result = grouped(df_input[self.input_col].to_frame()).pct_change(
            periods=self.lags, fill_method=None
        ).squeeze()

        df_input[self.output_col] = result
        return df_input

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to compute percent change, implementing the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)


class CumulativeReturn(BaseTransform):
    """
    Computes cumulative return: (p_t / p_0) - 1.

    Implements the Phase 1: Accumulation Contract.

    Parameters
    ----------
    input_col : str, default 'close'
        Column name of the price series to compute returns on.
    output_col: str, default 'cum_ret'
        Column name for the computed output.
    base_index : int, default 0
        Index to use as the base price for computing cumulative returns.
    """
    def __init__(self, input_col: str = 'close', output_col: str = 'cum_ret', base_index: int = 0):
        super().__init__(name="CumulativeReturn")

        self.input_col = input_col
        self.output_col = output_col
        self.base_index = base_index

    @property
    def inputs(self) -> List[str]:
        """Required input columns for this transform."""
        return [self.input_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'CumulativeReturn':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for calculating cumulative return and accumulating the result.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'cum_ret' column appended.
        """
        price_series = df_input[self.input_col]

        # Check bounds
        if not (0 <= self.base_index < len(price_series)):
            raise IndexError(f"base_index {self.base_index} out of bounds for DataFrame of length {len(price_series)}")

        if isinstance(price_series.index, pd.MultiIndex):
            def _get_base(g):
                return g.iloc[self.base_index]
            base = price_series.to_frame().groupby(level=1).transform(_get_base).squeeze()
        else:
            base = price_series.iloc[self.base_index]
            base = pd.Series([base] * len(price_series), index=price_series.index)

        # Calculate cumulative return: (p_t / p_0) - 1
        cum_ret = (price_series / base) - 1

        df_input[self.output_col] = cum_ret
        return df_input

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to compute cumulative return, implementing the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)


class TotalReturn(BaseTransform):
    """
    Computes total return by combining pre-calculated price returns,
    financing costs, and optional dividend/yield.

    TotalReturn = PriceReturn - FinancingCost + DividendYield

    Implements the Phase 1: Accumulation Contract.

    Parameters
    ----------
    ret_col : str, default 'ret'
        Column name of the *price return* series (e.g., the output of PctChange).
    financing_col : str, default 'funding_rate'
        Column name of the financing cost series (e.g. funding rate, repo rate, etc)
    dividend_col : str, optional
        Column name of the dividend/yield series. If None, no dividend component is included.
    output_col: str, default 'total_ret'
        Column name for the computed output.
    """
    def __init__(self,
                 ret_col: str = 'ret',
                 financing_col: str = 'funding_rate',
                 dividend_col: Optional[str] = None,
                 output_col: str = 'total_ret'):
        super().__init__(name="TotalReturn",
                         description="Computes total return by combining existing return components.")

        self.ret_col = ret_col
        self.financing_col = financing_col
        self.dividend_col = dividend_col
        self.output_col = output_col

    @property
    def inputs(self) -> list[str]:
        required = [self.ret_col, self.financing_col]
        if self.dividend_col is not None:
            required.append(self.dividend_col)
        return required

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'TotalReturn':
        """Stateless fit: validates input and marks the transform as fitted."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def _transform(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """
        Core logic for calculating total return and accumulating the result.

        Parameters
        ----------
        df_input : pd.DataFrame
            The input data containing all columns (context).

        Returns
        -------
        pd.DataFrame
            The input data with the new 'total_ret' column appended.
        """
        tr = df_input[self.ret_col] - df_input[self.financing_col]

        if self.dividend_col is not None:
            tr += df_input[self.dividend_col]

        df_input[self.output_col] = tr
        return df_input

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Public method to compute total return, implementing the Accumulation Contract.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        return self._transform(df_input)
