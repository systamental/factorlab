import pandas as pd
from typing import Union, Dict, Any, List
from factorlab.core.base_transform import BaseTransform
from factorlab.utils import to_dataframe


class MarketReturns(BaseTransform):
    """
    Factory for computing a market return series (a portfolio) from
    individual asset returns and corresponding weights.

    Delegates the work to specific aggregation classes (EW, MCW, etc.).
    """

    def __init__(self, method: str = "equal_weighted", **kwargs):
        # The factory itself inherits from BaseTransform
        super().__init__(name="MarketReturns", description="Calculates aggregated market portfolio returns.")
        self.method = method
        self.kwargs = kwargs

        self._method_map: Dict[str, Any] = {
            'equal_weighted': EqualWeightedMarketReturn,
            'generic_weighted': BaseWeightedMarketReturn,
        }

        if self.method not in self._method_map:
            raise ValueError(f"Invalid method '{self.method}', must be one of {list(self._method_map.keys())}")

        # transformer
        self._transformer = self._method_map[self.method](**self.kwargs)

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'MarketReturns':
        """Fit the delegated market return transformer (may be stateless or stateful)."""
        self.validate_inputs(X)
        self._transformer.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """Apply the delegated market return aggregation."""
        if not self._is_fitted:
            raise RuntimeError("MarketReturns transformer must be fitted before transform()")
        return self._transformer.transform(X)


class BaseWeightedMarketReturn(BaseTransform):
    """
    Base class for market returns that aggregate using an existing weight column.
    Handles the core weighted sum and broadcasting to the MultiIndex (Time, AssetID).
    """

    def __init__(self, return_col: str = 'ret', weight_col: str = 'weight', output_col: str = 'market'):
        super().__init__(name="BaseWeightedMarketReturn",
                         description="Aggregates asset returns using a weight column.")
        self.return_col = return_col
        self.weight_col = weight_col
        self.output_col = output_col

    @property
    def inputs(self) -> List[str]:
        return [self.return_col, self.weight_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'BaseWeightedMarketReturn':
        """Minimal fit implementation: validate inputs."""
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy()
        self.validate_inputs(df)

        # Core aggregation logic
        df['weighted_return'] = df[self.weight_col] * df[self.return_col]

        # Use transform('sum') to ensure alignment (broadcasting) back to the MultiIndex
        df[self.output_col] = df.groupby(level=0)['weighted_return'].transform('sum')

        # Drop the temporary column and return the aligned MultiIndex DataFrame
        return df.drop(columns='weighted_return')


class EqualWeightedMarketReturn(BaseTransform):
    """
    Computes the Equal-Weighted (EW) market return.
    This is a stateless transformation.
    """

    def __init__(self, return_col: str = 'ret', output_col: str = 'market'):

        super().__init__(name="EqualWeightedMarketReturn", description="Calculates equal-weighted market returns.")
        self.return_col = return_col
        self.output_col = output_col

    @property
    def inputs(self) -> List[str]:
        """
        Required input columns.
        """
        return [self.return_col]

    def fit(self, X: Union[pd.Series, pd.DataFrame], y=None) -> 'EqualWeightedMarketReturn':
        """
        Minimal fit implementation for a stateless class.
        We only validate inputs and set the fitted flag.
        """
        df_input = to_dataframe(X)
        self.validate_inputs(df_input)
        self._is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Computes the equal-weighted market return by averaging returns across all assets at each time step.

        Parameters
        ----------
        X: Union[pd.Series, pd.DataFrame]
            Input data with a MultiIndex (Time, AssetID) and a column for returns.

        Returns
        -------
        pd.DataFrame
            DataFrame with a single column 'market' representing the equal-weighted market return at each time step.

        """
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        df = to_dataframe(X).copy()
        self.validate_inputs(df)

        # compute mean return across assets at each time step
        df[self.output_col] = df.groupby(level=0)[self.return_col].transform('mean')

        return df
