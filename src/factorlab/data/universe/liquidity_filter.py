import pandas as pd

from factorlab.core.base_transform import BaseTransform
from factorlab.factors.liquidity.notional_value import NotionalValue
from factorlab.utils import to_dataframe


class LiquidityFilter(BaseTransform):
    """
    Filters the asset universe based on liquidity measures.

    Parameters
    ----------
    price_col : str
        The name of the price column.
    volume_col : str
        The name of the volume column.
    filter_col : str
        The name of the column to filter on (e.g., 'signal').
    thresh : float
        The liquidity threshold for filtering assets.
    rank : bool
        Whether to filter assets by liquidity ranking.
    n_assets : int
        The number of top most liquid assets to select if ranking is enabled.
    """

    def __init__(self,
                 price_col: str = "close",
                 volume_col: str = "volume",
                 filter_col: str = 'signal',
                 window_type: str = "rolling",
                 window_size: int = 30,
                 smooth: bool = True,
                 thresh: float = 1e6,
                 rank: bool = False,
                 n_assets: int = 10
                 ):
        super().__init__(name="LiquidityFilter", description="Filters assets based on liquidity measures.")

        self.price_col = price_col
        self.volume_col = volume_col
        self.filter_col = filter_col
        self.liquidity_threshold = thresh
        self.rank = rank
        self.n_assets = n_assets
        self.output_col = filter_col

        # Initialize the NotionalValue factor to compute liquidity
        self.liquidity_factor = NotionalValue(price_col=self.price_col,
                                              volume_col=self.volume_col,
                                              window_type=window_type,
                                              window_size=window_size,
                                              smooth=smooth)

    @property
    def inputs(self):
        return [self.price_col, self.volume_col, self.filter_col]

    def fit(self, X: pd.DataFrame, y=None) -> 'LiquidityFilter':
        """Fit method to validate inputs."""
        self.validate_inputs(X)
        self.liquidity_factor.fit(X)
        self._is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the liquidity filter to select eligible assets."""
        if not self._is_fitted:
            raise RuntimeError(f"Transform '{self.name}' must be fitted before calling transform()")

        # validate and create copy
        df_input = to_dataframe(X).copy(deep=True)
        self.validate_inputs(df_input)

        # Compute liquidity measure
        df = self.liquidity_factor.transform(df_input)

        # Filter assets based on liquidity threshold
        if self.rank:
            # rank assets by liquidity
            df['rank'] = df['NotionalValue'].groupby(level=0).rank(ascending=False, method='first')
            filtered_df = df[df['rank'] <= self.n_assets].drop(columns=['rank'])
        else:
            # Filter based on threshold
            filtered_df = df[df['NotionalValue'] >= self.liquidity_threshold]

        X[self.filter_col] = filtered_df[self.filter_col]

        return X
