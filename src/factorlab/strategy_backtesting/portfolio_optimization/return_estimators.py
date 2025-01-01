import pandas as pd
import numpy as np

from typing import Union, Optional


class ReturnEstimators:
    """
    Return estimators class.

    This class computes the expected returns of the assets or strategies
    using different methods.
    """
    def __init__(self,
                 returns: Union[pd.DataFrame, pd.Series],
                 method: str = 'historical_mean',
                 as_excess_returns: bool = False,
                 risk_free_rate: Optional[Union[pd.Series, pd.DataFrame, float]] = None,
                 as_ann_returns: bool = False,
                 ann_factor: Optional[int] = None,
                 window_size: Optional[int] = None
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.DataFrame or pd.Series
            The returns of the assets or strategies. If not provided, the returns are computed from the prices.
        method: str, {'historical_mean', 'historical_median', 'rolling_mean', 'rolling_median', 'ewma',
                      'rolling_sharpe', 'rolling_sortino'}
            Method to compute the expected returns.
        as_excess_returns: bool, default False
            Whether to compute excess returns.
        risk_free_rate: pd.Series, pd.DataFrame, float, default None
            Risk-free rate series for net returns computation.
        as_ann_returns: bool, default False
            Whether to annualize the expected returns.
        ann_factor: int, default None
            Annualization factor.
        window_size: int, default None
            Window size.
        """
        self.returns = returns
        self.method = method
        self.as_excess_returns = as_excess_returns
        self.risk_free_rate = risk_free_rate
        self.as_ann_returns = as_ann_returns
        self.ann_factor = ann_factor
        self.window_size = window_size
        self.exp_returns = None
        self.freq = None
        self.preprocess_data()

    def preprocess_data(self) -> None:
        """
        Preprocess the data for the return estimation.
        """
        # returns
        if not isinstance(self.returns, pd.DataFrame) and not isinstance(self.returns, pd.Series):  # check data type
            raise ValueError('rets must be a pd.DataFrame or pd.Series')
        # convert data type to float64
        if isinstance(self.returns, pd.Series):  # convert to df
            self.returns = self.returns.to_frame().astype('float64')
        elif isinstance(self.returns, pd.DataFrame):
            self.returns = self.returns.astype('float64')
        if isinstance(self.returns.index, pd.MultiIndex):  # convert to single index
            self.returns = self.returns.unstack()
        self.returns.index = pd.to_datetime(self.returns.index)  # convert to index to datetime
        self.returns = self.returns.dropna(how='all')  # drop missing rows

        # method
        if self.method not in ['historical_mean', 'historical_median', 'rolling_mean', 'rolling_median', 'ewma',
                               'rolling_sharpe', 'rolling_sortino']:
            raise ValueError("Method is not supported. Valid methods are: 'historical_mean', 'historical_median', "
                             "'rolling_mean', 'rolling_median', 'ewma', 'rolling_sharpe', 'rolling_sortino.")

        # risk-free rate
        if self.risk_free_rate is None:
            self.risk_free_rate = 0.0
        elif isinstance(self.risk_free_rate, (pd.Series, pd.DataFrame)):
            self.risk_free_rate = self.returns.join(self.risk_free_rate).ffill().iloc[:, -1]

        # freq
        self.freq = pd.infer_freq(self.returns.index)

        # ann_factor
        if self.ann_factor is None:
            self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().max()

        # window size
        if self.window_size is None:
            self.window_size = self.ann_factor

    def historical_mean(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute expected returns using mean method.
        """
        self.exp_returns = self.returns.mean()

    def historical_median(self) -> Union[pd.Series, pd.DataFrame]:
        """
        Compute expected returns using median method.
        """
        self.exp_returns = self.returns.median()

    def rolling_mean(self):
        """
        Compute expected returns using rolling mean method.
        """
        self.exp_returns = self.returns.rolling(window=self.window_size).mean().iloc[-1]

    def rolling_median(self):
        """
        Compute expected returns using rolling median method.
        """
        self.exp_returns = self.returns.rolling(window=self.window_size).median().iloc[-1]

    def ewma(self):
        """
        Compute expected returns using EWMA method.
        """
        self.exp_returns = self.returns.ewm(span=self.window_size).mean().iloc[-1]

    def rolling_sharpe(self):
        """
        Compute expected returns using rolling Sharpe ratio.
        """
        self.exp_returns = (self.returns.rolling(window=self.window_size).mean() /
                            self.returns.rolling(window=self.window_size).std()).iloc[-1]

    def rolling_sortino(self):
        """
        Compute expected returns using rolling Sortino ratio.
        """
        self.exp_returns = (
                self.returns.rolling(window=self.window_size, min_periods=1).mean() /
                self.returns[self.returns < 0].rolling(window=self.window_size, min_periods=2).std()
        ).iloc[-1]

    def annualize_returns(self):
        """
        Annualize the expected returns.
        """
        if self.as_ann_returns:
            if self.method in ['rolling_sharpe', 'rolling_sortino']:
                self.exp_returns = self.exp_returns * np.sqrt(self.ann_factor)
            else:
                self.exp_returns = self.exp_returns * self.ann_factor

    def deannualize_rf_rate(self):
        """
        De-annualize the risk-free rate.
        """
        self.risk_free_rate = np.log(1 + self.risk_free_rate) / self.ann_factor

    def compute_excess_returns(self):
        """
        Compute excess returns.
        """
        if self.as_excess_returns:
            if isinstance(self.risk_free_rate, (pd.Series, pd.DataFrame)):
                self.deannualize_rf_rate()
                self.returns = self.returns.sub(self.risk_free_rate, axis=0)

    def compute_expected_returns(self):
        """
        Compute expected returns.
        """
        # excess returns
        self.compute_excess_returns()

        # expected returns
        getattr(self, self.method)()

        # annualize returns
        self.annualize_returns()

        # excess returns for annualized returns
        if self.as_excess_returns and self.method not in ['rolling_sharpe', 'rolling_sortino']:
            if isinstance(self.risk_free_rate, float):
                if self.as_ann_returns:
                    self.exp_returns = self.exp_returns - self.risk_free_rate
                else:
                    self.deannualize_rf_rate()
                    self.exp_returns = self.exp_returns - self.risk_free_rate

        return self.exp_returns
