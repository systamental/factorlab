import pandas as pd
from typing import Optional, Union, Tuple, Dict

from factorlab.feature_engineering.transformations import Transform
from factorlab.strategy_backtesting.metrics import Metrics


class PortfolioSort:
    """
    Portfolio sort class.
    """
    def __init__(self,
                 returns: Union[pd.Series, pd.DataFrame],
                 factors: Union[pd.Series, pd.DataFrame],
                 factor_bins: Dict[str, Tuple[str, int]],
                 lags: int = 1,
                 as_conditional: bool = False,
                 central_tendency: str = 'mean',
                 perc: float = 0.5,
                 ls_portfolio: bool = False,
                 fill_na: bool = False,
                 window_type: str = 'expanding',
                 window_size: int = 365,
                 ann_factor: Optional[int] = None,
                 ):
        """
        Constructor

        Parameters
        ----------
        returns: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and returns (cols).
        factors: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and factors (cols).
        factor_bins: dict
            Strategy and number of bins to use for quantization. Factor_bins must be a dictionary with the name of each
            factor (key) along with the strategy ('ts' or 'cs') and  the number of quantiles (int)
            as a key-values pairs, e.g. {'factor1': ('cs', 5), 'factor2': ('ts', 10)}.
        lags: int, default 1
            Number of lags to apply to the factors.
        as_conditional: bool, default False
            Whether to compute conditional quantiles. If True, the factors are quantized in order to compute
            conditional quantiles. If False, the factors are quantized simultaneously to get unconditional quantiles.
        central_tendency: str, {'mean', 'median', 'quantile', 'min', 'max', etc.}, default 'mean'
            Central tendency to compute.
        perc: float, default 0.5
            Percentile to compute. It can be 0.5 for the median, 0.25 for the 25th percentile,
            0.75 for the 75th percentile, etc.
        ls_portfolio: bool, default False
            Whether to compute the long-short portfolio.
        fill_na: bool, default False
            Whether to fill missing values with 0.
        window_type: str, {'expanding', 'rolling'}, default 'expanding'
            The type of the window. It can be 'expanding' or 'rolling'.
        window_size: int, default 365
            The size of the window.
        ann_factor: int, default None
            Annualization factor, e.g. 365 or 252 for daily, 52 for weekly, 12 for monthly, etc.
        """
        self.returns = returns
        self.factors = factors
        self.factor_bins = factor_bins
        self.lags = lags
        self.as_conditional = as_conditional
        self.central_tendency = central_tendency
        self.perc = perc
        self.ls_portfolio = ls_portfolio
        self.fill_na = fill_na
        self.window_type = window_type
        self.window_size = window_size
        self.ann_factor = ann_factor
        self.factor_names = None
        self.factor_quantiles = pd.DataFrame()
        self.quantile_rets = pd.DataFrame()
        self.portfolio_rets = None
        self.freq = None
        self.index = None
        self.metric_df = None
        self.preprocess_data()
        self.check_factor_bins()

    def check_factor_bins(self) -> Union[int, Dict[str, int]]:
        """
        Check factor bins.
        """
        if self.factors.shape[1] > 3:
            raise ValueError("Only single, double and tripple sorts are supported.")
        elif self.factors.shape[1] != len(self.factor_bins):
            raise ValueError("Number of keys in factor_bins must be the same as the number of factors. "
                             "Factor_bins must be a dictionary with the name of each factor (key) along with the "
                             "strategy ('ts' or 'cs') and  the number of quantiles (int) as a key-values pairs, "
                             "e.g. {'factor1': ['cs', 5], 'factor2': ['ts', 10]}")
        elif isinstance(self.factor_bins, dict):
            return self.factor_bins
        else:
            raise ValueError("Factor_bins must be a dictionary with the name of each factor (key) along with the "
                             "strategy ('ts' or 'cs') and  the number of quantiles (int) as a key-values pairs, "
                             "e.g. {'factor1': ['cs', 5], 'factor2': ['ts', 10]}.")

    def preprocess_data(self) -> None:
        """
        Preprocesses data.
        """
        # factors
        if isinstance(self.factors, pd.Series):
            self.factors = self.factors.to_frame()

        # rets
        if isinstance(self.returns, pd.Series):
            self.returns = self.returns.to_frame()

        # index
        self.index = self.returns.index
        if isinstance(self.index, pd.MultiIndex):
            if not isinstance(self.index.levels[0], pd.DatetimeIndex):
                self.index = self.index.set_levels(pd.to_datetime(self.index.levels[0]), level=0)
        else:
            self.index = pd.to_datetime(self.index)

        # ann_factor
        if self.ann_factor is None:
            if isinstance(self.index, pd.MultiIndex):
                self.ann_factor = self.returns.unstack().groupby(
                    self.returns.unstack().index.get_level_values(0).year).count().max().mode()[0]
            else:
                self.ann_factor = self.returns.groupby(self.returns.index.year).count().max().mode()[0]

        # freq
        if isinstance(self.index, pd.MultiIndex):
            self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
        else:
            self.freq = pd.infer_freq(self.index)
        if self.freq is None:
            if self.ann_factor == 1:
                self.freq = 'Y'
            elif self.ann_factor == 4:
                self.freq = 'Q'
            elif self.ann_factor == 12:
                self.freq = 'M'
            elif self.ann_factor == 52:
                self.freq = 'W'
            else:
                self.freq = 'D'

        # factor names
        self.factor_names = self.factors.columns

    def quantize_factor(self, factor: Union[pd.Series, pd.DataFrame], strategy: str, n_bins: int) -> pd.DataFrame:
        """
        Quantize factors into quantiles.

        Parameters
        ----------
        factor: str
            Name of the factor to quantize.
        strategy: str
            Strategy to use for quantization. It can be 'ts' for time series or 'cs' for cross-sectional.
        n_bins: int
            Number of bins to use for quantization.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with quantized signals.
        """
        if not isinstance(factor.index, pd.MultiIndex) and strategy == 'cs':
            raise ValueError("Cross-sectional quantization requires a MultiIndex with DatetimeIndex as level 0 and "
                             "tickers as level 1.")
        # quantize
        factor_quantiles = Transform(factor).quantize(bins=n_bins, axis=strategy, window_type=self.window_type,
                                              window_size=self.window_size)

        return factor_quantiles

    def unconditional_quantization(self) -> pd.DataFrame:
        """
        Unconditional quantization of factors.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with quantized factors.
        """
        for factor, [strategy, bins] in self.factor_bins.items():
            self.factor_quantiles = pd.concat([self.factor_quantiles,
                                               self.quantize_factor(self.factors[factor], strategy, bins)], axis=1)

        return self.factor_quantiles

    def conditional_quantization(self) -> pd.DataFrame:
        """
        Conditional quantization of factors.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with quantized factors.
        """
        # factor names
        self.factor_quantiles = self.factors.copy()

        # first factor
        self.factor_quantiles[self.factor_names[0]] = self.quantize_factor(self.factors[self.factor_names[0]],
                                                                      self.factor_bins[self.factor_names[0]][0],
                                                                      self.factor_bins[self.factor_names[0]][1])

        # second factor
        if self.factors.shape[1] > 1:

            # quantize 2nd factor
            factor_2 = pd.DataFrame()
            # loop through quantiles
            for quant in range(1, self.factor_bins[self.factor_names[0]][1] + 1):
                factor_1_quantile = self.factor_quantiles[self.factor_quantiles[self.factor_names[0]] == quant]
                # quantize factor 2, conditional on first
                quant = self.quantize_factor(factor_1_quantile[self.factor_names[1]],
                                             self.factor_bins[self.factor_names[1]][0],
                                             self.factor_bins[self.factor_names[1]][1])
                factor_2 = pd.concat([factor_2, quant]).sort_index()  # add to df
            # update factor quantiles
            self.factor_quantiles[self.factor_names[1]] = factor_2

        # third factor
        if self.factors.shape[1] == 3:

            # quantize 3rd factor
            factor_3 = pd.DataFrame()
            # loop through quantiles
            for quant in range(1, self.factor_bins[self.factor_names[1]][1] + 1):
                factor_2_quantile = self.factor_quantiles[self.factor_quantiles[self.factor_names[1]] == quant]
                # quantize factor 3, conditional on first and second
                quant = self.quantize_factor(factor_2_quantile[self.factor_names[2]],
                                             self.factor_bins[self.factor_names[2]][0],
                                             self.factor_bins[self.factor_names[2]][1])
                factor_3 = pd.concat([factor_3, quant]).sort_index()  # add to df
            # update factor quantiles
            self.factor_quantiles[self.factor_names[2]] = factor_3

        return self.factor_quantiles

    def quantize_factors(self) -> pd.DataFrame:
        """
        Quantizes factors into quantiles.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with quantized factors.
        """
        # conditional
        if self.as_conditional:
            self.conditional_quantization()
        else:
            self.unconditional_quantization()

        return self.factor_quantiles

    def join_quantile_rets(self) -> pd.DataFrame:
        """
        Join quantiles and returns.

        Returns
        -------
        df: pd.DataFrame
            Dataframe with quantiles and returns.
        """
        # quantize factors
        self.quantize_factors()

        # quantiles and returns
        if isinstance(self.factor_quantiles.index, pd.MultiIndex):
            quant_df = pd.concat([self.factor_quantiles.groupby('ticker').shift(self.lags), self.returns], axis=1,
                                 join='inner')
        else:
            quant_df = pd.concat([self.factor_quantiles.shift(self.lags), self.returns], axis=1, join='inner')

        return quant_df

    def sort(self) -> pd.DataFrame:
        """
        Sorts factors into quantiles.

        Returns
        -------
        quantile_rets: pd.DataFrame
            Dataframe with sorted factor quantiles.
        """
        # quantile and returns
        quant_df = self.join_quantile_rets()

        # loop through factors
        for factor in self.factor_names:
            factor_quant_df = pd.DataFrame()
            for quant in range(1, self.factor_bins[factor][1] + 1):
                rets = quant_df[quant_df[factor] == quant].iloc[:, -1].to_frame(f"{quant}")
                factor_quant_df = pd.concat([factor_quant_df, rets], axis=1)
            factor_quant_df.columns = pd.MultiIndex.from_product([[factor], factor_quant_df.columns])
            self.quantile_rets = pd.concat([self.quantile_rets, factor_quant_df], axis=1)

        # sort index
        self.quantile_rets = self.quantile_rets.sort_index()

        return self.quantile_rets

    def compute_quantile_portfolios(self) -> pd.DataFrame:
        """
        Computes portfolio returns for each factor quantile.

        Returns
        -------
        portfolio_rets: pd.DataFrame
            Dataframe with portfolio returns.
        """
        # sort quantile returns
        self.sort()

        # compute quantile portfolios
        rets = self.quantile_rets.groupby('date')
        self.portfolio_rets = getattr(rets, self.central_tendency)(self.perc)

        # stack
        self.portfolio_rets = self.portfolio_rets.stack(future_stack=True)
        self.portfolio_rets.index.names = ['date', 'quantile']

        # fill na
        if self.fill_na:
            self.portfolio_rets.fillna(0, inplace=True)

        return self.portfolio_rets

    def performance(self, metric: str = 'sharpe_ratio') -> pd.DataFrame:
        """
        Computes portfolio performance.

        Parameters
        ----------
        metric: str, {'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown', 'annualized_return',
        'annualized_vol', 'cumulative_returns'},
         default 'sharpe_ratio'
            Performance metric to compute.

        Returns
        -------
        performance: pd.DataFrame
            Dataframe with portfolio performance metrics.
        """
        # compute portfolio returns
        self.compute_quantile_portfolios()

        # unstack portfolio returns
        df = self.portfolio_rets.unstack()

        # single sort
        if self.factors.shape[1] == 1:

            # store perf metric in df
            self.metric_df = pd.DataFrame(
                index=[str(i) for i in range(1, self.factor_bins[self.factor_names[0]][1] + 1)],
                columns=[self.factor_names[0]]
            )

            # compute performance metric for each factor quantile
            for factor in self.metric_df.columns:
                for quantile in self.metric_df.index:
                    quant_df = df[(factor, quantile)]
                    self.metric_df.loc[quantile, factor] = getattr(Metrics(quant_df, ret_type='log', ann_factor=365),
                                                             metric)().iloc[0].round(decimals=4)

        # double sort
        elif self.factors.shape[1] == 2:

            # store perf metric in df
            idx = pd.MultiIndex.from_product([[self.factor_names[0]],
                                             [str(i) for i in range(1, self.factor_bins[self.factor_names[0]][1] + 1)]])
            cols = pd.MultiIndex.from_product([[self.factor_names[1]],
                                             [str(i) for i in range(1, self.factor_bins[self.factor_names[1]][1] + 1)]])
            self.metric_df = pd.DataFrame(index=idx, columns=cols)

            # compute performance metric for each factor quantile
            for factor_1, quantile_1 in self.metric_df.index:
                for factor_2, quantile_2 in self.metric_df.columns:
                    quant_df = pd.concat([df[(factor_1, quantile_1)], df[(factor_2, quantile_2)]], axis=1).mean(axis=1)
                    self.metric_df.loc[(factor_1, quantile_1), (factor_2, quantile_2)] = \
                        getattr(Metrics(quant_df, ret_type='log', ann_factor=365), metric)().iloc[0].round(
                            decimals=4)

        # tripple sort
        elif self.factors.shape[1] == 3:

            # store perf metric in df
            idx_list = []
            for name in self.factor_names:
                idx_list.append([f"{name}_{quant}" for quant in range(1, self.factor_bins[name][1] + 1)])
            idx = pd.MultiIndex.from_product(idx_list)
            self.metric_df = pd.DataFrame(index=idx, columns=[metric])

            # compute performance metric for each factor quantile
            for row in self.metric_df.index:
                factor_1 = ('_'.join(row[0].split('_')[:-1]), row[0].split('_')[-1])
                factor_2 = ('_'.join(row[1].split('_')[:-1]), row[1].split('_')[-1])
                factor_3 = ('_'.join(row[2].split('_')[:-1]), row[2].split('_')[-1])
                quant_df = pd.concat([df[factor_1], df[factor_2], df[factor_3]], axis=1).mean(axis=1)
                self.metric_df.loc['_'.join(factor_1), '_'.join(factor_2), '_'.join(factor_3)] = \
                    getattr(Metrics(quant_df, ret_type='log', ann_factor=365), metric)().iloc[0].round(
                        decimals=4)

        else:
            raise ValueError("Only single, double and tripple sorts are supported.")

        return self.metric_df.astype(float)

