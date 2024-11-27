import pandas as pd
import numpy as np
from typing import Optional, Union, Any, Callable, Dict
import inspect
import statsmodels
from statsmodels.tsa.tsatools import add_trend
from statsmodels.api import OLS, RecursiveLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

from factorlab.feature_engineering.transformations import Transform


def rolling_window(callable_obj: Callable,
                   data: Union[pd.DataFrame, np.array],
                   window_size: int,
                   method: Optional[str] = None,
                   *args,
                   **kwargs) -> Any:
    """
    Computes rolling window function.

    Parameters
    ----------
    callable_obj: Callable
        Function or class to be implemented as rolling window.
    data: DataFrame or np.array
        DataFrame or array of data.
    window_size: int
        Size of rolling window (number of observations).
    method: str, default None
        Method to be applied to callable_obj if callable_obj is a class.
    *args: Optional arguments, for function.
    **kwargs: Optional keyword arguments, for function.

    Returns
    -------
    out: np.array
        Output from function applied to rolling window.
    """
    # check window size
    if window_size > data.shape[0]:
        raise ValueError(f"Window size {window_size} is greater than length of data {data.shape[0]}.")
    if window_size < 1:
        raise ValueError(f"Window size {window_size} must be greater than 0.")

    # output
    out = None

    # loop through rows of df
    for i in range(data.shape[0] - window_size + 1):

        # get results from callable
        if inspect.isfunction(callable_obj):
            res = callable_obj(data[i:i + window_size], *args, **kwargs)
        elif inspect.isclass(callable_obj) and method is not None:
            # extract relevant kwargs for the class and method
            class_kwargs = {k: v for k, v in kwargs.items() if hasattr(callable_obj, k)}
            method_kwargs = {k: v for k, v in kwargs.items() if hasattr(callable_obj(data[i:i + window_size],
                                                                                     **class_kwargs), method)}
            res = getattr(callable_obj(data[i:i + window_size], **class_kwargs), method)(**method_kwargs)
        else:
            raise TypeError(f"Object {callable_obj} is not a function or class. If class, method must be specified.")

        # pd.dataframe
        if isinstance(res, pd.DataFrame):
            if i == 0:
                out = pd.DataFrame(res.iloc[-1]).T
            else:
                out = pd.concat([out, pd.DataFrame(res.iloc[-1]).T])

        # pd.series
        if isinstance(res, pd.Series):
            if i == 0:
                out = pd.DataFrame(res, columns=[data[i:i + window_size].index[-1]]).T
            else:
                out = pd.concat([out, pd.DataFrame(res, columns=[data[i:i + window_size].index[-1]]).T])

        # np array
        if isinstance(res, np.ndarray):
            if i == 0:
                if len(res.shape) == 1:
                    out = res.reshape(1, -1)
                else:
                    out = res[-1]
            else:
                # add output to array
                if len(res.shape) == 1:  # reshape to 2d arr
                    res = res.reshape(1, -1)
                out = np.vstack([out, res[-1]])

        # list
        if isinstance(res, list):
            pass
            # TODO: add condition to check if first output

        # tuple
        if isinstance(res, tuple):
            pass
            # TODO: add condition to check if first output is tuple

    return out


def expanding_window(callable_obj: Callable, data: Union[pd.DataFrame, np.ndarray], min_obs: int, *args,
                     method: Optional[str] = None, **kwargs) -> Any:
    """
    Computes expanding window function.

    Parameters
    ----------
    callable_obj: Callable
        Function or class to be implemented as rolling window.
    data: DataFrame or np.ndarray
        DataFrame or array with data.
    min_obs: int
        Minimum number of observations in the expanding window.
    method: str, default None
        Method to be applied to callable_obj.
    *args: Optional arguments, for function.
    **kwargs: Optional keyword arguments, for function.

    Returns
    -------
    out: np.array
        Output from function applied to rolling window.

    """
    # check min obs
    if min_obs > data.shape[0]:
        raise ValueError(f"Minimum observations {min_obs} is greater than length of data {data.shape[0]}.")
    if min_obs < 1:
        raise ValueError(f"Minimum observations {min_obs} must be greater than 0.")

    # output
    out = None

    # loop through rows of df
    for row in range(min_obs, data.shape[0] + 1):

        # get results from callable
        if inspect.isfunction(callable_obj):
            res = callable_obj(data[:row], *args, **kwargs)
        elif inspect.isclass(callable_obj) and method is not None:
            # extract relevant kwargs for the class and method
            class_kwargs = {k: v for k, v in kwargs.items() if hasattr(callable_obj, k)}
            method_kwargs = {k: v for k, v in kwargs.items() if hasattr(callable_obj(data[:row],
                                                                                     **class_kwargs), method)}
            res = getattr(callable_obj(data[:row], **class_kwargs), method)(**method_kwargs)
        else:
            raise TypeError(f"Object {callable_obj} is not a function or class. If class, method must be specified.")

        # pd.dataframe
        if isinstance(res, pd.DataFrame):
            if row == min_obs:
                out = pd.DataFrame(res.iloc[-1]).T
            else:
                out = pd.concat([out, pd.DataFrame(res.iloc[-1]).T])

        # pd.series
        if isinstance(res, pd.Series):
            if row == min_obs:
                out = pd.DataFrame(res, columns=[data[:row].index[-1]]).T
            else:
                out = pd.concat([out, pd.DataFrame(res, columns=[data[:row].index[-1]]).T])

        # np array
        if isinstance(res, np.ndarray):
            if row == min_obs:
                if len(res.shape) == 1:
                    out = res.reshape(1, -1)
                else:
                    out = res[-1]
            else:
                # add output to array
                if len(res.shape) == 1:  # reshape to 2d arr
                    res = res.reshape(1, -1)
                out = np.vstack([out, res[-1]])

        # list
        if isinstance(res, list):
            pass
            # TODO: add condition to check if first output

        # tuple
        if isinstance(res, tuple):
            pass
            # TODO: add condition to check if first output is tuple

    return out


def add_lags(data: Union[pd.DataFrame, pd.Series], n_lags: int) -> pd.DataFrame:
    """
    Adds lags to time series data.

    Parameters
    ----------
    data: DataFrame or Series
        DataFrame or Series with time series data.
    n_lags: int
        Number of lags to include in the model.

    Returns
    -------
    new_df: pd.DataFrame
        DataFrame with the time series data and lags of those data added.
    """
    # check n lags
    if n_lags < 1:
        raise ValueError(f"Number of lags {n_lags} must be greater than 0.")

    # convert series to dataframe
    if isinstance(data, pd.Series):
        data = data.to_frame()

    # create emtpy list for lagged columns
    lagged_cols_list = []

    # loop through each column
    for col in data.columns:
        if isinstance(data.index, pd.MultiIndex):
            lagged_cols = [data.groupby(level=1)[col].shift(lag).rename(f"{str(col)}_L{str(lag)}")
                           for lag in range(1, n_lags + 1)]
        else:
            lagged_cols = [data[col].shift(lag).rename(f"{str(col)}_L{str(lag)}")
                           for lag in range(1, n_lags + 1)]
        lagged_cols_list.extend(lagged_cols)

    # concat data and lagged values
    new_df = pd.concat([data] + lagged_cols_list, axis=1)

    return new_df


class TimeSeriesAnalysis:
    """
    Class for time series analysis.
    """
    def __init__(self,
                 target: Union[pd.DataFrame, pd.Series],
                 features: Optional[Union[pd.DataFrame, pd.Series]] = None,
                 log: bool = False,
                 diff: bool = False,
                 n_lags: Optional[int] = None,
                 trend: Optional[str] = None,
                 window_type: str = 'fixed',
                 window_size: Optional[int] = 365
                 ):
        """
        Initialize TimeSeriesAnalysis class.

        Parameters
        ----------
        target: DataFrame or Series
            DataFrame or Series with DatetimeIndex and target variable (y) (column).
        features: DataFrame or Series, default None
            DataFrame or Series with DatetimeIndex and predictor variables (X) (columns).
        log: bool, default False
            Computes log of series.
        diff: bool, default False
            Computes difference of series.
        n_lags: int, default None
            Number of lags to include in the model.
        trend: str, {'c', 't', ct', 'ctt'}, default None
            Adds constant and time trend variables to X.
            'n': no constant or trend
            'c': adds intercept term/constant.
            't': adds time trend only.
            'ct': adds intercept term and time trend.
            'ctt': adds intercept term, time trend, and quadratic trend.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
            Window type used in the linear regression estimation procedure.
        window_size: int, default 365
            Size of rolling or expanding window (number of observations).
        """
        self.target = target
        self.features = features
        self.log = log
        self.diff = diff
        self.n_lags = n_lags
        self.trend = trend
        self.window_type = window_type
        self.window_size = window_size
        self.data = None
        self.index = None
        self.freq = None
        self.model = {}
        self.results = None
        self.preprocess_data()

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess time series data.

        Returns
        -------
        data: pd.DataFrame
            DataFrame with preprocessed time series data.
        """
        # check data type
        if isinstance(self.features, pd.Series):
            self.features = self.features.to_frame()

        # log
        if self.log:
            self.target = Transform(self.target).log()
            if self.features is not None:
                self.features = Transform(self.features).log()

        # diff
        if self.diff:
            self.target = Transform(self.target).diff().dropna()
            if self.features is not None:
                self.features = Transform(self.features).diff().dropna()

        # concat features and target
        self.data = pd.concat([self.target, self.features], axis=1).dropna()
        self.target = self.data.iloc[:, 0]
        self.features = self.data.iloc[:, 1:]

        # index
        self.index = self.data.index
        if isinstance(self.index, pd.MultiIndex):
            if not isinstance(self.index.levels[0], pd.DatetimeIndex):
                self.index = self.index.set_levels(pd.to_datetime(self.index.levels[0]), level=0)
        else:
            self.index = pd.to_datetime(self.index)

        # freq
        if isinstance(self.index, pd.MultiIndex):
            self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
        else:
            self.freq = pd.infer_freq(self.index)

        return self.data

    def create_lags(self) -> pd.DataFrame:
        """
        Create lags for features and target.

        Returns
        -------
        new_df: pd.DataFrame
            DataFrame with the time series data and lags of those data added.
        """
        if self.n_lags is not None:
            self.features = add_lags(self.features, self.n_lags)
            target = add_lags(self.target, self.n_lags)
            self.data = pd.concat([target, self.features], axis=1).dropna()
            self.target = self.data.iloc[:, 0]
            self.features = self.data.iloc[:, 1:]

            return self.data

    def add_trend(self) -> pd.DataFrame:
        """
        Add intercept to time series data.

        Returns
        -------
        new_df: pd.DataFrame
            DataFrame with the time series data and intercept added.
        """
        if self.trend is not None:
            if self.trend not in ['n', 'c', 't', 'ct', 'ctt']:
                raise ValueError(f"Trend {self.trend} is not a valid trend. Select from ['n', 'c', 't', 'ct', 'ctt'].")

            if isinstance(self.target.index, pd.MultiIndex):
                if self.features is None:
                    self.features = self.target.to_frame().groupby(level=1, group_keys=False).\
                        apply(add_trend, trend=self.trend, prepend=True).iloc[:, :-1]
                else:
                    self.features = self.features.groupby(level=1, group_keys=False).\
                        apply(add_trend, trend=self.trend, prepend=True)
            else:
                if self.features is None:
                    self.features = add_trend(self.target, trend=self.trend, prepend=True).iloc[:, :-1]
                else:
                    self.features = add_trend(self.features, trend=self.trend, prepend=True)

            self.data = pd.concat([self.target, self.features], axis=1).dropna()
            self.target = self.data.iloc[:, 0]
            self.features = self.data.iloc[:, 1:]

            return self.data

    def _fit_ols(self,
                 cov_type: Optional[str] = 'nonrobust',
                 cov_kwds: Optional[dict] = None
                 ) -> Any:
        """
        Fit ordinary least squares regression model using Statsmodels.

        See statsmodels.regression.linear_model.OLS for more information.

        Parameters
        ----------
        cov_type: str, {'nonrobust', 'HC0', 'HC1', 'HC2', 'HC3', 'HAC'}, default None
            Covariance estimator to use.
        cov_kwds: any, default None
            Keywords for alternative covariance estimators.

        Returns
        -------
        out: Any
            Output from linear regression estimation.
        """
        if isinstance(self.target.index, pd.MultiIndex):
            for ticker, ticker_df in self.data.groupby(level=1):
                self.model[ticker] = OLS(self.target.loc[:, ticker, :], self.features.loc[:, ticker, :],
                                        missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)

        else:
            self.model = OLS(self.target, self.features, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)

        return self.model

    def _fit_recursive_ols(self) -> Any:
        """
        Fit recursive least squares regression model using Statsmodels.

        See statsmodels.regression.linear_model.RecursiveLS for more information.
        -------
        out: Any
            Output from linear regression estimation.
        """
        if isinstance(self.target.index, pd.MultiIndex):
            for ticker, ticker_df in self.target.groupby(level=1):
                self.model[ticker] = RecursiveLS(self.target.loc[:, ticker, :], self.features.loc[:, ticker, :],
                                                missing='drop').fit()
        else:
            self.model = RecursiveLS(self.target, self.features, missing='drop').fit()

        return self.model

    def _fit_rolling_ols(self, cov_type: Optional[str] = 'nonrobust', cov_kwds: Optional[dict] = None) -> Any:
        """
        Fit rolling ordinary least squares regression model using Statsmodels.

        See statsmodels.regression.linear_model.RollingOLS for more information.

        Parameters
        ----------
        cov_type: str, {'nonrobust', 'HCCM', 'HC0'}, default 'nonrobust'
            Covariance estimator to use.
        cov_kwds: any, default None
            Keywords for alternative covariance estimators.

        Returns
        -------
        out: Any
            Output from linear regression estimation.
        """
        if isinstance(self.target.index, pd.MultiIndex):
            for ticker, ticker_df in self.target.groupby(level=1):
                self.model[ticker] = RollingOLS(self.target.loc[:, ticker, :], self.features.loc[:, ticker, :],
                                               window=self.window_size, missing='drop').fit(cov_type=cov_type,
                                                                                            cov_kwds=cov_kwds)
        else:
            self.model = RollingOLS(self.target, self.features, window=self.window_size,
                                    missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)

        return self.model

    @staticmethod
    def get_model_output(model_res: Union[statsmodels.regression.linear_model.RegressionResultsWrapper,
                                Dict[str, statsmodels.regression.linear_model.RegressionResultsWrapper]],
                         output: str = 'predict') -> Any:
        """
        Get output from linear regression model.

        Parameters
        ----------
        model_res: statsmodels.regression.linear_model.RegressionResultsWrapper
            Linear regression model.
        output: str, {'params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', 'summary'}, default 'predict'
            'params': coefficients from linear regression fit.
            'recursive_coefficients': coefficients from recursive least squares regression.
            'pvalues': p-values of coefficients.
            'predict': predictions from linear regression fit (y_hat).
            'fittedvalues': predictions from recursive least squares regression.
            'resid': residuals from linear regression fit (y - y_hat).
            'resid_recursive': residuals from recursive least squares regression.
            'rsquared': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
            'f_pvalue': The p-value of the F-statistic.
            'summary': summary results from linear regression estimation.

        Returns
        -------
        out: Any
            Output from linear regression estimation.
        """
        # output
        if hasattr(model_res, output):
            if callable(getattr(model_res, output)):
                out = getattr(model_res, output)()
            else:
                out = getattr(model_res, output)
        else:
            raise ValueError(f"Output {output} is not a valid output. "
                             f"Select from ['params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', "
                             f"'summary'].")

        return out

    def ols(self,
            output: str = 'predict',
            cov_type: Optional[str] = 'nonrobust',
            cov_kwds: Optional[dict] = None
            ) -> Any:
        """
        Ordinary least squares regression of target (y) on predictors (X).

        See statsmodels.regression.linear_model.OLS for more information.

        Parameters
        ----------
        output: str, {'params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', 'summary'}, default 'predict'
            'params': coefficients from linear regression fit.
            'pvalues': p-values of coefficients.
            'predict': predictions from linear regression fit (y_hat).
            'resid': residuals from linear regression fit (y - y_hat).
            'rsquared': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
            'f_pvalue': The p-value of the F-statistic.
            'summary': summary results from linear regression estimation.
        cov_type: str, {None, 'HAC'}, default None
            Covariance estimator to use.
        cov_kwds: any, default None
            Keywords for alternative covariance estimators.

        Returns
        -------
        results: Any
            Output from linear regression estimation.
        """
        # lags
        self.create_lags()

        # constant and trend
        self.add_trend()

        # fit ols
        self._fit_ols(cov_type=cov_type, cov_kwds=cov_kwds)

        # get output
        if isinstance(self.target.index, pd.MultiIndex):
            if output in ['params', 'pvalues', 'rsquared', 'f_pvalue', 'summary']:
                self.results = pd.DataFrame(index=self.model.keys(),
                                   data=[self.get_model_output(self.model[ticker], output=output)
                                         for ticker in self.model.keys()]).round(decimals=4)
                if output in ['rsquared', 'f_pvalue', 'summary']:
                    self.results.columns = [output]
            else:
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output=output)
                    if output == 'resid':
                        vals = vals.values
                    idx = self.target.loc[:, ticker, :].index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=vals,
                                      columns=[output])
                    self.results = pd.concat([self.results, df])
                # sort index
                self.results.sort_index(inplace=True)
                if output == 'predict':
                    self.results.columns = ['y_hat']
        else:
            if output in ['predict', 'resid']:
                self.results = pd.DataFrame(index=self.index, data=self.get_model_output(self.model, output=output),
                                   columns=[output])
                if output == 'predict':
                    self.results.columns = ['y_hat']
            else:
                self.results = self.get_model_output(self.model, output=output)

        return self.results

    def expanding_ols(self, output: str = 'predict') -> Any:
        """
        Expanding window ordinary least squares regression of target (y) on predictors (X).

        See statsmodels.regression.linear_model.RecursiveLS for more information.

        Parameters
        ----------
        output: str, {'params', 'pvalues', 'predict', 'resid', 'rsquared', 'summary'}, default 'predict'
            'params': coefficients from linear regression fit.
            'pvalues': p-values of coefficients.
            'predict': predictions from linear regression fit (y_hat).
            'resid': residuals from linear regression fit (y - y_hat).
            'rsquared': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
            'summary': summary results from linear regression estimation.

        Returns
        -------
        results: Any
            Output from linear regression estimation.
        """
        # lags
        self.create_lags()

        # constant and trend
        self.add_trend()

        # fit ols
        self._fit_recursive_ols()

        # get output
        if isinstance(self.target.index, pd.MultiIndex):
            if output == 'params':
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output='recursive_coefficients')['filtered'].T
                    idx = self.target.loc[:, ticker, :].index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=vals, columns=self.features.columns)
                    self.results = pd.concat([self.results, df])
                # sort index
                self.results.sort_index(inplace=True)

            elif output == 'resid':
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output=output).values
                    idx = self.target.loc[:, ticker, :].index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=vals,
                                      columns=[output])
                    self.results = pd.concat([self.results, df])
                # sort index
                self.results.sort_index(inplace=True)

            elif output == 'predict':
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output='fittedvalues')
                    idx = vals.index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=vals.values, columns=['y_hat'])
                    self.results = pd.concat([self.results, df])
                # sort index
                self.results.sort_index(inplace=True)

            elif output in ['pvalues', 'summary']:
                self.results = pd.DataFrame(index=self.model.keys(),
                                            data=[self.get_model_output(self.model[ticker], output=output)
                                                  for ticker in self.model.keys()]).round(decimals=4)
                if output == 'summary':
                    self.results.columns = [output]

            elif output == 'rsquared':
                rsq_dict = {ticker: self.get_model_output(self.model[ticker], output=output)
                            for ticker in self.model.keys()}
                self.results = pd.DataFrame(rsq_dict, index=[output]).T.round(decimals=4)

        else:
            if output == 'params':
                vals = self.get_model_output(self.model, output='recursive_coefficients')['filtered'].T
                self.results = pd.DataFrame(index=self.target.index, data=vals, columns=self.features.columns)
            elif output == 'resid':
                vals = self.get_model_output(self.model, output=output)
                self.results = pd.DataFrame(index=self.target.index, data=vals, columns=[output])
            elif output == 'predict':
                vals = self.get_model_output(self.model, output='fittedvalues')
                self.results = pd.DataFrame(index=vals.index, data=vals.values, columns=['y_hat'])
            else:
                self.results = self.get_model_output(self.model, output=output)

        return self.results

    # TODO: fix bug in TSA rolling regression for multivariate regression, missing param output
    def rolling_ols(self,
                    output: str = 'predict',
                    cov_type: Optional[str] = 'nonrobust',
                    cov_kwds: Optional[dict] = None
                    ) -> Any:
        """
        Rolling window ordinary least squares regression of target (y) on predictors (X).

        See statsmodels.regression.linear_model.RollingOLS for more information.

        Parameters
        ----------
        output: str, {'params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', 'summary'}, default 'predict'
            'params': coefficients from linear regression fit.
            'pvalues': p-values of coefficients.
            'predict': predictions from linear regression fit (y_hat).
            'resid': residuals from linear regression fit (y - y_hat).
            'rsquared': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
            'f_pvalue': The p-value of the F-statistic.
            'summary': summary results from linear regression estimation.
        cov_type: str, {'nonrobust', 'HCCM', 'HC0'}, default 'nonrobust'
            Covariance estimator to use.
        cov_kwds: any, default None
            Keywords for alternative covariance estimators.

        Returns
        -------
        results: Any
            Output from linear regression estimation.

        """
        # lags
        self.create_lags()

        # constant and trend
        self.add_trend()

        # fit ols
        self._fit_rolling_ols(cov_type=cov_type, cov_kwds=cov_kwds)

        # get output
        if isinstance(self.target.index, pd.MultiIndex):
            if output in ['params', 'rsquared', 'f_pvalue']:
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output=output)
                    if isinstance(vals, pd.Series):
                        vals = vals.to_frame()
                    idx = vals.index
                    if output == 'params':
                        df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                          data=vals.values, columns=self.features.columns)
                    else:
                        df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                          data=vals.clip(0).values, columns=[output])
                    self.results = pd.concat([self.results, df])

            elif output in ['pvalues']:
                for ticker in self.model.keys():
                    vals = self.get_model_output(self.model[ticker], output=output)
                    idx = self.target.loc[:, ticker, :].index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=vals, columns=self.features.columns)
                    self.results = pd.concat([self.results, df])

            elif output in ['predict', 'resid']:
                for ticker in self.model.keys():
                    coef = self.get_model_output(self.model[ticker], output='params')
                    df = coef.mul(self.features.loc[:, ticker, :], axis=0).sum(axis=1).to_frame(name='y_hat')
                    if output == 'resid':
                        df = self.target.loc[:, ticker, :].sub(df.y_hat, axis=0).to_frame(name='resid')
                    # add ticker to index
                    idx = df.index
                    df = pd.DataFrame(index=pd.MultiIndex.from_product([idx, [ticker]], names=['date', 'ticker']),
                                      data=df.values, columns=df.columns)
                    self.results = pd.concat([self.results, df])

            else:
                raise ValueError(f"Output {output} is not a valid output. "
                                 f"Select from ['params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue'].")
            # sort index
            self.results.sort_index(inplace=True)

        else:
            if output == 'pvalues':
                vals = self.get_model_output(self.model, output=output)
                self.results = pd.DataFrame(index=self.target.index, data=vals, columns=self.features.columns)

            elif output in ['predict', 'resid']:
                coef = self.get_model_output(self.model, output='params')
                self.results = coef.mul(self.features, axis=0).sum(axis=1).to_frame(name='y_hat')
                if output == 'resid':
                    self.results = self.target.sub(self.results.y_hat, axis=0).to_frame(name='resid')

            elif output in ['rsquared', 'f_pvalue']:
                self.results = self.get_model_output(self.model, output=output).to_frame(name=output).clip(0)

            elif output == 'params':
                self.results = self.get_model_output(self.model, output=output)

            else:
                raise ValueError(f"Output {output} is not a valid output. "
                                 f"Select from ['params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue'].")

        return self.results

    def linear_regression(self,
                          output: str = 'predict',
                          cov_type: Optional[str] = 'nonrobust',
                          cov_kwds: Optional[dict] = None
                          ) -> Any:
        """
        Linear regression of target (y) on predictors (X).

        Parameters
        ----------
        output: str, {'params', 'pvalues', 'predict', 'resid', 'rsquared', 'f_pvalue', 'summary'}, default 'predict'
            'params': coefficients from linear regression fit.
            'pvalues': p-values of coefficients.
            'predict': predictions from linear regression fit (y_hat).
            'resid': residuals from linear regression fit (y - y_hat).
            'rsquared': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
            'f_pvalue': The p-value of the F-statistic.
            'summary': summary results from linear regression estimation.
        cov_type: str, {'nonrobust', 'HAC'}, default None
            Covariance estimator to use.
        cov_kwds: any, default None
            Keywords for alternative covariance estimators.

        Returns
        -------
        out: Any
            Output from linear regression estimation.
        """
        # fixed window
        if self.window_type == 'fixed':
            return self.ols(output=output, cov_type=cov_type, cov_kwds=cov_kwds)
        # expanding window
        elif self.window_type == 'expanding':
            return self.expanding_ols(output=output)
        # rolling window
        elif self.window_type == 'rolling':
            return self.rolling_ols(output=output, cov_type=cov_type, cov_kwds=cov_kwds)

    def adf_test(self, coef: str = 'c', autolag: Optional[str] = 'AIC') -> pd.DataFrame:
        """
        Augmented Dickey-Fuller test for unit root.

        Parameters
        ----------
        coef: str, {'c', 'ct', 'ctt'}, default 'c'
            Constant and trend order to include in regression.
            'c': constant only.
            'ct': constant and trend.
            'ctt': constant, trend, and quadratic trend.
        autolag: str, {“AIC”, “BIC”, “t-stat”, None}, default, 'AIC'
            Method to use when automatically determining the lag length among the values 0, 1, …, maxlag.
            If “AIC” (default) or “BIC”, then the number of lags is chosen to minimize the corresponding information
            criterion. "t-stat” based choice of maxlag starts with maxlag and drops a lag until the t-statistic
            on the last lag length is significant using a 5%-sized test. If None, then the number of included lags
            is set to maxlag.

        Returns
        -------
        results: pd.DataFrame
            Results from Augmented Dickey-Fuller test.
        """
        # store results
        self.results = None

        # drop const and/or trend
        cols = [col for col in self.data.columns if col not in ['const', 'trend', 'trend_squared']]

        if isinstance(self.target.index, pd.MultiIndex):
            # loop through tickers, cols
            for ticker, ticker_df in self.data.groupby(level=1):
                for col in cols:
                    # adf test
                    stats = adfuller(self.data.loc[:, ticker, :][col], maxlag=self.n_lags,
                                     regression=coef, autolag=autolag)
                    # dict for test stats
                    adf_dict = {'adf': stats[0], 'p-val': stats[1], 'lags': stats[2], 'nobs': stats[3],
                                '1%': stats[4]['1%'].round(decimals=4), '5%': stats[4]['5%'], '10%': stats[4]['10%']}
                    # create df
                    # adf_df = pd.DataFrame(adf_dict, index=[col]).round(decimals=4)
                    adf_df = pd.DataFrame(adf_dict, index=pd.MultiIndex.from_product([[ticker], [col]],
                                                                                     names=['ticker', 'series']))
                    self.results = pd.concat([self.results, adf_df])

        else:
            # loop through cols
            for col in cols:
                # adf test
                stats = adfuller(self.data[col], maxlag=self.n_lags, regression=coef, autolag=autolag)
                # dict for test stats
                adf_dict = {'adf': stats[0], 'p-val': stats[1], 'lags': stats[2], 'nobs': stats[3],
                            '1%': stats[4]['1%'].round(decimals=4), '5%': stats[4]['5%'], '10%': stats[4]['10%']}
                # create df
                adf_df = pd.DataFrame(adf_dict, index=[col]).round(decimals=4)
                self.results = pd.concat([self.results, adf_df])

        return self.results

    def granger_causality(self,
                          test: str = 'ssr_ftest',
                          add_const: bool = True,
                          alpha: float = 0.05
                          ) -> pd.DataFrame:
        """
        Runs four tests for granger non causality of 2 time series (target, feature).

        All four tests give similar results. params_ftest and ssr_ftest are equivalent based on F test
        which is identical to lmtest:grangertest in R.

        Notes
        -----
        The Null hypothesis for grangercausalitytests is that the time series in the second column,
        x2, does NOT Granger cause the time series in the first column, x1.
        Grange causality means that past values of x2 have a statistically significant effect on
        the current value of x1, taking past values of x1 into account as regressors. We reject the null hypothesis
        that x2 does not Granger cause x1 if the pvalues are below a desired size of the test.

        The null hypothesis for
        all four test is that the coefficients corresponding to past values of the second time series are zero.

        params_ftest, ssr_ftest are based on F distribution
        ssr_chi2test, lrtest are based on chi-square distribution

        References

        [1]
        https://en.wikipedia.org/wiki/Granger_causality

        [2]
        Greene: Econometric Analysis

        Parameters
        ----------
        test: str, {'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'}, default 'ssr_ftest'
            Type of test.
            'ssr_ftest': F-test for joint hypothesis that coefficients are zero.
            'ssr_chi2test': Chi-squared test for joint hypothesis that coefficients are zero.
            'lrtest': Likelihood ratio test.
            'params_ftest': F-test for joint hypothesis that coefficients are zero with restricted model.
        add_const: bool, default True
            Flag indicating whether to add a constant to the model.
        alpha: float, default 0.05
            Level of confidence/significance.

        Returns
        -------
        results: pd.DataFrame
            Dataframe with granger causality F-test statistic and p-value.
        """
        # check index
        if isinstance(self.target.index, pd.MultiIndex):
            raise TypeError("Granger causality test is not supported for MultiIndex dataframes.")
        # lags
        if self.n_lags is None:
            self.n_lags = int(np.log(len(self.features)))

        # gc stats df
        self.results = pd.DataFrame(columns=[test, 'p-val'])

        for feature in self.features.columns:  # loop through features
            print(f"Running Granger causality test for feature {feature}.")
            # concat ret and feature
            data = pd.concat([self.target, self.features[feature]], axis=1)
            # granger causality
            res = grangercausalitytests(data, maxlag=self.n_lags, addconst=add_const)
            # add to df
            for i in range(1, self.n_lags + 1):
                if res[i][0][test][1] <= alpha:
                    print(f"p-value {res[i][0][test][1]} for lag {i} is less than alpha {alpha}.")
                    self.results.loc[f"{feature}_L{i}", test] = res[i][0][test][0]
                    self.results.loc[f"{feature}_L{i}", 'p-val'] = res[i][0][test][1]

        return self.results.astype(float).round(decimals=4).sort_values(by=test, ascending=False)
