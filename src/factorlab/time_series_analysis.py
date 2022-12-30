import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from statsmodels.tsa.tsatools import add_trend
from statsmodels.api import OLS, RecursiveLS
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


def linear_reg(target: Union[pd.Series, pd.DataFrame],
               predictors: Optional[Union[pd.Series, pd.DataFrame]],
               window_type: str = 'fixed',
               cov_type: Optional[str] = 'nonrobust',
               cov_kwds: Optional[dict] = None,
               output: str = 'pred',
               log: bool = False,
               trend: str = 'c',
               lookback: Optional[int] = None) -> Any:

    """
    Linear regression of target (y) on predictors (X).

    Parameters
    ----------
    target: Series or Dataframe
        Series or DataFrame with DatetimeIndex and target variable (y) (column).
    predictors: Series or Dataframe
        Series or DataFrame with DatetimeIndex and predictor variables (X) (columns).
    window_type: str, {'fixed', 'expanding', 'rolling'}, default 'fixed'
        Window type used in the linear regression estimation procedure.
    cov_type: str, {None, 'HAC'}, default None
        Covariance estimator to use.
    cov_kwds: any, default None
        Keywords for alternative covariance estimators.
    output: str, {'coef', 'pred', resid', 'rsq', 'summary'}, default 'pred'
        'coef': coefficients from linear regression fit.
        'pred': predictions from linear regression fit (y_hat).
        'resid': residuals from linear regression fit (y - y_hat).
        'rsq': coefficient of determination, aka R-squared of linear regression. Measures goodness of fit.
        'summary': summary results from linear regression estimation.
        Output values from fitting linear regression.
    log: bool, default False
        Computes log of series.
    trend: str, {'c', 't', ct', 'ctt', 'cttt'}, default 'c'
        Adds constant and time trend variables to X.
        'n': no constant or trend
        'c': adds intercept term/constant.
        't': adds time trend only.
        'ct': adds intercept term and time trend.
        'ctt': adds intercept term, time trend and time trend squared.
        'cttt': adds intercept term, time trend, time trend squared and time trend cubed.
    lookback: int
        Number of observations to include in the rolling window.

    Returns
    -------
    output: DataFrame
        DataFrame with DatetimeIndex and selected output values.
    """
    y, X = target, predictors

    # convert y to DataFrame
    if isinstance(y, pd.Series):
        y = y.to_frame()
    # log
    if log:
        if X is not None:
            X = np.log(X).replace([np.inf, -np.inf], np.nan).ffill()
            y = np.log(y).replace([np.inf, -np.inf], np.nan).ffill()
        else:
            y = np.log(y).replace([np.inf, -np.inf], np.nan).ffill()

    # align data
    data = pd.concat([y, X], axis=1, join='inner').dropna().astype(float)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    # intercept
    if X is not None:
        if trend is not None:
            if trend == 'cttt':
                X = add_trend(X, trend='ctt')
                X['trend_cubed'] = X.trend ** 3
            else:
                X = add_trend(X, trend=trend)
    else:
        if trend is not None:
            if trend == 'cttt':
                X = add_trend(y, trend='ctt').iloc[:, 1:]
                X['trend_cubed'] = X.trend ** 3
            else:
                X = add_trend(y, trend=trend).iloc[:, 1:]

    # window type
    out = None
    # fixed
    if window_type == 'fixed':
        model = OLS(y, X, missing='drop').fit(cov_type=cov_type, cov_kwds=cov_kwds)
        if output == 'coef':
            out = model.params
        elif output == 'pval':
            out = model.pvalues
        elif output == 'pred':
            out = model.predict()
        elif output == 'resid':
            out = model.resid
        elif output == 'rsq':
            out = model.rsquared
        else:
            out = model.summary()

    # expanding
    elif window_type == 'expanding':
        model = RecursiveLS(y, X, missing='drop').fit()
        if output == 'coef':
            out = pd.DataFrame(model.recursive_coefficients.filtered.T, index=X.index, columns=X.columns)
        elif output == 'pval':
            out = model.pvalues
        elif output == 'pred':
            out = model.fittedvalues.to_frame(name='pred')
        elif output == 'resid':
            out = pd.DataFrame(model.resid_recursive, index=X.index, columns=['resid'])
        elif output == 'rsq':
            out = model.rsquared
        else:
            out = model.summary()

    # rolling
    elif window_type == 'rolling':
        model = RollingOLS(y, X, window=lookback, missing='drop').fit()
        if output == 'coef':
            out = model.params
        elif output == 'pval':
            out = pd.DataFrame(model.pvalues, index=X.index, columns=X.columns)
        elif output == 'pred':
            out = model.params.mul(X, axis=0).sum(axis=1).to_frame(name='pred')
        elif output == 'resid':
            y_hat = model.params.mul(X, axis=0).sum(axis=1)
            y_hat = y_hat[y_hat != 0]
            out = y.sub(y_hat, axis=0).to_frame(name='resid')
        elif output == 'rsq':
            out = model.rsquared.to_frame(name='rsq')
        else:
            print('This model output is not available. Select another output.\n')
            return

    return out


def granger_causality(
        target: pd.Series,
        factors: Union[pd.Series, pd.DataFrame],
        lags: Optional[int] = None,
        test: str = 'ssr_ftest',
) -> pd.DataFrame:
    """
    Runs four tests for granger non causality of 2 time series (target, predictor).

    Parameters
    ----------
    target: pd.Series
        Target variable. Must be stationary, e.g. log returns.
    factors: pd.Series or pd.DataFrame
        Factors/features to test.
    lags: int, optional, default None
        Maximum number of lags to include in the vector autoregression. If none, uses log(n) where n is number of obs.
    test: str, {'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'}, default 'ssr_ftest'
        Type of test.

    Returns
    -------
    gc_stats: pd.DataFrame
        Dataframe with granger causality F-test statistic and p-value.
    """
    # convert to df
    if isinstance(factors, pd.Series):
        factors = factors.to_frame()
    # lags
    if lags is None:
        lags = int(np.log(len(factors)))

    # gc stats df
    gc_stats = pd.DataFrame(columns=[test, 'p-val'])

    for factor in factors.columns:  # loop through factors
        # concat ret and feature
        data = pd.concat([target, factors[factor]], axis=1).replace([-np.inf, np.inf], 0).dropna()
        # granger causality
        res = grangercausalitytests(data, maxlag=lags)
        # add to df
        gc_stats.loc[factor, test] = res[lags][0][test][0]
        gc_stats.loc[factor, 'p-val'] = res[lags][0][test][1]

    return gc_stats.astype(float).round(decimals=4).sort_values(by=test, ascending=False)


def adf(df: Union[pd.Series, pd.DataFrame],
        lags: Optional[int] = None,
        coef: str = 'c',
        autolag: Optional[str] = 'AIC'
        ) -> pd.DataFrame:
    """
    Augmented Dickey-Fuller unit root test. The Augmented Dickey-Fuller test can be used to test for a unit root
    in a univariate process in the presence of serial correlation.

    df: pd.Series or pd.DataFrame
        Series or dataframe with time series.
    lags: int, optional, default 1
        Number of lags to be used in ADF test.
    coef: str, default 'c'
        Constant and trend order to include in regression.
    autolag: str, {“AIC”, “BIC”, “t-stat”, None}, default, 'AIC'
        Method to use when automatically determining the lag length among the values 0, 1, …, maxlag. If “AIC” (default)
         or “BIC”, then the number of lags is chosen to minimize the corresponding information criterion.
         "t-stat” based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag
         length is significant using a 5%-sized test. If None, then the number of included lags is set to maxlag.
    """

    # conver to df
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # store results in df
    res_df = pd.DataFrame()

    # loop through cols
    for col in df.columns:
        # adf test
        stats = adfuller(df[col].dropna(), maxlag=lags, regression=coef, autolag=autolag)
        # create dict for test stats
        adf_dict = {'adf': stats[0], 'p-val': stats[1], 'lags': stats[2], 'nobs': stats[3],
                    '1%': stats[4]['1%'].round(decimals=4), '5%': stats[4]['5%'], '10%': stats[4]['10%']}
        # create df
        adf_df = pd.DataFrame(adf_dict, index=[col]).round(decimals=4)
        res_df = pd.concat([res_df, adf_df])

    return res_df.sort_values(by='adf')


def hurst(df: pd.DataFrame, window_size: int = 365) -> pd.DataFrame:
    """
    Computes the hurst exponent of a time series.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame with time series.
    window_size: int
        Lookback window for hurst exponent calculation.

    Returns
    -------
    hurst: pd.DataFrame
        DataFrame with Hurst exponents for each series/col.
    """

    def he(series, window_size=window_size):

        # create the range of lag values
        lags = range(2, window_size)

        # calculate the array of the variances of the lagged differences
        tau = [np.sqrt(np.std(np.subtract(series[lag:].values, series[:-lag].values))) for lag in lags]

        # use a linear fit to estimate the Hurst Exponent
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0] * 2.0

    # conver to df
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # store results in df
    res_df = pd.DataFrame()

    # loop through cols
    for col in df.columns:
        hurst_exp = {'hurst': he(df[col], window_size=window_size)}
        res_df = res_df.append(hurst_exp, ignore_index=True)

    # add index
    res_df.index = [df.columns]

    return res_df.sort_values(by='hurst')


def ols_betas(data):
    """
    Computes factor betas using OLS regression.

    Parameters
    ----------
    data: pd.DataFrame
        Dataframe with returns and factors.

    Returns
    -------
    betas: pd.Series
        Estimated betas for each factor.
    """
    # convert type
    data = data
    # estimate beta
    betas = OLS(data.iloc[:, 0], data.iloc[:, 1:], missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 1}).params

    return betas


def fm_reg(returns: Union[pd.Series, pd.DataFrame],
           factors: Union[pd.Series, pd.DataFrame],
           nobs: int = 5
           ) -> pd.DataFrame:
    """
    Runs cross-sectional Fama Macbeth regressions for each time period to compute factor/characteristic betas.

    Parameters
    ----------
    returns: pd.Series or pd.DataFrame
        Returns.
    factors: pd.Series or pd.DataFrame
        Factor values
    nobs: int, default 5
        Minimum number of observations in the cross section to run the Fama Macbeth regression.

    Returns
    -------
    betas: pd.DataFrame
        Dataframe with DatetimeIndex and estimated factor betas.
    """
    # check n obs for factors in cross-section
    if factors.groupby('date').count()[(factors.groupby('date').count() > nobs)].dropna(how='all').empty:
        raise Exception(f"Cross-section does not meet minimum number of observations. Change nobs parameter or "
                        f"increase asset universe.")
    else:
        start_idx = factors.groupby('date').count()[(factors.groupby('date').count() > nobs)].index[0]
    # y, X
    y, X = returns, factors.loc[start_idx:]
    # add constant and join X, y
    data = pd.concat([y, X], axis=1, join='inner').astype(float).dropna()
    # estimate cross sectional betas
    betas = data.groupby(level=0).apply(ols_betas)

    return betas


def fm_summary(returns: Union[pd.Series, pd.DataFrame],
               factors: Union[pd.Series, pd.DataFrame],
               nobs: int = 5
               ) -> pd.DataFrame:
    """
    Computes test statistics for betas from Fama Macbeth cross sectional regressions.

    Parameters
    ----------
    returns: pd.Series
        Returns series.
    factors: pd.Series or pd.DataFrame
        Factors.
    nobs: int, default 5
        Minimum number of observations in the cross section to run the Fama Macbeth regression.

    Returns
    -------
    stats: pd.DataFrame
        DataFrame with mean estimated betas, std errors and t-stats.
    """
    # compute betas
    betas = fm_reg(returns, factors, nobs=nobs)
    # get stats
    stats = betas.describe().T
    stats['std_error'] = stats['std'] / np.sqrt(stats['count'])
    stats['t-stat'] = stats['mean'] / stats['std_error']
    # keep mean, std error and t-stat
    stats = stats[['mean', 'std_error', 't-stat']]
    stats.columns = ['beta', 'std_error', 't-stat']

    return stats
