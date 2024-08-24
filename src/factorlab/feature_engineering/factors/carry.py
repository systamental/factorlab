from __future__ import annotations
import pandas as pd
from typing import Union, Optional

from factorlab.feature_engineering.transformations import Transform


class Carry:
    """
    Carry factor.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 sign_flip: bool = True
                 ):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame
            DataFrame with DatetimeIndex (level 0), ticker (level 1) and spot, fwd and rate columns (cols).
        sign_flip: bool, default False
            Flips sign of rate series if true.
        """
        # convert data types
        if isinstance(df, pd.Series):
            df = df.to_frame()
        if not isinstance(df.index, pd.MultiIndex):
            df = df.stack()
        # check fields
        if 'spot' not in df.columns:
            raise ValueError("'spot' price series must be provided in dataframe.")
        if 'rate' not in df.columns and 'fwd' not in df.columns:
            raise ValueError("'fwd' price or interest 'rate' series must be provided in dataframe.")

        self.df = df.copy()
        self.sign_flip = sign_flip
        self.spot_price = None
        self.rate = None
        self.fwd_price = None
        self.carry = self.compute_carry()
        self.disp = None
        self.norm_carry = None

    def compute_carry(self):
        """
        Computes the carry factor values.

        Returns
        -------
        carry: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and carry values (cols).
        """
        # spot, fwd and rate series
        self.spot_price = self.df.loc[:, 'spot']
        if 'rate' in self.df.columns:
            self.rate = self.df.loc[:, 'rate']
        else:
            self.fwd_price = self.df.loc[:, 'fwd']

        # compute carry
        if self.rate is None:
            self.carry = (self.fwd_price/self.spot_price) - 1
        else:
            self.carry = self.rate

        # flip sign
        if self.sign_flip:
            self.carry = self.carry * -1

        # drop NaNs
        self.carry = self.carry.unstack().dropna(how='all').stack(future_stack=True).to_frame('carry').sort_index()

        return self.carry

    def dispersion(self,
                   method: str = 'std',
                   window_size: int = 30,
                   window_fcn: str = None
                   ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the dispersion of the spot returns.

        Parameters
        ----------
        method: str, {'std', 'iqr', 'range', 'atr', 'mad'}, default 'std'
            std:  divides by standard deviation.
            iqr:  divides by inter-quartile range.
            range: divides by the range.
            atr:  divides by the average true range.
            mad: divides median absolute deviation.
        window_size: int, default 30
            Length of lookback window for normalization.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.

        Returns
        -------
        norm_factor: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and normalization factor (cols).
        """
        # dispersion
        self.disp = Transform(self.carry).dispersion(method=method, window_type='rolling', window_size=window_size,
                                                     min_periods=window_size, window_fcn=window_fcn).dropna()

        return self.disp

    def smooth(self,
               window_size: int = 3,
               window_type: str = 'rolling',
               central_tendency: str = 'mean',
               window_fcn: Optional[str] = None
               ) -> Union[pd.Series, pd.DataFrame]:
        """
        Smooths the carry factor values.

        Parameters
        ----------
        window_size: int, default 3
            Number of observations in moving window for smoothing.
        window_type: str, {'rolling', 'expanding', 'ewm'}, default 'rolling'
            Type of window.
        central_tendency: str, {'mean', 'median'}, default 'mean'
            Measure of central tendency used for the rolling window.
        window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.

        Returns
        -------
        carry: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and carry values (cols).
        """
        # smoothing
        self.carry = Transform(self.carry).smooth(window_size, window_type=window_type,
                                                  central_tendency=central_tendency, window_fcn=window_fcn).dropna()
        # name
        self.carry.columns = [f"carry_{central_tendency}_{window_size}"]

        return self.carry

    def carry_risk_ratio(self,
                         smoothing: bool = False,
                         sm_window_type: str = 'rolling',
                         sm_window_size: int = 3,
                         sm_central_tendency: str = 'mean',
                         sm_window_fcn: Optional[str] = None,
                         norm_method: str = 'std',
                         norm_window_size: int = 30,
                         norm_window_fcn: Optional[str] = None
                         ) -> Union[pd.Series, pd.DataFrame]:
        """
        Computes the carry/volatility ratio.

        Parameters
        ----------
        smoothing: bool, default False
            If True, carry series is smoothed.
        sm_window_type: str, {'rolling', 'expanding', 'ewm'}, default 'rolling'
            Type of window for smoothing.
        sm_window_size: int, optional, default 3
            Number of observations in moving window for smoothing.
        sm_central_tendency: str, {'mean', 'median'}, default 'mean'
            Measure of central tendency used for the smoothing rolling window.
        sm_window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.
        norm_method: str, {'std', 'iqr', 'range', 'atr', 'mad'}, default 'std'
            std:  divides by standard deviation.
            iqr:  divides by inter-quartile range.
            range: divides by the range.
            atr:  divides by the average true range.
            mad: divides median absolute deviation.
        norm_window_size: int, optional, default 30
            Length of lookback window for normalization.
        norm_window_fcn: str, default None
            Provide a window function. If None, observations are equally-weighted in the rolling computation.
            See scipy.signal.windows for more information.

        Returns
        -------
        df: pd.Series or pd.DataFrame
            Series or DataFrame with DatetimeIndex (level 0), ticker (level 1) and carry/volatility ratio values (cols).
        """
        # name
        name = 'carry'

        # normalization
        self.dispersion(method=norm_method, window_size=norm_window_size, window_fcn=norm_window_fcn)

        # smoothing
        if smoothing:
            self.smooth(sm_window_size, window_type=sm_window_type, central_tendency=sm_central_tendency,
                        window_fcn=sm_window_fcn)
            # name
            name = f"{name}_{sm_central_tendency}_{sm_window_size}"

        # carry to risk
        self.norm_carry = self.carry.divide(self.disp.reindex(index=self.carry.index).values).dropna()

        # name
        name = f"{name}_to_{norm_method}_{norm_window_size}"
        self.norm_carry.columns = [name]

        return self.norm_carry
