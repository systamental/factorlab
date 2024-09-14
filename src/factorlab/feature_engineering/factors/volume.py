class Volume:
    """
    Volume Factor class to compute various volume-based indicators.
    """
    def __init__(self, df: pd.DataFrame):
        """
        Constructor

        Parameters
        ----------
        df: pd.DataFrame - MultiIndex
            DataFrame with MultiIndex (Date, Ticker) and columns ['open', 'high', 'low', 'close', 'volume'].
        """
        if not isinstance(df.index, pd.MultiIndex):
            raise TypeError("DataFrame must have a MultiIndex with levels [Date, Ticker].")
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        self.df = df.copy()
        self.volume = self.df['volume']
        self.close = self.df['close']
        self.price = self.close  # Default price used in some indicators

    @staticmethod
    def _scale_to_range(series: pd.Series, lower: float = -50, upper: float = 50) -> pd.Series:
        """
        Scales a pandas Series to a specified range.

        Parameters
        ----------
        series: pd.Series
            Series to scale.
        lower: float
            Lower bound of the target range.
        upper: float
            Upper bound of the target range.

        Returns
        -------
        scaled_series: pd.Series
            Scaled Series.
        """
        min_val = series.min()
        max_val = series.max()
        if pd.isna(min_val) or pd.isna(max_val):
            return pd.Series(np.nan, index=series.index)
        if max_val - min_val == 0:
            return pd.Series(0, index=series.index)
        scaled_series = (series - min_val) / (max_val - min_val)
        scaled_series = scaled_series * (upper - lower) + lower
        return scaled_series

    def volume_momentum(self, hist_length: int = 20, multiplier: int = 4) -> pd.Series:
        """
        Computes Volume Momentum (VMOM).

        Parameters
        ----------
        hist_length: int
            Short-term historical length.
        multiplier: int
            Multiplier to determine long-term historical length.

        Returns
        -------
        vmom_scaled: pd.Series
            Volume Momentum indicator scaled between -50 and 50.
        """
        # Compute short-term moving average of volume
        transform_short = Transform(self.volume)
        short_ma = transform_short.smooth(window_size=hist_length, window_type='rolling')

        # Compute long-term moving average of volume
        transform_long = Transform(self.volume)
        long_ma = transform_long.smooth(window_size=hist_length * multiplier, window_type='rolling')

        # Compute Volume Momentum ratio
        vmom_ratio = short_ma / long_ma

        # Scale the VMOM ratio
        vmom_scaled = self._scale_to_range(vmom_ratio).rename(f'VMOM_{hist_length}_{multiplier}')
        return vmom_scaled

    def delta_volume_momentum(self, hist_length: int = 20, multiplier: int = 4, delta_len: int = 100) -> pd.Series:
        """
        Computes Delta Volume Momentum (DVMOM).

        Parameters
        ----------
        hist_length: int
            Short-term historical length.
        multiplier: int
            Multiplier to determine long-term historical length.
        delta_len: int
            Number of bars to lag.

        Returns
        -------
        dvmom_scaled: pd.Series
            Delta Volume Momentum indicator scaled between -50 and 50.
        """
        # Current VMOM
        vmom_current = self.volume_momentum(hist_length, multiplier)

        # Lagged VMOM
        vmom_lagged = vmom_current.groupby(level=1).shift(delta_len)

        # Delta VMOM
        dvmom = vmom_current - vmom_lagged

        # Scale the Delta VMOM
        dvmom_scaled = self._scale_to_range(dvmom).rename(f'DVMOM_{hist_length}_{multiplier}_{delta_len}')
        return dvmom_scaled

    def volume_weighted_ma_over_ma(self, hist_length: int = 50) -> pd.Series:
        """
        Computes Volume Weighted Moving Average Over MA (VWMAMA).

        Parameters
        ----------
        hist_length: int
            Lookback period.

        Returns
        -------
        vwmama_scaled: pd.Series
            VWMAMA indicator scaled between -50 and 50.
        """
        # Compute Volume Weighted Moving Average (VWMA)
        transform_vwma = Transform(self.df)
        vwma = transform_vwma.vwap()['vwap']

        # Compute Ordinary Moving Average (MA)
        transform_ma = Transform(self.close)
        ma = transform_ma.smooth(window_size=hist_length, window_type='rolling')

        # Compute Ratio of VWMA to MA
        ratio = vwma / ma

        # Compute Log Ratio to stabilize variance
        log_ratio = np.log(ratio).replace([np.inf, -np.inf], np.nan)

        # Scale the Log Ratio
        vwmama_scaled = self._scale_to_range(log_ratio).rename(f'VWMAMA_{hist_length}')
        return vwmama_scaled

    def diff_volume_weighted_ma_over_ma(self, short_dist: int = 20, long_dist: int = 100) -> pd.Series:
        """
        Computes Diff Volume Weighted Moving Average Over MA (DVWMAMA).

        Parameters
        ----------
        short_dist: int
            Short-term lookback.
        long_dist: int
            Long-term lookback.

        Returns
        -------
        dvwmama_scaled: pd.Series
            DVWMAMA indicator scaled between -50 and 50.
        """
        # Short-term VWMAMA
        vwmama_short = self.volume_weighted_ma_over_ma(hist_length=short_dist)

        # Long-term VWMAMA
        vwmama_long = self.volume_weighted_ma_over_ma(hist_length=long_dist)

        # Difference between Short-term and Long-term VWMAMA
        dvwmama = vwmama_short - vwmama_long

        # Scale the DVWMAMA
        dvwmama_scaled = self._scale_to_range(dvwmama).rename(f'DVWMAMA_{short_dist}_{long_dist}')
        return dvwmama_scaled

    def price_volume_fit(self, hist_length: int = 50) -> pd.Series:
        """
        Computes the Price Volume Fit (PVF) indicator using Rolling OLS regression.

        Parameters
        ----------
        hist_length: int
            Lookback period for regression.

        Returns
        -------
        pvf_scaled: pd.Series
            Price Volume Fit indicator scaled between -50 and 50.
        """
        # Prepare data
        data = self.df[['close', 'volume']].copy()
        data['log_close'] = np.log(data['close'].replace(0, np.nan))
        data['log_volume'] = np.log(data['volume'].replace(0, np.nan))

        # Initialize an empty list to store results
        results = []

        # Iterate over each ticker
        for ticker, group in data.groupby(level=1):
            # Drop NaNs
            group = group.dropna(subset=['log_close', 'log_volume'])
            if len(group) < hist_length:
                # Not enough data points for regression
                slope = pd.Series(index=group.index, dtype='float64')
                slope[:] = np.nan
            else:
                # Prepare dependent and independent variables
                y = group['log_close']
                X = sm.add_constant(group['log_volume'])

                # Perform Rolling OLS regression
                try:
                    rolling_ols = RollingOLS(y, X, window=hist_length)
                    rres = rolling_ols.fit()
                    # Extract the slope coefficient for 'log_volume'
                    slope = rres.params['log_volume']
                except Exception as e:
                    print(f"Error in RollingOLS for ticker {ticker}: {e}")
                    slope = pd.Series(index=group.index, dtype='float64')
                    slope[:] = np.nan

                # Clip the slope to range [-50, 50]
                slope = slope.clip(-50, 50)

            # Assign ticker to the slope Series
            slope.name = 'slope'
            # No need to assign ticker as a column since it's part of the MultiIndex
            slope = slope.to_frame()

            # Append to results
            results.append(slope)

        # Combine all results
        if results:
            slopes = pd.concat(results)
            slopes.reset_index(inplace=True)
            # Ensure that 'ticker' is already part of the MultiIndex and not duplicated
            # If 'ticker' is already in the DataFrame, avoid re-inserting it
            if 'ticker' not in slopes.columns:
                slopes['ticker'] = slopes['level_1']
            slopes.set_index(['date', 'ticker'], inplace=True)
            slopes.rename(columns={'slope': f'PVF_{hist_length}'}, inplace=True)
            pvf_series = slopes[f'PVF_{hist_length}']
        else:
            # Return empty Series with appropriate name
            pvf_series = pd.Series(dtype='float64', name=f'PVF_{hist_length}')

        # Scale the PVF
        pvf_scaled = self._scale_to_range(pvf_series).rename(f'PVF_{hist_length}')
        return pvf_scaled

    def diff_price_volume_fit(self, short_dist: int = 20, long_dist: int = 100) -> pd.Series:
        """
        Computes Diff Price Volume Fit (DIFPVF).

        Parameters
        ----------
        short_dist: int
            Short-term lookback.
        long_dist: int
            Long-term lookback.

        Returns
        -------
        difpvf_scaled: pd.Series
            DIFPVF indicator scaled between -50 and 50.
        """
        # Short-term PVF
        pvf_short = self.price_volume_fit(hist_length=short_dist)

        # Long-term PVF
        pvf_long = self.price_volume_fit(hist_length=long_dist)

        # Difference between Short-term and Long-term PVF
        difpvf = pvf_short - pvf_long

        # Scale the DIFPVF
        difpvf_scaled = self._scale_to_range(difpvf).rename(f'DIFPVF_{short_dist}_{long_dist}')
        return difpvf_scaled

    def delta_price_volume_fit(self, hist_length: int = 20, delta_dist: int = 30) -> pd.Series:
        """
        Computes Delta Price Volume Fit (DELPVF).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        delpvf_scaled: pd.Series
            DELPVF indicator scaled between -50 and 50.
        """
        # Current PVF
        pvf_current = self.price_volume_fit(hist_length=hist_length)

        # Lagged PVF
        pvf_lagged = pvf_current.groupby(level=1).shift(delta_dist)

        # Delta PVF
        delpvf = pvf_current - pvf_lagged

        # Scale the DELPVF
        delpvf_scaled = self._scale_to_range(delpvf).rename(f'DELPVF_{hist_length}_{delta_dist}')
        return delpvf_scaled

    def on_balance_volume(self, hist_length: int = 50) -> pd.Series:
        """
        Computes the On-Balance Volume (OBV) indicator.

        Returns
        -------
        obv_scaled: pd.Series
            Smoothed OBV indicator scaled between -50 and 50, with a MultiIndex [Date, Ticker].
        """
        # Compute daily price difference per ticker
        price_diff = self.df['close'].groupby(level='ticker').diff()

        # Compute signed volume
        signed_volume = self.df['volume'] * np.sign(price_diff)

        # Compute OBV by cumulative sum per ticker
        obv = signed_volume.groupby(level='ticker').cumsum()

        # Smooth the OBV using the Transform class
        obv_smooth = Transform(obv).smooth(
            window_size=hist_length,
            window_type='rolling',
            central_tendency='mean',
            window_fcn=None
        )

        # Scale the OBV to the range [-50, 50]
        obv_scaled = self._scale_to_range(obv_smooth).rename(f'OBV_{hist_length}')

        return obv_scaled.sort_index()

    def delta_on_balance_volume(self, hist_length: int = 50, delta_dist: int = 45) -> pd.Series:
        """
        Computes Delta On Balance Volume (DOBV).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        dobv_scaled: pd.Series
            DOBV indicator scaled between -50 and 50.
        """
        # Current OBV
        obv_current = self.on_balance_volume(hist_length=hist_length)

        # Lagged OBV
        obv_lagged = obv_current.groupby(level=1).shift(delta_dist)

        # Delta OBV
        dobv = obv_current - obv_lagged

        # Scale the DOBV
        dobv_scaled = self._scale_to_range(dobv).rename(f'DOBV_{hist_length}_{delta_dist}')
        return dobv_scaled

    def positive_volume_indicator(self, hist_length: int = 40) -> pd.Series:
        """
        Computes Positive Volume Indicator (POSVOL).

        Parameters
        ----------
        hist_length: int
            Lookback period.

        Returns
        -------
        posvol_scaled: pd.Series
            POSVOL indicator scaled between -50 and 50.
        """
        # Compute relative price changes per ticker
        rel_price_change = self.close.groupby(level='ticker').pct_change()

        # Identify where volume is increasing
        increasing_volume = self.volume > self.volume.groupby(level='ticker').shift(1)

        # Apply the condition: relative price change where volume is increasing, else 0
        pos_changes = rel_price_change.where(increasing_volume, 0)

        # Compute rolling average of positive changes over hist_length
        avg_pos_change = pos_changes.groupby(level='ticker').rolling(window=hist_length, min_periods=hist_length).mean().reset_index(level=1, drop=True)

        # Normalize by rolling standard deviation over a longer window (e.g., 250 days)
        norm_window = max(2 * hist_length, 250)
        std_dev = rel_price_change.groupby(level='ticker').rolling(window=norm_window, min_periods=hist_length).std().reset_index(level=1, drop=True)

        # Avoid division by zero by replacing 0 with NaN
        std_dev.replace(0, np.nan, inplace=True)

        # Compute POSVOL as the ratio of average positive change to standard deviation
        posvol = avg_pos_change / std_dev

        # Scale POSVOL to the range [-50, 50]
        posvol_scaled = self._scale_to_range(posvol).rename(f'POSVOL_{hist_length}')

        return posvol_scaled


    def delta_positive_volume_indicator(self, hist_length: int = 40, delta_dist: int = 35) -> pd.Series:
        """
        Computes Delta Positive Volume Indicator (DPOSVOL).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        dposvol_scaled: pd.Series
            DPOSVOL indicator scaled between -50 and 50.
        """
        # Current POSVOL
        posvol_current = self.positive_volume_indicator(hist_length=hist_length)

        # Lagged POSVOL
        posvol_lagged = posvol_current.groupby(level=1).shift(delta_dist)

        # Delta POSVOL
        dposvol = posvol_current - posvol_lagged

        # Scale the DPOSVOL
        dposvol_scaled = self._scale_to_range(dposvol).rename(f'DPOSVOL_{hist_length}_{delta_dist}')
        return dposvol_scaled

    def negative_volume_indicator(self, hist_length: int = 40) -> pd.Series:
        """
        Computes Negative Volume Indicator (NEGVOL).

        Parameters
        ----------
        hist_length: int
            Lookback period.

        Returns
        -------
        negvol_scaled: pd.Series
            NEGVOL indicator scaled between -50 and 50.
        """
        # Compute relative price changes
        rel_price_change = self.close.groupby(level=1).pct_change()

        # Filter where volume is decreasing
        decreasing_volume = self.volume < self.volume.groupby(level=1).shift(1)
        neg_changes = rel_price_change.where(decreasing_volume, 0)

        # Compute average relative price change over hist_length
        avg_neg_change = neg_changes.groupby(level=1).rolling(window=hist_length).mean().reset_index(level=1, drop=True)

        # Normalize by standard deviation over a longer window (e.g., 250 days or 2*hist_length)
        norm_window = max(2 * hist_length, 250)
        std_dev = rel_price_change.groupby(level=1).rolling(window=norm_window).std().reset_index(level=1, drop=True)

        # Avoid division by zero
        std_dev.replace(0, np.nan, inplace=True)

        negvol = avg_neg_change / std_dev

        # Scale the NEGVOL
        negvol_scaled = self._scale_to_range(negvol).rename(f'NEGVOL_{hist_length}')
        return negvol_scaled

    def delta_negative_volume_indicator(self, hist_length: int = 40, delta_dist: int = 35) -> pd.Series:
        """
        Computes Delta Negative Volume Indicator (DNEGVOL).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        dnegvol_scaled: pd.Series
            DNEGVOL indicator scaled between -50 and 50.
        """
        # Current NEGVOL
        negvol_current = self.negative_volume_indicator(hist_length=hist_length)

        # Lagged NEGVOL
        negvol_lagged = negvol_current.groupby(level=1).shift(delta_dist)

        # Delta NEGVOL
        dnegvol = negvol_current - negvol_lagged

        # Scale the DNEGVOL
        dnegvol_scaled = self._scale_to_range(dnegvol).rename(f'DNEGVOL_{hist_length}_{delta_dist}')
        return dnegvol_scaled

    def product_price_volume(self, hist_length: int = 25) -> pd.Series:
        """
        Computes Product Price Volume (PPV).

        Parameters
        ----------
        hist_length: int
            Lookback period.

        Returns
        -------
        ppv_scaled: pd.Series
            PPV indicator scaled between -50 and 50.
        """
        # Step 1: Normalize volume
        median_volume = self.volume.groupby(level=1).rolling(window=250, min_periods=1).median().reset_index(level=1, drop=True)
        normalized_volume = self.volume / median_volume

        # Step 2: Normalize price change
        log_price = np.log(self.close.replace(0, np.nan))
        price_change = log_price.diff()
        median_price_change = price_change.groupby(level=1).rolling(window=250, min_periods=1).median().reset_index(level=1, drop=True)
        iqr_price_change = (
            price_change.groupby(level=1).rolling(window=250, min_periods=1).quantile(0.75) -
            price_change.groupby(level=1).rolling(window=250, min_periods=1).quantile(0.25)
        ).reset_index(level=1, drop=True)
        normalized_price_change = (price_change - median_price_change) / iqr_price_change

        # Handle division by zero or inf
        normalized_price_change.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 3: Compute precursor
        precursor = normalized_volume * normalized_price_change

        # Compute rolling mean over hist_length
        ppv = precursor.groupby(level=1).rolling(window=hist_length).mean().reset_index(level=1, drop=True)

        # Scale the PPV
        ppv_scaled = self._scale_to_range(ppv, lower=-50, upper=50).rename(f'PPV_{hist_length}')
        return ppv_scaled

    def sum_price_volume(self, hist_length: int = 25) -> pd.Series:
        """
        Computes Sum Price Volume (SPV).

        Parameters
        ----------
        hist_length: int
            Lookback period.

        Returns
        -------
        spv_scaled: pd.Series
            SPV indicator scaled between -50 and 50.
        """
        # Step 1: Normalize volume
        median_volume = self.volume.groupby(level=1).rolling(window=250, min_periods=1).median().reset_index(level=1, drop=True)
        normalized_volume = self.volume / median_volume

        # Step 2: Normalize price change
        log_price = np.log(self.close.replace(0, np.nan))
        price_change = log_price.diff()
        median_price_change = price_change.groupby(level=1).rolling(window=250, min_periods=1).median().reset_index(level=1, drop=True)
        iqr_price_change = (
            price_change.groupby(level=1).rolling(window=250, min_periods=1).quantile(0.75) -
            price_change.groupby(level=1).rolling(window=250, min_periods=1).quantile(0.25)
        ).reset_index(level=1, drop=True)
        normalized_price_change = (price_change - median_price_change) / iqr_price_change

        # Handle division by zero or inf
        normalized_price_change.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Step 3: Compute precursor
        precursor = normalized_volume + normalized_price_change.abs()
        precursor = precursor.where(price_change > 0, -precursor)

        # Compute rolling mean over hist_length
        spv = precursor.groupby(level=1).rolling(window=hist_length).mean().reset_index(level=1, drop=True)

        # Scale the SPV
        spv_scaled = self._scale_to_range(spv, lower=-50, upper=50).rename(f'SPV_{hist_length}')
        return spv_scaled

    def delta_product_price_volume(self, hist_length: int = 40, delta_dist: int = 35) -> pd.Series:
        """
        Computes Delta Product Price Volume (DSPV).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        dspv_scaled: pd.Series
            DSPV indicator scaled between -50 and 50.
        """
        # Current PPV
        ppv_current = self.product_price_volume(hist_length=hist_length)

        # Lagged PPV
        ppv_lagged = ppv_current.groupby(level=1).shift(delta_dist)

        # Delta PPV
        dspv = ppv_current - ppv_lagged

        # Scale the DSPV
        dspv_scaled = self._scale_to_range(dspv).rename(f'DSPV_{hist_length}_{delta_dist}')
        return dspv_scaled

    def delta_sum_price_volume(self, hist_length: int = 40, delta_dist: int = 35) -> pd.Series:
        """
        Computes Delta Sum Price Volume (DSUMPV).

        Parameters
        ----------
        hist_length: int
            Lookback period.
        delta_dist: int
            Number of bars to lag.

        Returns
        -------
        dsumpv_scaled: pd.Series
            DSUMPV indicator scaled between -50 and 50.
        """
        # Current SPV
        spv_current = self.sum_price_volume(hist_length=hist_length)

        # Lagged SPV
        spv_lagged = spv_current.groupby(level=1).shift(delta_dist)

        # Delta SPV
        dsumpv = spv_current - spv_lagged

        # Scale the DSUMPV
        dsumpv_scaled = self._scale_to_range(dsumpv).rename(f'DSUMPV_{hist_length}_{delta_dist}')
        return dsumpv_scaled
