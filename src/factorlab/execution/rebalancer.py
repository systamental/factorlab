import pandas as pd


class Rebalancer:
    """
    Manages the policy for portfolio rebalancing.

    This class encapsulates:
    1. The logic for determining if a rebalance is due (scheduling).
    2. The process of fitting the optimizer on historical data.
    3. The calculation of target weights based on signals.
    4. The computation of transaction costs.
    """

    def __init__(self, config):
        """
        Initializes the Rebalancer with the strategy configuration.

        Parameters
        ----------
        config : StrategyConfig
            The strategy blueprint containing the optimizer and cost model instances.
        """
        self.config = config
        self.freq_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}

    def is_rebalance_day(self, date_t: pd.Timestamp, dates: pd.Index) -> bool:
        """Determines if a rebalance should occur on the given date based on config."""

        # Retrieve the rebalance frequency from the configuration
        freq = getattr(self.config, 'rebal_freq', 'daily')

        if isinstance(freq, int) and freq > 0:
            # Check for N-day frequency
            start_index = dates.get_loc(dates[self.config.optimizer.window_size])
            current_index = dates.get_loc(date_t)
            return (current_index - start_index) % freq == 0
        elif isinstance(freq, str):
            if freq.lower() in self.freq_map:
                return date_t.dayofweek == self.freq_map[freq.lower()]
            if freq.lower() == '15th':
                return date_t.day == 15
            if freq.lower() == 'month_start':
                return date_t.day == 1
            if freq.lower() == 'month_end':
                # Check if current date is the last trading day of the month
                if date_t == dates[-1]:
                    return True
                try:
                    next_date = dates[dates.get_loc(date_t) + 1]
                    return date_t.month != next_date.month
                except IndexError:
                    # If it's the very last day of the whole backtest, rebalance
                    return True
            if freq.lower() == 'quarter_end':
                # Check if current date is the last trading day of the quarter
                if date_t == dates[-1]:
                    return True
                try:
                    next_date = dates[dates.get_loc(date_t) + 1]
                    return (date_t.month in [3, 6, 9, 12]) and (date_t.month != next_date.month)
                except IndexError:
                    return True
            if freq is None or freq.lower() in ['daily', '1d', 'd']:
                return True
        else:
            print(f"Warning: Unsupported rebalancing frequency '{freq}'. Defaulting to daily.")
            return True

    def rebalance(self,
                  lookback_returns: pd.DataFrame,
                  signal_series: pd.Series,
                  current_weights: pd.Series,
                  prices: pd.Series,
                  market_value: float) -> tuple[pd.Series, float]:
        """
        Performs the full rebalancing step: optimization and cost calculation.

        Parameters
        ----------
        date_t : pd.Timestamp
            The current trading date (the date of execution).
        i : int
            The index position of date_t in the full date index.
        dates : pd.Index
            The full date index of the simulation.
        lookback_returns : pd.DataFrame
            The returns data needed for the optimizer fit.
        signal_series : pd.Series
            The signals available at the time of the decision.
        current_weights : pd.Series
            The weights held just before rebalancing (W_t-1).
        prices : pd.Series
            The prices used to calculate the cost (P_t-1 close or t-1 open).
        market_value : float
            The current total market value of the portfolio.

        Returns
        -------
        tuple[pd.Series, float]
            (target_weights, transaction_cost)
        """

        # 1. Fit the Optimizer (Risk Estimation)
        self.config.optimizer.fit(lookback_returns)

        # 2. Compute Target Weights (Optimization)
        # W*_t uses S_t-1 and previous weights W_t-1
        target_weights = self.config.optimizer.transform(signal_series, current_weights)

        # 3. Compute Transaction Costs
        # Cost is calculated based on the difference between current_weights and target_weights
        transaction_cost = self.config.cost_model.compute_cost(
            current_weights=current_weights,
            target_weights=target_weights,
            prices=prices,
            portfolio_value=market_value
        )

        return target_weights, transaction_cost
