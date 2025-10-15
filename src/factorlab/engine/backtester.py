import pandas as pd
import numpy as np
from typing import Dict

from factorlab.strategy.config import StrategyConfig
from factorlab.execution.rebalancer import Rebalancer


class BacktesterEngine:
    """
    The main execution engine for running a time-series simulation (backtest).

    It consumes a StrategyConfig object and iterates over historical data,
    managing the simulation clock, rebalancing, and P&L accounting.
    """

    def __init__(self,
                 config: StrategyConfig,
                 data: pd.DataFrame,
                 initial_capital: float = 1_000_000.0,
                 verbose: bool = True,
                 ann_factor: int = 365):
        """
        Initializes the Backtester Engine.

        Parameters
        ----------
        config : StrategyConfig
            The strategy blueprint defining the data pipeline, optimizer, and cost models.
        data : pd.DataFrame
            The raw historical asset data (MultiIndex: Date, Asset) for the
            pipeline to transform.
        initial_capital : float, default 1,000,000.0
            The starting cash balance.
        verbose : bool, default True
            If True, prints progress updates to the console.
        ann_factor : int, default 365
            The annualization factor to use for volatility calculations.
        """

        self.config = config
        self.data = data
        self.initial_capital = initial_capital
        self.verbose = verbose
        self.ann_factor = ann_factor

        # internal state
        self.pipeline = None
        self.returns = None
        self.prices = None
        self.signals = None
        self.current_weights = None
        self.net_exposure = None
        self.gross_exposure = None
        self.portfolio_return = None
        self.results: Dict[str, pd.DataFrame] = {}

        # Frequency map for runtime checks
        self.freq_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}

        print(f"Backtesting Engine ready for strategy: {self.config.name}")

    def _compute_pipeline(self):
        """
        Runs the full data Pipeline for the backtest.

        Best Practice: The signal component MUST ensure causality (e.g., using
        shift(1) internally) so the signal for date 't' is based only on data
        available at the close of 't-1'.
        """
        print("--- Computing data pipeline... ---")

        # run the full data pipeline
        self.pipeline = self.config.data_pipeline.fit_transform(self.data)

        print("--- Data pipeline complete. ---\n")

        return self.pipeline

    def _validate_pipeline(self):
        """Validates the output of the data pipeline."""
        required_columns = {'signal', 'ret', 'funding_rate', 'close'}

        if not required_columns.issubset(self.pipeline.columns):
            missing = required_columns - set(self.pipeline.columns)
            raise ValueError(f"Pipeline output is missing required columns: {missing}")

    def _extract_features(self):
        """Extracts signals, returns, and prices from the pipeline output."""
        # extract features from data pipeline
        self.signals = self.pipeline['signal'].unstack()
        self.returns = self.pipeline['ret'].unstack()
        self.funding_rate = self.pipeline['funding_rate'].unstack()
        self.prices = self.pipeline['close'].unstack()

        if self.signals.empty:
            raise ValueError("Signal generator returned an empty DataFrame.")
        if self.returns.empty:
            raise ValueError("Returns DataFrame is empty.")
        if self.funding_rate.empty:
            raise ValueError("Funding Rate DataFrame is empty.")
        if self.prices.empty:
            raise ValueError("Prices DataFrame is empty.")

    def _initialize_state(self):
        """Sets up initial portfolio holdings and tracking tables."""

        self.dates = self.signals.index.intersection(self.returns.index).unique()
        self.assets = self.returns.columns
        self.market_value = self.initial_capital

        # tracking tables
        self.portfolio_return = pd.Series(0.0, index=self.dates)
        self.pnl = pd.Series(dtype=float, index=self.dates)
        self.account_value = pd.Series(dtype=float, index=self.dates)
        self.weights = pd.DataFrame(index=self.dates, columns=self.assets, dtype=float)
        self.costs = pd.Series(dtype=float, index=self.dates)

    def _is_rebalance_day(self, date_t: pd.Timestamp) -> bool:
        """Determines if a rebalance should occur on the given date."""

        freq = getattr(self.config, 'rebal_freq', 'daily')

        if freq is None or freq.lower() in ['daily', '1d', 'd']:
            return True
        if freq.lower() in self.freq_map:
            return date_t.dayofweek == self.freq_map[freq.lower()]
        if freq.lower() == '15th':
            return date_t.day == 15
        if freq.lower() == 'month_end':
            if date_t == self.dates[-1]:
                return True
            try:
                next_date = self.dates[self.dates.get_loc(date_t) + 1]
                return date_t.month != next_date.month
            except IndexError:
                return True
        if freq.lower() == 'month_start':
            return date_t.day == 1
        if isinstance(freq, int) and freq > 0:
            start_index = self.dates.get_loc(self.dates[self.config.optimizer.window_size])
            current_index = self.dates.get_loc(date_t)
            return (current_index - start_index) % freq == 0
        else:
            print(f"Warning: Unsupported rebalancing frequency '{freq}'. Defaulting to daily.")
            return True

    def run_backtest(self) -> Dict[str, pd.DataFrame]:
        """
        The main simulation loop.
        """
        self._compute_pipeline()
        self._validate_pipeline()
        self._extract_features()
        self._initialize_state()

        # Determine minimum required lookback data for a clean start
        min_start_index = getattr(self.config.optimizer, 'window_size', 1)

        if len(self.dates) <= min_start_index:
            print("Error: Not enough data for the required lookback window.")
            return {}

        trading_dates = self.dates[min_start_index:]

        print(
            f"Starting simulation from {trading_dates[0].strftime('%Y-%m-%d')} to "
            f"{trading_dates[-1].strftime('%Y-%m-%d')}...")

        # iteration from the first trading day
        for i, date_t in enumerate(trading_dates, start=min_start_index):
            date_t_minus_1 = self.dates[i - 1]  # last period

            # --- START OF TRADING DAY T ---
            # signals at S_t-1
            if date_t_minus_1 not in self.signals.index:
                raise KeyError(f"Signal not available for previous date: {date_t_minus_1.strftime('%Y-%m-%d')}")
            signal_series = self.signals.loc[date_t_minus_1]

            # weights at W_t-1
            if self.current_weights is None:
                self.current_weights = pd.Series(0.0, index=signal_series.index)

            try:
                # --- REBALANCING at the OPEN of day T ---
                rebalancer = Rebalancer(self.config)

                # lookback returns for risk estimators at time t-1
                fit_start_index = i - self.config.optimizer.window_size
                lookback_returns = self.returns.loc[self.dates[fit_start_index]:date_t_minus_1]

                # portfolio optimization: target weights W*_t for signals at S_t-1 and current weights W*_t-1
                # execution costs based on market value at the close of t-1 (or open of t)
                if rebalancer.is_rebalance_day(date_t, self.dates):
                    target_weights, transaction_cost = rebalancer.rebalance(lookback_returns,
                                                                            signal_series,
                                                                            self.current_weights,
                                                                            self.prices.loc[date_t_minus_1],
                                                                            self.market_value)
                else:
                    target_weights = self.current_weights.copy()
                    transaction_cost = 0.0

                # update market value for transaction cost
                self.market_value -= transaction_cost
                # update current weights to target weights
                self.current_weights = target_weights

                # --- END of TRADING DAY T ---
                # Returns over the current period (t-1 close to t close)
                period_returns = self.returns.loc[date_t] - self.funding_rate.loc[date_t]

                # PNL for period t, over the period (t-1 to t)
                portfolio_return = (self.current_weights * period_returns).sum()
                pnl = self.market_value * portfolio_return
                self.market_value += pnl

                # update weights for drift in market value
                # W_drift,t = W_t-1 * (1 + R_t) / (1 + R_p,t)
                self.current_weights = ((self.current_weights * (1 + period_returns)) / (1 + portfolio_return))

                # --- record state ---
                self.portfolio_return.loc[date_t] = portfolio_return
                self.pnl.loc[date_t] = pnl
                self.account_value.loc[date_t] = self.market_value
                self.weights.loc[date_t] = self.current_weights
                self.costs.loc[date_t] = transaction_cost

            except Exception as e:
                # log the error but ensure the backtest doesn't crash entirely
                print(f"\nCritical error on date {date_t}: {e}. Skipping period. State carried forward.")
                portfolio_return = 0.0
                self.account_value.loc[date_t] = self.market_value
                self.weights.loc[date_t] = self.current_weights
                self.portfolio_return.loc[date_t] = portfolio_return
                self.pnl.loc[date_t] = 0.0
                self.costs.loc[date_t] = 0.0

            # console status update
            if self.verbose:
                print(
                    f'Date: {date_t.strftime("%Y-%m-%d")} | '
                    f'Return: {portfolio_return:.4f} | '
                    f'Value: ${self.market_value:,.2f}',
                    end='\r')

        # store final results
        self.results['account_value'] = self.account_value.dropna()
        self.results['pnl'] = self.pnl.dropna()
        self.results['weights'] = self.weights.dropna(how='all')
        self.results['t-costs'] = self.costs.dropna()

        self.gross_exposure = self.weights.abs().sum(axis=1)
        self.net_exposure = self.weights.sum(axis=1)

        print("\n\nBacktest finished successfully. Summary:")

        return self.results

    def get_summary(self):
        """Calculates and prints key performance metrics."""
        if 'account_value' in self.results and not self.results['account_value'].empty:
            # Calculation of daily returns from value history
            daily_returns = self.results['account_value'].pct_change().dropna()

            # Simple metrics placeholder
            total_return = (self.results['account_value'].iloc[-1] / self.initial_capital) - 1
            # Assuming 252 trading days per year
            annual_volatility = daily_returns.std() * np.sqrt(self.ann_factor)
            # Assuming zero risk-free rate for simplicity
            sharpe_ratio = (daily_returns.mean() * self.ann_factor) / annual_volatility

            print("\n--- Backtest Summary ---")
            print(f"Strategy: {self.config.name}")
            print(f"Total Return: {total_return:.2%}")
            print(f'Annualized Return: {(daily_returns.mean() * self.ann_factor):.2%}')
            print(f"Annualized Volatility: {annual_volatility:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Total Periods Simulated: {len(self.results['account_value'])}")
            print("------------------------")
