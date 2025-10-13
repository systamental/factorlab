import pandas as pd
import numpy as np
from typing import Dict

from factorlab.strategy.config import StrategyConfig


class BacktesterEngine:
    """
    The main execution engine for running a time-series simulation (backtest).

    It consumes a StrategyConfig object and iterates over historical data,
    managing the simulation clock, rebalancing, and P&L accounting.
    """

    def __init__(self,
                 config: StrategyConfig,
                 data: pd.DataFrame,
                 initial_capital: float = 1_000_000.0):
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
        """

        self.config = config
        self.data = data
        self.initial_capital = initial_capital

        # internal state
        self.pipeline = None
        self.returns = None
        self.prices = None
        self.signals = None
        self.current_weights = None
        self.total_returns = None
        self.portfolio_return = None
        self.results: Dict[str, pd.DataFrame] = {}

        print(f"Engine ready for strategy: {self.config.name}")

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

        print("--- Data pipeline computation complete. ---")

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
        self.pnl_history = pd.Series(dtype=float, index=self.dates)
        self.value_history = pd.Series(dtype=float, index=self.dates)
        self.weights_history = pd.DataFrame(index=self.dates, columns=self.assets, dtype=float)
        self.cost_history = pd.Series(dtype=float, index=self.dates)

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

            try:
                # Returns over the current period (t-1 close to t close)
                period_returns = self.returns.loc[date_t]  # - self.funding_rate.loc[date_t]

                # retrieve signal based on data up to date_t_minus_1
                if date_t_minus_1 not in self.signals.index:
                    raise KeyError(f"Signal not available for previous date: {date_t_minus_1.strftime('%Y-%m-%d')}")

                # signal_series vector S_t
                signal_series = self.signals.loc[date_t_minus_1]

                if self.current_weights is None:
                    self.current_weights = pd.Series(0.0, index=signal_series.index)

                # Portfolio Optimization
                # risk estimators using historical lookback
                fit_start_index = i - self.config.optimizer.window_size
                lookback_returns = self.returns.loc[self.dates[fit_start_index]:date_t_minus_1]
                self.config.optimizer.fit(lookback_returns)

                # compute W*_t using S_t and previous weights (W*_t-1)
                target_weights = self.config.optimizer.transform(
                    signal_series,
                    self.current_weights
                )

                # execution costs
                transaction_cost = self.config.cost_model.compute_cost(
                    current_weights=self.current_weights,
                    target_weights=target_weights,
                    prices=self.prices.loc[date_t_minus_1],
                    portfolio_value=self.market_value
                )

                # update state with new weights and cost
                self.market_value -= transaction_cost
                self.current_weights = target_weights

                # P&L for period
                # portfolio return earned over period (t-1 to t)
                portfolio_return = (self.current_weights * period_returns).sum()
                pnl = self.market_value * portfolio_return
                self.market_value += pnl

                # --- record state ---
                self.portfolio_return.loc[date_t] = portfolio_return
                self.pnl_history.loc[date_t] = pnl
                self.value_history.loc[date_t] = self.market_value
                self.weights_history.loc[date_t] = target_weights
                self.cost_history.loc[date_t] = transaction_cost

            except Exception as e:
                # log the error but ensure the backtest doesn't crash entirely
                print(f"\nCritical error on date {date_t}: {e}. Skipping period. State carried forward.")
                self.value_history.loc[date_t] = self.market_value
                self.weights_history.loc[date_t] = self.current_weights
                self.portfolio_return.loc[date_t] = 0.0
                self.pnl_history.loc[date_t] = 0.0
                self.cost_history.loc[date_t] = 0.0

            # console status update
            print(
                f"Date: {date_t.strftime('%Y-%m-%d')} | "
                f"Return: {portfolio_return:.4f} | "
                f"Value: ${self.market_value:,.2f}",
                end='\r')

        # store final results
        self.results['value_history'] = self.value_history.dropna()
        self.results['pnl_history'] = self.pnl_history.dropna()
        self.results['weights_history'] = self.weights_history.dropna(how='all')
        self.results['cost_history'] = self.cost_history.dropna()

        print("\n\nBacktest finished successfully. Summary:")

        return self.results

    def get_summary(self):
        """Calculates and prints key performance metrics."""
        if 'value_history' in self.results and not self.results['value_history'].empty:
            # Calculation of daily returns from value history
            daily_returns = self.results['value_history'].pct_change().dropna()

            # Simple metrics placeholder
            total_return = (self.results['value_history'].iloc[-1] / self.initial_capital) - 1
            # Assuming 252 trading days per year
            annual_volatility = daily_returns.std() * np.sqrt(365)
            # Assuming zero risk-free rate for simplicity
            sharpe_ratio = (daily_returns.mean() * 365) / annual_volatility

            print("\n--- Backtest Summary ---")
            print(f"Strategy: {self.config.name}")
            print(f"Total Return: {total_return:.2%}")
            print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Total Periods Simulated: {len(self.results['value_history'])}")
            print("------------------------")
