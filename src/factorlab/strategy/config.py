from factorlab.core.pipeline import Pipeline
from factorlab.portfolio_opt.base import PortfolioOptimizerBase
from factorlab.cost_model.base import CostModelBase
from typing import Union


class StrategyConfig:
    """
    Configuration container for a specific trading strategy.
    It defines the combination of research components (Signal, Optimizer)
    and the execution parameters (rebalance frequency).

    This class serves as the blueprint, isolating the definition of the
    strategy from the mechanics of the simulation loop.
    """

    def __init__(self,
                 name: str,
                 data_pipeline: Pipeline,
                 optimizer: PortfolioOptimizerBase,
                 cost_model: CostModelBase,
                 rebal_freq: Union[str, int] = 'd'):
        """
        Parameters
        ----------
        name : str
            A descriptive name for the strategy (e.g., "MomentumInverseVol").
        data_pipeline : Pipeline
            The instance responsible for generating the data pipeline.
        optimizer : PortfolioOptimizerBase
            The instance responsible for translating signals into weights.
        cost_model : CostModelBase
            The instance responsible for calculating transaction costs.
        rebal_freq : str or int
            The frequency at which the portfolio weights are recalculated.
            Can be a pandas offset alias (e.g., 'd' for daily, 'w' for weekly)
            or an integer representing the number of days between rebalances.

        """
        self.name = name
        self.data_pipeline = data_pipeline
        self.optimizer = optimizer
        self.cost_model = cost_model
        self.rebal_freq = rebal_freq

        # add other global constraints here, like max_leverage,
        # max_turnover, etc., as the strategy becomes more complex.

        print(f"Strategy Config '{self.name}' created.")
