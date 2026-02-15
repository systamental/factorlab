from typing import Dict

from factorlab.portfolio.optimization.base import PortfolioOptimizerBase
from factorlab.portfolio.optimization.equal_weighted import EqualWeighted
from factorlab.portfolio.optimization.inverse_vol import InverseVolatility
from factorlab.portfolio.optimization.inverse_variance import InverseVariance
from factorlab.portfolio.optimization.signal_weighted import SignalWeighted
from factorlab.portfolio.optimization.min_vol import MinVolOptimizer
from factorlab.portfolio.optimization.risk_parity import RiskParity
from factorlab.portfolio.optimization.max_diversification import MaxDiversification
from factorlab.portfolio.optimization.mvo import MeanVarianceOptimizer


class PortfolioOptimizer:
    """
    Factory class to centralize the creation and configuration of all
    PortfolioOptimizerBase implementations.
    """

    # Maps a user-friendly string to the concrete class
    _OPTIMIZER_MAP: Dict[str, type(PortfolioOptimizerBase)] = {
        "equal_weighted": EqualWeighted,
        'inverse_volatility': InverseVolatility,
        'inverse_variance': InverseVariance,
        'signal_weighted': SignalWeighted,
        "min_vol": MinVolOptimizer,
        "risk_parity": RiskParity,
        "max_diversification": MaxDiversification,
        "mean_variance": MeanVarianceOptimizer,
    }

    @classmethod
    def get_optimizer_metadata(cls) -> Dict[str, str]:
        """
        Returns a dictionary of available optimizers and their descriptions.
        """
        return list(cls._OPTIMIZER_MAP.keys())

    @classmethod
    def create_optimizer(cls, method: str, **kwargs) -> PortfolioOptimizerBase:
        """
        Factory method to instantiate a PortfolioOptimizerBase subclass
        based on the provided method name and parameters.

        Parameters
        ----------
        method : str
            The name of the optimizer method to instantiate (e.g., 'risk_parity').
        **kwargs : Any
            Additional keyword arguments (e.g., 'window_size', 'risk_target')
            to pass to the optimizer constructor.

        Returns
        -------
        PortfolioOptimizerBase
            An initialized instance of the specified optimizer.

        Raises
        ------
        ValueError
            If the specified method is not recognized.
        """
        if method not in cls._OPTIMIZER_MAP:
            raise ValueError(f"Invalid optimizer method '{method}'. Must be one of {list(cls._OPTIMIZER_MAP.keys())}")

        optimizer_class = cls._OPTIMIZER_MAP[method]
        return optimizer_class(**kwargs)