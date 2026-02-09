import pandas as pd
import numpy as np
from typing import Optional, Any, Union
from copy import deepcopy
from tqdm import tqdm
from abc import abstractmethod

from factorlab.core.base_transform import BaseTransform


class WindowTransform(BaseTransform):
    """
    Base class for time-windowed transformations.

    This wrapper enforces a point-in-time contract by fitting on a window
    and transforming only the current (and optionally forward-filled) rows.
    """

    def __init__(self, base_transform: BaseTransform, step: int = 1, show_progress: bool = True):
        super().__init__()
        self.base_transform = base_transform
        self.step = step
        self.show_progress = show_progress
        self._prev_fitted_params = None

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Any = None) -> 'WindowTransform':
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Subclasses must implement rolling/expanding transform behavior."""
        raise NotImplementedError

    def _apply_window_logic(self,
                            X: pd.DataFrame,
                            is_rolling: bool,
                            window_size: Optional[int] = None) -> pd.DataFrame:
        results = []
        expl_var_frames = []
        align_frames = []
        eig_frames = []
        n_obs = len(X)
        start_idx = window_size if is_rolling else 1

        total_steps = range(start_idx, n_obs + 1, self.step)
        pbar = tqdm(total_steps, desc="Window Computation", disable=not self.show_progress)

        for i in pbar:
            if is_rolling:
                window = X.iloc[i - window_size: i]
            else:
                window = X.iloc[:i]

            self.base_transform.fit(window)

            if self._prev_fitted_params is not None:
                self.base_transform.align(self._prev_fitted_params)

            fill_until = min(i + self.step, n_obs + 1)
            target_slice = X.iloc[i - 1: fill_until - 1]
            res = self.base_transform.transform(target_slice)
            results.append(res)

            if hasattr(self.base_transform, "_window_outputs"):
                wo = self.base_transform._window_outputs
                target_k = wo['target_k']
                universe_cols = wo['universe_cols']
                pc_cols = [f"PC{k + 1}" for k in range(target_k)]
                ev_cols = [f"EV{k + 1}" for k in range(target_k)]

                n_rows = len(target_slice)
                evr = np.tile(wo['expl_var_ratio'], (n_rows, 1))
                expl_var_frames.append(pd.DataFrame(evr, index=target_slice.index, columns=pc_cols))

                aln = np.tile(wo['alignment_scores'], (n_rows, 1))
                align_frames.append(pd.DataFrame(aln, index=target_slice.index, columns=pc_cols))

                ev_full = wo['eigenvectors_full']
                ev_rep = np.vstack([ev_full for _ in range(n_rows)])
                ev_index = pd.MultiIndex.from_product([target_slice.index, universe_cols], names=['date', 'ticker'])
                eig_frames.append(pd.DataFrame(ev_rep, index=ev_index, columns=ev_cols))

            self._prev_fitted_params = deepcopy(self.base_transform._fitted_params)

        pcs = pd.concat(results)

        if expl_var_frames:
            self.expl_var_ratio = pd.concat(expl_var_frames)
            self.alignment_scores = pd.concat(align_frames)
            self.eigenvecs = pd.concat(eig_frames).sort_index()

            self.base_transform.expl_var_ratio = self.expl_var_ratio
            self.base_transform.alignment_scores = self.alignment_scores
            self.base_transform.eigenvecs = self.eigenvecs

        self.pcs = pcs
        self.base_transform.pcs = pcs

        return pcs


class RollingTransform(WindowTransform):
    """Applies a transformation over a sliding window of fixed size."""

    def __init__(self, base_transform: BaseTransform, window: int, step: int = 1, show_progress: bool = True):
        super().__init__(base_transform, step, show_progress)
        self.window = window

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._apply_window_logic(X, is_rolling=True, window_size=self.window)


class ExpandingTransform(WindowTransform):
    """Applies a transformation over a window that grows with time."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._apply_window_logic(X, is_rolling=False)
