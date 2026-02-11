import numpy as np
import pandas as pd
from typing import Dict


class WindowOutputCollector:
    """
    Base collector for windowed outputs.
    """

    def reset(self) -> None:
        return

    def append(self, output: Dict, target_index: pd.Index) -> None:
        raise NotImplementedError

    def finalize(self) -> Dict:
        return {}


class PCAWindowOutputCollector(WindowOutputCollector):
    """
    Collector for PCA window outputs: expl_var_ratio, alignment_scores, eigenvecs.
    """

    def __init__(self):
        self._expl_var_frames = []
        self._align_frames = []
        self._eig_frames = []

    def reset(self) -> None:
        self._expl_var_frames = []
        self._align_frames = []
        self._eig_frames = []

    def append(self, output: Dict, target_index: pd.Index) -> None:
        target_k = output["target_k"]
        universe_cols = output["universe_cols"]
        pc_cols = [f"PC{k + 1}" for k in range(target_k)]
        ev_cols = [f"EV{k + 1}" for k in range(target_k)]

        n_rows = len(target_index)
        evr = np.tile(output["expl_var_ratio"], (n_rows, 1))
        self._expl_var_frames.append(pd.DataFrame(evr, index=target_index, columns=pc_cols))

        aln = np.tile(output["alignment_scores"], (n_rows, 1))
        self._align_frames.append(pd.DataFrame(aln, index=target_index, columns=pc_cols))

        ev_full = output["eigenvectors_full"]
        ev_rep = np.vstack([ev_full for _ in range(n_rows)])
        ev_index = pd.MultiIndex.from_product([target_index, universe_cols], names=["date", "ticker"])
        self._eig_frames.append(pd.DataFrame(ev_rep, index=ev_index, columns=ev_cols))

    def finalize(self) -> Dict:
        if not self._expl_var_frames:
            return {}
        return {
            "expl_var_ratio": pd.concat(self._expl_var_frames),
            "alignment_scores": pd.concat(self._align_frames),
            "eigenvecs": pd.concat(self._eig_frames).sort_index(),
        }
