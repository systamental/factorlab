from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import pandas as pd

from factorlab.core.pipeline import Pipeline
from factorlab.core.walk_forward_runner import WalkForwardRunner
from factorlab.learning.splitters import BasePanelSplit
from factorlab.targets.forward import ForwardTargetSpec


def parameter_grid(parameters: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """
    Expand a parameter grid into a list of parameter combinations.

    Parameters are specified using sklearn-style keys:
    ``<step_name>__<attribute>``.
    """
    if not parameters:
        return [{}]
    keys = list(parameters.keys())
    values = [list(parameters[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def set_pipeline_params(pipeline: Pipeline, params: Mapping[str, Any]) -> None:
    """
    Apply sklearn-style step parameters in-place to a Pipeline copy.
    """
    steps = dict(pipeline.steps)
    for key, value in params.items():
        if "__" not in key:
            raise ValueError(f"Invalid param key '{key}'. Use '<step_name>__<attribute>'.")
        step_name, attr = key.split("__", 1)
        if step_name not in steps:
            raise ValueError(f"Unknown pipeline step '{step_name}' in param key '{key}'.")
        setattr(steps[step_name], attr, value)


class WalkForwardGridSearch:
    """
    Pipeline-native parameter search using walk-forward evaluation.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        splitter: BasePanelSplit,
        param_grid: Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]],
        scorer: Callable[[pd.DataFrame], float],
        y: Optional[Union[pd.Series, pd.DataFrame]] = None,
        target_spec: Optional[ForwardTargetSpec] = None,
        strict_temporal: bool = True,
        date_level: int = 0,
        show_progress: bool = False,
    ):
        self.pipeline = pipeline
        self.splitter = splitter
        self.param_grid = (
            parameter_grid(param_grid)
            if isinstance(param_grid, Mapping)
            else [dict(p) for p in param_grid]
        )
        self.scorer = scorer
        self.y = y
        self.target_spec = target_spec
        self.strict_temporal = strict_temporal
        self.date_level = date_level
        self.show_progress = show_progress
        self.results_: Optional[pd.DataFrame] = None

    def run(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("WalkForwardGridSearch expects X as a pandas DataFrame.")

        records: list[dict[str, Any]] = []
        for params in self.param_grid:
            pipe = deepcopy(self.pipeline)
            set_pipeline_params(pipe, params)
            runner = WalkForwardRunner(
                pipeline=pipe,
                splitter=self.splitter,
                y=self.y,
                target_spec=self.target_spec,
                strict_temporal=self.strict_temporal,
                date_level=self.date_level,
                show_progress=self.show_progress,
            )
            out = runner.run(X)
            score = float(self.scorer(out))
            record = dict(params)
            record["score"] = score
            record["n_folds"] = len(runner.fold_info)
            records.append(record)

        results = pd.DataFrame(records).sort_values("score", ascending=False).reset_index(drop=True)
        self.results_ = results
        return results
