import pandas as pd
from typing import List, Tuple, Union, Optional
from factorlab.core.base_transform import BaseTransform  # Assuming BaseTransform path


class Pipeline:
    """
    Sequentially applies a list of transformers to the input data.

    This class adopts the Scikit-learn API (fit/transform) for chaining
    transformations, ensuring proper data flow and preventing look-ahead bias.
    The output of one step becomes the input of the next step.

    Parameters
    ----------
    steps : List[Tuple[str, BaseTransform]]
        List of (name, transformer) tuples that are applied sequentially.
        All transformers must inherit from BaseTransform and implement
        fit() and transform().
    """

    def __init__(self, steps: List[Tuple[str, BaseTransform]]):
        self.steps = steps
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step.")

        # Ensure all steps are BaseTransform instances
        for name, transformer in self.steps:
            if not hasattr(transformer, 'fit') or not hasattr(transformer, 'transform'):
                raise TypeError(f"Step '{name}' is not a valid transformer (missing fit/transform methods).")

    def fit(self, X: Union[pd.Series, pd.DataFrame], y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> 'Pipeline':
        """
        Fit all steps in the pipeline sequentially.

        Data is transformed and passed to the next component during the fitting process.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input training data.
        y : Optional[Union[pd.Series, pd.DataFrame]], optional
            Optional target data.

        Returns
        -------
        self : Pipeline
            The fitted pipeline instance.
        """
        Xt = X.copy(deep=True)

        # Iterate through all steps
        for i, (name, transformer) in enumerate(self.steps):
            print(f"Pipeline: Fitting step {i + 1}/{len(self.steps)}: {name} ({transformer.__class__.__name__})")

            # 1. Fit the current transformer on the (previously transformed) data Xt
            transformer.fit(Xt, y)

            # 2. Immediately transform the data for the NEXT component's fit step
            # This is crucial for chaining learning (e.g., fitting a Scaler on transformed returns).
            if i < len(self.steps):
                Xt = transformer.transform(Xt)

        return self

    def transform(self, X: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply transformation to the data by running all steps sequentially.

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data to transform.

        Returns
        -------
        pd.DataFrame
            The final transformed data.
        """
        Xt = X.copy(deep=True)

        # Apply transform for all steps
        for name, transformer in self.steps:
            Xt = transformer.transform(Xt)

        return Xt

    def fit_transform(self, X: Union[pd.Series, pd.DataFrame],
                      y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> pd.DataFrame:
        """
        Fit all steps and then transform the input data X.

        This is a convenience method that combines fit() and transform(X).

        Parameters
        ----------
        X : Union[pd.Series, pd.DataFrame]
            Input data to fit and transform.
        y : Optional[Union[pd.Series, pd.DataFrame]], optional
            Optional target data passed to fit().

        Returns
        -------
        pd.DataFrame
            The final transformed data.
        """
        # Fit the pipeline first
        self.fit(X, y)

        # Then transform the original data X using the newly fitted pipeline
        return self.transform(X)
