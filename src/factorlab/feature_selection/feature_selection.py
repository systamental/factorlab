import pandas as pd
import numpy as np
from typing import Optional, Union, Any
# from scipy.stats import chi2_contingency, spearmanr, kendalltau, contingency
# from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoLarsIC, Lasso, Ridge, ElasticNet, Lars
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from factorlab.feature_analysis.time_series_analysis import add_lags


class FeatureSelection:
    """
    Feature selection methods for any set of features and target.
    """
    def __init__(self,
                 target: Union[pd.Series, pd.DataFrame, np.array],
                 features: Union[pd.DataFrame, np.array],
                 n_feat: Optional[int] = None,
                 n_lookahead: int = 1,
                 n_lags: int = 4,
                 ):
        """
        Initialize FeatureSelection object.

        Parameters
        ----------
        target: pd.Series, pd.DataFrame or np.array
            Series of target variable.
        features: pd.DataFrame or np.array
            Dataframe of features.
        n_feat: int, default None
            Number of features to select.
        n_lookahead: int, default 1
            Number of periods to lookahead for the target variable.
        n_lags: int, default 4
            Number of lags to add to the target variable and features.
        """
        self.target = target
        self.features = features
        self.n_feat = n_feat
        self.n_lookahead = n_lookahead
        self.n_lags = n_lags
        self.lagged_target = None
        self.data = self.preprocess_data()  # pre-process data, target and features attributes
        self.index = self.data.index
        self.ranked_features = None
        self.ranked_features_list = None
        self.selected_features = None
        self.feature_importance = None
        self.check_n_feat()

    def check_n_feat(self) -> None:
        """
        Check if number of features to select is specified.

        Returns
        -------
        None
        """
        if self.n_feat is None:
            self.n_feat = self.features.shape[1]
        elif self.n_feat > self.features.shape[1]:
            raise ValueError("Number of features to select must be less than or equal to the number of features.")

    def preprocess_data(self) -> Union[pd.DataFrame, np.array]:
        """
        Pre-process data, target and features attributes.

        Returns
        -------
        data: pd.DataFrame or np.array
            Data matrix.
        """
        if isinstance(self.target, pd.Series) and isinstance(self.features, pd.DataFrame):
            if self.n_lags > 0:
                self.lagged_target = add_lags(self.target, self.n_lags)
            self.target = self.target.shift(-self.n_lookahead).rename(f"{self.target.name}_F{self.n_lookahead}")
            self.data = pd.concat([self.target, self.lagged_target, self.features], axis=1).dropna()
            self.target = self.data.iloc[:, 0]
            self.features = self.data.iloc[:, 1:]
        else:
            raise TypeError("Target and features must be a pandas Series or DataFrame.")

        return self.data

    def stepwise(self) -> np.array:
        """
        Stepwise supervised learning feature selection.

        Returns
        -------

        """
        pass

    def backward(self) -> np.array:
        """
        Backward supervised learning feature selection.

        Returns
        -------

        """
        pass

    def forward(self) -> np.array:
        """
        Forward supervised learning feature selection.

        Returns
        -------

        """
        pass

    def exhaustive(self) -> np.array:
        """
        Exhaustive supervised learning feature selection.

        Returns
        -------

        """
        pass

    def mutual_info(self) -> np.array:
        """
        Mutual information supervised learning feature selection.

        Returns
        -------

        """
        pass

    def chi2(self) -> np.array:
        """
        Chi-squared supervised learning feature selection.

        Returns
        -------

        """
        pass

    def spearman(self) -> np.array:
        """
        Spearman supervised learning feature selection.

        Returns
        -------

        """
        pass

    def kendall(self) -> np.array:
        """
        Kendall supervised learning feature selection.

        Returns
        -------

        """
        pass

    def get_selected_features(self, model: Any, attr: str) -> Union[np.array, pd.DataFrame]:
        """
        Get selected features from a model.

        Parameters
        ----------
        model: Any
            Model to get selected features from.
        attr: str
            Attribute of the model to get selected features from.

        Returns
        -------
        selected_features: np.array or pd.DataFrame
            Selected features from the model.
        """
        # coefficients or feature importances
        if not hasattr(model, attr):
            raise AttributeError(f"Model must have attribute {attr}.")
        if not isinstance(getattr(model, attr), np.ndarray):
            raise TypeError(f"Attribute {attr} must be a numpy array.")
        coef = getattr(model, attr)

        # feature importance
        sorted_coef_idxs = np.argsort(np.abs(coef))[::-1]
        sorted_coefs = coef[sorted_coef_idxs]
        self.feature_importance = sorted_coefs[sorted_coefs != 0]

        # ranked features
        self.ranked_features = self.features.iloc[:, sorted_coef_idxs].copy()
        self.ranked_features_list = self.ranked_features.columns.tolist()
        self.feature_importance = pd.DataFrame(self.feature_importance,
                                               index=self.ranked_features_list[: len(self.feature_importance)],
                                               columns=['feature_importance'])
        self.ranked_features.drop(columns=self.lagged_target.columns, inplace=True, errors='ignore')
        self.feature_importance.drop(index=self.lagged_target.columns, inplace=True, errors='ignore')
        self.selected_features = self.ranked_features.iloc[:, :len(self.feature_importance)].iloc[:, :self.n_feat]

        return self.selected_features

    def lars(self, **kwargs) -> pd.DataFrame:
        """
        Least Angle Regression (LARS) supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html

        Parameters
        ----------
        kwargs: dict

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the LARS regression.
        """
        # fit lars model
        lars = Lars(n_nonzero_coefs=self.n_feat, normalize=False, **kwargs)
        lars.fit(self.features, self.target)

        # selected features
        self.get_selected_features(lars, 'coef_')

        return self.selected_features

    def lasso(self, alpha: float = 0.05, auto_selection: bool = True, criterion: str = 'aic', **kwargs) -> pd.DataFrame:
        """
        LASSO supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html

        Parameters
        ----------
        alpha: float, default 0.05
            Constant that multiplies the L1 regularization term. Alpha = 0 is equivalent to an OLS regression.
        auto_selection: bool, default True
            Lasso model fit with Lars using BIC or AIC for model selection.
        criterion: str, {'aic', 'bic'}, default 'aic'
            AIC is the Akaike information criterion and BIC is the Bayes Information criterion. Such criteria are useful
            to select the value of the regularization parameter by making a trade-off between the goodness of fit and
            the complexity of the model. A good model should explain the data well while being simple.\
        **kwargs: dict
            Additional keyword arguments to pass to the Lasso or LassoLarsIC model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the LASSO regression.
        """
        # fit lasso model
        if auto_selection and self.features.shape[0] > self.features.shape[1]:
            lasso = LassoLarsIC(criterion=criterion, normalize=False, **kwargs)  # auto selection
        else:
            lasso = Lasso(alpha=alpha, **kwargs)
        lasso.fit(self.features, self.target)

        # selected features
        self.get_selected_features(lasso, 'coef_')

        return self.selected_features

    def ridge(self, alpha: float = 1, **kwargs) -> pd.DataFrame:
        """
        Ridge supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

        Parameters
        ----------
        alpha: float, default 1
            Constant that multiplies the L2 regularization term. Alpha = 0 is equivalent to an OLS regression.
        **kwargs: dict
            Additional keyword arguments to pass to the Ridge model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the Ridge regression.
        """
        # fit ridge model
        ridge = Ridge(alpha=alpha, **kwargs)
        ridge.fit(self.features, self.target)

        # selected features
        self.get_selected_features(ridge, 'coef_')

        return self.selected_features

    def elastic_net(self, alpha: float = 0.05, l1_ratio: float = 0.5, **kwargs) -> pd.DataFrame:
        """
        Elastic Net supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html

        Parameters
        ----------
        alpha: float, default 1
            Constant that multiplies the L1 regularization term. Alpha = 0 is equivalent to an OLS regression.
        l1_ratio: float, default 0.5
            The ElasticNet mixing parameter, with 0 <= l1_ratio <= 1. For l1_ratio = 0 the penalty is an L2 penalty.
            For l1_ratio = 1 it is an L1 penalty. For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.
        **kwargs: dict
            Additional keyword arguments to pass to the ElasticNet model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the Elastic Net regression.
        """
        # fit elastic net model
        elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, **kwargs)
        elastic_net.fit(self.features, self.target)

        # selected features
        self.get_selected_features(elastic_net, 'coef_')

        return self.selected_features

    def random_forest(self, n_estimators: int = 100, criterion: str = 'squared_error', max_depth: int = 5,
                      **kwargs) -> pd.DataFrame:
        """
        Random Forest supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

        Parameters
        ----------
        n_estimators: int, default 100
            The number of trees in the forest.
        criterion: str, {“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default ”squared_error”
            The function to measure the quality of a split. Supported criteria are “squared_error” for regression,
            which devolves to mean squared error (mse), and “poisson” which uses reduction in Poisson deviance
            to find splits.
        max_depth: int, default 5
            The maximum depth of the tree.
        **kwargs: dict
            Additional keyword arguments to pass to the Random Forest model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the Random Forest regression.
        """
        # fit random forest model
        rf = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, **kwargs)
        rf.fit(self.features, self.target)

        # selected features
        self.get_selected_features(rf, 'feature_importances_')

        return self.selected_features[:]

    def xgboost(self, n_estimators: int = 100, max_depth: int = 5, **kwargs) -> pd.DataFrame:
        """
        XGBoost supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

        Parameters
        ----------
        n_estimators: int, default 100
            The number of trees in the forest.
        max_depth: int, default 5
            The maximum depth of the tree.
        **kwargs: dict
            Additional keyword arguments to pass to the XGBoost model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the XGBoost regression.
        """
        # fit xgboost model
        xgb = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, **kwargs)
        xgb.fit(self.features, self.target)

        # selected features
        self.get_selected_features(xgb, 'feature_importances_')

        return self.selected_features

    def catboost(self, loss_function='RMSE', iterations=2, depth=2, learning_rate=1, **kwargs) -> pd.DataFrame:
        """
        CatBoost supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        See scitkit-learn documentation for more details:
        https://catboost.ai/docs/concepts/python-reference_catboostregressor.html

        Parameters
        ----------
        loss_function: str, default 'RMSE'
            The loss function to be optimized.
        iterations: int, default 2
            The number of trees in the forest.
        depth: int, default 2
            Depth of the trees.
        learning_rate: float, default 1
            The learning rate.

        **kwargs: dict
            Additional keyword arguments to pass to the CatBoost model.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the CatBoost regression.
        """
        pass

    def mrmr(self, n_features: int = 10) -> pd.DataFrame:
        """
        Minimum Redundancy Maximum Relevance (mRMR) unsupervised feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.
        The mRMR method is an unsupervised feature selection method that selects features with maximum relevance
        to the target and minimum redundancy to the features already selected.

        Parameters
        ----------
        n_features: int, default 10
            Number of features to select.

        Returns
        -------
        selected_features: list
            List of the subset of selected features from the mRMR feature selection.
        """
        pass

    def mifs(self, n_features: int = 10) -> pd.DataFrame:
        """
        Mutual Information Feature Selection (MIFS) unsupervised feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.
        The MIFS method is an unsupervised feature selection method that selects features with maximum relevance
        to the target and minimum redundancy to the features already selected.

        Parameters
        ----------
        n_features: int, default 10
            Number of features to select.

        Returns
        -------
        selected_features: list
            List of the subset of selected features from the MIFS feature selection.
        """
        pass

    def mrmr_mifs(self, n_features: int = 10) -> pd.DataFrame:
        """
        Minimum Redundancy Maximum Relevance (mRMR) and Mutual Information Feature Selection (MIFS)
        unsupervised feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.
        The mRMR and MIFS method is an unsupervised feature selection method that selects features with
        maximum relevance to the target and minimum redundancy to the features already selected.

        Parameters
        ----------
        n_features: int, default 10
            Number of features to select.

        Returns
        -------
        selected_features: list
            List of the subset of selected features from the mRMR and MIFS feature selection.
        """
        pass

    def spearman_mrmr(self, n_features: int = 10) -> pd.DataFrame:
        """
        Spearman Minimum Redundancy Maximum Relevance (mRMR) unsupervised feature selection.

        Parameters
        ----------
        n_features

        Returns
        -------

        """
        pass
