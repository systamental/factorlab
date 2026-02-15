import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression, Lasso, LassoCV, LassoLars, LassoLarsCV, LassoLarsIC, Lars, LarsCV, \
    Ridge, RidgeCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from factorlab.signal_generation.unsupervised_learning import PCAWrapper
from factorlab.signal_generation.time_series_analysis import add_lags


class SuperviseLearning(ABC):
    """
    Abstract class for supervised learning.
    """
    def __init__(self,
                 target: Union[np.array, pd.Series, pd.DataFrame],
                 features: Union[np.array, pd.DataFrame],
                 method: str,
                 target_lookahead: int = 1,
                 feature_lags: int = 6
                 ):
        """
        Initialize SupervisedLearning object.

        Parameters
        ----------
        target: pd.Series, pd.DataFrame or np.ndarray
            Factor to select features for.
        features: pd.DataFrame or np.ndarray
            Features to select from. If None, target is used as the only feature.
        method: str
            Supervised learning model method to use.
        target_lookahead: int, default 1
            Number of look-ahead periods for target variable.
        feature_lags: int, default 4
            Number of lags to include for features.
        """
        self.target = target
        self.features = features
        self.method = method
        self.target_lookahead = target_lookahead
        self.feature_lags = feature_lags

    @abstractmethod
    def preprocess_data(self):
        """
        Pre-process data into a format suitable for supervised learning using scikit-learn.
        """
        # to be implemented by subclasses

    @abstractmethod
    def fit(self) -> pd.DataFrame:
        """
        Fit data.
        """
        # to be implemented by subclasses

    @abstractmethod
    def predict(self) -> pd.DataFrame:
        """
        Predict target.
        """
        # to be implemented by subclasses


class Regression(SuperviseLearning):
    """
    Regression model class.

    Wrapper for scikit-learn regression methods where the target value is expected to be a
    continuous variable (returns).

    See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model for details.

    See Also
    --------
    sklearn.linear_model.LinearRegression
    sklearn.linear_model.Lasso
    sklearn.linear_model.LassoCV
    sklearn.linear_model.LassoLars
    sklearn.linear_model.LassoLarsCV
    sklearn.linear_model.LassoLarsIC
    sklearn.linear_model.Lars
    sklearn.linear_model.LarsCV
    sklearn.linear_model.Ridge
    sklearn.linear_model.RidgeCV
    sklearn.linear_model.ElasticNet
    sklearn.linear_model.ElasticNetCV
    sklearn.ensemble.RandomForestRegressor
    xgboost.XGBRegressor
    """
    def __init__(self,
                 target: Union[np.array, pd.Series, pd.DataFrame],
                 features: Union[np.array, pd.DataFrame],
                 method: str,
                 target_lookahead: int = 1,
                 feature_lags: int = 6,
                 **kwargs: Any):
        """
        Initialize LinearRegression object.

        Parameters
        ----------
        target: pd.Series, pd.DataFrame or np.ndarray
            Factor to select features for.
        features: pd.DataFrame or np.ndarray
            Features to select from. If None, target is used as the only feature.
        method: str, {'ols', 'lasso', 'lasso_cv', 'lasso_lars', 'lasso_lars_cv', 'lasso_lars_ic', 'lars', 'lars_cv',
                        'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'random_forest', 'xgboost'}
            Linear model method to use.
        target_lookahead: int, default 1
            Number of look-ahead periods for target variable.
        feature_lags: int, default 4
            Number of lags to include for features.
        **kwargs: Optional keyword arguments, for model object. See sklearn.linear_model for details.
        """
        super().__init__(target, features, method, target_lookahead, feature_lags)
        self.target = target
        self.features = features
        self.method = method
        self.target_lookahead = target_lookahead
        self.feature_lags = feature_lags
        self.target_lags = None
        self.feature_lags = None
        self.predictors = None
        self.target_fcst = None
        self.features_window = None
        self.model = None
        self.yhat = None
        self.yhat_name = None
        self.score = None
        self.selected_features = None
        self.feature_importance = None
        self.data = self.preprocess_data()  # pre-process data, target and features attributes
        self.index = self.features.index
        self.kwargs = kwargs

    def preprocess_data(self) -> Union[pd.DataFrame, np.array]:
        """
        Pre-process data into a format suitable for supervised learning using scikit-learn.

        Returns
        -------
        data: pd.DataFrame or np.ndarray
            Data matrix.
        """
        if not isinstance(self.target, (pd.Series,  np.ndarray)):
            raise TypeError("Target must be a pandas Series or np.array.")
        elif isinstance(self.target, pd.Series) and (isinstance(self.features, pd.DataFrame) or self.features is None):
            if self.features is not None:
                self.feature_lags = add_lags(self.features, n_lags=self.feature_lags).copy()  # features + L lags
            self.target_lags = add_lags(self.target, n_lags=self.feature_lags).copy()  # target + L lags
            self.features = pd.concat([self.target_lags, self.feature_lags], axis=1).dropna().copy()  # features
            self.target_fcst = self.target.shift(-self.target_lookahead).\
                rename(f"{self.target.name}_F{self.target_lookahead}").copy()  # target forecast
            self.data = pd.concat([self.target_fcst, self.features], axis=1).dropna().copy()
            self.target_fcst = self.data.iloc[:, 0]
            self.predictors = self.data.iloc[:, 1:]
            self.features_window = self.features.copy()
        elif isinstance(self.target, np.ndarray) and isinstance(self.features, np.ndarray):
            n = min(self.target.shape[0], self.features.shape[0])
            self.data = np.concatenate([self.target[-n:].reshape(-1, 1), self.features[-n:]], axis=1)
            self.target = self.data[:, 0]
            self.features = self.data[:, 1:]
        else:
            raise TypeError("Target and features must be a pandas Series, DataFrame or np.array.")

        return self.data

    def fit(self) -> None:
        """
        Fit data.

        """
        #TODO: review and refactor
        # out of sample oos
        if self.oos:
            self.predictors = self.predictors.iloc[:-1]
            self.target_fcst = self.target_fcst.iloc[:-1]

        # model fit
        if self.method == 'ols':
            self.model = LinearRegression(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lasso':
            self.model = Lasso(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lasso_cv':
            self.model = LassoCV(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lasso_lars':
            self.model = LassoLars(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lasso_lars_cv':
            self.model = LassoLarsCV(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lasso_lars_ic':
            self.model = LassoLarsIC(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lars':
            self.model = Lars(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'lars_cv':
            self.model = LarsCV(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'ridge':
            self.model = Ridge(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'ridge_cv':
            self.model = RidgeCV(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'elastic_net':
            self.model = ElasticNet(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'elastic_net_cv':
            self.model = ElasticNetCV(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'random_forest':
            self.model = RandomForestRegressor(**self.kwargs).fit(self.predictors, self.target_fcst)
        elif self.method == 'xgboost':
            self.model = XGBRegressor(**self.kwargs).fit(self.predictors, self.target_fcst)
        # TODO add regression models

        return self.model

    def get_selected_features(self, drop_target: bool = False, drop_feat_lags: bool = False) -> np.array:
        """
        Get selected features.

        Parameters
        ----------
        drop_target: bool, default False
            Drop target and target lags from selected features.
        drop_feat_lags: bool, default False
            Drop feature lags from selected features.

        Returns
        -------
        selected_features: np.ndarray
            Selected features.
        """
        # fit
        self.fit()

        # feature importance
        if self.method in ['random_forest', 'xgboost']:
            coef = self.model.feature_importances_
        else:
            coef = self.model.coef_

        # sort features
        sorted_coef_idxs = np.argsort(np.abs(coef))[::-1]
        sorted_coefs = coef[sorted_coef_idxs]
        self.feature_importance = sorted_coefs[sorted_coefs != 0]

        # ranked features
        self.selected_features = self.features_window.iloc[:, sorted_coef_idxs].copy()
        ranked_features_list = self.selected_features.columns.tolist()
        self.feature_importance = pd.DataFrame(self.feature_importance,
                                               index=ranked_features_list[: len(self.feature_importance)],
                                               columns=['feature_importance'])
        if drop_target:
            self.selected_features.drop(columns=self.target_lags.columns, inplace=True, errors='ignore')
            self.feature_importance.drop(index=self.target_lags.columns, inplace=True, errors='ignore')
        if drop_feat_lags:
            self.selected_features.drop(columns=self.feature_lags.columns, inplace=True, errors='ignore')
            self.feature_importance.drop(index=self.feature_lags.columns, inplace=True, errors='ignore')

        # remove features with zero importance
        self.selected_features = self.selected_features.iloc[:, :len(self.feature_importance)]

        return self.selected_features

    def predict(self) -> np.array:
        """
        Predict target.

        Returns
        -------
        target: np.array
            Target.
        """
        # fit
        self.fit()

        # predict target
        self.yhat = self.model.predict(self.features_window)

        # add index and col name if not np.array
        if self.index is not None:
            if self.h_lookahead > 0:
                self.yhat_name = f"{self.target.name}_fcst_t+{self.h_lookahead}"
            else:
                self.yhat_name = f"{self.target.name}_nowcast"
        self.yhat = pd.DataFrame(self.yhat, index=self.index[-self.yhat.shape[0]:], columns=[self.yhat_name])

        return self.yhat

    def compute_score(self, metric: str = 'mse') -> float:
        """
        Model score.

        Parameters
        ----------
        metric: str, {'mse', 'rmse', 'mae', 'r2', 'adj_r2', 'chg_accuracy'}, default='mse'
            Score metric.

        Returns
        -------
        score: float
            Model score.
        """
        # predict target
        if self.yhat is None:
            self.predict()

        # error
        data = pd.concat([self.yhat, self.target_fcst], axis=1).dropna().copy()
        error = data.diff(axis=1).dropna(axis=1, how='all').copy()

        # score
        if metric == 'mse':
            self.score = (error ** 2).mean()[0]
        elif metric == 'rmse':
            self.score = np.sqrt((error ** 2).mean())[0]
        elif metric == 'mae':
            self.score = error.abs().mean()[0]
        elif metric == 'r2':
            self.score = self.model.score(self.predictors, self.target_fcst)
        elif metric == 'adj_r2':
            n = self.target_fcst.shape[0]
            p = self.predictors.shape[1]
            r2 = self.model.score(self.predictors, self.target_fcst)
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            if adj_r2 >= 1:
                self.score = np.float64(1.0)
            else:
                self.score = adj_r2
        elif metric == 'chg_accuracy':
            chg = data.diff().dropna()
            sign = np.sign(chg)
            self.score = (sign.iloc[:, 0] == sign.iloc[:, 1]).sum() / sign.shape[0]
        else:
            raise ValueError("Invalid score. Must be one of 'mse', 'rmse', 'r2', 'adj_r2', 'chg_accuracy'.")

        return self.score.round(decimals=4)

    def expanding_window_data(self, row: int) -> None:
        """
        Preprocess data for expanding window computation.

        Parameters
        ----------
        row: int
            Row index.
        """
        # get data window
        if isinstance(self.data, pd.DataFrame):
            self.predictors = self.data.iloc[:row, 1:]
            self.target_fcst = self.data.iloc[:row, 0]
            self.features_window = self.features.iloc[:row]
        else:
            self.predictors = self.data[:row, 1:]
            self.target_fcst = self.data[:row, 0]
            self.features_window = self.features[:row]

    def expanding_predict(self, min_obs: int) -> float:
        """
        Expanding window model prediction.

        Parameters
        ----------
        min_obs: int
            Mininum number of observations for expanding window computation.

        Returns
        -------
        yhat: float
            Model prediction.
        """
        # min obs
        if min_obs > self.data.shape[0]:
            raise ValueError(f"Minimum observations {min_obs} is greater than length of data {self.data.shape[0]}.")
        if min_obs < 1:
            raise ValueError(f"Minimum observations {min_obs} must be greater than 0.")

        # store predictions
        yhat_df = pd.DataFrame()

        # loop through rows of df
        for row in range(min_obs, self.features.shape[0] + 1):

            # set expanding window
            self.expanding_window_data(row)
            # predict
            self.predict()
            # add to df if not np.array
            if self.index is not None:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat.iloc[-1].to_numpy(dtype=np.float64))])
            else:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat[-1])])

        # add index and col name
        yhat_df.index = self.index[-yhat_df.shape[0]:]
        yhat_df.columns = [self.yhat_name]

        # yhat
        self.yhat = yhat_df

        return self.yhat

    def rolling_window_data(self, row: int, window_size: int) -> None:
        """
        Preprocess data for rolling window computation.

        Parameters
        ----------
        row: int
            Row index.
        window_size: int
            Size of rolling window (number of observations).
        """
        # get data window
        if isinstance(self.data, pd.DataFrame):
            self.predictors = self.data.iloc[row: row + window_size, 1:]
            self.target_fcst = self.data.iloc[row: row + window_size, 0]
            self.features_window = self.features.iloc[row: row + window_size]
        else:
            self.predictors = self.data[row: row + window_size, 1:]
            self.target_fcst = self.data[row: row + window_size, 0]
            self.features_window = self.features[row: row + window_size]

    def rolling_predict(self, window_size: int) -> float:
        """
        Rolling window model prediction.

        Parameters
        ----------
        window_size: int
            Size of rolling window (number of observations).

        Returns
        -------
        yhat: float
            Model prediction.
        """
        # window size
        if window_size > self.data.shape[0]:
            raise ValueError(f"Window size {window_size} is greater than length of data {self.data.shape[0]}.")
        if window_size < 1:
            raise ValueError(f"Window size {window_size} must be greater than 0.")

        # store predictions
        yhat_df = pd.DataFrame()

        # loop through rows of df
        for row in range(self.features.shape[0] - window_size + 1):

            # set rolling window
            self.rolling_window_data(row, window_size)
            # predict
            self.predict()
            # add to df if not np.array
            if self.index is not None:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat.iloc[-1].to_numpy(dtype=np.float64))])
            else:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat[-1])])

        # add index and col name
        yhat_df.index = self.index[-yhat_df.shape[0]:]
        yhat_df.columns = [self.yhat_name]

        # yhat
        self.yhat = yhat_df

        return self.yhat


class SPCA:
    """
    Supervised PCA class.

    Wrapper for scikit-learn PCA and Linear regression methods where the target value is expected to be a
    linear combination of the principal components.

    See https://www.sciencedirect.com/science/article/abs/pii/S0304407608001085 and
    https://hastie.su.domains/Papers/spca_JASA.pdf for details.

    """
    def __init__(self,
                 target: Union[np.array, pd.Series, pd.DataFrame],
                 features: Union[np.array, pd.DataFrame],
                 method: str = 'lars',
                 oos: bool = False,
                 n_feat: int = 30,
                 n_components: Optional[int] = None,
                 t_lags: int = 6,
                 h_lookahead: int = 1,
                 **kwargs: Any
                 ):
        """
        Initialize Targeted PCA object.

        Parameters
        ----------
        target: pd.Series, pd.DataFrame or np.ndarray
            Factor to select features for.
        features: pd.DataFrame or np.ndarray
            Features to select from.
        method: str, {'ols', 'lasso', 'lasso_cv', 'lasso_lars', 'lasso_lars_cv', 'lasso_lars_ic', 'lars', 'lars_cv',
                        'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'random_forest', 'xgboost'}
            Linear model method to use for feature selection.
        oos: bool, default False
            Out-of-sample prediction. If True, uses previous period coefficients to predict next period target.
        n_feat: int, default=30
            Number of features/non-zero coefficients to select.
        n_components: int
            Number of principal components.
        n_lags: int, default 4
            Number of lags of features/predictors to include in forecast.
        n_lookahead: int, default 1
            Number of look-ahead periods for target forecast.
        **kwargs: Optional keyword arguments, for model object. See sklearn.linear_model for details.
        """
        self.target = target
        self.features = features
        self.method = method
        self.oos = oos
        self.n_feat = n_feat
        self.n_components = n_components
        self.t_lags = t_lags
        self.h_lookahead = h_lookahead
        self.features_window = features.copy()
        self.selected_features = None
        self.feature_importance = None
        self.pcs = None
        self.model = None
        self.yhat = None
        self.score = None
        self.index = self.features.index
        self.kwargs = kwargs

    def get_selected_features(self) -> pd.DataFrame:
        """
        Select features with the most predictive power.

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features.
        """
        # model
        lm = Regression(self.target, self.features_window, method=self.method, oos=self.oos, t_lags=self.t_lags,
                         h_lookahead=self.h_lookahead, **self.kwargs)

        # drop feature lags for feature selection step
        columns_to_drop = [col for col in lm.predictors.columns if '_L' in col]
        lm.predictors = lm.predictors.drop(columns=columns_to_drop)

        # get features
        lm.get_selected_features(drop_target=True)
        # n features
        self.selected_features = lm.selected_features.iloc[:, :self.n_feat]
        self.feature_importance = lm.feature_importance.iloc[:self.n_feat]

        return self.selected_features

    def get_pcs(self) -> pd.DataFrame:
        """
        Get targeted principal components.

        Returns
        -------
        pcs: pd.DataFrame
            Principal components.
        """
        # feature selection
        self.get_selected_features()

        # n components
        if self.n_components is None or self.n_components > self.selected_features.shape[1]:
            self.n_components = self.selected_features.shape[1]

        # PCA
        self.pcs = PCAWrapper(self.selected_features, n_components=self.n_components).get_pcs()

        # add index and cols if available
        if self.index is not None:
            self.pcs.columns = [f"PC{i+1}" for i in range(self.pcs.shape[1])]

        return self.pcs

    def predict(self, method: str = 'lasso_lars_ic', **kwargs) -> pd.DataFrame:
        """
        Predict target.

        Parameters
        ----------
        method: str, {'ols', 'lasso', 'lasso_cv', 'lasso_lars', 'lasso_lars_cv', 'lasso_lars_ic', 'lars', 'lars_cv',
                        'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'random_forest', 'xgboost'}
            Linear model method to use for prediction.
        **kwargs: Optional keyword arguments, for model object. See sklearn.linear_model for details.

        Returns
        -------
        yhat: pd.DataFrame
            Predicted target (yhat).
        """
        # pcs
        self.pcs = self.get_pcs()

        # linear model
        self.model = Regression(self.target, self.pcs, method=method, oos=self.oos, t_lags=self.t_lags,
                         h_lookahead=self.h_lookahead, **kwargs)
        # predict
        self.yhat = self.model.predict()

        return self.yhat

    def compute_score(self, metric: str = 'mse') -> float:
        """
        Model score.

        Parameters
        ----------
        metric: str, {'mse', 'rmse', 'mae', 'r2', 'adj_r2', 'chg_accuracy'}, default='mse'
            Score metric.

        Returns
        -------
        score: float
            Model score.
        """
        # predict target
        if self.yhat is None:
            self.predict()

        # error
        data = pd.concat([self.yhat, self.model.target_fcst], axis=1).dropna().copy()
        error = data.diff(axis=1).dropna(axis=1, how='all').copy()

        # score
        if metric == 'mse':
            self.score = (error ** 2).mean()[0]
        elif metric == 'rmse':
            self.score = np.sqrt((error ** 2).mean())[0]
        elif metric == 'mae':
            self.score = error.abs().mean()[0]
        elif metric == 'r2':
            self.score = self.model.model.score(self.model.predictors, self.model.target_fcst)
        elif metric == 'adj_r2':
            n = self.model.target_fcst.shape[0]
            p = self.selected_features.shape[1]
            r2 = self.model.model.score(self.model.predictors, self.model.target_fcst)
            adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            if adj_r2 >= 1:
                self.score = np.float64(1.0)
            else:
                self.score = adj_r2
        elif metric == 'chg_accuracy':
            chg = data.diff().dropna().copy()
            sign = np.sign(chg).copy()
            self.score = (sign.iloc[:, 0] == sign.iloc[:, 1]).sum() / sign.shape[0]
        else:
            raise ValueError("Invalid score. Must be one of 'mse', 'rmse', 'r2', 'adj_r2'. 'chg_accuracy")

        return self.score.round(decimals=4)

    def expanding_window_data(self, row: int) -> None:
        """
        Preprocess data for expanding window computation.

        Parameters
        ----------
        row: int
            Row index.
        """
        # get data window
        if isinstance(self.features_window, pd.DataFrame):
            self.features_window = self.features.iloc[:row]
        else:
            self.features_window = self.features[:row]

    def expanding_predict(self, min_obs: int, method: str = 'lasso_lars_ic', **kwargs) \
            -> Union[pd.Series, pd.DataFrame]:
        """
        Expanding window model prediction.

        Parameters
        ----------
        min_obs: int
            Mininum number of observations for expanding window computation.
        method: str, {'ols', 'lasso', 'lasso_cv', 'lasso_lars', 'lasso_lars_cv', 'lasso_lars_ic', 'lars', 'lars_cv',
                      'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'random_forest', 'xgboost'}
            Linear model method to use.
        **kwargs: Optional keyword arguments, for model object. See sklearn.linear_model for details.

        Returns
        -------
        yhat: float
            Model prediction.
        """
        # min obs
        if min_obs > self.features.shape[0]:
            raise ValueError(f"Minimum observations {min_obs} is greater than length of data {self.features.shape[0]}.")
        if min_obs < 1:
            raise ValueError(f"Minimum observations {min_obs} must be greater than 0.")

        # store predictions
        yhat_df = pd.DataFrame()

        # loop through rows of df
        for row in range(min_obs, self.features.shape[0] + 1):

            # set expanding window
            self.expanding_window_data(row)

            # predict
            self.predict(method=method, **kwargs)

            # add to df if not np.array
            if self.index is not None:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat.iloc[-1]).T])
            else:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat[-1])])

        # add index and col name
        yhat_df.index = self.index[-yhat_df.shape[0]:]
        yhat_df.columns = [self.model.yhat_name]

        # yhat
        self.yhat = yhat_df

        return self.yhat

    def rolling_window_data(self, row: int, window_size: int) -> None:
        """
        Preprocess data for rolling window computation.

        Parameters
        ----------
        row: int
            Row index.
        window_size: int
            Size of rolling window (number of observations).
        """
        # get data window
        if isinstance(self.features_window, pd.DataFrame):
            self.features_window = self.features.iloc[row: row + window_size]
        else:
            self.features_window = self.features[row: row + window_size]

    def rolling_predict(self, window_size: int, method: str = 'lasso_lars_ic', **kwargs) \
            -> Union[pd.Series, pd.DataFrame]:
        """
        Rolling window model prediction.

        Parameters
        ----------
        window_size: int
            Size of rolling window (number of observations).
        method: str, {'ols', 'lasso', 'lasso_cv', 'lasso_lars', 'lasso_lars_cv', 'lasso_lars_ic', 'lars', 'lars_cv',
                      'ridge', 'ridge_cv', 'elastic_net', 'elastic_net_cv', 'random_forest', 'xgboost'}
            Linear model method to use.
        **kwargs: Optional keyword arguments, for model object. See sklearn.linear_model for details.

        Returns
        -------
        yhat: float
            Model prediction.
        """
        # window size
        if window_size > self.features.shape[0]:
            raise ValueError(f"Window size {window_size} is greater than length of data {self.features.shape[0]}.")
        if window_size < 1:
            raise ValueError(f"Window size {window_size} must be greater than 0.")

        # store predictions
        yhat_df = pd.DataFrame()

        # loop through rows of df
        for row in range(self.features.shape[0] - window_size + 1):

            # set rolling window
            self.rolling_window_data(row, window_size)
            # predict
            self.predict(method=method, **kwargs)
            # add to df if not np.array
            if self.index is not None:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat.iloc[-1]).T])
            else:
                yhat_df = pd.concat([yhat_df, pd.DataFrame(self.yhat[-1])])

        # add index and col name
        yhat_df.index = self.index[-yhat_df.shape[0]:]
        yhat_df.columns = [self.model.yhat_name]

        # yhat
        self.yhat = yhat_df

        return self.yhat


class Classification(SuperviseLearning):
    """
    Classification model class.

    Wrapper for scikit-learn classification methods where the target value is expected to be a binary or multi-class
    classification (up/down, buy/sell/hold, etc.).

    See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model for details.

    """
    def __init__(self,
                 target: Union[np.array, pd.Series, pd.DataFrame],
                 features: Union[np.array, pd.DataFrame],
                 method: str, target_lookahead: int = 1,
                 feature_lags: int = 6,
                 **kwargs: Any):
        """
        Initialize Classification object.

        Parameters
        ----------
        target
        features
        method
        target_lookahead
        feature_lags
        kwargs
        """
        super().__init__(target, features, method, target_lookahead, feature_lags)
        self.target = target
        self.features = features
        self.method = method
        self.target_lookahead = target_lookahead
        self.feature_lags = feature_lags
        self.target_lags = None
        self.feature_lags = None
        self.predictors = None
        self.target_fcst = None
        self.features_window = None
        self.model = None
        self.yhat = None
        self.yhat_name = None
        self.score = None
        self.selected_features = None
        self.feature_importance = None
        self.data = self.preprocess_data()
        self.index = self.features.index
        self.kwargs = kwargs

    def preprocess_data(self) -> Union[pd.DataFrame, np.array]:
        """
        Pre-process data into a format suitable for supervised learning using scikit-learn.

        Returns
        -------
        data: pd.DataFrame or np.ndarray
            Data matrix.
        """
        # if not isinstance(self.target, (pd.Series,  np.ndarray)):
        #     raise TypeError("Target must be a pandas Series or np.array.")
        # elif isinstance(self.target, pd.Series) and (isinstance(self.features, pd.DataFrame) or self.features is None):
        #     if self.features is not None:
        #         self.feature_lags = add_lags(self.features, n_lags=self.feature_lags).copy()
        #

    def fit(self) -> None:
        """
        Fit data.

        """
        pass

    def predict(self) -> None:
        """
        Predict target.

        """
        pass


class Forecast(SuperviseLearning):
    """
    Forecast model class.

    Wrapper for scikit-learn regression methods where the target value is expected to be a continuous variable (returns).

    See https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model for details.

    See Also
    --------
    sklearn.linear_model.LinearRegression
    sklearn.linear_model.Lasso
    sklearn.linear_model.LassoCV
    sklearn.linear_model.LassoLars
    sklearn.linear_model.LassoLarsCV
    sklearn.linear_model.LassoLarsIC
    sklearn.linear_model.Lars
    sklearn.linear_model.LarsCV
    sklearn.linear_model.Ridge
    sklearn.linear_model.RidgeCV
    sklearn.linear_model.ElasticNet
    sklearn.linear_model.ElasticNetCV
    sklearn.ensemble.RandomForestRegressor
    xgboost.XGBRegressor
    """
    def __init__(self,
                 target: Union[np.array, pd.Series, pd.DataFrame],
                 features: Union[np.array, pd.DataFrame],
                 method: str,
                 target_lookahead: int = 1,
                 feature_lags: int = 6,
                 **kwargs: Any):
        """
        Initialize Forecast object.

        Parameters
        ----------
        target
        features
        method
        target_lookahead
        feature_lags
        kwargs
        """
        super().__init__(target, features, method, target_lookahead, feature_lags)
        self.target = target
        self.features = features
        self.method = method
        self.target_lookahead = target_lookahead
        self.feature_lags = feature_lags
        self.target_lags = None
        self.feature_lags = None
        self.predictors = None
        self.target_fcst = None
        self.features_window = None
        self.model = None
        self.yhat = None
        self.yhat_name = None
        self.score = None
        self.selected_features = None
        self.feature_importance = None
        self.data = self.preprocess_data()
        self.index = self.features.index
        self.kwargs = kwargs

    def preprocess_data(self) -> Union[pd.DataFrame, np.array]:
        """
        Pre-process data into a format suitable for supervised learning using scikit-learn.

        Returns
        -------
        data: pd.DataFrame or np.ndarray
            Data matrix.
        """
        pass

    def fit(self) -> None:
        """
        Fit data.

        """
        pass

    def predict(self) -> None:
        """
        Predict target.

        """
        pass

    def compute_score(self, metric: str = 'mse') -> float:
        """
        Model score.

        Parameters
        ----------
        metric: str, {'mse', 'rmse', 'mae', 'r2', 'adj_r2', 'chg_accuracy'}, default='mse'
            Score metric.

        Returns
        -------
        score: float
            Model score.
        """
        pass

