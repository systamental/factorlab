import pandas as pd
import numpy as np
from typing import Optional, Union, Any
from scipy.stats import chi2_contingency, spearmanr, kendalltau, contingency
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoLarsIC, Lasso, Ridge, ElasticNet, Lars
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from factorlab.feature_engineering.transformations import Transform
from factorlab.signal_generation.time_series_analysis import TimeSeriesAnalysis as TSA


class FeatureSelection:
    """
    Filter and wrapper methods for feature selection.
    """
    def __init__(self,
                 target: pd.Series,
                 features: pd.DataFrame,
                 n_feat: Optional[int] = None,
                 n_lookahead: Optional[int] = None,
                 n_lags: Optional[int] = None,
                 strategy: str = 'ts',
                 normalize: bool = False,
                 quantize: bool = False,
                 feature_bins: int = 5,
                 target_bins: int = 3,
                 window_type: str = 'expanding',
                 window_size: int = 30,
                 ):
        """
        Initialize Filter object.

        Parameters
        ----------
        target: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe or series with DatetimeIndex (level 0), tickers (level 1) and target (cols).
        features: pd.Series or pd.DataFrame - Single or MultiIndex
            Dataframe with DatetimeIndex (level 0), tickers (level 1) and features (cols).
        n_feat: int, default None
            Number of features to select.
        n_lookahead: int, default 1
            Number of periods to lookahead for the target variable.
        n_lags: int, default 4
            Number of lags to add to the features and lagged target.
        strategy: str, {'ts', 'cs}, default 'ts'
            Time series or cross-sectional strategy.
        normalize: bool, default False
            Normalize features and target.
        quantize: bool, default False
            Quantize features and target.
        feature_bins: int, default 5
            Number of bins for feature quantiles.
        target_bins: int, default 3
            Number of bins for target quantiles.
        window_type: str, {'fixed', 'expanding', 'rolling'}, default 'expanding'
            Window type for normalization.
        window_size: int, default 90
            Minimal number of observations to include in moving window (rolling or expanding).
        """
        self.target = target
        self.features = features
        self.n_feat = n_feat
        self.n_lookahead = n_lookahead
        self.n_lags = n_lags
        self.strategy = strategy
        self.normalize = normalize
        self.quantize = quantize
        self.feature_bins = feature_bins
        self.target_bins = target_bins
        self.window_type = window_type
        self.window_size = window_size

        self.lagged_target = None
        self.data = None
        self.index = None
        self.freq = None
        self.ranked_features = None
        self.ranked_features_list = None
        self.selected_features = None
        self.feature_importance = None
        self.check_n_feat()
        self.preprocess_data()
        self.normalize_data()
        self.quantize_data()

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
        # features
        if isinstance(self.features, pd.Series):
            self.features = self.features.to_frame()

        # target
        if isinstance(self.target, pd.Series):
            self.target = self.target.to_frame()

        # add target lags
        if self.n_lags is not None:
            tsa = TSA(self.target, self.features, n_lags=self.n_lags)
            self.lagged_target = tsa.features.iloc[:, :self.n_lags]

        # target look ahead
        if self.n_lookahead is not None:
            if isinstance(self.target.index, pd.MultiIndex):
                self.target = self.target.groupby('ticker').shift(-self.n_lookahead)
            else:
                self.target = self.target.shift(-self.n_lookahead)
            self.target.columns = [f"{self.target.columns[0]}_F{self.n_lookahead}"]

        # concat features, lags and target
        self.data = pd.concat([self.features, self.lagged_target, self.target], join='inner', axis=1).dropna()
        self.target = self.data.iloc[:, -1].to_frame()
        self.features = self.data.iloc[:, :-1]

        # index
        self.index = self.data.index
        if isinstance(self.index, pd.MultiIndex):
            if not isinstance(self.index.levels[0], pd.DatetimeIndex):
                self.index = self.index.set_levels(pd.to_datetime(self.index.levels[0]), level=0)
        else:
            self.index = pd.to_datetime(self.index)

        # freq
        if isinstance(self.index, pd.MultiIndex):
            self.freq = pd.infer_freq(self.index.get_level_values(0).unique())
        else:
            self.freq = pd.infer_freq(self.index)

    def normalize_data(self) -> pd.DataFrame:
        """
        Normalize factors and/or targets.
        """
        if self.normalize:
            # time series
            if self.strategy == 'ts':
                self.features = Transform(self.features).normalize(window_type=self.window_type, axis='ts',
                                                                 window_size=self.window_size)
                self.target = Transform(self.target).normalize(window_type=self.window_type, axis='ts',
                                                         window_size=self.window_size)
            # cross-sectional
            elif self.strategy == 'cs':
                self.features = Transform(self.features).normalize(axis='cs')
                self.target = Transform(self.target).normalize(axis='cs')

    def quantize_data(self) -> pd.DataFrame:
        """
        Quantize factors and/or targets.
        """
        if self.quantize:
            # time series
            if self.strategy == 'ts':
                self.features = Transform(self.features).quantize(bins=self.feature_bins, axis='ts',
                                                                  window_type=self.window_type,
                                                                  window_size=self.window_size)
                self.target = Transform(self.target).quantize(bins=self.target_bins, axis='ts',
                                                              window_type=self.window_type,
                                                              window_size=self.window_size)
            # cross-sectional
            elif self.strategy == 'cs':
                self.features = Transform(self.features).quantize(bins=self.feature_bins, axis='cs')
                self.target = Transform(self.target).quantize(bins=self.target_bins, axis='cs')

    def spearman_rank(self) -> pd.DataFrame:
        """
        Computes the Spearman Rank correlation coefficient with associated p-values.

        The Spearman Rank correlation is a non-parametric measure of rank correlation (statistical dependence between
        the rankings of two variables). It assesses how well the relationship between two variables can be described
        using a monotonic function. The Spearman Rank correlation is robust to outliers and does not assume a linear
        relationship between the variables. Like other correlation coefficients, this one varies between -1 and +1
        with 0 implying no correlation. Correlations of -1 or +1 imply an exact monotonic relationship.

        See scipy.stats.spearmanr for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

        Returns
        -------
        spearman_df: pd.DataFrame
            DataFrame with factors (rows), Spearman Rank correlation and p-values (cols).
        """
        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            self.ranked_features.loc[feat, 'spearman_rank'] = spearmanr(self.features[feat], self.target)[0]
            self.ranked_features.loc[feat, 'p-val'] = spearmanr(self.features[feat], self.target)[1]

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='spearman_rank', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def kendall_tau(self) -> pd.DataFrame:
        """
        Computes the Kendall Tau correlation with associated p-values.

        Kendall’s tau is a measure of the correspondence between two rankings.
        Values close to 1 indicate strong agreement, and values close to -1 indicate strong disagreement.

        See scipy.stats.kendalltau for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html

        Returns
        -------
        kendall_df: pd.DataFrame
            DataFrame with factors (rows), Kendall Tau correlation and p-values (cols).
        """
        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            self.ranked_features.loc[feat, 'kendall_tau'] = kendalltau(self.features[feat], self.target)[0]
            self.ranked_features.loc[feat, 'p-val'] = kendalltau(self.features[feat], self.target)[1]

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='kendall_tau', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def cramer_v(self) -> pd.DataFrame:
        """
        Computes the Cramer's V correlation.

        Cramer's V is a measure of association between two nominal variables.
        It is based on the chi-square statistic and is used to determine the strength of association between two
        nominal variables. The value of Cramer's V ranges from 0 to 1, where 0 means no association and 1 is full
        association.

        See scipy.stats.contingency.association for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html

        Returns
        -------
        cramer_df: pd.DataFrame
            DataFrame with factors (rows), Cramer's V correlation (cols).
        """
        # quantize
        self.quantize = True
        self.quantize_data()

        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            cont_table = pd.crosstab(self.features[feat], self.target.iloc[:, 0])
            self.ranked_features.loc[feat, 'cramer_v'] = contingency.association(cont_table, method='cramer')

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='cramer_v', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def tschuprow(self) -> pd.DataFrame:
        """
        Computes the Tschuprow's T correlation.

        Tschuprow's T correlation is a measure of association between two ordinal variables.
        It is closely related to Cramer's V, coinciding with it for square contingency tables.

        See scipy.stats.contingency.association for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html

        Returns
        -------
        tschuprow_df: pd.DataFrame
            DataFrame with factors (rows), Tschuprow's T correlation (cols).
        """
        # quantize
        self.quantize = True
        self.quantize_data()

        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            cont_table = pd.crosstab(self.features[feat], self.target.iloc[:, 0])
            self.ranked_features.loc[feat, 'tschuprow'] = contingency.association(cont_table, method='tschuprow')

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='tschuprow', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def pearson_cc(self) -> pd.DataFrame:
        """
        Computes the Pearson's Contingency Coefficient.

        Pearson's contingency coefficient is a measure of association between two categorical variables.
        It measures the strength of association between two nominal or ordinal variables.
        The value of Pearson's contingency coefficient ranges from 0 to 1, where 0 means no association
        and 1 is full association.

        See scipy.stats.contingency.association for more details:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.association.html

        Returns
        -------
        pearson_df: pd.DataFrame
            DataFrame with factors (rows), Pearson's correlation coefficient (cols).
        """
        # quantize
        self.quantize = True
        self.quantize_data()

        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            # contingency table
            cont_table = pd.crosstab(self.features[feat], self.target.iloc[:, 0])
            self.ranked_features.loc[feat, 'pearson_cc'] = contingency.association(cont_table, method='pearson')

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='pearson_cc', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def chi2(self) -> pd.DataFrame:
        """
        Computes the Chi-squared correlation between factors and forward returns.

        The Chi-squared correlation is a measure of association between two categorical variables, giving a value
        between 0 and +1. It is based on the chi-square statistic and is used to determine the strength of association

        Returns
        -------
        chi2_df: pd.DataFrame
            DataFrame with factors (rows), Chi-squared correlation (cols).
        """
        # quantize
        self.quantize = True
        self.quantize_data()

        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            # contingency table
            cont_table = pd.crosstab(self.features[feat], self.target.iloc[:, 0])
            self.ranked_features.loc[feat, 'chi2'] = chi2_contingency(cont_table)[0]

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='chi2', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def mutual_info(self) -> pd.DataFrame:
        """
        Computes the Mutual Information correlation between factors and forward returns.

        Returns
        -------
        mutual_info_df: pd.DataFrame
            DataFrame with factors (rows), Mutual Information correlation (cols).
        """
        # quantize
        self.quantize = True
        self.quantize_data()

        # create empty df for correlation measures
        self.ranked_features = pd.DataFrame(index=self.features.columns)

        # loop through factors, compute spearman rank corr
        for feat in self.features.columns:
            self.ranked_features.loc[feat, 'mutual_info'] = mutual_info_classif(self.features[[feat]],
                                                                                self.target.iloc[:, 0])

        # sort features
        self.ranked_features = self.ranked_features.sort_values(by='mutual_info', ascending=False).round(decimals=2)
        self.ranked_features_list = self.ranked_features.columns.tolist()
        # feature importance
        self.feature_importance = self.ranked_features.iloc[: self.n_feat]

        return self.feature_importance

    def filter(self,
               method: str = 'spearman_rank',
               ) -> pd.DataFrame:
        """
        Computes measures of correlation and dependence for factor/target pairs.

        The spearman rank correlation, aka information coefficient (IC), is often used to assess the predictive power
        of a factor. It measures the degree of correlation between factor quantiles and forward returns.
        The higher (lower) the IC, the stronger the relationship between higher (lower) factor values and
        higher (lower) returns. Unlike measures of linear correlation (e.g. Pearson), the spearman rank correlation
        captures the monotonicity and non-linearity of the relationship between factor quantiles and forward returns.

        We also compare the Spearman Rank correlation to other statistical measures of association, e.g. Cramer's V,
        Chi-square, etc. Correlation measures in what way two variables are related, whereas association measures how
        related the variables are.

        Parameters
        ----------
        method: str, {'spearman_r', 'kendall_tau', 'cramer_v', 'tschuprow', 'pearson_cc', 'chi2', 'mutual_info'}
            Metric to compute.

        Returns
        -------
        metrics_df: pd.DataFrame
            Dataframe with factors (rows) and stats (cols), ranked by metric.
        """
        # check method
        if method not in ['spearman_rank', 'kendall_tau', 'cramer_v', 'tschuprow', 'pearson_cc', 'chi2', 'mutual_info']:
            raise ValueError("Method must be one of 'spearman_rank', 'kendall_tau', 'cramer_v', 'tschuprow', "
                             "'pearson_cc', 'chi2', 'mutual_info'.")

        # filter
        self.feature_importance = getattr(self, method)()

        return self.feature_importance

    # TODO: add threshold parameter to ic method
    def ic(self, feature: str) -> pd.DataFrame:
        """
        Computes the Information Coefficient (IC) for factor and forward returns over a moving window.

        Parameters
        ----------
        feature: str
            Name of feature (column) for which compute IC.

        Returns
        -------
        ic : pd.DataFrame
            Information coefficient between factor and forward returns over time.
        """
        # create df
        df = pd.concat([self.features[feature], self.target], join='inner', axis=1).dropna()

        # check if df is empty
        if df.empty:
            raise ValueError("Dataframe is empty. Check if feature and target are aligned.")
        if self.strategy == 'cs' and not isinstance(df.index, pd.MultiIndex):
            raise ValueError("Cross-sectional strategy requires MultiIndex dataframe.")

        # compute spearman rank corr
        def spearman_r(data):
            stat = data.apply(lambda x: spearmanr(data.iloc[:, 0], data.iloc[:, 1], nan_policy='omit')[0])
            return stat

        # cs strategy
        if self.strategy == 'cs':
            ic_df = df.groupby('date').apply(spearman_r).rolling(self.window_size).mean().iloc[:, :-1]

        # ts strategy
        else:
            # create dates list to iterate over
            if isinstance(df.index, pd.MultiIndex):
                dates = df.index.droplevel(1).unique().to_list()
            else:
                dates = df.index.unique().to_list()
            # empty np array
            corr_arr = np.full((len(dates), 2), np.nan)

            # loop through rows of df
            for i in range(len(dates) - self.window_size + 1):
                # get corr
                corr = spearmanr(df.loc[dates[i]: dates[i + self.window_size - 1], df.iloc[:, 0].name],
                                 df.loc[dates[i]: dates[i + self.window_size - 1], df.iloc[:, 1].name],
                                 nan_policy='omit')
                # add corr arr
                corr_arr[i + self.window_size - 1, 0] = corr[0]

            # create ic df
            ic_df = pd.DataFrame(corr_arr, index=dates).iloc[:, 0].to_frame(self.features[feature].name)
            ic_df.index.name = 'date'

        return ic_df

    def get_feature_importance(self, model: Any, attr: str) -> Union[np.array, pd.DataFrame]:
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
        selected_features: np.ndarray or pd.DataFrame
            Selected features from the model.
        """
        # coefficients or feature importance
        if not hasattr(model, attr):
            raise AttributeError(f"Model must have attribute {attr}.")
        if not isinstance(getattr(model, attr), np.ndarray):
            raise TypeError(f"Attribute {attr} must be a numpy array.")
        coef = getattr(model, attr)

        # feature importance
        sorted_coef_idxs = np.argsort(np.abs(coef), axis=None)[::-1]
        sorted_coefs = coef[sorted_coef_idxs]
        self.feature_importance = sorted_coefs[sorted_coefs != 0]

        # ranked features
        self.ranked_features = self.features.iloc[:, sorted_coef_idxs].copy()
        self.ranked_features_list = self.ranked_features.columns.tolist()

        # feature importance
        self.feature_importance = pd.DataFrame(self.feature_importance.astype(float),
                                               index=self.ranked_features_list[: len(self.feature_importance)],
                                               columns=['feature_importance'])
        self.feature_importance = self.feature_importance.iloc[: self.n_feat]

        return self.feature_importance

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
        lars = Lars(n_nonzero_coefs=self.n_feat, **kwargs)
        lars.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(lars, 'coef_')

        return self.feature_importance

    def lasso(self, alpha: float = 0.05, auto_selection: bool = True, criterion: str = 'aic',
              **kwargs) -> pd.DataFrame:
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
            lasso = LassoLarsIC(criterion=criterion, **kwargs)  # auto selection
        else:
            lasso = Lasso(alpha=alpha, **kwargs)
        lasso.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(lasso, 'coef_')

        return self.feature_importance

    def ridge(self, alpha: float = 1.0, **kwargs) -> pd.DataFrame:
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
        ridge.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(ridge, 'coef_')

        return self.feature_importance

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
        elastic_net.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(elastic_net, 'coef_')

        return self.feature_importance

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
        rf.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(rf, 'feature_importances_')

        return self.feature_importance

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
        xgb.fit(self.features, self.target.iloc[:, 0])

        # selected features
        self.get_feature_importance(xgb, 'feature_importances_')

        return self.feature_importance

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

    def rfe(self) -> np.array:
        """
        Recursive Feature Elimination (RFE) supervised learning feature selection.

        Selects a subset of relevant features from a broader set of features by removing the redundant
        or irrelevant features, or features which are strongly correlated in the data without much loss of information.

        Parameters
        ----------

        Returns
        -------
        selected_features: pd.DataFrame
            Selected features from the RFE regression.
        """
        pass
