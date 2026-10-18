"""Preprocessing pipelines (fit on training rows only, always inside a sklearn Pipeline).

Linear / NN models need: missing values filled, skew tamed, features on one scale, categories encoded.
Trees need almost none of that: they split on order, not distance, and handle NaN natively.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.features import NUMERIC_FEATURES

# Not learnable during model selection (notebook 04): trend is missing for 5 of 6 training cutoffs;
# history_days and peak_days_in_window are collinear (rho 0.997) across them. Revisit at the final refit.
NOT_LEARNABLE_YET = ["spend_trend", "days_trend", "history_days", "peak_days_in_window"]
NUM_FEATURES = [c for c in NUMERIC_FEATURES if c not in NOT_LEARNABLE_YET + ["is_uk"]]
# |skew| > 2 on training rows, non-binary (notebook 04 hand-off table)
SKEWED = ["spend_30d", "spend_90d", "spend_180d", "n_days_30d", "n_days_90d", "n_days_180d",
          "days_per_month", "spend_per_month", "avg_spend_per_day", "max_spend_day_180d",
          "avg_items_per_day", "n_products_180d", "return_value_share", "overdue_ratio"]
OTHER_NUM = [c for c in NUM_FEATURES if c not in SKEWED]
MODEL_INPUTS = NUM_FEATURES + ["is_uk", "country", "Customer ID"]  # Customer ID only feeds the grouped encoder


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize: clip each column to its [lower, upper] quantiles, learned on training rows."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower, self.upper = lower, upper

    def fit(self, X, y=None):
        X = np.asarray(X, float)
        self.lo_, self.hi_ = np.nanquantile(X, self.lower, axis=0), np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, float), self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


class GroupTargetEncoder(BaseEstimator, TransformerMixin):
    """Smoothed mean-target encoding of one categorical column.

    Expects two columns: [category, group]. The encoding of a category is
    (sum_y + smoothing * prior) / (count + smoothing): rare categories shrink to the overall rate.
    fit_transform (training rows) is out-of-fold by GROUP, so a row's encoding never uses its own
    customer's targets (the same customer appears in several snapshots). transform uses the full mapping.
    oof=False, smoothing=0 reproduces the naive, leaky version.
    """

    def __init__(self, smoothing: float = 20.0, n_splits: int = 5, oof: bool = True):
        self.smoothing, self.n_splits, self.oof = smoothing, n_splits, oof

    def _mapping(self, cat, y):
        prior = float(np.mean(y))
        stats = pd.DataFrame({"c": cat, "y": y}).groupby("c")["y"].agg(["sum", "count"])
        return (stats["sum"] + self.smoothing * prior) / (stats["count"] + self.smoothing), prior

    def _apply(self, cat, mapping, prior):
        return pd.Series(cat).map(mapping).fillna(prior).to_numpy().reshape(-1, 1)

    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.mapping_, self.prior_ = self._mapping(X.iloc[:, 0].to_numpy(), np.asarray(y))
        return self

    def fit_transform(self, X, y=None, **fit_params):
        X, y = pd.DataFrame(X), np.asarray(y)
        cat, groups = X.iloc[:, 0].to_numpy(), X.iloc[:, 1].to_numpy()
        self.fit(X, y)
        if not self.oof:
            return self._apply(cat, self.mapping_, self.prior_)
        out = np.empty((len(X), 1))
        for fit_idx, enc_idx in GroupKFold(self.n_splits).split(cat, y, groups):
            mapping, prior = self._mapping(cat[fit_idx], y[fit_idx])
            out[enc_idx] = self._apply(cat[enc_idx], mapping, prior)
        return out

    def transform(self, X):
        return self._apply(pd.DataFrame(X).iloc[:, 0].to_numpy(), self.mapping_, self.prior_)

    def get_feature_names_out(self, input_features=None):
        return np.array(["country_te"], dtype=object)


def _country_block(country: str):
    if country == "none":
        return []
    if country == "is_uk":
        return [("country", "passthrough", ["is_uk"])]
    if country == "onehot":
        return [("country", OneHotEncoder(handle_unknown="ignore"), ["country"])]
    if country == "onehot_rare":  # levels with < 30 training rows share one "infrequent" column
        return [("country", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30), ["country"])]
    if country == "target_oof":
        return [("country", GroupTargetEncoder(), ["country", "Customer ID"])]
    if country == "target_naive":
        return [("country", GroupTargetEncoder(smoothing=0, oof=False), ["country", "Customer ID"])]
    raise ValueError(country)


def make_linear_preprocessor(skew: str = "none", missing: str = "median", scale: bool = True,
                             country: str = "is_uk", exclude: tuple = ()) -> ColumnTransformer:
    """skew: 'log' | 'clip' | 'none'; missing: 'indicator' | 'median'; exclude: feature names to leave out.
    Defaults = the choices notebook 05 supported across time folds (tie or win in every fold)."""
    shape = {"log": FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
             "clip": QuantileClipper(), "none": "passthrough"}[skew]

    def num_pipe(transform_skew):
        steps = [("skew", shape)] if transform_skew else []
        steps.append(("impute", SimpleImputer(strategy="median", add_indicator=(missing == "indicator"))))
        if scale:
            steps.append(("scale", StandardScaler()))
        return Pipeline(steps)

    blocks = [("skewed", num_pipe(True), [c for c in SKEWED if c not in exclude]),
              ("other", num_pipe(False), [c for c in OTHER_NUM if c not in exclude])]
    return ColumnTransformer(blocks + _country_block(country), remainder="drop", verbose_feature_names_out=False)


def make_tree_preprocessor(country: str = "is_uk") -> ColumnTransformer:
    """Trees: raw numerics with NaN left as NaN (handled natively); no log, no scaling."""
    return ColumnTransformer([("num", "passthrough", NUM_FEATURES)] + _country_block(country),
                             remainder="drop", verbose_feature_names_out=False)
