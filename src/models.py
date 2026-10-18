"""Model builders and a time-fold runner shared by notebooks 06–08 and 12."""
import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from src.preprocess import MODEL_INPUTS, make_linear_preprocessor, make_tree_preprocessor

SEED = 42

# Regularised starting point for ~2.8k customers behind ~11k correlated rows (tuned in notebook 08).
LGBM_BASE = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=SEED, verbose=-1)


def make_logreg(C: float = 1.0, exclude: tuple = ()) -> Pipeline:
    return Pipeline([("prep", make_linear_preprocessor(exclude=exclude)), ("model", LogisticRegression(C=C, max_iter=2000))])


def make_lgbm_classifier(**params) -> Pipeline:
    return Pipeline([("prep", make_tree_preprocessor()), ("model", LGBMClassifier(**{**LGBM_BASE, **params}))])


def make_lgbm_regressor(**params) -> Pipeline:
    return Pipeline([("prep", make_tree_preprocessor()), ("model", LGBMRegressor(**{**LGBM_BASE, **params}))])


class LogTargetRegressor(BaseEstimator, RegressorMixin):
    """Fit `base` on log1p(y); predict on the £ scale.

    Naive back-transform expm1(prediction) estimates the *median*, not the mean, so totals come out too
    low (Jensen's inequality). Duan's smearing multiplies by mean(exp(residuals)) from the training fit:
    E[y + 1] = exp(mu) * E[exp(eps)].
    """

    def __init__(self, base, smearing: bool = True):
        self.base, self.smearing = base, smearing

    def fit(self, X, y):
        y_log = np.log1p(np.asarray(y, float))
        self.model_ = clone(self.base).fit(X, y_log)
        resid = y_log - self.model_.predict(X)
        self.smear_ = float(np.mean(np.exp(resid))) if self.smearing else 1.0
        return self

    def predict(self, X):
        return np.maximum(np.exp(self.model_.predict(X)) * self.smear_ - 1, 0.0)


def make_ridge_log(alpha: float = 1.0, smearing: bool = True, skew: str = "clip") -> LogTargetRegressor:
    """Log-linear model. Inputs are clipped to training quantiles: with a log target, an extreme input
    becomes an extreme log prediction and exp() turns it into an absurd £ value (notebook 07)."""
    prep = make_linear_preprocessor(skew=skew)
    return LogTargetRegressor(Pipeline([("prep", prep), ("model", Ridge(alpha=alpha))]), smearing)


def make_lgbm_log(smearing: bool = True, **params) -> LogTargetRegressor:
    return LogTargetRegressor(make_lgbm_regressor(**params), smearing)


def make_lgbm_tweedie(power: float = 1.5, **params) -> Pipeline:
    return make_lgbm_regressor(objective="tweedie", tweedie_variance_power=power, **params)


def _predict(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def run_folds(table: pd.DataFrame, make_model, target: str, folds, train_filter=None,
              inputs=MODEL_INPUTS, return_models: bool = False):
    """Fit a fresh model per time fold on that fold's training cutoffs; predict the fold's validation cutoff.

    train_filter: optional function(df) -> boolean mask applied to training rows (e.g. buyers only).
    Returns one row per validation customer: fold, cutoff, Customer ID, y, pred (and the fitted models).
    """
    parts, models = [], {}
    for train_cuts, v in folds:
        tr, va = table[table["cutoff"].isin(train_cuts)], table[table["cutoff"] == v]
        if train_filter is not None:
            tr = tr[train_filter(tr)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = make_model().fit(tr[inputs], tr[target])
            pred = _predict(model, va[inputs])
        parts.append(pd.DataFrame({"fold": v.strftime("%b"), "cutoff": v, "Customer ID": va["Customer ID"].values,
                                   "y": va[target].values, "pred": pred}))
        models[v.strftime("%b")] = model
    out = pd.concat(parts, ignore_index=True)
    return (out, models) if return_models else out


def per_fold(preds: pd.DataFrame, metric_fn) -> pd.Series:
    """Apply metric_fn(y, pred) -> float within each fold; adds the mean."""
    s = pd.Series({f: metric_fn(d["y"], d["pred"]) for f, d in preds.groupby("fold", sort=False)})
    return pd.concat([s, pd.Series({"mean": s.mean()})])


def run_hurdle(table: pd.DataFrame, folds, make_classifier, make_spend_if_buy) -> pd.DataFrame:
    """Two-part model: P(buy) from a classifier fit on all rows x E[spend | buy] from a regressor fit on
    buyers only (so its back-transform correction sees only buyers' residuals)."""
    p_buy = run_folds(table, make_classifier, "y_buy", folds)
    spend = run_folds(table, make_spend_if_buy, "y_spend", folds, train_filter=lambda d: d["y_spend"] > 0)
    assert (p_buy["Customer ID"].values == spend["Customer ID"].values).all()
    return spend.assign(p_buy=p_buy["pred"].values, spend_if_buy=spend["pred"].values,
                        pred=p_buy["pred"].values * spend["pred"].values)
