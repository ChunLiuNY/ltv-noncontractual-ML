"""Metrics shared by all notebooks: classification, regression/ranking, deciles, bootstrap CIs."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             mean_absolute_error, roc_auc_score)


def classification_metrics(y, score, is_probability: bool = True) -> dict:
    """Ranking metrics always; log loss and Brier only when `score` is a probability."""
    out = {"roc_auc": roc_auc_score(y, score), "pr_auc": average_precision_score(y, score)}
    if is_probability:
        p = np.clip(score, 1e-6, 1 - 1e-6)
        out |= {"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p)}
    return out


def top_share(y, pred, frac: float = 0.10) -> float:
    """Share of total actual value captured by the top `frac` of customers ranked by `pred`."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    k = max(1, int(round(len(y) * frac)))
    top = np.argsort(-pred, kind="stable")[:k]
    return y[top].sum() / y.sum()


def normalized_gini(y, pred) -> float:
    """Gini of the gains curve when ranking by `pred`, divided by the Gini of a perfect ranking.
    1 = perfect order, 0 = random order."""
    def gini(order_by):
        order = np.argsort(-np.asarray(order_by, float), kind="stable")
        cum = np.cumsum(np.asarray(y, float)[order]) / np.sum(y)
        return (cum - np.arange(1, len(y) + 1) / len(y)).sum() / len(y)
    return gini(pred) / gini(y)


def regression_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return {
        "mae": mean_absolute_error(y, pred),
        "rmse": float(np.sqrt(np.mean((y - pred) ** 2))),
        "spearman": spearmanr(y, pred).correlation,
        "gini": normalized_gini(y, pred),
        "top10_capture": top_share(y, pred, 0.10),
        "top10_lift": top_share(y, pred, 0.10) / 0.10,
        "total_pct_error": pred.sum() / y.sum() - 1,
    }


def decile_table(y, pred) -> pd.DataFrame:
    """Customers split into 10 equal groups by predicted value (1 = highest)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "pred": np.asarray(pred, float)})
    df["decile"] = pd.qcut(df["pred"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    t = df.groupby("decile", observed=True).agg(customers=("y", "size"), mean_pred=("pred", "mean"),
                                                mean_actual=("y", "mean"), total_actual=("y", "sum"))
    t["share_of_actual"] = t["total_actual"] / t["total_actual"].sum()
    t["lift"] = t["mean_actual"] / df["y"].mean()
    return t


def bootstrap_ci(metric_fn, y, pred, n_boot: int = 500, seed: int = 42, alpha: float = 0.05) -> tuple:
    """Percentile CI by resampling customers (rows) with replacement.
    Valid for one snapshot (one row per customer); stacked snapshots need a customer-level bootstrap."""
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    stats = [metric_fn(y[i], pred[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))]
    return metric_fn(y, pred), *np.quantile(stats, [alpha / 2, 1 - alpha / 2])
