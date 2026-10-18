"""Customer features as of a cutoff date.

Every feature is a function of (purchases on or before `as_of`, returns on or before `as_of`, the
calendar). `build_features` filters by `as_of` itself, so callers can pass the full tables.
Unit of behaviour = a purchase day (customer x date), as in notebook 02.

Design rule against train/serve skew: training cutoffs see at most ~190 days of history, the test
cutoff sees ~555. So sums/counts/maxima use fixed windows (<= 180 days) or are converted to rates;
only tenure is allowed to grow with history, and it is flagged.
"""
import numpy as np
import pandas as pd

from src.split import DATA_START, HORIZON_MONTHS, ONE_DAY, target_end

WINDOWS = (30, 90, 180)
PEAK_MONTHS = (9, 10, 11)  # Sep–Nov trade stocking peak (notebook 02)

FEATURE_CATALOG = [
    # feature, family, definition, why it should predict future value, when it is missing
    ("recency_days", "RFM", "days since last purchase day", "silence is the main sign of lapsing (nb 02: 83% → 53%)", "never"),
    *[(f"n_days_{w}d", "RFM", f"purchase days in last {w} days", "recent frequency = current engagement", "never (0)") for w in WINDOWS],
    *[(f"spend_{w}d", "RFM", f"£ spend in last {w} days", "recent value; spend_180d is the naive baseline", "never (0)") for w in WINDOWS],
    ("days_per_month", "Rate", "purchase days per 30 days since first purchase", "long-run habit, comparable across cutoffs", "never"),
    ("spend_per_month", "Rate", "£ per 30 days since first purchase", "long-run value; catches dormant big customers", "never"),
    ("tenure_days", "Tenure", "days since first purchase (≤ history available)", "survivors of early churn are stickier; grows with history → skew risk", "never"),
    ("tenure_censored", "Tenure", "first purchase within 30 days of data start", "tenure is only a lower bound for these customers", "never"),
    ("history_days", "Tenure", "days of data available before the cutoff", "tells the model how far the history features can be trusted", "never"),
    ("gap_mean", "Timing", "mean days between purchase days", "the customer's own rhythm", "one-day buyers"),
    ("gap_std", "Timing", "std of days between purchase days", "regular vs. erratic buyers", "< 3 purchase days"),
    ("gap_last", "Timing", "days between the last two purchase days", "is the rhythm speeding up or slowing down?", "one-day buyers"),
    ("overdue_ratio", "Timing", "recency_days ÷ gap_mean", "60 days' silence is alarming for a weekly buyer, normal for a quarterly one", "one-day buyers"),
    ("spend_trend", "Trend", "(spend last 90d − prior 90d) ÷ (sum); range −1…1", "growing vs. shrinking accounts", "history < 180 days"),
    ("days_trend", "Trend", "same, for purchase days", "frequency growing or shrinking", "history < 180 days"),
    ("avg_spend_per_day", "Basket", "£ per purchase day (all history)", "wholesaler vs. small shop", "never"),
    ("max_spend_day_180d", "Basket", "largest single-day spend in last 180 days", "capacity for big orders", "never (0)"),
    ("avg_items_per_day", "Basket", "units bought per purchase day (all history)", "bulk buying = trade customer", "never"),
    ("n_products_180d", "Breadth", "distinct StockCodes bought in last 180 days", "broad catalogue use = embedded supplier relationship", "never (0)"),
    ("return_value_share", "Returns", "£ returned ÷ £ purchased (all history)", "dissatisfaction, or value that won't stick", "never (0)"),
    ("has_returned", "Returns", "any return before the cutoff", "sparse signal (nb 02: 40% have one, median share 0%)", "never"),
    ("peak_days_in_window", "Calendar", "days of the target window that fall in Sep–Nov", "known in advance: does the next 6 months contain the peak?", "never"),
    ("country", "Country", "customer's most frequent country (raw; encoded in nb 05)", "market differences, shipping cost", "never"),
    ("is_uk", "Country", "country == United Kingdom", "the one level with enough data on its own", "never"),
]
CATALOG = pd.DataFrame(FEATURE_CATALOG, columns=["feature", "family", "definition", "why_it_should_predict", "missing_when"])
NUMERIC_FEATURES = [c for c in CATALOG["feature"] if c != "country"]


def _sym_trend(recent: pd.Series, prior: pd.Series) -> pd.Series:
    """(recent − prior) / (recent + prior): bounded in [−1, 1], 0 when both are zero.
    A plain ratio recent/prior explodes when prior = 0, which is common."""
    total = recent + prior
    return ((recent - prior) / total.where(total > 0)).fillna(0.0)


def peak_days_in_window(as_of: pd.Timestamp, months: int = HORIZON_MONTHS) -> int:
    days = pd.date_range(as_of + ONE_DAY, target_end(as_of, months))
    return int(days.month.isin(PEAK_MONTHS).sum())


def purchase_days(purchases: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Purchase lines on or before `as_of`, collapsed to one row per customer per day."""
    hist = purchases[purchases["InvoiceDate"] < as_of + ONE_DAY]
    days = (hist.assign(day=hist["InvoiceDate"].dt.normalize())
            .groupby(["Customer ID", "day"], as_index=False)
            .agg(spend=("Spend", "sum"), items=("Quantity", "sum"))
            .sort_values(["Customer ID", "day"]))
    days["age"] = (as_of - days["day"]).dt.days  # 0 = bought on the cutoff day
    return days


def build_features(purchases: pd.DataFrame, returns: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per customer with >= 1 purchase on or before `as_of`; index = Customer ID."""
    hist = purchases[purchases["InvoiceDate"] < as_of + ONE_DAY]
    ret = returns[returns["InvoiceDate"] < as_of + ONE_DAY]
    days = purchase_days(purchases, as_of)
    g = days.groupby("Customer ID")
    history_days = (as_of - DATA_START).days

    f = pd.DataFrame(index=g.size().index)
    f["recency_days"] = g["age"].min()
    for w in WINDOWS:
        in_w = days[days["age"] < w].groupby("Customer ID")
        f[f"n_days_{w}d"] = in_w.size().reindex(f.index).fillna(0)
        f[f"spend_{w}d"] = in_w["spend"].sum().reindex(f.index).fillna(0)

    first = g["day"].min()
    f["tenure_days"] = (as_of - first).dt.days
    months_seen = np.maximum(f["tenure_days"], 30) / 30
    f["days_per_month"] = g.size() / months_seen
    f["spend_per_month"] = g["spend"].sum() / months_seen
    f["tenure_censored"] = (first < DATA_START + pd.Timedelta(days=30)).astype(int)
    f["history_days"] = history_days

    days["gap"] = g["day"].diff().dt.days
    gaps = days.dropna(subset=["gap"]).groupby("Customer ID")["gap"]
    f["gap_mean"] = gaps.mean()
    f["gap_std"] = gaps.std()
    f["gap_last"] = gaps.last()
    f["overdue_ratio"] = f["recency_days"] / f["gap_mean"]

    if history_days >= 180:  # otherwise the "prior 90 days" window is partly before the data starts
        prior = days[(days["age"] >= 90) & (days["age"] < 180)].groupby("Customer ID")
        f["spend_trend"] = _sym_trend(f["spend_90d"], prior["spend"].sum().reindex(f.index).fillna(0))
        f["days_trend"] = _sym_trend(f["n_days_90d"], prior.size().reindex(f.index).fillna(0))
    else:
        f["spend_trend"] = f["days_trend"] = np.nan

    f["avg_spend_per_day"] = g["spend"].sum() / g.size()
    f["max_spend_day_180d"] = days[days["age"] < 180].groupby("Customer ID")["spend"].max().reindex(f.index).fillna(0)
    f["avg_items_per_day"] = g["items"].sum() / g.size()
    recent_lines = hist[hist["InvoiceDate"] >= as_of - pd.Timedelta(days=179)]
    f["n_products_180d"] = recent_lines.groupby("Customer ID")["StockCode"].nunique().reindex(f.index).fillna(0)

    returned = ret.groupby("Customer ID")["ReturnValue"].sum().reindex(f.index).fillna(0)
    f["return_value_share"] = returned / g["spend"].sum()
    f["has_returned"] = (returned > 0).astype(int)

    f["peak_days_in_window"] = peak_days_in_window(as_of)
    f["country"] = hist.groupby("Customer ID")["Country"].agg(lambda c: c.mode().iloc[0]).astype(str)
    f["is_uk"] = (f["country"] == "United Kingdom").astype(int)

    assert set(f.columns) == set(CATALOG["feature"]), "catalog and features out of sync"
    return f[CATALOG["feature"].tolist()]


def build_feature_table(purchases: pd.DataFrame, returns: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    """Attach as-of features to a stacked snapshot table (columns: cutoff, Customer ID, targets)."""
    parts = []
    for cutoff, snap in snapshots.groupby("cutoff"):
        feats = build_features(purchases, returns, cutoff)
        merged = snap.merge(feats, left_on="Customer ID", right_index=True, how="left", validate="1:1")
        assert merged["recency_days"].notna().all(), f"customer without features at {cutoff.date()}"
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)
