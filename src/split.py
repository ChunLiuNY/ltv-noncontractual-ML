"""Cutoff dates, snapshot (customer x cutoff) builder, and time-based folds.

Label-maturity rule: a snapshot at cutoff c has a target that is only known at c + horizon.
A model "built on date T" may only train on snapshots whose target window ends on or before T.
"""
import pandas as pd

HORIZON_MONTHS = 6
MARGIN = 0.30                             # assumed gross margin: data has revenue only
DATA_START = pd.Timestamp("2009-12-01")
VAL_CUTOFF = pd.Timestamp("2010-12-09")   # model selection and tuning
TEST_CUTOFF = pd.Timestamp("2011-06-09")  # touched once, in notebook 12
DATA_END = pd.Timestamp("2011-12-09")
ONE_DAY = pd.Timedelta(days=1)


def before_test(df: pd.DataFrame, date_col: str = "InvoiceDate") -> pd.DataFrame:
    """Rows on or before the test cutoff day; the test target window stays unseen."""
    return df[df[date_col] < TEST_CUTOFF + ONE_DAY]


def target_end(cutoff: pd.Timestamp, months: int = HORIZON_MONTHS) -> pd.Timestamp:
    """Last day (inclusive) of the target window that starts the day after `cutoff`."""
    return cutoff + pd.DateOffset(months=months)


def monthly_cutoffs(first_month: str, last_month: str) -> list[pd.Timestamp]:
    """Cutoffs on the 9th of each month from `first_month` to `last_month` ("YYYY-MM"), inclusive.
    The data ends on the 9th, so 6-month windows line up exactly."""
    months = pd.period_range(first_month, last_month, freq="M")
    return [m.to_timestamp() + pd.Timedelta(days=8) for m in months]


def cutoffs_with_mature_labels(as_of: pd.Timestamp, first_month: str = "2010-01") -> list[pd.Timestamp]:
    """All monthly cutoffs whose target window ends on or before `as_of`."""
    return [c for c in monthly_cutoffs(first_month, as_of.strftime("%Y-%m")) if target_end(c) <= as_of]


# Selection stage: what we'd have on the validation date. Final refit: what we'd have on the test date.
TRAIN_CUTOFFS = cutoffs_with_mature_labels(VAL_CUTOFF)    # 2010-01-09 ... 2010-06-09
FINAL_CUTOFFS = cutoffs_with_mature_labels(TEST_CUTOFF)   # 2010-01-09 ... 2010-12-09


def build_snapshot(purchases: pd.DataFrame, cutoff: pd.Timestamp,
                   months: int = HORIZON_MONTHS, allow_test: bool = False) -> pd.DataFrame:
    """One row per customer with >= 1 purchase on or before `cutoff`, plus 6-month targets.

    History = purchases on or before the cutoff day. Target window = the day after the
    cutoff through `target_end(cutoff)`, inclusive. y_spend is gross purchase value
    (returns are not netted). Refuses to build targets inside the test window unless
    `allow_test=True` (notebook 12 only).
    """
    end = target_end(cutoff, months)
    if end > DATA_END:
        raise ValueError(f"Target window for {cutoff.date()} ends {end.date()}, after the data ends.")
    if end > TEST_CUTOFF and not allow_test:
        raise ValueError(f"Target window for {cutoff.date()} overlaps the test window; pass allow_test=True only in notebook 12.")

    t = purchases["InvoiceDate"]
    history = purchases[t < cutoff + ONE_DAY]
    future = purchases[(t >= cutoff + ONE_DAY) & (t < end + ONE_DAY)]

    snap = history.groupby("Customer ID").agg(first_purchase=("InvoiceDate", "min"))
    snap["y_spend"] = future.groupby("Customer ID")["Spend"].sum().reindex(snap.index).fillna(0.0)
    snap["y_buy"] = (snap["y_spend"] > 0).astype(int)
    snap["y_ltv"] = snap["y_spend"] * MARGIN
    snap.insert(0, "cutoff", cutoff)
    return snap.drop(columns="first_purchase").reset_index()


def build_snapshots(purchases: pd.DataFrame, cutoffs: list[pd.Timestamp], **kwargs) -> pd.DataFrame:
    """Stack snapshots for several cutoffs (same customer appears once per cutoff)."""
    return pd.concat([build_snapshot(purchases, c, **kwargs) for c in cutoffs], ignore_index=True)


def time_folds(val_cutoffs: list[pd.Timestamp]) -> list[tuple[list[pd.Timestamp], pd.Timestamp]]:
    """Expanding-window folds: validate at each cutoff, train on every cutoff whose
    labels had matured by then. Used for tuning (notebook 08) and baseline stability."""
    return [(cutoffs_with_mature_labels(v), v) for v in val_cutoffs]
