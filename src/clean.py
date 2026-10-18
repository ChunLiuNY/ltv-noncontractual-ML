"""Load and clean UCI Online Retail II.

Every cleaning rule is a named step so notebook 01 can log rows removed per rule.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = PROJECT_ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_PARQUET = PROJECT_ROOT / "data" / "processed" / "transactions_raw.parquet"
CLEAN_PARQUET = PROJECT_ROOT / "data" / "processed" / "transactions_clean.parquet"
RETURNS_PARQUET = PROJECT_ROOT / "data" / "processed" / "returns_clean.parquet"

# StockCodes that are fees, postage, adjustments, or vouchers -- not products.
NON_PRODUCT_CODES = {
    "POST", "DOT", "M", "m", "BANK CHARGES", "AMAZONFEE", "ADJUST", "ADJUST2",
    "D", "C2", "CRUK", "S", "B", "PADS", "TEST001", "TEST002",
}


def load_raw(force_xlsx: bool = False) -> pd.DataFrame:
    """Read both xlsx sheets once and cache as parquet; afterwards read the parquet."""
    if RAW_PARQUET.exists() and not force_xlsx:
        return pd.read_parquet(RAW_PARQUET)
    sheets = pd.read_excel(RAW_XLSX, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df["Description"] = df["Description"].astype("string")
    df["Country"] = df["Country"].astype("string")
    RAW_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_PARQUET, index=False)
    return df


def is_non_product(stock_code: pd.Series) -> pd.Series:
    """Fee/postage/adjustment codes plus gift vouchers (gift_0001_10 etc.)."""
    code = stock_code.astype(str).str.strip()
    return code.isin(NON_PRODUCT_CODES) | code.str.lower().str.startswith("gift_")


def is_cancellation(invoice: pd.Series) -> pd.Series:
    return invoice.astype(str).str.upper().str.startswith("C")


def find_immediate_reversals(purchases: pd.DataFrame, returns: pd.DataFrame,
                             max_hours: float = 24) -> pd.DataFrame:
    """Pair purchase lines with a cancellation of the exact same line soon after.

    Same customer, StockCode, Price and |Quantity|, cancelled within `max_hours`.
    These are order-entry errors (e.g. 80,995 units cancelled 12 minutes later), not
    demand. Each purchase and each cancellation is used at most once.
    Returns a frame with the matched index labels: `purchase_idx`, `return_idx`.
    """
    keys = ["Customer ID", "StockCode", "Price"]
    p = purchases[keys + ["Quantity", "InvoiceDate"]].rename_axis("purchase_idx").reset_index()
    r = (returns[keys + ["Quantity", "InvoiceDate"]].assign(Quantity=lambda d: -d["Quantity"])
         .rename_axis("return_idx").reset_index())
    pairs = p.merge(r, on=keys + ["Quantity"], suffixes=("_p", "_r"))
    lag_h = (pairs["InvoiceDate_r"] - pairs["InvoiceDate_p"]).dt.total_seconds() / 3600
    pairs = pairs[(lag_h >= 0) & (lag_h <= max_hours)].assign(lag_h=lag_h)
    pairs = (pairs.sort_values("lag_h")
             .drop_duplicates("purchase_idx").drop_duplicates("return_idx"))
    return pairs[["purchase_idx", "return_idx", "lag_h"]]


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply cleaning rules in order.

    Returns
    -------
    purchases : product purchase lines, with Spend = Quantity * Price
    returns   : product cancellation lines (positive ReturnValue), kept for return-rate features
    log       : rows removed per rule, in the order applied
    """
    log = []

    def step(frame, keep_mask, rule, why):
        removed = int((~keep_mask).sum())
        log.append({"rule": rule, "rows_removed": removed, "rows_left": len(frame) - removed, "why": why})
        return frame[keep_mask]

    out = df.copy()
    log.append({"rule": "raw (both sheets)", "rows_removed": 0, "rows_left": len(out), "why": ""})

    out = step(out, ~out.duplicated(), "exact duplicate rows",
               "same invoice line twice = data-entry/export artefact; would double-count spend")
    out = step(out, out["Customer ID"].notna(), "missing Customer ID",
               "can't attribute to a customer; guest/offline sales -> population bias")
    out = step(out, ~is_non_product(out["StockCode"]), "non-product StockCode",
               "postage, fees, manual adjustments, vouchers are not customer demand")
    out = step(out, out["Price"] > 0, "Price <= 0",
               "free samples / bad-debt adjustments; no revenue meaning")

    out = out.assign(**{"Customer ID": out["Customer ID"].astype(int)})
    cancel = is_cancellation(out["Invoice"])

    returns = out[cancel].assign(ReturnValue=lambda d: -d["Quantity"] * d["Price"])
    out = step(out, ~cancel, "cancellations (set aside)",
               "negative lines are returns, not purchases; kept separately for return-rate features")
    out = step(out, out["Quantity"] > 0, "Quantity <= 0 (non-cancel)",
               "stock corrections without a 'C' invoice")

    pairs = find_immediate_reversals(out, returns)
    out = step(out, ~out.index.isin(pairs["purchase_idx"]), "purchase reversed within 24h",
               "order-entry error: exact same line cancelled within a day; never real demand")
    returns = returns.drop(index=pairs["return_idx"])

    purchases = out.assign(Spend=out["Quantity"] * out["Price"]).reset_index(drop=True)
    return purchases, returns.reset_index(drop=True), pd.DataFrame(log)
