from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import read_csv_if_exists, reduce_memory_usage, safe_divide

OPTIONAL_FILES = {
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
}

BUREAU_COLS = [
    "SK_ID_CURR",
    "SK_ID_BUREAU",
    "CREDIT_ACTIVE",
    "DAYS_CREDIT",
    "CREDIT_DAY_OVERDUE",
    "DAYS_CREDIT_ENDDATE",
    "DAYS_ENDDATE_FACT",
    "AMT_CREDIT_MAX_OVERDUE",
    "AMT_CREDIT_SUM",
    "AMT_CREDIT_SUM_DEBT",
    "AMT_CREDIT_SUM_OVERDUE",
    "AMT_CREDIT_SUM_LIMIT",
    "CREDIT_TYPE",
]

BUREAU_BALANCE_COLS = ["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"]

POS_CASH_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "CNT_INSTALMENT",
    "CNT_INSTALMENT_FUTURE",
    "NAME_CONTRACT_STATUS",
    "SK_DPD",
    "SK_DPD_DEF",
]

CREDIT_CARD_COLS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "MONTHS_BALANCE",
    "AMT_BALANCE",
    "AMT_CREDIT_LIMIT_ACTUAL",
    "AMT_DRAWINGS_CURRENT",
    "AMT_INST_MIN_REGULARITY",
    "AMT_PAYMENT_CURRENT",
    "AMT_PAYMENT_TOTAL_CURRENT",
    "AMT_RECEIVABLE_PRINCIPAL",
    "AMT_RECIVABLE",
    "AMT_TOTAL_RECEIVABLE",
    "CNT_DRAWINGS_CURRENT",
    "SK_DPD",
    "SK_DPD_DEF",
    "NAME_CONTRACT_STATUS",
]


def _load_optional(path: Path, usecols: list[str], max_rows: int | None) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return read_csv_if_exists(path, usecols=usecols, nrows=max_rows)


def load_optional_tables(raw_dir: Path, max_rows: int | None = None) -> dict[str, pd.DataFrame | None]:
    return {
        "bureau": _load_optional(raw_dir / OPTIONAL_FILES["bureau"], BUREAU_COLS, max_rows),
        "bureau_balance": _load_optional(raw_dir / OPTIONAL_FILES["bureau_balance"], BUREAU_BALANCE_COLS, max_rows),
        "pos_cash": _load_optional(raw_dir / OPTIONAL_FILES["pos_cash"], POS_CASH_COLS, max_rows),
        "credit_card": _load_optional(raw_dir / OPTIONAL_FILES["credit_card"], CREDIT_CARD_COLS, max_rows),
    }


def _flatten_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{prefix}_{a}_{b}" if b else f"{prefix}_{a}" for a, b in df.columns]
    return df


def build_bureau_features(bureau: pd.DataFrame | None, bureau_balance: pd.DataFrame | None = None) -> pd.DataFrame:
    if bureau is None or bureau.empty or "SK_ID_CURR" not in bureau.columns:
        return pd.DataFrame()
    b = bureau.copy()
    numeric_cols = [c for c in b.columns if c not in {"SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE", "CREDIT_TYPE"}]
    for col in numeric_cols:
        b[col] = pd.to_numeric(b[col], errors="coerce")

    b["bureau_is_active"] = (b.get("CREDIT_ACTIVE", "") == "Active").astype(int)
    b["bureau_is_closed"] = (b.get("CREDIT_ACTIVE", "") == "Closed").astype(int)
    b["bureau_debt_to_credit"] = safe_divide(b.get("AMT_CREDIT_SUM_DEBT", 0), b.get("AMT_CREDIT_SUM", 0), 0)
    b["bureau_has_overdue"] = (b.get("AMT_CREDIT_SUM_OVERDUE", 0).fillna(0) > 0).astype(int)

    agg = b.groupby("SK_ID_CURR").agg(
        bureau_n_credits=("SK_ID_BUREAU", "nunique"),
        bureau_active_count=("bureau_is_active", "sum"),
        bureau_closed_count=("bureau_is_closed", "sum"),
        bureau_days_credit_min=("DAYS_CREDIT", "min"),
        bureau_days_credit_max=("DAYS_CREDIT", "max"),
        bureau_credit_day_overdue_max=("CREDIT_DAY_OVERDUE", "max"),
        bureau_credit_sum_total=("AMT_CREDIT_SUM", "sum"),
        bureau_credit_debt_total=("AMT_CREDIT_SUM_DEBT", "sum"),
        bureau_credit_overdue_total=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        bureau_max_overdue_max=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_debt_to_credit_mean=("bureau_debt_to_credit", "mean"),
        bureau_has_overdue_rate=("bureau_has_overdue", "mean"),
    ).reset_index()

    if bureau_balance is not None and not bureau_balance.empty and {"SK_ID_BUREAU", "STATUS"}.issubset(bureau_balance.columns):
        bb = bureau_balance.copy()
        status_map = {"C": 0, "X": np.nan, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}
        bb["bb_status_num"] = bb["STATUS"].map(status_map)
        bb_agg = bb.groupby("SK_ID_BUREAU").agg(
            bb_months_count=("MONTHS_BALANCE", "count"),
            bb_status_max=("bb_status_num", "max"),
            bb_status_mean=("bb_status_num", "mean"),
            bb_delinquent_rate=("bb_status_num", lambda s: float((s.fillna(0) > 0).mean())),
        ).reset_index()
        link = b[["SK_ID_CURR", "SK_ID_BUREAU"]].drop_duplicates().merge(bb_agg, on="SK_ID_BUREAU", how="left")
        bb_curr = link.groupby("SK_ID_CURR").agg(
            bb_status_max=("bb_status_max", "max"),
            bb_status_mean=("bb_status_mean", "mean"),
            bb_delinquent_rate_mean=("bb_delinquent_rate", "mean"),
            bb_months_total=("bb_months_count", "sum"),
        ).reset_index()
        agg = agg.merge(bb_curr, on="SK_ID_CURR", how="left")

    return reduce_memory_usage(agg)


def build_pos_cash_features(pos_cash: pd.DataFrame | None) -> pd.DataFrame:
    if pos_cash is None or pos_cash.empty or "SK_ID_CURR" not in pos_cash.columns:
        return pd.DataFrame()
    p = pos_cash.copy()
    for col in ["MONTHS_BALANCE", "CNT_INSTALMENT", "CNT_INSTALMENT_FUTURE", "SK_DPD", "SK_DPD_DEF"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")
    p["pos_is_active"] = p.get("NAME_CONTRACT_STATUS", pd.Series("", index=p.index)).isin(["Active", "Signed"]).astype(int)
    agg = p.groupby("SK_ID_CURR").agg(
        pos_records_count=("SK_ID_PREV", "count"),
        pos_prev_count=("SK_ID_PREV", "nunique"),
        pos_months_min=("MONTHS_BALANCE", "min"),
        pos_months_max=("MONTHS_BALANCE", "max"),
        pos_cnt_instalment_mean=("CNT_INSTALMENT", "mean"),
        pos_cnt_instalment_future_mean=("CNT_INSTALMENT_FUTURE", "mean"),
        pos_skd_dpd_mean=("SK_DPD", "mean"),
        pos_skd_dpd_max=("SK_DPD", "max"),
        pos_skd_dpd_def_mean=("SK_DPD_DEF", "mean"),
        pos_skd_dpd_def_max=("SK_DPD_DEF", "max"),
        pos_active_rate=("pos_is_active", "mean"),
    ).reset_index()
    return reduce_memory_usage(agg)


def build_credit_card_features(credit_card: pd.DataFrame | None) -> pd.DataFrame:
    if credit_card is None or credit_card.empty or "SK_ID_CURR" not in credit_card.columns:
        return pd.DataFrame()
    cc = credit_card.copy()
    numeric_cols = [c for c in cc.columns if c not in {"SK_ID_PREV", "SK_ID_CURR", "NAME_CONTRACT_STATUS"}]
    for col in numeric_cols:
        cc[col] = pd.to_numeric(cc[col], errors="coerce")
    cc["cc_utilization"] = safe_divide(cc.get("AMT_BALANCE", 0), cc.get("AMT_CREDIT_LIMIT_ACTUAL", 0), 0).clip(0, 5)
    cc["cc_payment_to_balance"] = safe_divide(cc.get("AMT_PAYMENT_CURRENT", 0), cc.get("AMT_BALANCE", 0), 0).clip(0, 5)
    agg = cc.groupby("SK_ID_CURR").agg(
        cc_records_count=("SK_ID_PREV", "count"),
        cc_prev_count=("SK_ID_PREV", "nunique"),
        cc_balance_mean=("AMT_BALANCE", "mean"),
        cc_balance_max=("AMT_BALANCE", "max"),
        cc_credit_limit_mean=("AMT_CREDIT_LIMIT_ACTUAL", "mean"),
        cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
        cc_payment_current_mean=("AMT_PAYMENT_CURRENT", "mean"),
        cc_receivable_principal_mean=("AMT_RECEIVABLE_PRINCIPAL", "mean"),
        cc_utilization_mean=("cc_utilization", "mean"),
        cc_utilization_max=("cc_utilization", "max"),
        cc_payment_to_balance_mean=("cc_payment_to_balance", "mean"),
        cc_skd_dpd_mean=("SK_DPD", "mean"),
        cc_skd_dpd_max=("SK_DPD", "max"),
        cc_skd_dpd_def_mean=("SK_DPD_DEF", "mean"),
        cc_skd_dpd_def_max=("SK_DPD_DEF", "max"),
    ).reset_index()
    return reduce_memory_usage(agg)


def build_optional_customer_features(optional_tables: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    parts = []
    bureau_features = build_bureau_features(optional_tables.get("bureau"), optional_tables.get("bureau_balance"))
    pos_features = build_pos_cash_features(optional_tables.get("pos_cash"))
    cc_features = build_credit_card_features(optional_tables.get("credit_card"))
    for part in [bureau_features, pos_features, cc_features]:
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on="SK_ID_CURR", how="outer")
    return reduce_memory_usage(out)
