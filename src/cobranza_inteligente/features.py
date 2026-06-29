from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_divide

INSTALLMENT_REQUIRED_COLUMNS = [
    "SK_ID_PREV",
    "SK_ID_CURR",
    "NUM_INSTALMENT_VERSION",
    "NUM_INSTALMENT_NUMBER",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
    "AMT_INSTALMENT",
    "AMT_PAYMENT",
]


def prepare_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """Crea variables básicas de pago por cuota.

    Importante:
    - DAYS_INSTALMENT es la fecha pactada de vencimiento.
    - DAYS_ENTRY_PAYMENT es la fecha real de pago.
    - Las fechas son días relativos a la solicitud actual; valores más cercanos a 0 son más recientes.
    """
    missing = sorted(set(INSTALLMENT_REQUIRED_COLUMNS) - set(installments.columns))
    if missing:
        raise ValueError(f"Faltan columnas en installments_payments: {missing}")

    df = installments[INSTALLMENT_REQUIRED_COLUMNS].copy()

    numeric_cols = [
        "NUM_INSTALMENT_VERSION",
        "NUM_INSTALMENT_NUMBER",
        "DAYS_INSTALMENT",
        "DAYS_ENTRY_PAYMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["AMT_PAYMENT"] = df["AMT_PAYMENT"].fillna(0)
    df["AMT_INSTALMENT"] = df["AMT_INSTALMENT"].fillna(0)

    df["DPD"] = (df["DAYS_ENTRY_PAYMENT"] - df["DAYS_INSTALMENT"]).clip(lower=0)
    df["DBD"] = (df["DAYS_INSTALMENT"] - df["DAYS_ENTRY_PAYMENT"]).clip(lower=0)
    df["PAYMENT_RATIO"] = safe_divide(df["AMT_PAYMENT"], df["AMT_INSTALMENT"], default=0).clip(0, 5)
    df["UNPAID_AMOUNT"] = (df["AMT_INSTALMENT"] - df["AMT_PAYMENT"]).clip(lower=0)
    df["OVERPAY_AMOUNT"] = (df["AMT_PAYMENT"] - df["AMT_INSTALMENT"]).clip(lower=0)
    df["IS_LATE"] = (df["DPD"] > 0).astype(int)
    df["IS_PARTIAL"] = (df["PAYMENT_RATIO"] < 0.98).astype(int)
    df["IS_FULL_PAYMENT"] = (df["PAYMENT_RATIO"] >= 0.98).astype(int)
    df["SEVERITY"] = df["DPD"].fillna(0) * df["AMT_INSTALMENT"].fillna(0)

    # Orden temporal: de más antiguo a más reciente.
    df = df.sort_values(["SK_ID_PREV", "DAYS_INSTALMENT", "NUM_INSTALMENT_NUMBER"]).reset_index(drop=True)
    return df


def _add_group_history_features(group: pd.DataFrame) -> pd.DataFrame:
    """Versión de compatibilidad para grupos pequeños."""
    return _add_history_features_fast(group)


def _rolling_by_group(shifted_col: pd.Series, group_key: pd.Series, window: int, stat: str) -> pd.Series:
    roller = shifted_col.groupby(group_key).rolling(window=window, min_periods=1)
    if stat == "mean":
        out = roller.mean()
    elif stat == "max":
        out = roller.max()
    elif stat == "sum":
        out = roller.sum()
    else:
        raise ValueError(stat)
    return out.reset_index(level=0, drop=True).sort_index()


def _add_history_features_fast(df: pd.DataFrame) -> pd.DataFrame:
    """Features históricas vectorizadas usando solo cuotas anteriores al evento actual.

    Reemplaza el loop por crédito, que es lento con el dataset real de Home Credit.
    """
    g = df.copy()
    key = g["SK_ID_PREV"]
    g["hist_n_installments"] = g.groupby("SK_ID_PREV").cumcount()

    base_cols = [
        "DPD",
        "PAYMENT_RATIO",
        "IS_LATE",
        "IS_PARTIAL",
        "IS_FULL_PAYMENT",
        "AMT_PAYMENT",
        "AMT_INSTALMENT",
        "UNPAID_AMOUNT",
        "SEVERITY",
    ]
    shifted = g.groupby("SK_ID_PREV")[base_cols].shift(1)
    count = g["hist_n_installments"].replace(0, np.nan)

    def cum_mean(col: str) -> pd.Series:
        return shifted[col].fillna(0).groupby(key).cumsum() / count

    def cum_sum(col: str) -> pd.Series:
        return shifted[col].fillna(0).groupby(key).cumsum()

    def cum_max(col: str) -> pd.Series:
        return shifted[col].groupby(key).cummax()

    # Std histórico: sqrt(E[x^2] - E[x]^2), calculado hasta cuota anterior.
    dpd_sum = shifted["DPD"].fillna(0).groupby(key).cumsum()
    dpd_sq_sum = (shifted["DPD"].fillna(0) ** 2).groupby(key).cumsum()
    dpd_mean = dpd_sum / count
    dpd_var = (dpd_sq_sum / count) - (dpd_mean ** 2)

    g["hist_dpd_mean"] = dpd_mean
    g["hist_dpd_max"] = cum_max("DPD")
    g["hist_dpd_std"] = np.sqrt(dpd_var.clip(lower=0))
    g["hist_late_rate"] = cum_mean("IS_LATE")
    g["hist_partial_rate"] = cum_mean("IS_PARTIAL")
    g["hist_full_payment_rate"] = cum_mean("IS_FULL_PAYMENT")
    g["hist_payment_ratio_mean"] = cum_mean("PAYMENT_RATIO")
    g["hist_payment_ratio_min"] = shifted["PAYMENT_RATIO"].groupby(key).cummin()
    g["hist_unpaid_sum"] = cum_sum("UNPAID_AMOUNT")
    g["hist_severity_sum"] = cum_sum("SEVERITY")
    g["hist_payment_sum"] = cum_sum("AMT_PAYMENT")
    g["hist_instalment_sum"] = cum_sum("AMT_INSTALMENT")

    g["last_dpd"] = shifted["DPD"]
    g["last_payment_ratio"] = shifted["PAYMENT_RATIO"]
    g["last_was_late"] = shifted["IS_LATE"]
    g["last_was_partial"] = shifted["IS_PARTIAL"]

    for window in [3, 6]:
        g[f"roll{window}_dpd_mean"] = _rolling_by_group(shifted["DPD"], key, window, "mean")
        g[f"roll{window}_dpd_max"] = _rolling_by_group(shifted["DPD"], key, window, "max")
        g[f"roll{window}_late_rate"] = _rolling_by_group(shifted["IS_LATE"], key, window, "mean")
        g[f"roll{window}_partial_rate"] = _rolling_by_group(shifted["IS_PARTIAL"], key, window, "mean")
        g[f"roll{window}_payment_ratio_mean"] = _rolling_by_group(shifted["PAYMENT_RATIO"], key, window, "mean")
        g[f"roll{window}_unpaid_sum"] = _rolling_by_group(shifted["UNPAID_AMOUNT"], key, window, "sum")
        g[f"roll{window}_severity_sum"] = _rolling_by_group(shifted["SEVERITY"], key, window, "sum")

    return g


def build_collection_snapshots(
    installments: pd.DataFrame,
    previous: pd.DataFrame | None = None,
    application: pd.DataFrame | None = None,
    min_prev_installments: int = 2,
) -> pd.DataFrame:
    """Construye una base modelable de eventos de cobranza.

    Cada fila representa una cuota que llega a su vencimiento y que, según el pago observado,
    fue tardía o parcial. Las variables predictoras se calculan usando solo historial anterior
    dentro del mismo crédito previo.
    """
    inst = prepare_installments(installments)

    # Crea variables históricas sin fuga temporal.
    # v4: vectorizado para correr con el dataset real de Home Credit.
    if inst.empty:
        return pd.DataFrame()
    featured = _add_history_features_fast(inst)

    # Evento de cobranza: cuota que no fue pagada completa y a tiempo.
    # Se usa para simular que en el vencimiento habría que gestionar esa cuota.
    collection_mask = (featured["IS_LATE"] == 1) | (featured["IS_PARTIAL"] == 1)
    snapshots = featured.loc[collection_mask].copy()
    snapshots = snapshots[snapshots["hist_n_installments"] >= min_prev_installments].copy()

    # Variables actuales conocidas en el vencimiento.
    snapshots["monto_vencido_actual"] = snapshots["AMT_INSTALMENT"].fillna(0)
    snapshots["cuota_numero_actual"] = snapshots["NUM_INSTALMENT_NUMBER"].fillna(0)
    snapshots["dias_relativos_vencimiento"] = snapshots["DAYS_INSTALMENT"].fillna(0)

    # Merges opcionales.
    if previous is not None and {"SK_ID_PREV", "SK_ID_CURR"}.issubset(previous.columns):
        prev = previous.copy()
        # Evita duplicados por seguridad.
        prev = prev.drop_duplicates("SK_ID_PREV")
        snapshots = snapshots.merge(prev, on=["SK_ID_PREV", "SK_ID_CURR"], how="left", suffixes=("", "_prev"))

    if application is not None and "SK_ID_CURR" in application.columns:
        app = application.copy().drop_duplicates("SK_ID_CURR")
        # TARGET se conserva para análisis, pero no debe usarse como feature del modelo de cobranza.
        snapshots = snapshots.merge(app, on="SK_ID_CURR", how="left", suffixes=("", "_app"))

    return snapshots.reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Devuelve columnas numéricas y categóricas candidatas para modelar.

    Excluye columnas objetivo y variables que se conocen solo después del vencimiento.
    """
    forbidden = {
        "regulariza_horizonte",
        "monto_recuperado_horizonte",
        "regulariza_60d",
        "monto_recuperado_60d",
        "TARGET",
        # Flags auxiliares de construcción de target: no deben entrar al modelo.
        "has_future_observation",
        # Identificadores puros
        "SK_ID_PREV",
        "SK_ID_CURR",
        # Variables crudas que pueden revelar fecha/monto exacto de pago.
        # En v4 sí permitimos variables derivadas del estado actual del evento
        # como DPD, PAYMENT_RATIO, UNPAID_AMOUNT y SEVERITY, porque el target es prospectivo.
        "DAYS_ENTRY_PAYMENT",
        "AMT_PAYMENT",
        "DBD",
        "OVERPAY_AMOUNT",
    }

    candidate_cols = [c for c in df.columns if c not in forbidden]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in candidate_cols if c not in numeric_cols]

    # Evita columnas con demasiadas categorías, que pueden inflar la demo innecesariamente.
    categorical_cols = [c for c in categorical_cols if df[c].nunique(dropna=True) <= 80]

    return numeric_cols, categorical_cols
