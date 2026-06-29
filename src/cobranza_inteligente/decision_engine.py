from __future__ import annotations

import numpy as np
import pandas as pd


def _quantile(series: pd.Series, q: float, default: float) -> float:
    """Retorna un cuantil robusto ante series vacías o no numéricas."""
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float(default)
    return float(values.quantile(q))


def _validate_non_negative_costs(**costs: float) -> None:
    invalid = {name: value for name, value in costs.items() if value < 0}
    if invalid:
        raise ValueError(f"Los costos de gestión deben ser no negativos: {invalid}")


def _safe_bucket_by_quantiles(
    series: pd.Series,
    q_low: float,
    q_high: float,
    labels: tuple[str, str, str],
    default_label: str,
) -> pd.Series:
    """Segmenta en tres tramos evitando fallas por cuantiles repetidos.

    pd.cut(..., duplicates="drop") puede fallar cuando los cuantiles colapsan y el
    número de etiquetas deja de coincidir con el número de intervalos. Esta función
    devuelve un tramo por defecto cuando la distribución no permite tres cortes válidos.
    """
    values = pd.to_numeric(series, errors="coerce")
    low = _quantile(values, q_low, default=np.nan)
    high = _quantile(values, q_high, default=np.nan)

    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return pd.Series(default_label, index=series.index, dtype="object")

    bucket = pd.cut(
        values,
        bins=[-np.inf, low, high, np.inf],
        labels=list(labels),
        include_lowest=True,
    )
    return bucket.astype("object").fillna(default_label)


def recommend_actions(
    df: pd.DataFrame,
    cost_sms: float = 150,
    cost_whatsapp: float = 100,
    cost_call: float = 1200,
    cost_specialist: float = 5000,
) -> pd.DataFrame:
    """Asigna acción, prioridad y valor esperado neto para una cartera de cobranza.

    Supuesto central del motor:
    - ``prob_regulariza`` estima P(regulariza = 1 | X).
    - ``monto_recuperado_pred`` estima E(monto recuperado | regulariza = 1, X).

    Por lo tanto, bajo un Hurdle Model, la recuperación esperada incondicional es:

        E[recuperación | X] = P(regulariza = 1 | X)
                            * E(monto recuperado | regulariza = 1, X)

    Los costos son supuestos de demo. En un proyecto real deben calibrarse con datos
    operacionales del cliente y, idealmente, con modelos de uplift/efecto incremental
    por canal de gestión.
    """
    _validate_non_negative_costs(
        cost_sms=cost_sms,
        cost_whatsapp=cost_whatsapp,
        cost_call=cost_call,
        cost_specialist=cost_specialist,
    )

    required = ["prob_regulariza", "monto_recuperado_pred", "monto_vencido_actual"]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas para motor de decisión: {missing}")

    out = df.copy()

    amount = pd.to_numeric(out["monto_vencido_actual"], errors="coerce").fillna(0).clip(lower=0)
    prob = pd.to_numeric(out["prob_regulariza"], errors="coerce").fillna(0).clip(0, 1)
    amount_pred = pd.to_numeric(out["monto_recuperado_pred"], errors="coerce").fillna(0).clip(lower=0)

    # Limpieza defensiva: mantiene las columnas principales en rangos económicamente válidos.
    out["monto_vencido_actual"] = amount
    out["prob_regulariza"] = prob
    out["monto_recuperado_pred"] = amount_pred

    # Hurdle Model estricto:
    # monto_recuperado_pred es condicional a regularizar, por lo que primero se acota
    # por la deuda vigente y luego se multiplica por la probabilidad de regularización.
    amount_pred_capped = np.minimum(amount_pred, amount)
    out["monto_recuperado_pred_acotado"] = amount_pred_capped
    out["recuperacion_esperada"] = prob * amount_pred_capped

    q_prob_low = _quantile(prob, 0.30, default=0.40)
    q_prob_high = _quantile(prob, 0.75, default=0.75)
    q_amount_high = _quantile(amount, 0.65, default=float(amount.mean()) if len(amount) else 0)
    q_amount_very_high = _quantile(amount, 0.85, default=float(amount.mean()) if len(amount) else 0)

    conditions = [
        (prob >= q_prob_high) & (amount < q_amount_high),
        (prob >= q_prob_high) & (amount >= q_amount_high),
        (prob >= q_prob_low) & (prob < q_prob_high) & (amount >= q_amount_high),
        (prob >= q_prob_low) & (prob < q_prob_high) & (amount < q_amount_high),
        (prob < q_prob_low) & (amount >= q_amount_very_high),
        (prob < q_prob_low) & (amount < q_amount_very_high),
    ]
    actions = [
        "WhatsApp/SMS automático",
        "Llamada prioritaria",
        "Llamada + propuesta de convenio",
        "Recordatorio automático y monitoreo",
        "Oferta de descuento / gestión especializada",
        "Automatizar y postergar",
    ]
    costs = [
        cost_sms + cost_whatsapp,
        cost_call,
        cost_call + cost_whatsapp,
        cost_sms,
        cost_specialist,
        cost_sms,
    ]

    out["accion_recomendada"] = np.select(conditions, actions, default="Revisar manualmente")
    out["costo_gestion_estimado"] = np.select(conditions, costs, default=cost_call).astype(float)

    # Si el pipeline trae una estimación explícita de recuperación sin gestión, se usa
    # para calcular beneficio incremental. Si no existe, se asume 0 para mantener la demo
    # autocontenida y evitar inventar un uplift no observado.
    if "recuperacion_base_estimada" in out.columns:
        base_recovery = pd.to_numeric(out["recuperacion_base_estimada"], errors="coerce").fillna(0).clip(lower=0)
        base_recovery = np.minimum(base_recovery, amount)
    elif "prob_regulariza_sin_gestion" in out.columns:
        prob_base = pd.to_numeric(out["prob_regulariza_sin_gestion"], errors="coerce").fillna(0).clip(0, 1)
        base_recovery = prob_base * amount_pred_capped
    else:
        base_recovery = pd.Series(0.0, index=out.index)

    out["recuperacion_base_estimada"] = base_recovery
    out["beneficio_incremental_estimado"] = (
        out["recuperacion_esperada"] - out["recuperacion_base_estimada"]
    ).clip(lower=0)
    out["valor_esperado_neto"] = out["beneficio_incremental_estimado"] - out["costo_gestion_estimado"]

    q50 = _quantile(out["valor_esperado_neto"], 0.50, default=0)
    q80 = _quantile(out["valor_esperado_neto"], 0.80, default=0)
    out["prioridad"] = np.select(
        [out["valor_esperado_neto"] >= q80, out["valor_esperado_neto"] >= q50],
        ["Alta", "Media"],
        default="Baja",
    )

    out["tramo_probabilidad"] = _safe_bucket_by_quantiles(
        prob,
        q_low=0.30,
        q_high=0.75,
        labels=("Baja", "Media", "Alta"),
        default_label="Media",
    )
    out["tramo_monto"] = _safe_bucket_by_quantiles(
        amount,
        q_low=0.33,
        q_high=0.66,
        labels=("Bajo", "Medio", "Alto"),
        default_label="Medio",
    )

    return out.sort_values("valor_esperado_neto", ascending=False).reset_index(drop=True)


def aggregate_to_current_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce múltiples eventos históricos a una fila por cliente-crédito.

    Home Credit contiene varias cuotas por el mismo SK_ID_CURR/SK_ID_PREV. Para una
    pantalla comercial de cobranza conviene mostrar una sola acción por crédito. Se conserva
    el evento con mayor valor esperado neto dentro de cada cliente-crédito.
    """
    if df.empty or not {"SK_ID_CURR", "SK_ID_PREV"}.issubset(df.columns):
        return df.copy()

    sort_cols = ["SK_ID_CURR", "SK_ID_PREV"]
    ascending = [True, True]

    if "valor_esperado_neto" in df.columns:
        sort_cols.append("valor_esperado_neto")
        ascending.append(False)
    if "cuota_numero_actual" in df.columns:
        sort_cols.append("cuota_numero_actual")
        ascending.append(False)

    ordered = df.sort_values(sort_cols, ascending=ascending)
    result = ordered.drop_duplicates(["SK_ID_CURR", "SK_ID_PREV"], keep="first")

    if "valor_esperado_neto" in result.columns:
        result = result.sort_values("valor_esperado_neto", ascending=False)

    return result.reset_index(drop=True)


def select_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "SK_ID_CURR",
        "SK_ID_PREV",
        "cuota_numero_actual",
        "dias_relativos_vencimiento",
        "monto_vencido_actual",
        "DPD",
        "PAYMENT_RATIO",
        "UNPAID_AMOUNT",
        "prob_regulariza",
        "prob_regulariza_raw",
        "prob_regulariza_sin_gestion",
        "monto_recuperado_pred",
        "monto_recuperado_pred_acotado",
        "recuperacion_esperada",
        "recuperacion_base_estimada",
        "beneficio_incremental_estimado",
        "costo_gestion_estimado",
        "valor_esperado_neto",
        "accion_recomendada",
        "prioridad",
        "requiere_revision_humana",
        "razones_recomendacion",
        "tramo_probabilidad",
        "tramo_monto",
        "regulariza_horizonte",
        "monto_recuperado_horizonte",
        "hist_n_installments",
        "hist_late_rate",
        "hist_partial_rate",
        "hist_dpd_mean",
        "hist_dpd_max",
        "roll3_late_rate",
        "roll6_late_rate",
        "last_dpd",
        "last_payment_ratio",
    ]
    existing = [c for c in preferred if c in df.columns]
    return df[existing].copy()
