from __future__ import annotations

import numpy as np
import pandas as pd


def add_collection_targets(df: pd.DataFrame, horizon_days: int = 60) -> pd.DataFrame:
    """Crea targets prospectivos de cobranza para cada evento.

    v4 cambia la lógica respecto del MVP:
    - El evento de cobranza es una cuota problemática ya observada: atrasada o parcial.
    - Como variables predictoras se permite usar el estado actual del evento: DPD, pago parcial, monto impago, etc.
    - El target mira hacia adelante dentro del mismo crédito: si en las próximas cuotas dentro del horizonte el cliente vuelve
      a pagar de forma razonablemente completa y sin atraso severo.

    Esto es más cercano a una pregunta real de cobranza:
    "dado el comportamiento observado hoy, ¿conviene gestionar este caso porque se recuperará en los próximos 60 días?".
    """
    required = [
        "SK_ID_PREV",
        "DAYS_INSTALMENT",
        "AMT_INSTALMENT",
        "AMT_PAYMENT",
        "PAYMENT_RATIO",
        "DPD",
    ]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas para crear targets: {missing}")

    out = df.sort_values(["SK_ID_PREV", "DAYS_INSTALMENT", "NUM_INSTALMENT_NUMBER"]).copy()
    group = out.groupby("SK_ID_PREV", sort=False)

    # Home Credit suele tener cuotas mensuales; 60 días equivale normalmente a mirar 1-2 cuotas futuras.
    future_cols = ["DAYS_INSTALMENT", "AMT_INSTALMENT", "AMT_PAYMENT", "PAYMENT_RATIO", "DPD"]
    future_parts = []
    for step in [1, 2, 3]:
        shifted = group[future_cols].shift(-step).add_prefix(f"f{step}_")
        future_parts.append(shifted)
    fut = pd.concat(future_parts, axis=1)
    out = pd.concat([out, fut], axis=1)

    recovered_amount = np.zeros(len(out), dtype=float)
    regularization_signal = np.zeros(len(out), dtype=bool)
    has_future = np.zeros(len(out), dtype=bool)

    for step in [1, 2, 3]:
        gap = out[f"f{step}_DAYS_INSTALMENT"] - out["DAYS_INSTALMENT"]
        in_horizon = gap.gt(0) & gap.le(horizon_days)
        has_future |= in_horizon.fillna(False).to_numpy()
        future_payment = out[f"f{step}_AMT_PAYMENT"].fillna(0)
        future_due = out[f"f{step}_AMT_INSTALMENT"].fillna(0)
        future_ratio = out[f"f{step}_PAYMENT_RATIO"].fillna(0)
        future_dpd = out[f"f{step}_DPD"].fillna(999)

        recovered_amount += np.where(in_horizon, np.minimum(future_payment, future_due), 0.0)
        good_future_payment = in_horizon & (future_ratio >= 0.98) & (future_dpd <= 15)
        regularization_signal |= good_future_payment.fillna(False).to_numpy()

    out["has_future_observation"] = has_future.astype(int)
    out["regulariza_horizonte"] = np.where(has_future, regularization_signal.astype(int), np.nan)
    out["monto_recuperado_horizonte"] = np.where(has_future, recovered_amount, np.nan)

    if horizon_days == 60:
        out["regulariza_60d"] = out["regulariza_horizonte"]
        out["monto_recuperado_60d"] = out["monto_recuperado_horizonte"]

    # Elimina columnas auxiliares futuras para evitar fuga de información en el modelamiento.
    future_aux_cols = [c for c in out.columns if c.startswith("f") and "_" in c]
    out = out.drop(columns=future_aux_cols)
    return out.reset_index(drop=True)
