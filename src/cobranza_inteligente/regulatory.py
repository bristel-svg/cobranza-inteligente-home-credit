from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Enfoque conservador para una demo de cobranza en Chile:
# estas variables quedan excluidas del entrenamiento y de la decisión automática.
# Se conservan solo para análisis interno si el dataset las trae, pero no entran al modelo.
CMF_EXCLUDED_FEATURES = {
    # Atributos personales o proxies directos que no deberían guiar acciones de cobranza automatizadas.
    "CODE_GENDER",
    "DAYS_BIRTH",
    "AGE_YEARS",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_FAMILY_STATUS",
    # Target externo de Kaggle: no corresponde usarlo para la decisión de cobranza.
    "TARGET",
    # Identificadores puros.
    "SK_ID_CURR",
    "SK_ID_PREV",
    "SK_ID_BUREAU",
    # Flags auxiliares/operacionales que pueden introducir fuga de información o sesgo de disponibilidad.
    "has_future_observation",
}

# Patrones para evitar que se cuelen identificadores o datos de contacto si el cliente real los agrega.
DIRECT_IDENTIFIER_PATTERNS = [
    r"rut",
    r"dni",
    r"passport",
    r"pasaporte",
    r"nombre",
    r"apellido",
    r"email",
    r"correo",
    r"phone",
    r"telefono",
    r"celular",
    r"address",
    r"direccion",
    r"domicilio",
    r"device_id",
    r"ip_address",
]


@dataclass
class RegulatoryFeatureReport:
    strict_mode: bool
    removed_explicit_columns: list[str]
    removed_identifier_like_columns: list[str]
    retained_numeric_columns: list[str]
    retained_categorical_columns: list[str]

    def to_dict(self) -> dict:
        return {
            "strict_mode": self.strict_mode,
            "removed_explicit_columns": self.removed_explicit_columns,
            "removed_identifier_like_columns": self.removed_identifier_like_columns,
            "retained_numeric_columns": self.retained_numeric_columns,
            "retained_categorical_columns": self.retained_categorical_columns,
            "policy_summary": [
                "No usar variables personales directas o altamente sensibles/proxy para decidir acciones de cobranza.",
                "No usar identificadores puros como variables predictivas.",
                "Mantener trazabilidad de variables excluidas y variables usadas.",
                "Entregar recomendaciones para decisión humana; no ejecutar acciones irrevocables automáticamente.",
                "Excluir flags auxiliares que puedan introducir fuga temporal o sesgo de disponibilidad.",
            ],
        }


def _looks_like_direct_identifier(col: str) -> bool:
    normalized = col.lower().strip()
    return any(re.search(pattern, normalized) for pattern in DIRECT_IDENTIFIER_PATTERNS)


def apply_cmf_feature_policy(
    numeric_cols: list[str],
    categorical_cols: list[str],
    strict_mode: bool = True,
) -> tuple[list[str], list[str], RegulatoryFeatureReport]:
    """Filtra variables de modelamiento bajo una política conservadora compatible con uso regulado.

    No declara cumplimiento legal; crea un baseline prudente para una demo orientada a instituciones financieras chilenas.
    """
    if not strict_mode:
        report = RegulatoryFeatureReport(
            strict_mode=False,
            removed_explicit_columns=[],
            removed_identifier_like_columns=[],
            retained_numeric_columns=numeric_cols,
            retained_categorical_columns=categorical_cols,
        )
        return numeric_cols, categorical_cols, report

    removed_explicit: list[str] = []
    removed_identifier_like: list[str] = []

    def keep(col: str) -> bool:
        if col in CMF_EXCLUDED_FEATURES:
            removed_explicit.append(col)
            return False
        if _looks_like_direct_identifier(col):
            removed_identifier_like.append(col)
            return False
        return True

    filtered_numeric = [c for c in numeric_cols if keep(c)]
    filtered_categorical = [c for c in categorical_cols if keep(c)]

    report = RegulatoryFeatureReport(
        strict_mode=True,
        removed_explicit_columns=sorted(set(removed_explicit)),
        removed_identifier_like_columns=sorted(set(removed_identifier_like)),
        retained_numeric_columns=filtered_numeric,
        retained_categorical_columns=filtered_categorical,
    )
    return filtered_numeric, filtered_categorical, report


def build_reason_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Genera explicaciones simples tipo reason codes para el gestor de cobranza.

    Son reglas interpretables, no explicaciones SHAP. Evitan depender de librerías pesadas y ayudan a justificar la recomendación.
    """
    out = df.copy()
    reasons: list[str] = []
    for _, row in out.iterrows():
        items: list[str] = []
        if row.get("monto_vencido_actual", 0) >= out["monto_vencido_actual"].quantile(0.75):
            items.append("monto vencido alto")
        if row.get("hist_late_rate", 0) >= 0.50:
            items.append("historial con alta frecuencia de atraso")
        if row.get("hist_partial_rate", 0) >= 0.50:
            items.append("pagos parciales recurrentes")
        if row.get("roll3_payment_ratio_mean", 1) < 0.75:
            items.append("deterioro reciente en ratio de pago")
        if row.get("bureau_credit_day_overdue_max", 0) > 0:
            items.append("señal externa de atraso en bureau")
        if row.get("pos_skd_dpd_max", 0) > 0:
            items.append("atrasos históricos en POS/CASH")
        if row.get("cc_skd_dpd_max", 0) > 0:
            items.append("atrasos históricos en tarjeta")
        if not items:
            items.append("priorización por valor esperado neto")
        reasons.append("; ".join(items[:4]))
    out["razones_recomendacion"] = reasons
    return out


def add_human_in_the_loop_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Marca casos que no deben automatizarse sin revisión humana."""
    out = df.copy()
    amount = out.get("monto_vencido_actual", pd.Series(0, index=out.index)).fillna(0)
    prob = out.get("prob_regulariza", pd.Series(0, index=out.index)).fillna(0)
    value_net = out.get("valor_esperado_neto", pd.Series(0, index=out.index)).fillna(0)

    high_amount = amount >= amount.quantile(0.90) if len(amount) else False
    low_confidence = (prob >= 0.45) & (prob <= 0.55)
    negative_value = value_net < 0
    specialized = out.get("accion_recomendada", pd.Series("", index=out.index)).astype(str).str.contains(
        "descuento|especializada", case=False, regex=True
    )
    out["requiere_revision_humana"] = (high_amount | low_confidence | negative_value | specialized).astype(bool)
    return out


def write_regulatory_report(
    path: Path,
    feature_report: RegulatoryFeatureReport,
    metrics: dict,
    extra_notes: Iterable[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = feature_report.to_dict()
    payload["model_metrics"] = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
    payload["notes"] = list(extra_notes or [])
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
