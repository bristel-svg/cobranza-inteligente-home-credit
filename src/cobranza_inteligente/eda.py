from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


RAW_TABLES = {
    "application_train": ["application_train", "application_train.csv", "application_train.parquet"],
    "previous_application": ["previous_application", "previous_application.csv", "previous_application.parquet"],
    "installments_payments": ["installments_payments", "installments_payments.csv", "installments_payments.parquet"],
    "bureau": ["bureau", "bureau.csv", "bureau.parquet"],
    "bureau_balance": ["bureau_balance", "bureau_balance.csv", "bureau_balance.parquet"],
    "POS_CASH_balance": ["POS_CASH_balance", "POS_CASH_balance.csv", "POS_CASH_balance.parquet"],
    "credit_card_balance": ["credit_card_balance", "credit_card_balance.csv", "credit_card_balance.parquet"],
}

KEY_RELATIONSHIPS = [
    ("application_train", "SK_ID_CURR", "previous_application", "SK_ID_CURR"),
    ("application_train", "SK_ID_CURR", "bureau", "SK_ID_CURR"),
    ("previous_application", "SK_ID_PREV", "installments_payments", "SK_ID_PREV"),
    ("previous_application", "SK_ID_PREV", "POS_CASH_balance", "SK_ID_PREV"),
    ("previous_application", "SK_ID_PREV", "credit_card_balance", "SK_ID_PREV"),
    ("bureau", "SK_ID_BUREAU", "bureau_balance", "SK_ID_BUREAU"),
]

SENSITIVE_OR_EXCLUDED_HINTS = [
    "gender",
    "birth",
    "children",
    "family",
    "fam_",
    "name_family_status",
    "code_gender",
]

LEAKAGE_HINTS = [
    "target",
    "future",
    "horizon",
    "regulariza_horizonte",
    "monto_recuperado_horizonte",
    "has_future_observation",
    "prob_",
    "pred",
    "score",
]


@dataclass
class EDAPaths:
    report_dir: Path
    table_profile: Path
    missing_values: Path
    numeric_profile: Path
    categorical_profile: Path
    key_integrity: Path
    target_profile: Path
    funnel: Path
    alerts: Path
    html_report: Path
    summary_json: Path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if math.isnan(float(obj)) or math.isinf(float(obj)):
            return None
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if pd.isna(obj):
        return None
    return str(obj)


def _safe_pct(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _resolve_table_path(raw_dir: Path, table: str) -> Path | None:
    candidates = RAW_TABLES.get(table, [table])
    for candidate in candidates:
        path = raw_dir / candidate
        if path.exists() and path.is_file():
            return path
    for ext in [".csv", ".csv.gz", ".parquet", ".pq", ".feather"]:
        path = raw_dir / f"{table}{ext}"
        if path.exists() and path.is_file():
            return path
    matches = sorted(raw_dir.glob(f"{table}*"))
    for path in matches:
        if path.is_file() and not path.name.startswith("."):
            return path
    return None


def _read_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet") or suffixes.endswith(".pq"):
        df = pd.read_parquet(path)
        if max_rows is not None:
            return df.head(max_rows).copy()
        return df
    if suffixes.endswith(".feather"):
        df = pd.read_feather(path)
        if max_rows is not None:
            return df.head(max_rows).copy()
        return df
    # CSV o archivo sin extensión visible en Windows.
    return pd.read_csv(path, nrows=max_rows, low_memory=False)


def _table_profile(table: str, path: Path, df: pd.DataFrame) -> dict[str, Any]:
    key_cols = [c for c in ["SK_ID_CURR", "SK_ID_PREV", "SK_ID_BUREAU"] if c in df.columns]
    return {
        "table": table,
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "memory_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
        "duplicate_rows": int(df.duplicated().sum()) if len(df) <= 1_000_000 else None,
        "duplicate_rows_note": None if len(df) <= 1_000_000 else "No calculado para evitar alto costo computacional.",
        "numeric_columns": int(len(df.select_dtypes(include=[np.number]).columns)),
        "categorical_columns": int(len(df.select_dtypes(exclude=[np.number]).columns)),
        "key_columns_present": ", ".join(key_cols),
        **{f"unique_{col}": int(df[col].nunique(dropna=True)) for col in key_cols},
    }


def _missing_profile(table: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = len(df)
    out: list[dict[str, Any]] = []
    for col in df.columns:
        miss = int(df[col].isna().sum())
        out.append(
            {
                "table": table,
                "column": col,
                "dtype": str(df[col].dtype),
                "missing_count": miss,
                "missing_pct": _safe_pct(miss, rows),
                "non_missing_count": int(rows - miss),
                "unique_count": int(df[col].nunique(dropna=True)),
            }
        )
    return out


def _numeric_profile(table: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce")
        non_null = s.dropna()
        if non_null.empty:
            out.append(
                {
                    "table": table,
                    "column": col,
                    "count": 0,
                    "missing_pct": 1.0,
                }
            )
            continue
        q = non_null.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
        iqr = float(q.loc[0.75] - q.loc[0.25])
        lower = float(q.loc[0.25] - 1.5 * iqr)
        upper = float(q.loc[0.75] + 1.5 * iqr)
        outlier_count = int(((non_null < lower) | (non_null > upper)).sum()) if iqr > 0 else 0
        out.append(
            {
                "table": table,
                "column": col,
                "count": int(non_null.shape[0]),
                "missing_pct": _safe_pct(s.isna().sum(), len(s)),
                "mean": float(non_null.mean()),
                "std": float(non_null.std(ddof=1)) if non_null.shape[0] > 1 else 0.0,
                "min": float(non_null.min()),
                "p01": float(q.loc[0.01]),
                "p05": float(q.loc[0.05]),
                "p25": float(q.loc[0.25]),
                "median": float(q.loc[0.50]),
                "p75": float(q.loc[0.75]),
                "p95": float(q.loc[0.95]),
                "p99": float(q.loc[0.99]),
                "max": float(non_null.max()),
                "zero_pct": _safe_pct((non_null == 0).sum(), non_null.shape[0]),
                "negative_pct": _safe_pct((non_null < 0).sum(), non_null.shape[0]),
                "outlier_iqr_count": outlier_count,
                "outlier_iqr_pct": _safe_pct(outlier_count, non_null.shape[0]),
            }
        )
    return out


def _categorical_profile(table: str, df: pd.DataFrame, max_examples: int = 5) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for col in cat_cols:
        s = df[col]
        vc = s.value_counts(dropna=True)
        examples = vc.head(max_examples).index.astype(str).tolist() if not vc.empty else []
        out.append(
            {
                "table": table,
                "column": col,
                "dtype": str(s.dtype),
                "count": int(s.notna().sum()),
                "missing_pct": _safe_pct(s.isna().sum(), len(s)),
                "unique_count": int(s.nunique(dropna=True)),
                "top_value": str(vc.index[0]) if not vc.empty else None,
                "top_freq": int(vc.iloc[0]) if not vc.empty else 0,
                "top_pct": _safe_pct(int(vc.iloc[0]), int(s.notna().sum())) if not vc.empty else 0.0,
                "example_values": " | ".join(examples),
            }
        )
    return out


def _key_integrity(loaded: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for parent_table, parent_key, child_table, child_key in KEY_RELATIONSHIPS:
        if parent_table not in loaded or child_table not in loaded:
            continue
        parent = loaded[parent_table]
        child = loaded[child_table]
        if parent_key not in parent.columns or child_key not in child.columns:
            continue
        parent_keys = set(parent[parent_key].dropna().unique())
        child_keys = child[child_key].dropna()
        child_unique = set(child_keys.unique())
        matched_child_rows = int(child_keys.isin(parent_keys).sum())
        orphan_child_rows = int((~child_keys.isin(parent_keys)).sum())
        parent_with_child = len(parent_keys.intersection(child_unique))
        out.append(
            {
                "parent_table": parent_table,
                "parent_key": parent_key,
                "child_table": child_table,
                "child_key": child_key,
                "parent_unique_keys": int(len(parent_keys)),
                "child_rows_non_null_key": int(child_keys.shape[0]),
                "child_unique_keys": int(len(child_unique)),
                "matched_child_rows": matched_child_rows,
                "matched_child_rows_pct": _safe_pct(matched_child_rows, child_keys.shape[0]),
                "orphan_child_rows": orphan_child_rows,
                "orphan_child_rows_pct": _safe_pct(orphan_child_rows, child_keys.shape[0]),
                "parent_keys_with_child": int(parent_with_child),
                "parent_keys_with_child_pct": _safe_pct(parent_with_child, len(parent_keys)),
            }
        )
    return out


def _target_profile_from_application(df: pd.DataFrame) -> list[dict[str, Any]]:
    if "TARGET" not in df.columns:
        return []
    s = df["TARGET"].dropna().astype(int)
    vc = s.value_counts().sort_index()
    return [
        {
            "source": "application_train",
            "target": "TARGET",
            "rows_with_target": int(s.shape[0]),
            "positive_count": int(vc.get(1, 0)),
            "negative_count": int(vc.get(0, 0)),
            "positive_rate": _safe_pct(int(vc.get(1, 0)), int(s.shape[0])),
            "definition": "Default target original de Home Credit; no es el target operativo de cobranza del pipeline.",
        }
    ]


def _target_profile_from_processed(processed_dir: Path | None) -> list[dict[str, Any]]:
    if processed_dir is None:
        return []
    path = processed_dir / "cartera_cobranza_modelable.csv"
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    out: list[dict[str, Any]] = []
    if "regulariza_horizonte" in df.columns:
        s = df["regulariza_horizonte"].dropna().astype(int)
        vc = s.value_counts().sort_index()
        out.append(
            {
                "source": "processed/cartera_cobranza_modelable.csv",
                "target": "regulariza_horizonte",
                "rows_with_target": int(s.shape[0]),
                "positive_count": int(vc.get(1, 0)),
                "negative_count": int(vc.get(0, 0)),
                "positive_rate": _safe_pct(int(vc.get(1, 0)), int(s.shape[0])),
                "definition": "Target prospectivo del pipeline: regularización dentro del horizonte definido.",
            }
        )
    if "monto_recuperado_horizonte" in df.columns:
        s = pd.to_numeric(df["monto_recuperado_horizonte"], errors="coerce")
        non_null = s.dropna()
        out.append(
            {
                "source": "processed/cartera_cobranza_modelable.csv",
                "target": "monto_recuperado_horizonte",
                "rows_with_target": int(non_null.shape[0]),
                "positive_amount_count": int((non_null > 0).sum()),
                "positive_amount_rate": _safe_pct((non_null > 0).sum(), non_null.shape[0]),
                "mean": float(non_null.mean()) if not non_null.empty else None,
                "median": float(non_null.median()) if not non_null.empty else None,
                "p90": float(non_null.quantile(0.90)) if not non_null.empty else None,
                "max": float(non_null.max()) if not non_null.empty else None,
                "definition": "Monto recuperado observado dentro del horizonte prospectivo.",
            }
        )
    return out


def _build_funnel(
    loaded: dict[str, pd.DataFrame],
    processed_dir: Path | None,
    outputs_dir: Path | None,
) -> list[dict[str, Any]]:
    steps: list[tuple[str, int | None, str]] = []
    app = loaded.get("application_train")
    prev = loaded.get("previous_application")
    inst = loaded.get("installments_payments")

    steps.append(("application_train cargada", len(app) if app is not None else None, "Clientes/aplicaciones principales cargadas."))
    steps.append(("previous_application cargada", len(prev) if prev is not None else None, "Créditos previos cargados."))
    steps.append(("installments_payments cargada", len(inst) if inst is not None else None, "Pagos/cuotas cargados."))

    if app is not None and prev is not None and "SK_ID_CURR" in app.columns and "SK_ID_CURR" in prev.columns:
        app_keys = set(app["SK_ID_CURR"].dropna().unique())
        prev_matched = int(prev["SK_ID_CURR"].dropna().isin(app_keys).sum())
        steps.append(("previous_application con SK_ID_CURR en application_train", prev_matched, "Mide pérdida por cruce cliente-crédito."))

    if prev is not None and inst is not None and "SK_ID_PREV" in prev.columns and "SK_ID_PREV" in inst.columns:
        prev_keys = set(prev["SK_ID_PREV"].dropna().unique())
        inst_matched = int(inst["SK_ID_PREV"].dropna().isin(prev_keys).sum())
        steps.append(("installments_payments con SK_ID_PREV en previous_application", inst_matched, "Mide pérdida por cruce crédito-cuotas."))

    if inst is not None and {"DAYS_ENTRY_PAYMENT", "DAYS_INSTALMENT"}.issubset(inst.columns):
        dpd = (pd.to_numeric(inst["DAYS_ENTRY_PAYMENT"], errors="coerce") - pd.to_numeric(inst["DAYS_INSTALMENT"], errors="coerce")).clip(lower=0)
        steps.append(("cuotas con atraso observado DPD > 0", int((dpd > 0).sum()), "Proxy exploratorio de eventos potenciales de cobranza."))

    processed_path = processed_dir / "cartera_cobranza_modelable.csv" if processed_dir else None
    if processed_path is not None and processed_path.exists():
        processed = pd.read_csv(processed_path, low_memory=False)
        steps.append(("snapshots/eventos procesados por pipeline", int(len(processed)), "Base modelable generada por feature engineering temporal."))
        if {"regulariza_horizonte", "monto_recuperado_horizonte"}.issubset(processed.columns):
            target_rows = int(processed[["regulariza_horizonte", "monto_recuperado_horizonte"]].dropna().shape[0])
            steps.append(("eventos con target prospectivo observado", target_rows, "Filas efectivamente elegibles para entrenamiento supervisado."))
        if "regulariza_horizonte" in processed.columns:
            pos = int(pd.to_numeric(processed["regulariza_horizonte"], errors="coerce").fillna(0).astype(int).sum())
            steps.append(("positivos de regularización", pos, "Casos positivos para el clasificador Hurdle."))

    if outputs_dir is not None:
        summary_path = outputs_dir / "resumen_ejecucion.json"
        metrics_path = outputs_dir / "metricas_modelo.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if "n_snapshots" in summary:
                steps.append(("n_snapshots reportado en resumen_ejecucion", int(summary["n_snapshots"]), "Valor final reportado por el pipeline."))
            if "n_current_portfolio" in summary:
                steps.append(("n_current_portfolio reportado en resumen_ejecucion", int(summary["n_current_portfolio"]), "Cartera actual puntuable/priorizable."))
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key, label in [
                ("n_rows_train", "filas train"),
                ("n_rows_validation", "filas validation"),
                ("n_rows_calibration", "filas calibration"),
                ("n_rows_test", "filas test"),
                ("n_positive_amount_train", "positivos monto train"),
                ("n_positive_amount_validation", "positivos monto validation"),
            ]:
                if key in metrics:
                    steps.append((label, int(metrics[key]), "Tamaño final usado por modelado."))

    out = []
    first = next((v for _, v, _ in steps if v is not None and v > 0), None)
    prev_rows: int | None = None
    for i, (step, rows, note) in enumerate(steps, start=1):
        out.append(
            {
                "step_order": i,
                "stage": step,
                "rows": None if rows is None else int(rows),
                "retention_vs_previous": None if rows is None or prev_rows in [None, 0] else _safe_pct(rows, prev_rows),
                "retention_vs_first": None if rows is None or first in [None, 0] else _safe_pct(rows, first),
                "note": note,
            }
        )
        if rows is not None:
            prev_rows = rows
    return out


def _build_alerts(
    table_profiles: list[dict[str, Any]],
    missing: list[dict[str, Any]],
    categorical: list[dict[str, Any]],
    numeric: list[dict[str, Any]],
    target_profile: list[dict[str, Any]],
    funnel: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    alerts: list[dict[str, Any]] = []

    for row in missing:
        if row.get("missing_pct", 0) >= 0.90:
            alerts.append(
                {
                    "severity": "alta",
                    "type": "missingness",
                    "table": row["table"],
                    "column": row["column"],
                    "message": "Columna con 90% o más de valores faltantes; evaluar exclusión o tratamiento específico.",
                    "value": row["missing_pct"],
                }
            )

    for row in categorical:
        if row.get("unique_count", 0) > 1000:
            alerts.append(
                {
                    "severity": "media",
                    "type": "high_cardinality",
                    "table": row["table"],
                    "column": row["column"],
                    "message": "Variable categórica de alta cardinalidad; revisar si es identificador, proxy o requiere encoding especial.",
                    "value": row["unique_count"],
                }
            )

    for row in missing:
        col_l = str(row["column"]).lower()
        if any(h in col_l for h in SENSITIVE_OR_EXCLUDED_HINTS):
            alerts.append(
                {
                    "severity": "media",
                    "type": "sensitive_or_proxy_hint",
                    "table": row["table"],
                    "column": row["column"],
                    "message": "Nombre de columna sugiere variable sensible o proxy; excluir del modelo de decisión si no hay justificación regulatoria.",
                }
            )
        if any(h in col_l for h in LEAKAGE_HINTS):
            alerts.append(
                {
                    "severity": "alta",
                    "type": "leakage_hint",
                    "table": row["table"],
                    "column": row["column"],
                    "message": "Nombre de columna sugiere target, futuro, predicción o fuga temporal; no usar como feature de entrenamiento.",
                }
            )

    modelable = next((r for r in funnel if r["stage"] == "eventos con target prospectivo observado"), None)
    if modelable and modelable.get("rows") is not None and modelable["rows"] < 5000:
        alerts.append(
            {
                "severity": "alta",
                "type": "low_modelable_sample",
                "message": "Muestra modelable menor a 5.000 eventos; suficiente para demo, débil para piloto productivo.",
                "value": modelable["rows"],
            }
        )

    for t in target_profile:
        if t.get("positive_rate") is not None:
            rate = float(t["positive_rate"])
            if rate < 0.05 or rate > 0.95:
                alerts.append(
                    {
                        "severity": "media",
                        "type": "target_imbalance",
                        "source": t.get("source"),
                        "target": t.get("target"),
                        "message": "Target muy desbalanceado; priorizar PR-AUC, lift/top-k y calibración.",
                        "value": rate,
                    }
                )

    for row in numeric:
        if row.get("outlier_iqr_pct", 0) >= 0.20:
            alerts.append(
                {
                    "severity": "baja",
                    "type": "outlier_iqr",
                    "table": row["table"],
                    "column": row["column"],
                    "message": "Alta proporción de outliers por regla IQR; revisar winsorización, log-transform o robustez del modelo.",
                    "value": row["outlier_iqr_pct"],
                }
            )

    return alerts


def _write_html_report(
    path: Path,
    table_profiles: pd.DataFrame,
    key_integrity: pd.DataFrame,
    target_profile: pd.DataFrame,
    funnel: pd.DataFrame,
    alerts: list[dict[str, Any]],
    run_params: dict[str, Any],
) -> None:
    def table_html(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "<p>No disponible.</p>"
        return df.head(max_rows).to_html(index=False, escape=False, border=0)

    alerts_df = pd.DataFrame(alerts)
    html = f"""
<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>EDA formal - Cobranza Inteligente</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 32px; color: #222; }}
h1, h2 {{ color: #1f2937; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 24px; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
th {{ background: #f3f4f6; }}
.badge {{ display:inline-block; padding: 3px 8px; border-radius: 10px; background:#e5e7eb; margin-right: 6px; }}
.note {{ background:#f9fafb; padding:12px; border-left:4px solid #6b7280; }}
</style>
</head>
<body>
<h1>EDA formal - Proyecto Cobranza Inteligente</h1>
<p class="note">Este reporte audita estructura de tablas, calidad de datos, llaves, target, embudo de pérdida de observaciones y alertas antes del modelamiento.</p>
<p>
<span class="badge">raw_dir: {run_params.get('raw_dir')}</span>
<span class="badge">processed_dir: {run_params.get('processed_dir')}</span>
<span class="badge">outputs_dir: {run_params.get('outputs_dir')}</span>
<span class="badge">max_rows: {run_params.get('max_rows')}</span>
</p>

<h2>1. Perfil de tablas</h2>
{table_html(table_profiles)}

<h2>2. Integridad referencial</h2>
{table_html(key_integrity)}

<h2>3. Perfil de targets</h2>
{table_html(target_profile)}

<h2>4. Embudo de datos</h2>
{table_html(funnel, max_rows=50)}

<h2>5. Alertas de calidad</h2>
{table_html(alerts_df, max_rows=50)}

<h2>6. Lectura ejecutiva</h2>
<ul>
<li>Si el embudo muestra fuerte caída entre tablas crudas y eventos con target, revisar joins, horizonte temporal y filtros de historial mínimo.</li>
<li>Si la muestra modelable es baja, los resultados deben tratarse como demo técnica y no como validación productiva.</li>
<li>Variables con posible fuga temporal o sensibilidad regulatoria deben excluirse del set de features productivo.</li>
<li>Para cobranza se recomienda evaluar ranking, lift, top-k, recuperación esperada y calibración, no solo accuracy.</li>
</ul>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def generate_eda_report(
    raw_dir: str | Path,
    processed_dir: str | Path | None = None,
    outputs_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    max_rows: int | None = None,
) -> EDAPaths:
    """Genera EDA formal para el proyecto de cobranza.

    Parameters
    ----------
    raw_dir:
        Carpeta con tablas crudas Home Credit: application_train, previous_application,
        installments_payments, bureau, bureau_balance, POS_CASH_balance y credit_card_balance.
    processed_dir:
        Carpeta processed del proyecto. Si existe cartera_cobranza_modelable.csv,
        se perfila el target operativo del pipeline.
    outputs_dir:
        Carpeta outputs del pipeline. Si existen metricas_modelo.json o resumen_ejecucion.json,
        se integran al embudo.
    report_dir:
        Carpeta de salida para reportes EDA. Por defecto outputs/eda.
    max_rows:
        Límite opcional de filas por tabla para EDA rápido. Para EDA definitivo, usar None.
    """
    raw_dir = Path(raw_dir)
    processed_path = Path(processed_dir) if processed_dir is not None else None
    outputs_path = Path(outputs_dir) if outputs_dir is not None else None
    if report_dir is None:
        report_path = (outputs_path / "eda") if outputs_path is not None else Path("outputs") / "eda"
    else:
        report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    loaded: dict[str, pd.DataFrame] = {}
    table_profiles: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    numeric_rows: list[dict[str, Any]] = []
    categorical_rows: list[dict[str, Any]] = []

    for table in RAW_TABLES:
        path = _resolve_table_path(raw_dir, table)
        if path is None:
            table_profiles.append(
                {
                    "table": table,
                    "path": None,
                    "rows": None,
                    "columns": None,
                    "status": "no_encontrada",
                }
            )
            continue
        df = _read_table(path, max_rows=max_rows)
        loaded[table] = df
        profile = _table_profile(table, path, df)
        profile["status"] = "ok"
        table_profiles.append(profile)
        missing_rows.extend(_missing_profile(table, df))
        numeric_rows.extend(_numeric_profile(table, df))
        categorical_rows.extend(_categorical_profile(table, df))

    key_rows = _key_integrity(loaded)
    target_rows: list[dict[str, Any]] = []
    if "application_train" in loaded:
        target_rows.extend(_target_profile_from_application(loaded["application_train"]))
    target_rows.extend(_target_profile_from_processed(processed_path))
    funnel_rows = _build_funnel(loaded, processed_path, outputs_path)
    alerts = _build_alerts(table_profiles, missing_rows, categorical_rows, numeric_rows, target_rows, funnel_rows)

    table_profile_df = pd.DataFrame(table_profiles)
    missing_df = pd.DataFrame(missing_rows).sort_values(["missing_pct", "table", "column"], ascending=[False, True, True]) if missing_rows else pd.DataFrame()
    numeric_df = pd.DataFrame(numeric_rows)
    categorical_df = pd.DataFrame(categorical_rows)
    key_df = pd.DataFrame(key_rows)
    target_df = pd.DataFrame(target_rows)
    funnel_df = pd.DataFrame(funnel_rows)
    alerts_df = pd.DataFrame(alerts)

    paths = EDAPaths(
        report_dir=report_path,
        table_profile=report_path / "eda_table_profile.csv",
        missing_values=report_path / "eda_missing_values.csv",
        numeric_profile=report_path / "eda_numeric_profile.csv",
        categorical_profile=report_path / "eda_categorical_profile.csv",
        key_integrity=report_path / "eda_key_integrity.csv",
        target_profile=report_path / "eda_target_profile.csv",
        funnel=report_path / "eda_funnel.csv",
        alerts=report_path / "eda_alertas_calidad.json",
        html_report=report_path / "eda_report.html",
        summary_json=report_path / "eda_resumen.json",
    )

    table_profile_df.to_csv(paths.table_profile, index=False)
    missing_df.to_csv(paths.missing_values, index=False)
    numeric_df.to_csv(paths.numeric_profile, index=False)
    categorical_df.to_csv(paths.categorical_profile, index=False)
    key_df.to_csv(paths.key_integrity, index=False)
    target_df.to_csv(paths.target_profile, index=False)
    funnel_df.to_csv(paths.funnel, index=False)
    paths.alerts.write_text(json.dumps(alerts, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    summary = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_path) if processed_path is not None else None,
        "outputs_dir": str(outputs_path) if outputs_path is not None else None,
        "report_dir": str(report_path),
        "max_rows": max_rows,
        "tables_found": int(sum(1 for r in table_profiles if r.get("status") == "ok")),
        "tables_missing": [r["table"] for r in table_profiles if r.get("status") != "ok"],
        "n_quality_alerts": int(len(alerts)),
        "n_high_severity_alerts": int(sum(1 for a in alerts if a.get("severity") == "alta")),
        "generated_files": {field: str(getattr(paths, field)) for field in paths.__dataclass_fields__ if field != "report_dir"},
    }
    paths.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    _write_html_report(
        paths.html_report,
        table_profile_df,
        key_df,
        target_df,
        funnel_df,
        alerts,
        {
            "raw_dir": raw_dir,
            "processed_dir": processed_path,
            "outputs_dir": outputs_path,
            "max_rows": max_rows,
        },
    )
    return paths
