from __future__ import annotations

import json
import warnings
from pathlib import Path

import pandas as pd

from .config import (
    APPLICATION_FEATURES,
    DATA_PROCESSED,
    DATA_RAW,
    DEFAULT_HORIZON_DAYS,
    MODELS_DIR,
    OUTPUTS_DIR,
    PREVIOUS_FEATURES,
    REQUIRED_FILES,
)
from .decision_engine import aggregate_to_current_portfolio, recommend_actions, select_output_columns
from .evaluation import build_recovery_curve, build_strategy_benchmark
from .features import INSTALLMENT_REQUIRED_COLUMNS, build_collection_snapshots, get_feature_columns
from .modeling import save_artifacts, score_dataset, train_models
from .optional_features import build_optional_customer_features, load_optional_tables
from .optimization import optimize_contact_plan, summarize_optimized_plan
from .regulatory import (
    add_human_in_the_loop_flags,
    apply_cmf_feature_policy,
    build_reason_codes,
    write_regulatory_report,
)
from .targets import add_collection_targets
from .utils import ensure_dirs, read_csv_if_exists


def load_minimum_data(raw_dir: Path = DATA_RAW, max_rows: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    app_path = raw_dir / REQUIRED_FILES["application"]
    prev_path = raw_dir / REQUIRED_FILES["previous"]
    inst_path = raw_dir / REQUIRED_FILES["installments"]

    application = read_csv_if_exists(app_path, usecols=APPLICATION_FEATURES, nrows=max_rows)
    previous = read_csv_if_exists(prev_path, usecols=PREVIOUS_FEATURES, nrows=max_rows)
    installments = read_csv_if_exists(inst_path, usecols=INSTALLMENT_REQUIRED_COLUMNS, nrows=max_rows)

    return application, previous, installments


def run_pipeline(
    raw_dir: Path = DATA_RAW,
    processed_dir: Path = DATA_PROCESSED,
    outputs_dir: Path = OUTPUTS_DIR,
    models_dir: Path = MODELS_DIR,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    max_rows: int | None = None,
    min_prev_installments: int = 2,
    strict_cmf: bool = True,
    use_optional_tables: bool = True,
    optimization_budget: float = 500_000,
    optimization_capacity: int | None = None,
    optimization_channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
    random_benchmark_sims: int = 200,
    verbose: bool = True,
) -> dict:
    """Ejecuta el flujo completo: features, targets, modelos, ranking y archivos finales."""
    ensure_dirs([processed_dir, outputs_dir, models_dir])

    def log(message: str) -> None:
        if verbose:
            print(message, flush=True)

    log("[1/9] Cargando tablas principales...")
    application, previous, installments = load_minimum_data(raw_dir=raw_dir, max_rows=max_rows)
    log(f"      application={len(application):,}, previous={len(previous):,}, installments={len(installments):,}")

    log("[2/9] Construyendo eventos de cobranza y features temporales...")
    snapshots = build_collection_snapshots(
        installments=installments,
        previous=previous,
        application=application,
        min_prev_installments=min_prev_installments,
    )
    if snapshots.empty:
        raise ValueError("No se generaron eventos de cobranza. Revisa datos o min_prev_installments.")
    log(f"      eventos de cobranza={len(snapshots):,}")

    log("[3/9] Construyendo targets prospectivos de regularización y recuperación...")
    dataset = add_collection_targets(snapshots, horizon_days=horizon_days)
    trainable_rows = int(dataset.dropna(subset=["regulariza_horizonte", "monto_recuperado_horizonte"]).shape[0])
    log(f"      filas modelables con target observado={trainable_rows:,}")

    optional_feature_count = 0
    if use_optional_tables:
        log("[4/9] Agregando tablas opcionales: bureau, bureau_balance, POS/CASH, credit_card...")
        optional_tables = load_optional_tables(raw_dir=raw_dir, max_rows=max_rows)
        optional_features = build_optional_customer_features(optional_tables)
        if not optional_features.empty:
            optional_feature_count = optional_features.shape[1] - 1
            dataset = dataset.merge(optional_features, on="SK_ID_CURR", how="left")
        log(f"      features opcionales agregadas={optional_feature_count}")
    else:
        log("[4/9] Tablas opcionales desactivadas por parámetro --no-optional-tables.")

    dataset_path = processed_dir / "cartera_cobranza_modelable.csv"
    dataset.to_csv(dataset_path, index=False)

    log("[5/9] Aplicando política de variables CMF-friendly y preparando features...")
    numeric_cols, categorical_cols = get_feature_columns(dataset)

    # Por seguridad, elimina columnas con todos los valores nulos.
    numeric_cols = [c for c in numeric_cols if not dataset[c].isna().all()]
    categorical_cols = [c for c in categorical_cols if not dataset[c].isna().all()]

    numeric_cols, categorical_cols, feature_policy_report = apply_cmf_feature_policy(
        numeric_cols, categorical_cols, strict_mode=strict_cmf
    )
    log(f"      features finales: numéricas={len(numeric_cols)}, categóricas={len(categorical_cols)}")

    log("[6/9] Entrenando modelos y calibrando probabilidades...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        artifacts = train_models(
            dataset,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_class="regulariza_horizonte",
            target_amount="monto_recuperado_horizonte",
        )

    save_artifacts(artifacts, models_dir)
    log(f"      ROC-AUC={artifacts.metrics.get('roc_auc', float('nan')):.3f}; AP={artifacts.metrics.get('average_precision', float('nan')):.3f}; Gini={artifacts.metrics.get('gini', float('nan')):.3f}")

    log("[7/9] Scoring de cartera, acciones recomendadas y reason codes...")
    scored = score_dataset(dataset, artifacts)
    ranked_events = recommend_actions(scored)
    ranked_events = build_reason_codes(ranked_events)
    ranked_events = add_human_in_the_loop_flags(ranked_events)

    ranked_current = aggregate_to_current_portfolio(ranked_events)
    final_output_events = select_output_columns(ranked_events)
    final_output_current = select_output_columns(ranked_current)

    log("[8/9] Backtesting económico contra reglas simples y aleatorio simulado...")
    # Backtesting económico en holdout: compara el ranking del modelo con reglas simples.
    holdout_scored = pd.DataFrame()
    benchmark = pd.DataFrame()
    recovery_curve = pd.DataFrame()
    if artifacts.holdout_predictions is not None and not artifacts.holdout_predictions.empty:
        holdout_scored = recommend_actions(artifacts.holdout_predictions)
        benchmark = build_strategy_benchmark(holdout_scored, n_random_simulations=random_benchmark_sims)
        recovery_curve = build_recovery_curve(holdout_scored, strategy="Mayor valor neto esperado")

    log("[9/9] Optimizando plan de contacto bajo presupuesto/capacidad/canales...")
    optimized_plan = optimize_contact_plan(
        final_output_current,
        budget=optimization_budget,
        capacity=optimization_capacity,
        min_expected_net=0,
        channel_limits=optimization_channel_limits,
        max_contacts_per_customer=max_contacts_per_customer,
    )
    optimization_summary = summarize_optimized_plan(
        optimized_plan,
        optimization_budget,
        optimization_capacity,
        channel_limits=optimization_channel_limits,
        max_contacts_per_customer=max_contacts_per_customer,
    )

    ranked_path = outputs_dir / "cartera_priorizada.csv"
    ranked_events_path = outputs_dir / "cartera_priorizada_eventos.csv"
    holdout_path = outputs_dir / "holdout_scored.csv"
    benchmark_path = outputs_dir / "benchmark_estrategias.csv"
    recovery_curve_path = outputs_dir / "curva_recuperacion.csv"
    optimized_plan_path = outputs_dir / "plan_optimo_presupuesto.csv"
    optimization_summary_path = outputs_dir / "resumen_optimizacion.json"
    metrics_path = outputs_dir / "metricas_modelo.csv"
    metrics_json_path = outputs_dir / "metricas_modelo.json"
    regulatory_path = outputs_dir / "reporte_cmf_modelo.json"
    run_summary_path = outputs_dir / "resumen_ejecucion.json"

    final_output_current.to_csv(ranked_path, index=False)
    final_output_events.to_csv(ranked_events_path, index=False)
    if not holdout_scored.empty:
        holdout_scored.to_csv(holdout_path, index=False)
    if not benchmark.empty:
        benchmark.to_csv(benchmark_path, index=False)
    if not recovery_curve.empty:
        recovery_curve.to_csv(recovery_curve_path, index=False)
    optimized_plan.to_csv(optimized_plan_path, index=False)
    optimization_summary_path.write_text(json.dumps(optimization_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    flat_metrics = {k: v for k, v in artifacts.metrics.items() if not isinstance(v, dict)}
    pd.DataFrame([flat_metrics]).to_csv(metrics_path, index=False)
    metrics_json_path.write_text(json.dumps(artifacts.metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    write_regulatory_report(
        regulatory_path,
        feature_policy_report,
        artifacts.metrics,
        extra_notes=[
            "El proyecto es una demo técnica y no constituye certificación de cumplimiento normativo.",
            "El modo estricto excluye variables personales/proxy de la decisión automatizada.",
            "Las acciones recomendadas incorporan bandera de revisión humana para casos sensibles o de alto impacto.",
            "Los datos reales del cliente deben contar con base legal/consentimiento y controles de seguridad antes de usarse.",
            "v6 agrega benchmark aleatorio con simulaciones, restricciones por canal/capacidad y exclusión de flags auxiliares como has_future_observation.",
        ],
    )

    quality_warnings: list[str] = []
    if max_rows is not None:
        quality_warnings.append("Ejecución con --max-rows: resultados útiles para prueba/demo rápida, no para evaluación definitiva.")
    if not use_optional_tables:
        quality_warnings.append("Ejecución con --no-optional-tables: menor riqueza predictiva; para versión final usar tablas opcionales.")
    if artifacts.metrics.get("n_rows_train", 0) < 2000:
        quality_warnings.append("Muestra de entrenamiento baja; aumentar datos o revisar filtros de target antes de vender como piloto productivo.")

    summary = {
        "dataset_path": str(dataset_path),
        "ranked_path": str(ranked_path),
        "ranked_events_path": str(ranked_events_path),
        "metrics_path": str(metrics_path),
        "metrics_json_path": str(metrics_json_path),
        "regulatory_report_path": str(regulatory_path),
        "benchmark_path": str(benchmark_path),
        "recovery_curve_path": str(recovery_curve_path),
        "optimized_plan_path": str(optimized_plan_path),
        "optimization_summary_path": str(optimization_summary_path),
        "run_summary_path": str(run_summary_path),
        "n_snapshots": int(len(dataset)),
        "n_current_portfolio": int(len(final_output_current)),
        "n_features_numeric": int(len(numeric_cols)),
        "n_features_categorical": int(len(categorical_cols)),
        "n_optional_features_added": int(optional_feature_count),
        "target_rate": float(dataset["regulariza_horizonte"].mean()),
        "metrics": flat_metrics,
        "strict_cmf": bool(strict_cmf),
        "removed_features": feature_policy_report.removed_explicit_columns + feature_policy_report.removed_identifier_like_columns,
        "n_benchmark_rows": int(len(benchmark)),
        "optimized_contacts": int(len(optimized_plan)),
        "optimization_summary": optimization_summary,
        "run_config": {
            "horizon_days": horizon_days,
            "max_rows": max_rows,
            "min_prev_installments": min_prev_installments,
            "strict_cmf": strict_cmf,
            "use_optional_tables": use_optional_tables,
            "optimization_budget": optimization_budget,
            "optimization_capacity": optimization_capacity,
            "optimization_channel_limits": optimization_channel_limits,
            "max_contacts_per_customer": max_contacts_per_customer,
            "random_benchmark_sims": random_benchmark_sims,
        },
        "quality_warnings": quality_warnings,
    }
    run_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("Pipeline finalizado correctamente.")
    return summary
