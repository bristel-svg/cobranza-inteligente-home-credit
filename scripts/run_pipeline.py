from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permite ejecutar sin instalar el paquete.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cobranza_inteligente.config import DATA_RAW
from cobranza_inteligente.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta pipeline de cobranza inteligente.")
    parser.add_argument("--horizon-days", type=int, default=60, help="Horizonte de regularización en días.")
    parser.add_argument("--max-rows", type=int, default=None, help="Número máximo de filas por CSV para pruebas.")
    parser.add_argument("--min-prev-installments", type=int, default=2, help="Mínimo de cuotas históricas antes del evento.")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Usa data/raw generada por scripts/make_sample_data.py. Es solo una bandera semántica.",
    )
    parser.add_argument(
        "--no-strict-cmf",
        action="store_true",
        help="Desactiva el filtro conservador de variables personales/proxy. No recomendado para demo regulada.",
    )
    parser.add_argument(
        "--no-optional-tables",
        action="store_true",
        help="No usar bureau, bureau_balance, POS_CASH_balance ni credit_card_balance aunque existan.",
    )
    parser.add_argument("--optimization-budget", type=float, default=500000.0, help="Presupuesto para plan optimizado de cobranza.")
    parser.add_argument("--optimization-capacity", type=int, default=None, help="Cantidad máxima total de contactos en plan optimizado.")
    parser.add_argument("--max-auto-contacts", type=int, default=None, help="Límite de gestiones automáticas en el plan optimizado.")
    parser.add_argument("--max-call-contacts", type=int, default=None, help="Límite de llamadas en el plan optimizado.")
    parser.add_argument("--max-specialist-contacts", type=int, default=None, help="Límite de gestiones especializadas/descuento en el plan optimizado.")
    parser.add_argument("--max-contacts-per-customer", type=int, default=1, help="Máximo de contactos por cliente en el plan optimizado. Usa 0 para sin límite.")
    parser.add_argument("--random-benchmark-sims", type=int, default=200, help="Simulaciones aleatorias para benchmark robusto.")
    parser.add_argument("--quiet", action="store_true", help="Reduce mensajes de progreso.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    channel_limits = {
        "automatico": args.max_auto_contacts,
        "llamada": args.max_call_contacts,
        "especializada": args.max_specialist_contacts,
    }
    max_contacts_per_customer = None if args.max_contacts_per_customer == 0 else args.max_contacts_per_customer
    summary = run_pipeline(
        raw_dir=DATA_RAW,
        horizon_days=args.horizon_days,
        max_rows=args.max_rows,
        min_prev_installments=args.min_prev_installments,
        strict_cmf=not args.no_strict_cmf,
        use_optional_tables=not args.no_optional_tables,
        optimization_budget=args.optimization_budget,
        optimization_capacity=args.optimization_capacity,
        optimization_channel_limits=channel_limits,
        max_contacts_per_customer=max_contacts_per_customer,
        random_benchmark_sims=args.random_benchmark_sims,
        verbose=not args.quiet,
    )

    print("\nPipeline ejecutado correctamente.\n")
    print(f"Eventos de cobranza generados: {summary['n_snapshots']:,}")
    print(f"Tasa de regularización: {summary['target_rate']:.2%}")
    print(f"Features numéricas: {summary['n_features_numeric']}")
    print(f"Features categóricas: {summary['n_features_categorical']}")
    print(f"Features opcionales agregadas: {summary['n_optional_features_added']}")
    print(f"Modo CMF estricto: {summary['strict_cmf']}")
    print(f"Filas benchmark estrategias: {summary['n_benchmark_rows']}")
    print(f"Contactos plan optimizado: {summary['optimized_contacts']}")
    print("\nMétricas principales:")
    important = [
        "roc_auc", "gini", "average_precision", "baseline_average_precision", "lift_average_precision",
        "brier_score", "top_10pct_target_rate", "top_10pct_recovery_capture",
        "top_20pct_target_rate", "top_20pct_recovery_capture",
    ]
    for key in important:
        if key in summary["metrics"]:
            print(f"- {key}: {summary['metrics'][key]}")
    if summary.get("quality_warnings"):
        print("\nAdvertencias de calidad:")
        for warning in summary["quality_warnings"]:
            print(f"- {warning}")
    print("\nArchivos generados:")
    print(f"- Dataset modelable: {summary['dataset_path']}")
    print(f"- Ranking cartera:   {summary['ranked_path']}")
    print(f"- Métricas:          {summary['metrics_path']}")
    print(f"- Benchmark:         {summary['benchmark_path']}")
    print(f"- Curva recuperación:{summary['recovery_curve_path']}")
    print(f"- Plan optimizado:   {summary['optimized_plan_path']}")
    print(f"- Reporte CMF:       {summary['regulatory_report_path']}")
    print(f"- Resumen ejecución: {summary['run_summary_path']}")


if __name__ == "__main__":
    main()
