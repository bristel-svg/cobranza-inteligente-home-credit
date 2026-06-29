from __future__ import annotations

import argparse
from pathlib import Path

from cobranza_inteligente.eda import generate_eda_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera EDA formal para el proyecto de cobranza inteligente.")
    parser.add_argument("--raw-dir", default="data/raw", help="Carpeta con tablas crudas.")
    parser.add_argument("--processed-dir", default="data/processed", help="Carpeta con datos procesados.")
    parser.add_argument("--outputs-dir", default="outputs", help="Carpeta de outputs del pipeline.")
    parser.add_argument("--report-dir", default=None, help="Carpeta de salida EDA. Default: outputs/eda.")
    parser.add_argument("--max-rows", type=int, default=None, help="Límite opcional de filas por tabla para EDA rápido.")
    args = parser.parse_args()

    paths = generate_eda_report(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        outputs_dir=Path(args.outputs_dir),
        report_dir=Path(args.report_dir) if args.report_dir else None,
        max_rows=args.max_rows,
    )
    print("EDA generado correctamente.")
    print(f"Reporte HTML: {paths.html_report}")
    print(f"Embudo: {paths.funnel}")
    print(f"Alertas: {paths.alerts}")


if __name__ == "__main__":
    main()
