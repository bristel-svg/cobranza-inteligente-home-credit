from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cobranza_inteligente.optimization import optimize_contact_plan, summarize_optimized_plan

OUTPUT_PATH = PROJECT_ROOT / "outputs" / "cartera_priorizada.csv"
EVENTS_PATH = PROJECT_ROOT / "outputs" / "cartera_priorizada_eventos.csv"
METRICS_PATH = PROJECT_ROOT / "outputs" / "metricas_modelo.csv"
BENCHMARK_PATH = PROJECT_ROOT / "outputs" / "benchmark_estrategias.csv"
CURVE_PATH = PROJECT_ROOT / "outputs" / "curva_recuperacion.csv"
OPT_PLAN_PATH = PROJECT_ROOT / "outputs" / "plan_optimo_presupuesto.csv"
OPT_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "resumen_optimizacion.json"
REG_REPORT_PATH = PROJECT_ROOT / "outputs" / "reporte_cmf_modelo.json"

st.set_page_config(page_title="Cobranza Inteligente", page_icon="📊", layout="wide")

st.title("Sistema de Cobranza Inteligente")
st.caption(
    "v6: priorización de cartera, benchmark económico con simulaciones aleatorias, curva de recuperación, "
    "optimización con restricciones por canal/cliente y capa CMF-friendly. Demo técnica; no constituye certificación normativa."
)

if not OUTPUT_PATH.exists():
    st.warning("No encontré outputs/cartera_priorizada.csv. Ejecuta primero: python scripts/run_pipeline.py")
    st.stop()


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def money(value: float) -> str:
    return f"${value:,.0f}"


df_current = load_data(OUTPUT_PATH)
df_events = load_data(EVENTS_PATH) if EVENTS_PATH.exists() else df_current.copy()
metrics = load_data(METRICS_PATH)
benchmark = load_data(BENCHMARK_PATH)
curve = load_data(CURVE_PATH)
opt_plan = load_data(OPT_PLAN_PATH)
opt_summary = load_json(OPT_SUMMARY_PATH)
reg_report = load_json(REG_REPORT_PATH)

view = st.sidebar.radio(
    "Vista base",
    ["Cartera actual (1 fila por cliente-crédito)", "Eventos históricos de cobranza"],
    index=0,
)
df = df_current if view.startswith("Cartera") else df_events

st.sidebar.header("Filtros")
prioridades = sorted(df["prioridad"].dropna().unique().tolist()) if "prioridad" in df.columns else []
acciones = sorted(df["accion_recomendada"].dropna().unique().tolist()) if "accion_recomendada" in df.columns else []
selected_prioridades = st.sidebar.multiselect("Prioridad", prioridades, default=prioridades)
selected_acciones = st.sidebar.multiselect("Acción", acciones, default=acciones)
min_prob = st.sidebar.slider("Probabilidad mínima de regularización", 0.0, 1.0, 0.0, 0.01)
min_monto = st.sidebar.number_input("Monto vencido mínimo", min_value=0.0, value=0.0, step=1000.0)
only_human_review = st.sidebar.checkbox("Solo casos con revisión humana", value=False)

filtered = df.copy()
if selected_prioridades and "prioridad" in filtered.columns:
    filtered = filtered[filtered["prioridad"].isin(selected_prioridades)]
if selected_acciones and "accion_recomendada" in filtered.columns:
    filtered = filtered[filtered["accion_recomendada"].isin(selected_acciones)]
if "prob_regulariza" in filtered.columns:
    filtered = filtered[filtered["prob_regulariza"] >= min_prob]
if "monto_vencido_actual" in filtered.columns:
    filtered = filtered[filtered["monto_vencido_actual"] >= min_monto]
if only_human_review and "requiere_revision_humana" in filtered.columns:
    filtered = filtered[filtered["requiere_revision_humana"] == True]

label_eventos = "Créditos priorizados" if view.startswith("Cartera") else "Eventos de cobranza"
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric(label_eventos, f"{len(filtered):,}")
col2.metric("Monto vencido", money(filtered.get("monto_vencido_actual", pd.Series(dtype=float)).sum()))
col3.metric("Recuperación esperada", money(filtered.get("recuperacion_esperada", pd.Series(dtype=float)).sum()))
col4.metric("Costo estimado", money(filtered.get("costo_gestion_estimado", pd.Series(dtype=float)).sum()))
col5.metric("Valor neto esperado", money(filtered.get("valor_esperado_neto", pd.Series(dtype=float)).sum()))
if "requiere_revision_humana" in filtered.columns:
    col6.metric("Revisión humana", f"{int(filtered['requiere_revision_humana'].sum()):,}")
else:
    col6.metric("Revisión humana", "N/D")

if not metrics.empty:
    auc = float(metrics.get("roc_auc", pd.Series([0.0])).iloc[0]) if "roc_auc" in metrics.columns else None
    ap = float(metrics.get("average_precision", pd.Series([0.0])).iloc[0]) if "average_precision" in metrics.columns else None
    base_ap = float(metrics.get("baseline_average_precision", pd.Series([0.0])).iloc[0]) if "baseline_average_precision" in metrics.columns else None
    lift = float(metrics.get("lift_average_precision", pd.Series([0.0])).iloc[0]) if "lift_average_precision" in metrics.columns else None
    if auc is not None:
        msg = f"Modelo: ROC-AUC={auc:.3f}, Average Precision={ap:.3f}, baseline={base_ap:.3f}, Lift AP={lift:.2f}x."
        if auc < 0.60:
            st.warning("Discriminación baja/moderada. " + msg)
        else:
            st.success(msg)

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Ranking",
    "Benchmark económico",
    "Curva de recuperación",
    "Plan optimizado",
    "Métricas y CMF-friendly",
])

show_cols = [
    "SK_ID_CURR", "SK_ID_PREV", "cuota_numero_actual", "monto_vencido_actual", "DPD",
    "prob_regulariza", "recuperacion_esperada", "valor_esperado_neto", "accion_recomendada",
    "prioridad", "requiere_revision_humana", "razones_recomendacion",
]
show_cols = [c for c in show_cols if c in filtered.columns]

with tab1:
    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("Ranking de cartera priorizada")
        st.dataframe(
            filtered[show_cols].head(300),
            use_container_width=True,
            hide_index=True,
            column_config={
                "prob_regulariza": st.column_config.NumberColumn("Prob. regulariza", format="%.2f"),
                "monto_vencido_actual": st.column_config.NumberColumn("Monto vencido", format="$ %.0f"),
                "recuperacion_esperada": st.column_config.NumberColumn("Recup. esperada", format="$ %.0f"),
                "valor_esperado_neto": st.column_config.NumberColumn("Valor neto", format="$ %.0f"),
            },
        )
    with right:
        st.subheader("Distribución por prioridad")
        if "prioridad" in filtered.columns and not filtered.empty:
            st.bar_chart(filtered["prioridad"].value_counts())
        st.subheader("Acciones recomendadas")
        if "accion_recomendada" in filtered.columns and not filtered.empty:
            action_counts = filtered["accion_recomendada"].value_counts().rename_axis("acción").reset_index(name="cantidad")
            st.dataframe(action_counts, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Benchmark económico contra reglas simples")
    st.caption(
        "Evalúa en holdout qué habría pasado al contactar el top-k de cada estrategia. "
        "La métrica relevante no es accuracy, sino captura de recuperación y lift frente a estrategias simples."
    )
    if benchmark.empty:
        st.info("No encontré benchmark_estrategias.csv. Ejecuta nuevamente el pipeline v6.")
    else:
        top_options = sorted(benchmark["top_pct"].dropna().unique().tolist())
        selected_top = st.selectbox("Top de cartera evaluado", top_options, format_func=lambda x: f"Top {int(x*100)}%")
        bench_top = benchmark[benchmark["top_pct"] == selected_top].copy()
        bench_top = bench_top.sort_values("recuperacion_observada", ascending=False)
        st.dataframe(
            bench_top,
            use_container_width=True,
            hide_index=True,
            column_config={
                "top_pct": st.column_config.NumberColumn("Top %", format="%.0%%"),
                "tasa_regularizacion_top": st.column_config.NumberColumn("Tasa regularización top", format="%.2%"),
                "tasa_regularizacion_base": st.column_config.NumberColumn("Tasa base", format="%.2%"),
                "lift_tasa_regularizacion": st.column_config.NumberColumn("Lift tasa", format="%.2f"),
                "recuperacion_observada": st.column_config.NumberColumn("Recuperación obs.", format="$ %.0f"),
                "captura_recuperacion": st.column_config.NumberColumn("Captura recuperación", format="%.2%"),
                "costo_estimado": st.column_config.NumberColumn("Costo", format="$ %.0f"),
                "neto_observado_aproximado": st.column_config.NumberColumn("Neto obs. aprox.", format="$ %.0f"),
                "mejora_modelo_vs_estrategia": st.column_config.NumberColumn("Mejora modelo vs estrategia", format="%.2%"),
                "lift_vs_aleatorio_promedio": st.column_config.NumberColumn("Lift vs aleatorio prom.", format="%.2f"),
                "aleatorio_recuperacion_p10": st.column_config.NumberColumn("Aleatorio p10", format="$ %.0f"),
                "aleatorio_recuperacion_p90": st.column_config.NumberColumn("Aleatorio p90", format="$ %.0f"),
            },
        )
        chart_data = bench_top.set_index("estrategia")[["recuperacion_observada", "neto_observado_aproximado"]]
        st.bar_chart(chart_data)

with tab3:
    st.subheader("Curva acumulada de recuperación")
    st.caption("Muestra cuánto se captura al contactar progresivamente más cartera según el ranking del modelo.")
    if curve.empty:
        st.info("No encontré curva_recuperacion.csv. Ejecuta nuevamente el pipeline v6.")
    else:
        plot = curve.set_index("contactado_pct")[["captura_recuperacion", "tasa_regularizacion_acumulada"]]
        st.line_chart(plot)
        st.dataframe(
            curve.head(120),
            use_container_width=True,
            hide_index=True,
            column_config={
                "contactado_pct": st.column_config.NumberColumn("Contactado %", format="%.0%%"),
                "captura_recuperacion": st.column_config.NumberColumn("Captura recuperación", format="%.2%"),
                "tasa_regularizacion_acumulada": st.column_config.NumberColumn("Tasa regularización", format="%.2%"),
                "recuperacion_acumulada": st.column_config.NumberColumn("Recup. acumulada", format="$ %.0f"),
                "costo_acumulado": st.column_config.NumberColumn("Costo acumulado", format="$ %.0f"),
                "neto_observado_aproximado": st.column_config.NumberColumn("Neto obs. aprox.", format="$ %.0f"),
            },
        )

with tab4:
    st.subheader("Plan optimizado de presupuesto")
    st.caption(
        "El pipeline genera un plan base con presupuesto definido por consola. Además, aquí puedes simular "
        "un plan greedy tipo knapsack sobre la cartera filtrada."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Presupuesto pipeline", money(opt_summary.get("presupuesto", 0.0)))
    c2.metric("Contactos pipeline", f"{opt_summary.get('contactos_seleccionados', 0):,}")
    c3.metric("Costo usado", money(opt_summary.get("costo_usado", 0.0)))
    c4.metric("Valor neto esperado", money(opt_summary.get("valor_neto_esperado", 0.0)))

    st.markdown("**Plan generado por el pipeline**")
    if opt_plan.empty:
        st.info("No encontré plan_optimo_presupuesto.csv.")
    else:
        opt_cols = [c for c in ["orden_contacto_optimizado"] + show_cols + ["roi_esperado", "costo_acumulado", "valor_neto_esperado_acumulado"] if c in opt_plan.columns]
        st.dataframe(opt_plan[opt_cols].head(250), use_container_width=True, hide_index=True)

    st.markdown("**Simulador interactivo sobre cartera filtrada**")
    budget = st.number_input("Presupuesto máximo", min_value=0.0, value=500000.0, step=50000.0)
    capacity = st.number_input("Capacidad máxima total de contactos (0 = sin límite)", min_value=0, value=0, step=50)
    l1, l2, l3, l4 = st.columns(4)
    max_auto = l1.number_input("Máx. automáticos (0 = sin límite)", min_value=0, value=0, step=100)
    max_calls = l2.number_input("Máx. llamadas (0 = sin límite)", min_value=0, value=0, step=50)
    max_special = l3.number_input("Máx. especializados (0 = sin límite)", min_value=0, value=0, step=10)
    max_per_customer = l4.number_input("Máx. contactos por cliente (0 = sin límite)", min_value=0, value=1, step=1)

    candidate = filtered.copy()
    if {"costo_gestion_estimado", "valor_esperado_neto", "recuperacion_esperada"}.issubset(candidate.columns):
        channel_limits = {
            "automatico": None if max_auto == 0 else int(max_auto),
            "llamada": None if max_calls == 0 else int(max_calls),
            "especializada": None if max_special == 0 else int(max_special),
        }
        selected = optimize_contact_plan(
            candidate,
            budget=float(budget),
            capacity=None if capacity == 0 else int(capacity),
            channel_limits=channel_limits,
            max_contacts_per_customer=None if max_per_customer == 0 else int(max_per_customer),
        )
        sim_summary = summarize_optimized_plan(
            selected,
            budget=float(budget),
            capacity=None if capacity == 0 else int(capacity),
            channel_limits=channel_limits,
            max_contacts_per_customer=None if max_per_customer == 0 else int(max_per_customer),
        )
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Contactos seleccionados", f"{sim_summary.get('contactos_seleccionados', 0):,}")
        s2.metric("Costo usado", money(sim_summary.get("costo_usado", 0.0)))
        s3.metric("Recuperación esperada", money(sim_summary.get("recuperacion_esperada", 0.0)))
        s4.metric("Valor neto", money(sim_summary.get("valor_neto_esperado", 0.0)))
        with st.expander("Resumen por canal/acción"):
            st.write("Contactos por canal:", sim_summary.get("contactos_por_canal", {}))
            st.write("Contactos por acción:", sim_summary.get("contactos_por_accion", {}))
        st.dataframe(selected[[c for c in ["orden_contacto_optimizado"] + show_cols + ["canal_gestion", "roi_esperado", "costo_acumulado"] if c in selected.columns]].head(200), use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Métricas del modelo")
    if not metrics.empty:
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        st.caption(
            "v6 usa split por cliente, calibración de probabilidad, modelos ExtraTrees, "
            "benchmark económico con aleatorio simulado, métricas top-k de negocio y optimización con restricciones."
        )
    else:
        st.info("No hay archivo de métricas disponible.")

    st.subheader("Panel regulatorio / CMF-friendly")
    if reg_report:
        r1, r2, r3 = st.columns(3)
        r1.metric("Modo estricto", str(reg_report.get("strict_mode", "N/D")))
        r2.metric("Variables excluidas", len(reg_report.get("removed_explicit_columns", [])) + len(reg_report.get("removed_identifier_like_columns", [])))
        r3.metric("Variables usadas", len(reg_report.get("retained_numeric_columns", [])) + len(reg_report.get("retained_categorical_columns", [])))
        with st.expander("Ver variables excluidas y notas"):
            st.write("Variables excluidas explícitamente:", reg_report.get("removed_explicit_columns", []))
            st.write("Identificadores detectados:", reg_report.get("removed_identifier_like_columns", []))
            st.write("Política:", reg_report.get("policy_summary", []))
            st.write("Notas:", reg_report.get("notes", []))
    else:
        st.info("No encontré outputs/reporte_cmf_modelo.json.")

st.caption("Nota: los costos de gestión son supuestos de demo. En implementación real se calibran con datos del cliente y revisión legal/compliance.")
