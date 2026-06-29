from __future__ import annotations

import numpy as np
import pandas as pd


TOP_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
DETERMINISTIC_STRATEGIES = [
    "Mayor monto vencido",
    "Mayor DPD",
    "Mayor prob. regularización",
    "Mayor recuperación esperada",
    "Mayor valor neto esperado",
]
MODEL_STRATEGY = "Mayor valor neto esperado"


def _actual_recovery(df: pd.DataFrame) -> pd.Series:
    """Recuperación observada usada para backtesting económico."""
    if "monto_recuperado_horizonte" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=float)
    return df["monto_recuperado_horizonte"].fillna(0).clip(lower=0).astype(float)


def _actual_success(df: pd.DataFrame) -> pd.Series:
    if "regulariza_horizonte" not in df.columns:
        return pd.Series(np.zeros(len(df)), index=df.index, dtype=int)
    return df["regulariza_horizonte"].fillna(0).astype(int)


def _score_series(df: pd.DataFrame, strategy: str, random_state: int = 42) -> pd.Series:
    """Devuelve un score de ordenamiento: mayor score = contactar antes."""
    n = len(df)
    if strategy == "Aleatorio":
        rng = np.random.default_rng(random_state)
        return pd.Series(rng.random(n), index=df.index)
    if strategy == "Mayor monto vencido":
        return df.get("monto_vencido_actual", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float)
    if strategy == "Mayor DPD":
        return df.get("DPD", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float)
    if strategy == "Mayor prob. regularización":
        return df.get("prob_regulariza", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float)
    if strategy == "Mayor recuperación esperada":
        return df.get("recuperacion_esperada", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float)
    if strategy == "Mayor valor neto esperado":
        return df.get("valor_esperado_neto", pd.Series(np.zeros(n), index=df.index)).fillna(0).astype(float)
    raise ValueError(f"Estrategia no reconocida: {strategy}")


def _evaluate_ordered(
    ordered: pd.DataFrame,
    fractions: list[float],
    total_recovery: float,
    base_rate: float,
) -> list[dict]:
    rows: list[dict] = []
    total_cases = len(ordered)
    for frac in fractions:
        n = max(1, int(np.ceil(total_cases * frac)))
        top = ordered.head(n)
        recovery = float(top["_actual_recovery"].sum())
        success_rate = float(top["_success"].mean()) if len(top) else 0.0
        cost = float(top.get("costo_gestion_estimado", pd.Series(np.zeros(len(top)), index=top.index)).fillna(0).sum())
        rows.append(
            {
                "top_pct": float(frac),
                "casos_contactados": int(n),
                "tasa_regularizacion_top": success_rate,
                "tasa_regularizacion_base": base_rate,
                "lift_tasa_regularizacion": success_rate / base_rate if base_rate > 0 else np.nan,
                "recuperacion_observada": recovery,
                "captura_recuperacion": recovery / total_recovery if total_recovery > 0 else 0.0,
                "costo_estimado": cost,
                "neto_observado_aproximado": recovery - cost,
            }
        )
    return rows


def build_strategy_benchmark(
    df: pd.DataFrame,
    fractions: list[float] | None = None,
    random_state: int = 42,
    n_random_simulations: int = 200,
) -> pd.DataFrame:
    """Compara el modelo contra reglas simples y contra aleatoriedad simulada.

    La evaluación usa recuperación observada futura, no recuperación predicha. Así responde una
    pregunta de negocio: si históricamente hubiéramos contactado el top-k de cada estrategia,
    ¿cuánto se habría recuperado?

    v6 agrega un benchmark aleatorio robusto: promedio, percentil 10 y percentil 90 sobre N simulaciones.
    Esto evita vender una mejora contra una sola corrida aleatoria accidentalmente favorable o desfavorable.
    """
    if df.empty:
        return pd.DataFrame()
    fractions = fractions or TOP_FRACTIONS

    actual_recovery = _actual_recovery(df)
    actual_success = _actual_success(df)
    total_recovery = float(actual_recovery.sum())
    total_cases = len(df)
    base_rate = float(actual_success.mean()) if total_cases else 0.0

    base_df = df.assign(_actual_recovery=actual_recovery, _success=actual_success)
    rows: list[dict] = []

    # Estrategias determinísticas.
    for strategy in DETERMINISTIC_STRATEGIES:
        score = _score_series(base_df, strategy, random_state=random_state)
        ordered = base_df.assign(_score=score).sort_values("_score", ascending=False)
        for row in _evaluate_ordered(ordered, fractions, total_recovery, base_rate):
            row["estrategia"] = strategy
            row["tipo_estrategia"] = "deterministica"
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # Simulaciones aleatorias por fracción.
    random_rows: list[dict] = []
    rng = np.random.default_rng(random_state)
    for frac in fractions:
        n = max(1, int(np.ceil(total_cases * frac)))
        rec_vals: list[float] = []
        net_vals: list[float] = []
        rate_vals: list[float] = []
        cost_vals: list[float] = []
        for _ in range(max(1, n_random_simulations)):
            chosen_pos = rng.choice(total_cases, size=n, replace=False)
            top = base_df.iloc[chosen_pos]
            rec = float(top["_actual_recovery"].sum())
            cost = float(top.get("costo_gestion_estimado", pd.Series(np.zeros(len(top)), index=top.index)).fillna(0).sum())
            rec_vals.append(rec)
            net_vals.append(rec - cost)
            rate_vals.append(float(top["_success"].mean()) if len(top) else 0.0)
            cost_vals.append(cost)
        random_rows.append(
            {
                "estrategia": f"Aleatorio promedio ({n_random_simulations} sims)",
                "tipo_estrategia": "aleatoria_simulada",
                "top_pct": float(frac),
                "casos_contactados": int(n),
                "tasa_regularizacion_top": float(np.mean(rate_vals)),
                "tasa_regularizacion_base": base_rate,
                "lift_tasa_regularizacion": float(np.mean(rate_vals) / base_rate) if base_rate > 0 else np.nan,
                "recuperacion_observada": float(np.mean(rec_vals)),
                "captura_recuperacion": float(np.mean(rec_vals) / total_recovery) if total_recovery > 0 else 0.0,
                "costo_estimado": float(np.mean(cost_vals)),
                "neto_observado_aproximado": float(np.mean(net_vals)),
                "aleatorio_recuperacion_p10": float(np.percentile(rec_vals, 10)),
                "aleatorio_recuperacion_p90": float(np.percentile(rec_vals, 90)),
                "aleatorio_neto_p10": float(np.percentile(net_vals, 10)),
                "aleatorio_neto_p90": float(np.percentile(net_vals, 90)),
            }
        )

    out = pd.concat([pd.DataFrame(random_rows), out], ignore_index=True)

    # Enriquecimiento comparativo por top_pct.
    random_lookup = out[out["tipo_estrategia"] == "aleatoria_simulada"].set_index("top_pct")
    model_lookup = out[out["estrategia"] == MODEL_STRATEGY].set_index("top_pct")

    mejoras = []
    lift_vs_random = []
    sobre_p90_random = []
    for _, row in out.iterrows():
        frac = row["top_pct"]
        model_recovery = float(model_lookup.loc[frac, "recuperacion_observada"]) if frac in model_lookup.index else np.nan
        baseline = float(row["recuperacion_observada"])
        mejoras.append((model_recovery / baseline - 1.0) if baseline > 0 else np.nan)

        if frac in random_lookup.index:
            random_mean = float(random_lookup.loc[frac, "recuperacion_observada"])
            random_p90 = float(random_lookup.loc[frac, "aleatorio_recuperacion_p90"])
            lift_vs_random.append(baseline / random_mean if random_mean > 0 else np.nan)
            sobre_p90_random.append(bool(baseline > random_p90))
        else:
            lift_vs_random.append(np.nan)
            sobre_p90_random.append(False)

    out["mejora_modelo_vs_estrategia"] = mejoras
    out["lift_vs_aleatorio_promedio"] = lift_vs_random
    out["supera_p90_aleatorio"] = sobre_p90_random
    return out.sort_values(["top_pct", "recuperacion_observada"], ascending=[True, False]).reset_index(drop=True)


def build_recovery_curve(
    df: pd.DataFrame,
    strategy: str = MODEL_STRATEGY,
    step: float = 0.01,
    random_state: int = 42,
) -> pd.DataFrame:
    """Curva acumulada de recuperación para la estrategia seleccionada."""
    if df.empty:
        return pd.DataFrame()
    actual_recovery = _actual_recovery(df)
    actual_success = _actual_success(df)
    score = _score_series(df, strategy, random_state=random_state)
    ordered = df.assign(_score=score, _actual_recovery=actual_recovery, _success=actual_success).sort_values(
        "_score", ascending=False
    ).reset_index(drop=True)

    total_recovery = float(ordered["_actual_recovery"].sum())
    total_cases = len(ordered)
    rows = []
    pct_values = np.unique(np.r_[np.arange(step, 1.0 + step, step), TOP_FRACTIONS])
    pct_values = [p for p in pct_values if 0 < p <= 1]
    for pct in pct_values:
        n = max(1, int(np.ceil(total_cases * pct)))
        top = ordered.head(n)
        recovery = float(top["_actual_recovery"].sum())
        cost = float(top.get("costo_gestion_estimado", pd.Series(np.zeros(len(top)), index=top.index)).fillna(0).sum())
        rows.append(
            {
                "estrategia": strategy,
                "contactado_pct": float(pct),
                "casos_contactados": int(n),
                "recuperacion_acumulada": recovery,
                "captura_recuperacion": recovery / total_recovery if total_recovery > 0 else 0.0,
                "costo_acumulado": cost,
                "neto_observado_aproximado": recovery - cost,
                "tasa_regularizacion_acumulada": float(top["_success"].mean()) if len(top) else 0.0,
            }
        )
    return pd.DataFrame(rows)
