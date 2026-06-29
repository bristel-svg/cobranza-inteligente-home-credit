from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import csr_matrix, vstack
except Exception:  # pragma: no cover - fallback when SciPy MILP is unavailable
    Bounds = None  # type: ignore[assignment]
    LinearConstraint = None  # type: ignore[assignment]
    milp = None  # type: ignore[assignment]
    csr_matrix = None  # type: ignore[assignment]
    vstack = None  # type: ignore[assignment]


DEFAULT_CHANNEL_LIMITS: dict[str, int | None] = {
    "automatico": None,
    "llamada": None,
    "especializada": None,
    "manual": None,
}

REQUIRED_OPTIMIZATION_COLUMNS = [
    "valor_esperado_neto",
    "recuperacion_esperada",
    "costo_gestion_estimado",
]


class OptimizationError(RuntimeError):
    """Error explícito para fallas de factibilidad o solver en optimización."""


def infer_action_channel(action: str) -> str:
    """Inferencia defensiva del canal cuando el pipeline no lo entrega explícitamente.

    En producción, lo recomendable es que `decision_engine.py` entregue una columna
    `canal_gestion` ya normalizada. Esta función se mantiene solo como compatibilidad
    con versiones anteriores del proyecto.
    """
    text = str(action).lower()
    if "descuento" in text or "especializada" in text or "especialista" in text:
        return "especializada"
    if "llamada" in text or "telef" in text:
        return "llamada"
    if (
        "whatsapp" in text
        or "sms" in text
        or "automático" in text
        or "automatico" in text
        or "monitoreo" in text
        or "postergar" in text
        or "recordatorio" in text
    ):
        return "automatico"
    return "manual"


def _validate_non_negative_number(name: str, value: float | int | None, allow_none: bool = False) -> None:
    if value is None and allow_none:
        return
    if value is None:
        raise ValueError(f"{name} no puede ser None.")
    if not np.isfinite(float(value)) or float(value) < 0:
        raise ValueError(f"{name} debe ser un número finito y no negativo. Valor recibido: {value!r}")


def _normalise_channel_limits(channel_limits: dict[str, int | None] | None) -> dict[str, int | None]:
    out = DEFAULT_CHANNEL_LIMITS.copy()
    if channel_limits:
        for key, value in channel_limits.items():
            channel = str(key).lower().strip()
            if value is not None:
                _validate_non_negative_number(f"límite del canal {channel}", value)
            out[channel] = None if value is None else int(value)
    return out


def _validate_common_inputs(
    budget: float,
    capacity: int | None,
    channel_limits: dict[str, int | None] | None,
    max_contacts_per_customer: int | None,
) -> dict[str, int | None]:
    _validate_non_negative_number("budget", budget)
    _validate_non_negative_number("capacity", capacity, allow_none=True)
    _validate_non_negative_number("max_contacts_per_customer", max_contacts_per_customer, allow_none=True)
    if capacity is not None and int(capacity) != capacity:
        raise ValueError("capacity debe ser entero o None.")
    if max_contacts_per_customer is not None and int(max_contacts_per_customer) != max_contacts_per_customer:
        raise ValueError("max_contacts_per_customer debe ser entero o None.")
    return _normalise_channel_limits(channel_limits)


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para optimización: {missing}")


def _safe_roi(value: pd.Series, cost: pd.Series) -> pd.Series:
    """ROI robusto para ranking heurístico.

    - Si costo > 0: valor/costo.
    - Si costo = 0 y valor > 0: +inf, porque aporta valor sin consumir presupuesto.
    - Si costo = 0 y valor <= 0: 0, para no priorizar acciones gratis pero inútiles.
    """
    value_arr = value.astype(float).to_numpy()
    cost_arr = cost.astype(float).to_numpy()
    roi = np.zeros(len(value_arr), dtype=float)
    positive_cost = cost_arr > 0
    roi[positive_cost] = value_arr[positive_cost] / cost_arr[positive_cost]
    roi[(~positive_cost) & (value_arr > 0)] = np.inf
    return pd.Series(roi, index=value.index)


def _prepare_candidates(
    df: pd.DataFrame,
    min_expected_net: float = 0,
    entity_cols: list[str] | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    _require_columns(df, REQUIRED_OPTIMIZATION_COLUMNS)

    candidates = df.copy()
    candidates["costo_gestion_estimado"] = (
        pd.to_numeric(candidates["costo_gestion_estimado"], errors="coerce").fillna(0).clip(lower=0)
    )
    candidates["valor_esperado_neto"] = pd.to_numeric(candidates["valor_esperado_neto"], errors="coerce").fillna(-np.inf)
    candidates["recuperacion_esperada"] = (
        pd.to_numeric(candidates["recuperacion_esperada"], errors="coerce").fillna(0).clip(lower=0)
    )

    if "canal_gestion" not in candidates.columns:
        actions = candidates.get("accion_recomendada", pd.Series("manual", index=candidates.index))
        candidates["canal_gestion"] = actions.map(infer_action_channel)
    candidates["canal_gestion"] = candidates["canal_gestion"].fillna("manual").astype(str).str.lower().str.strip()

    candidates = candidates[candidates["valor_esperado_neto"] >= float(min_expected_net)].copy()
    candidates = candidates[np.isfinite(candidates["valor_esperado_neto"].to_numpy())].copy()
    if candidates.empty:
        return candidates

    candidates["roi_esperado"] = _safe_roi(candidates["valor_esperado_neto"], candidates["costo_gestion_estimado"])
    candidates["_candidate_id"] = np.arange(len(candidates), dtype=int)

    if entity_cols:
        existing = [c for c in entity_cols if c in candidates.columns]
        if existing:
            candidates["_entity_key"] = candidates[existing].astype(str).agg("|".join, axis=1)
        else:
            candidates["_entity_key"] = candidates["_candidate_id"].astype(str)
    elif {"SK_ID_CURR", "SK_ID_PREV"}.issubset(candidates.columns):
        candidates["_entity_key"] = candidates[["SK_ID_CURR", "SK_ID_PREV"]].astype(str).agg("|".join, axis=1)
    elif "SK_ID_CURR" in candidates.columns:
        candidates["_entity_key"] = candidates["SK_ID_CURR"].astype(str)
    else:
        candidates["_entity_key"] = candidates["_candidate_id"].astype(str)

    return candidates.reset_index(drop=False).rename(columns={"index": "_original_index"})


def _finalize_selected_plan(selected: pd.DataFrame, method: str, summary: dict[str, Any]) -> pd.DataFrame:
    if selected.empty:
        selected = selected.copy()
        selected.attrs["optimization_summary"] = summary
        return selected

    selected = selected.sort_values(
        ["valor_esperado_neto", "roi_esperado", "recuperacion_esperada"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    selected["costo_acumulado"] = selected["costo_gestion_estimado"].cumsum()
    selected["recuperacion_esperada_acumulada"] = selected["recuperacion_esperada"].cumsum()
    selected["valor_neto_esperado_acumulado"] = selected["valor_esperado_neto"].cumsum()
    selected["orden_contacto_optimizado"] = np.arange(1, len(selected) + 1)
    selected["metodo_optimizacion"] = method

    for key, value in summary.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            selected[key] = value

    selected.attrs["optimization_summary"] = summary
    drop_cols = ["_candidate_id", "_entity_key"]
    selected = selected.drop(columns=[c for c in drop_cols if c in selected.columns])
    return selected


def optimize_contact_plan_greedy(
    df: pd.DataFrame,
    budget: float = 500_000,
    capacity: int | None = None,
    min_expected_net: float = 0,
    channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
) -> pd.DataFrame:
    """Baseline heurístico tipo knapsack.

    Ordena candidatos por ROI esperado y selecciona mientras cumplan presupuesto,
    capacidad, límites por canal y máximo de contactos por cliente.

    No garantiza optimalidad. Se mantiene como benchmark contra el MILP.
    """
    limits = _validate_common_inputs(budget, capacity, channel_limits, max_contacts_per_customer)
    candidates = _prepare_candidates(df, min_expected_net=min_expected_net)
    if candidates.empty:
        summary = _empty_solver_summary("greedy", budget, capacity, limits, max_contacts_per_customer)
        return _finalize_selected_plan(candidates, "greedy", summary)

    candidates = candidates.sort_values(
        ["roi_esperado", "valor_esperado_neto", "recuperacion_esperada"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_idx: list[int] = []
    used_budget = 0.0
    used_by_channel: dict[str, int] = defaultdict(int)
    used_by_customer: dict[Any, int] = defaultdict(int)

    for idx, row in candidates.iterrows():
        cost = float(row["costo_gestion_estimado"])
        channel = str(row["canal_gestion"])
        customer = row.get("SK_ID_CURR", row.get("_entity_key", idx))

        if used_budget + cost > float(budget) + 1e-9:
            continue
        if capacity is not None and len(selected_idx) >= int(capacity):
            break
        channel_limit = limits.get(channel)
        if channel_limit is not None and used_by_channel[channel] >= channel_limit:
            continue
        if max_contacts_per_customer is not None and used_by_customer[customer] >= int(max_contacts_per_customer):
            continue

        selected_idx.append(idx)
        used_budget += cost
        used_by_channel[channel] += 1
        used_by_customer[customer] += 1

    selected = candidates.loc[selected_idx].copy()
    summary = _build_plan_summary(
        selected,
        method="greedy",
        budget=budget,
        capacity=capacity,
        channel_limits=limits,
        max_contacts_per_customer=max_contacts_per_customer,
        solver_status="heuristic",
        solver_success=True,
        objective_value=float(selected["valor_esperado_neto"].sum()) if not selected.empty else 0.0,
        optimality_gap=None,
    )
    return _finalize_selected_plan(selected, "greedy", summary)


def _empty_solver_summary(
    method: str,
    budget: float,
    capacity: int | None,
    channel_limits: dict[str, int | None],
    max_contacts_per_customer: int | None,
) -> dict[str, Any]:
    return {
        "metodo_optimizacion": method,
        "solver_status": "empty_candidates",
        "solver_success": True,
        "objective_value": 0.0,
        "optimality_gap": None,
        "presupuesto": float(budget),
        "capacidad": None if capacity is None else int(capacity),
        "limites_por_canal": channel_limits,
        "max_contactos_por_cliente": None if max_contacts_per_customer is None else int(max_contacts_per_customer),
        "contactos_seleccionados": 0,
        "costo_usado": 0.0,
        "recuperacion_esperada": 0.0,
        "valor_neto_esperado": 0.0,
        "contactos_por_canal": {},
        "contactos_por_accion": {},
    }


def _add_sparse_constraint(rows: list[Any], lb: list[float], ub: list[float], coeffs: np.ndarray, lower: float, upper: float) -> None:
    rows.append(csr_matrix(coeffs.reshape(1, -1)))
    lb.append(float(lower))
    ub.append(float(upper))


def _solve_binary_selection_milp(
    candidates: pd.DataFrame,
    budget: float,
    capacity: int | None,
    channel_limits: dict[str, int | None],
    max_per_entity: int | None,
    entity_col: str,
    time_limit_seconds: float | None,
    mip_rel_gap: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if milp is None or Bounds is None or LinearConstraint is None or csr_matrix is None or vstack is None:
        raise OptimizationError(
            "scipy.optimize.milp no está disponible. Instala una versión reciente de SciPy "
            "o usa optimize_contact_plan_greedy como fallback."
        )

    n = len(candidates)
    values = candidates["valor_esperado_neto"].astype(float).to_numpy()
    costs = candidates["costo_gestion_estimado"].astype(float).to_numpy()

    rows: list[Any] = []
    lb: list[float] = []
    ub: list[float] = []

    # Restricción presupuestaria.
    _add_sparse_constraint(rows, lb, ub, costs, 0.0, float(budget))

    # Restricción de capacidad total.
    if capacity is not None:
        _add_sparse_constraint(rows, lb, ub, np.ones(n, dtype=float), 0.0, float(capacity))

    # Restricciones por canal.
    for channel, limit in channel_limits.items():
        if limit is None:
            continue
        mask = (candidates["canal_gestion"].astype(str).to_numpy() == str(channel)).astype(float)
        if mask.sum() > 0:
            _add_sparse_constraint(rows, lb, ub, mask, 0.0, float(limit))

    # Restricción de exclusividad por entidad: cliente, crédito o cliente-acción.
    if max_per_entity is not None:
        for _, idx in candidates.groupby(entity_col, sort=False).groups.items():
            mask = np.zeros(n, dtype=float)
            mask[list(idx)] = 1.0
            _add_sparse_constraint(rows, lb, ub, mask, 0.0, float(max_per_entity))

    constraints = LinearConstraint(vstack(rows), np.array(lb, dtype=float), np.array(ub, dtype=float))

    # scipy.optimize.milp minimiza; por eso usamos -valor para maximizar.
    objective = -values
    options: dict[str, Any] = {"disp": False}
    if time_limit_seconds is not None:
        _validate_non_negative_number("time_limit_seconds", time_limit_seconds)
        options["time_limit"] = float(time_limit_seconds)
    if mip_rel_gap is not None:
        _validate_non_negative_number("mip_rel_gap", mip_rel_gap)
        options["mip_rel_gap"] = float(mip_rel_gap)

    result = milp(
        c=objective,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=constraints,
        options=options,
    )

    status_text = str(getattr(result, "message", ""))
    success = bool(getattr(result, "success", False))
    if result.x is None:
        raise OptimizationError(f"El solver MILP no entregó solución. Status: {status_text}")

    selected_mask = np.rint(result.x).astype(int).clip(0, 1).astype(bool)
    objective_value = float(values[selected_mask].sum())
    gap = getattr(result, "mip_gap", None)
    lower_bound = getattr(result, "mip_dual_bound", None)

    diagnostics = {
        "solver_status": status_text,
        "solver_success": success,
        "solver_status_code": int(getattr(result, "status", -1)),
        "objective_value": objective_value,
        "optimality_gap": None if gap is None else float(gap),
        "solver_fun": None if getattr(result, "fun", None) is None else float(result.fun),
        "solver_dual_bound": None if lower_bound is None else float(lower_bound),
    }
    return selected_mask, diagnostics


def optimize_contact_plan_milp(
    df: pd.DataFrame,
    budget: float = 500_000,
    capacity: int | None = None,
    min_expected_net: float = 0,
    channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
    time_limit_seconds: float | None = 30,
    mip_rel_gap: float | None = 0.001,
    fallback_to_greedy: bool = True,
) -> pd.DataFrame:
    """Optimiza el plan de contacto mediante Programación Lineal Entera Mixta.

    Formulación:
        max  sum_i valor_esperado_neto_i x_i
        s.a. sum_i costo_i x_i <= presupuesto
             sum_i x_i <= capacidad, si existe
             sum_i 1{canal_i = c} x_i <= limite_c, si existe
             sum_i 1{cliente_i = k} x_i <= max_contactos_cliente, si existe
             x_i ∈ {0, 1}

    Esta función reemplaza el ranking greedy por una solución exacta/near-exacta
    según el gap configurado del solver. El greedy se mantiene como fallback y benchmark.
    """
    limits = _validate_common_inputs(budget, capacity, channel_limits, max_contacts_per_customer)
    candidates = _prepare_candidates(df, min_expected_net=min_expected_net)
    if candidates.empty:
        summary = _empty_solver_summary("milp", budget, capacity, limits, max_contacts_per_customer)
        return _finalize_selected_plan(candidates, "milp", summary)

    try:
        selected_mask, diagnostics = _solve_binary_selection_milp(
            candidates=candidates,
            budget=budget,
            capacity=capacity,
            channel_limits=limits,
            max_per_entity=max_contacts_per_customer,
            entity_col="SK_ID_CURR" if "SK_ID_CURR" in candidates.columns else "_entity_key",
            time_limit_seconds=time_limit_seconds,
            mip_rel_gap=mip_rel_gap,
        )
        selected = candidates.loc[selected_mask].copy()
        summary = _build_plan_summary(
            selected,
            method="milp",
            budget=budget,
            capacity=capacity,
            channel_limits=limits,
            max_contacts_per_customer=max_contacts_per_customer,
            **diagnostics,
        )
        return _finalize_selected_plan(selected, "milp", summary)
    except Exception as exc:
        if not fallback_to_greedy:
            raise
        fallback = optimize_contact_plan_greedy(
            df=df,
            budget=budget,
            capacity=capacity,
            min_expected_net=min_expected_net,
            channel_limits=channel_limits,
            max_contacts_per_customer=max_contacts_per_customer,
        )
        summary = fallback.attrs.get("optimization_summary", {}).copy()
        summary.update(
            {
                "metodo_optimizacion": "greedy_fallback",
                "solver_status": f"MILP falló; se usó greedy. Error: {exc}",
                "solver_success": False,
            }
        )
        fallback["metodo_optimizacion"] = "greedy_fallback"
        fallback["solver_status"] = summary["solver_status"]
        fallback["solver_success"] = False
        fallback.attrs["optimization_summary"] = summary
        return fallback


def optimize_action_assignment_milp(
    action_candidates: pd.DataFrame,
    budget: float = 500_000,
    capacity: int | None = None,
    min_expected_net: float = 0,
    channel_limits: dict[str, int | None] | None = None,
    entity_cols: list[str] | None = None,
    time_limit_seconds: float | None = 30,
    mip_rel_gap: float | None = 0.001,
) -> pd.DataFrame:
    """Asignación óptima acción-entidad.

    Entrada esperada: una fila por alternativa factible. Por ejemplo, para cada cliente
    pueden existir filas SMS, llamada, convenio y especialista, cada una con su costo,
    recuperación esperada y valor neto esperado.

    Formulación:
        max  sum_{i,a} valor_neto_{i,a} x_{i,a}
        s.a. sum_a x_{i,a} <= 1                  para cada entidad i
             sum_{i,a} costo_{i,a} x_{i,a} <= presupuesto
             sum_{i,a in canal c} x_{i,a} <= limite_c
             sum_{i,a} x_{i,a} <= capacidad
             x_{i,a} ∈ {0,1}

    Esta es la versión más profesional: el optimizador no solo elige a quién contactar,
    sino también qué acción asignar.
    """
    limits = _validate_common_inputs(budget, capacity, channel_limits, max_contacts_per_customer=1)
    candidates = _prepare_candidates(action_candidates, min_expected_net=min_expected_net, entity_cols=entity_cols)
    if candidates.empty:
        summary = _empty_solver_summary("milp_action_assignment", budget, capacity, limits, 1)
        return _finalize_selected_plan(candidates, "milp_action_assignment", summary)

    selected_mask, diagnostics = _solve_binary_selection_milp(
        candidates=candidates,
        budget=budget,
        capacity=capacity,
        channel_limits=limits,
        max_per_entity=1,
        entity_col="_entity_key",
        time_limit_seconds=time_limit_seconds,
        mip_rel_gap=mip_rel_gap,
    )
    selected = candidates.loc[selected_mask].copy()
    summary = _build_plan_summary(
        selected,
        method="milp_action_assignment",
        budget=budget,
        capacity=capacity,
        channel_limits=limits,
        max_contacts_per_customer=1,
        **diagnostics,
    )
    return _finalize_selected_plan(selected, "milp_action_assignment", summary)


def optimize_contact_plan(
    df: pd.DataFrame,
    budget: float = 500_000,
    capacity: int | None = None,
    min_expected_net: float = 0,
    channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
    method: str = "milp",
    time_limit_seconds: float | None = 30,
    mip_rel_gap: float | None = 0.001,
    fallback_to_greedy: bool = True,
) -> pd.DataFrame:
    """API principal compatible con versiones anteriores.

    Por defecto usa MILP. Para comparar o mantener comportamiento antiguo:
        optimize_contact_plan(..., method="greedy")
    """
    method_clean = method.lower().strip()
    if method_clean == "greedy":
        return optimize_contact_plan_greedy(
            df=df,
            budget=budget,
            capacity=capacity,
            min_expected_net=min_expected_net,
            channel_limits=channel_limits,
            max_contacts_per_customer=max_contacts_per_customer,
        )
    if method_clean == "milp":
        return optimize_contact_plan_milp(
            df=df,
            budget=budget,
            capacity=capacity,
            min_expected_net=min_expected_net,
            channel_limits=channel_limits,
            max_contacts_per_customer=max_contacts_per_customer,
            time_limit_seconds=time_limit_seconds,
            mip_rel_gap=mip_rel_gap,
            fallback_to_greedy=fallback_to_greedy,
        )
    raise ValueError("method debe ser 'milp' o 'greedy'.")


def _build_plan_summary(
    plan: pd.DataFrame,
    method: str,
    budget: float,
    capacity: int | None,
    channel_limits: dict[str, int | None],
    max_contacts_per_customer: int | None,
    solver_status: str,
    solver_success: bool,
    objective_value: float,
    optimality_gap: float | None,
    solver_status_code: int | None = None,
    solver_fun: float | None = None,
    solver_dual_bound: float | None = None,
) -> dict[str, Any]:
    base = {
        "metodo_optimizacion": method,
        "solver_status": solver_status,
        "solver_success": bool(solver_success),
        "solver_status_code": solver_status_code,
        "objective_value": float(objective_value),
        "optimality_gap": None if optimality_gap is None else float(optimality_gap),
        "solver_fun": solver_fun,
        "solver_dual_bound": solver_dual_bound,
        "presupuesto": float(budget),
        "capacidad": None if capacity is None else int(capacity),
        "limites_por_canal": channel_limits,
        "max_contactos_por_cliente": None if max_contacts_per_customer is None else int(max_contacts_per_customer),
    }
    if plan.empty:
        return {
            **base,
            "contactos_seleccionados": 0,
            "costo_usado": 0.0,
            "presupuesto_no_usado": float(budget),
            "recuperacion_esperada": 0.0,
            "valor_neto_esperado": 0.0,
            "roi_esperado_promedio": None,
            "contactos_por_canal": {},
            "contactos_por_accion": {},
        }
    roi = plan["roi_esperado"].replace([np.inf, -np.inf], np.nan) if "roi_esperado" in plan else pd.Series(dtype=float)
    cost_used = float(plan["costo_gestion_estimado"].sum())
    return {
        **base,
        "contactos_seleccionados": int(len(plan)),
        "costo_usado": cost_used,
        "presupuesto_no_usado": float(budget) - cost_used,
        "recuperacion_esperada": float(plan["recuperacion_esperada"].sum()),
        "valor_neto_esperado": float(plan["valor_esperado_neto"].sum()),
        "roi_esperado_promedio": float(roi.mean()) if roi.notna().any() else None,
        "contactos_por_canal": {
            str(k): int(v) for k, v in plan.get("canal_gestion", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
        "contactos_por_accion": {
            str(k): int(v) for k, v in plan.get("accion_recomendada", pd.Series(dtype=str)).value_counts().to_dict().items()
        },
    }


def summarize_optimized_plan(
    plan: pd.DataFrame,
    budget: float,
    capacity: int | None = None,
    channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
) -> dict[str, Any]:
    """Resumen ejecutivo del plan.

    Si el plan fue generado por este módulo, reutiliza los diagnostics guardados en
    `plan.attrs["optimization_summary"]`. Si se perdió ese atributo al guardar/cargar CSV,
    reconstruye el resumen desde las columnas disponibles.
    """
    if isinstance(plan, pd.DataFrame) and plan.attrs.get("optimization_summary"):
        return plan.attrs["optimization_summary"]

    limits = _normalise_channel_limits(channel_limits)
    return _build_plan_summary(
        plan=plan,
        method=str(plan["metodo_optimizacion"].iloc[0]) if not plan.empty and "metodo_optimizacion" in plan else "unknown",
        budget=budget,
        capacity=capacity,
        channel_limits=limits,
        max_contacts_per_customer=max_contacts_per_customer,
        solver_status=str(plan["solver_status"].iloc[0]) if not plan.empty and "solver_status" in plan else "not_available",
        solver_success=bool(plan["solver_success"].iloc[0]) if not plan.empty and "solver_success" in plan else False,
        objective_value=float(plan["valor_esperado_neto"].sum()) if not plan.empty and "valor_esperado_neto" in plan else 0.0,
        optimality_gap=float(plan["optimality_gap"].iloc[0]) if not plan.empty and "optimality_gap" in plan and pd.notna(plan["optimality_gap"].iloc[0]) else None,
    )


def compare_greedy_vs_milp(
    df: pd.DataFrame,
    budget: float = 500_000,
    capacity: int | None = None,
    min_expected_net: float = 0,
    channel_limits: dict[str, int | None] | None = None,
    max_contacts_per_customer: int | None = 1,
    time_limit_seconds: float | None = 30,
    mip_rel_gap: float | None = 0.001,
) -> pd.DataFrame:
    """Compara la heurística greedy contra MILP en una tabla compacta."""
    greedy = optimize_contact_plan_greedy(
        df=df,
        budget=budget,
        capacity=capacity,
        min_expected_net=min_expected_net,
        channel_limits=channel_limits,
        max_contacts_per_customer=max_contacts_per_customer,
    )
    milp_plan = optimize_contact_plan_milp(
        df=df,
        budget=budget,
        capacity=capacity,
        min_expected_net=min_expected_net,
        channel_limits=channel_limits,
        max_contacts_per_customer=max_contacts_per_customer,
        time_limit_seconds=time_limit_seconds,
        mip_rel_gap=mip_rel_gap,
        fallback_to_greedy=False,
    )
    rows = []
    for name, plan in [("greedy", greedy), ("milp", milp_plan)]:
        summary = summarize_optimized_plan(
            plan,
            budget=budget,
            capacity=capacity,
            channel_limits=channel_limits,
            max_contacts_per_customer=max_contacts_per_customer,
        )
        rows.append(
            {
                "metodo": name,
                "contactos_seleccionados": summary["contactos_seleccionados"],
                "costo_usado": summary["costo_usado"],
                "recuperacion_esperada": summary["recuperacion_esperada"],
                "valor_neto_esperado": summary["valor_neto_esperado"],
                "solver_success": summary["solver_success"],
                "optimality_gap": summary["optimality_gap"],
            }
        )
    comparison = pd.DataFrame(rows)
    greedy_value = float(comparison.loc[comparison["metodo"] == "greedy", "valor_neto_esperado"].iloc[0])
    milp_value = float(comparison.loc[comparison["metodo"] == "milp", "valor_neto_esperado"].iloc[0])
    improvement = (milp_value - greedy_value) / abs(greedy_value) if greedy_value != 0 else np.nan
    comparison["mejora_milp_vs_greedy_pct"] = np.where(comparison["metodo"] == "milp", improvement * 100, np.nan)
    return comparison
