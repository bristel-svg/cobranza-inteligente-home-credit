from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .config import DEFAULT_RANDOM_STATE

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover - se valida en runtime con mensaje claro.
    lgb = None
    LGBMClassifier = None
    LGBMRegressor = None


MISSING_CATEGORY = "__MISSING__"


@dataclass
class ModelArtifacts:
    classifier: Any
    regressor: Any
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    metrics: dict[str, Any]
    categorical_levels: dict[str, list[str]] | None = None
    probability_calibrator: LogisticRegression | None = None
    holdout_predictions: pd.DataFrame | None = None
    amount_target_log_transformed: bool = True


class DataFrameColumnSelector(BaseEstimator, TransformerMixin):
    """Asegura que los baselines reciban columnas en el mismo orden."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


def _require_lightgbm() -> None:
    if lgb is None or LGBMClassifier is None or LGBMRegressor is None:
        raise ImportError(
            "LightGBM no está instalado. Instálalo con `pip install lightgbm` "
            "o agrégalo al requirements.txt del proyecto."
        )


def _clean_column_lists(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Valida columnas y evita duplicados entre numéricas y categóricas."""
    numeric = [c for c in dict.fromkeys(numeric_cols) if c in df.columns]
    categorical = [c for c in dict.fromkeys(categorical_cols) if c in df.columns and c not in numeric]
    feature_columns = numeric + categorical
    return feature_columns, numeric, categorical


def _fit_categorical_levels(X: pd.DataFrame, categorical_cols: list[str]) -> dict[str, list[str]]:
    """Guarda categorías observadas en entrenamiento para reproducibilidad en scoring."""
    levels: dict[str, list[str]] = {}
    for col in categorical_cols:
        values = X[col].astype("string").fillna(MISSING_CATEGORY)
        cats = sorted(values.unique().tolist())
        if MISSING_CATEGORY not in cats:
            cats.append(MISSING_CATEGORY)
        levels[col] = cats
    return levels


def _prepare_lgbm_frame(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_cols: list[str],
    categorical_cols: list[str],
    categorical_levels: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """
    LightGBM maneja valores faltantes y variables categóricas de pandas. Para que el
    scoring sea estable, las categorías se fijan según entrenamiento; categorías nuevas
    en producción quedan como missing.
    """
    missing = sorted(set(feature_columns) - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas de features para score/entrenamiento: {missing}")

    X = df[feature_columns].copy()
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    for col in categorical_cols:
        if categorical_levels is None:
            X[col] = X[col].astype("string").fillna(MISSING_CATEGORY).astype("category")
        else:
            cats = categorical_levels.get(col, [MISSING_CATEGORY])
            dtype = CategoricalDtype(categories=cats, ordered=False)
            X[col] = X[col].astype("string").fillna(MISSING_CATEGORY).astype(dtype)

    return X


def _make_baseline_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """Preprocesador solo para baseline RandomForest.

    No se escala porque los árboles no lo necesitan. El OneHotEncoder queda restringido
    al baseline; el modelo productivo usa categóricas nativas de LightGBM.
    """
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, min_frequency=20)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.30,
        verbose_feature_names_out=False,
    )


def _lgbm_callbacks(stopping_rounds: int = 75) -> list[Any]:
    _require_lightgbm()
    return [
        lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False),
        lgb.log_evaluation(period=0),
    ]


def build_classifier(random_state: int) -> Any:
    """Modelo principal de probabilidad de regularización.

    Se usa LightGBM por desempeño industrial en datos tabulares, manejo nativo de
    valores faltantes/categóricos y soporte de early stopping. El problema completo se
    modela como Hurdle Model: esta pieza estima P(regulariza | X).
    """
    _require_lightgbm()
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=60,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.00,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=4,
        importance_type="gain",
        verbosity=-1,
    )


def build_regressor(random_state: int) -> Any:
    """Modelo condicional de monto recuperado positivo.

    Esta pieza estima E(monto recuperado | regulariza = 1, X). Se entrena sobre log1p
    del monto para estabilizar colas y luego se invierte con expm1 al predecir.
    """
    _require_lightgbm()
    return LGBMRegressor(
        objective="regression_l1",
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=45,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.00,
        random_state=random_state,
        n_jobs=4,
        importance_type="gain",
        verbosity=-1,
    )


def _stratify_if_possible(y: pd.Series) -> pd.Series | None:
    y_non_null = y.dropna()
    if y_non_null.nunique() < 2:
        return None
    if y_non_null.value_counts().min() < 2:
        return None
    return y


def _group_split(
    X: pd.DataFrame,
    y: pd.Series,
    amount: pd.Series,
    groups: pd.Series | None,
    test_size: float,
    random_state: int,
):
    """Split por cliente para reducir fuga del mismo SK_ID_CURR entre particiones."""
    if groups is not None and groups.nunique(dropna=True) > 10:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        return (
            X.iloc[train_idx],
            X.iloc[test_idx],
            y.iloc[train_idx],
            y.iloc[test_idx],
            amount.iloc[train_idx],
            amount.iloc[test_idx],
            groups.iloc[train_idx],
            groups.iloc[test_idx],
        )

    split = train_test_split(
        X,
        y,
        amount,
        test_size=test_size,
        random_state=random_state,
        stratify=_stratify_if_possible(y),
    )
    X_train, X_test, y_train, y_test, amount_train, amount_test = split
    return X_train, X_test, y_train, y_test, amount_train, amount_test, None, None


def _calibrate_probabilities(raw_proba_cal: np.ndarray, y_cal: pd.Series) -> LogisticRegression | None:
    if len(np.unique(y_cal)) < 2:
        return None
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(raw_proba_cal.reshape(-1, 1), y_cal.astype(int))
    return calibrator


def _apply_calibrator(raw_proba: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    if calibrator is None:
        return np.clip(raw_proba, 0, 1)
    return calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


def _safe_roc_auc(y_true: pd.Series, proba: np.ndarray) -> float:
    if pd.Series(y_true).nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def _safe_average_precision(y_true: pd.Series, proba: np.ndarray) -> float:
    if pd.Series(y_true).sum() <= 0:
        return float("nan")
    return float(average_precision_score(y_true, proba))


def _capture_metrics(y_true: pd.Series, proba: np.ndarray, amounts: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"y": y_true.astype(int).to_numpy(), "p": proba, "amount": amounts.astype(float).to_numpy()})
    frame = frame.sort_values("p", ascending=False).reset_index(drop=True)
    metrics: dict[str, float] = {}
    for frac in [0.05, 0.10, 0.20, 0.30]:
        n = max(1, int(len(frame) * frac))
        top = frame.head(n)
        metrics[f"top_{int(frac*100)}pct_target_rate"] = float(top["y"].mean())
        denom = float(frame["amount"].sum())
        metrics[f"top_{int(frac*100)}pct_recovery_capture"] = float(top["amount"].sum() / denom) if denom > 0 else 0.0
    return metrics


def _business_threshold_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    """Métricas top-k, más útiles en cobranza que un threshold fijo 0.5."""
    y = y_true.astype(int).to_numpy()
    out: dict[str, float] = {}
    if len(y) == 0:
        return out
    order = np.argsort(-proba)
    positives = max(1, int(y.sum()))
    for frac in [0.05, 0.10, 0.20, 0.30]:
        n = max(1, int(np.ceil(len(y) * frac)))
        selected = order[:n]
        tp = int(y[selected].sum())
        out[f"top_{int(frac*100)}pct_precision_business"] = float(tp / n)
        out[f"top_{int(frac*100)}pct_recall_business"] = float(tp / positives)
        out[f"top_{int(frac*100)}pct_threshold"] = float(np.min(proba[selected]))
    return out


def _expected_recovery_metrics(
    y_amount_true: pd.Series,
    expected_recovery_pred: np.ndarray,
    pred_amount_conditional: np.ndarray,
) -> dict[str, float]:
    """Métricas del Hurdle Model: monto condicional y recuperación esperada total."""
    y_amount = y_amount_true.astype(float).clip(lower=0).to_numpy()
    metrics: dict[str, float] = {}
    metrics["expected_recovery_mae_all"] = float(mean_absolute_error(y_amount, expected_recovery_pred))
    metrics["expected_recovery_rmse_all"] = float(np.sqrt(mean_squared_error(y_amount, expected_recovery_pred)))

    positive_mask = y_amount > 0
    metrics["n_positive_amount_test"] = int(positive_mask.sum())
    if positive_mask.any():
        metrics["amount_mae_positive"] = float(mean_absolute_error(y_amount[positive_mask], pred_amount_conditional[positive_mask]))
        metrics["amount_rmse_positive"] = float(
            np.sqrt(mean_squared_error(y_amount[positive_mask], pred_amount_conditional[positive_mask]))
        )
    else:
        metrics["amount_mae_positive"] = float("nan")
        metrics["amount_rmse_positive"] = float("nan")
    return metrics


def _fit_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
) -> np.ndarray:
    """Baseline RandomForest para medir ganancia incremental del modelo productivo."""
    baseline = Pipeline(
        steps=[
            ("select", DataFrameColumnSelector(numeric_cols + categorical_cols)),
            ("preprocessor", _make_baseline_preprocessor(numeric_cols, categorical_cols)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=80,
                    max_depth=9,
                    min_samples_leaf=80,
                    n_jobs=4,
                    random_state=random_state,
                    class_weight="balanced_subsample",
                ),
            ),
        ]
    )
    baseline.fit(X_train, y_train)
    return baseline.predict_proba(X_test)[:, 1]


def _predict_amount_conditional(regressor: Any, X: pd.DataFrame) -> np.ndarray:
    pred_log = regressor.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


def _cap_by_current_debt(pred_amount: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    if "monto_vencido_actual" not in frame.columns:
        return pred_amount
    debt = pd.to_numeric(frame["monto_vencido_actual"], errors="coerce").fillna(0).clip(lower=0).to_numpy()
    return np.minimum(pred_amount, debt)


def train_models(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_class: str = "regulariza_horizonte",
    target_amount: str = "monto_recuperado_horizonte",
    test_size: float = 0.25,
    calibration_size: float = 0.20,
    validation_size: float = 0.15,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ModelArtifacts:

    _require_lightgbm()

    feature_columns, numeric_cols, categorical_cols = _clean_column_lists(df, numeric_cols, categorical_cols)
    if not feature_columns:
        raise ValueError("No hay columnas de features para entrenar.")

    data = df.dropna(subset=[target_class, target_amount]).copy()
    if data[target_class].nunique() < 2:
        raise ValueError("El target de clasificación tiene una sola clase. Revisa el horizonte o los filtros.")

    X_raw = data[feature_columns]
    y_class = data[target_class].astype(int)
    y_amount = data[target_amount].astype(float).clip(lower=0)
    groups = data["SK_ID_CURR"] if "SK_ID_CURR" in data.columns else None

    X_train_dev, X_test, y_train_dev, y_test, amt_train_dev, amt_test, grp_train_dev, _ = _group_split(
        X_raw, y_class, y_amount, groups, test_size=test_size, random_state=random_state
    )

    X_train_val, X_cal, y_train_val, y_cal, amt_train_val, amt_cal, grp_train_val, _ = _group_split(
        X_train_dev,
        y_train_dev,
        amt_train_dev,
        grp_train_dev if isinstance(grp_train_dev, pd.Series) else None,
        test_size=calibration_size,
        random_state=random_state + 1,
    )

    X_train, X_valid, y_train, y_valid, amt_train, amt_valid, _, _ = _group_split(
        X_train_val,
        y_train_val,
        amt_train_val,
        grp_train_val if isinstance(grp_train_val, pd.Series) else None,
        test_size=validation_size,
        random_state=random_state + 2,
    )

    categorical_levels = _fit_categorical_levels(X_train, categorical_cols)
    X_train_lgb = _prepare_lgbm_frame(X_train, feature_columns, numeric_cols, categorical_cols, categorical_levels)
    X_valid_lgb = _prepare_lgbm_frame(X_valid, feature_columns, numeric_cols, categorical_cols, categorical_levels)
    X_cal_lgb = _prepare_lgbm_frame(X_cal, feature_columns, numeric_cols, categorical_cols, categorical_levels)
    X_test_lgb = _prepare_lgbm_frame(X_test, feature_columns, numeric_cols, categorical_cols, categorical_levels)
    categorical_feature = [c for c in categorical_cols if c in X_train_lgb.columns]

    classifier = build_classifier(random_state)
    classifier.fit(
        X_train_lgb,
        y_train,
        eval_set=[(X_valid_lgb, y_valid)],
        eval_metric="auc",
        categorical_feature=categorical_feature,
        callbacks=_lgbm_callbacks(stopping_rounds=50),
    )

    reg_train_mask = (y_train.astype(int) == 1) & (amt_train.astype(float) > 0)
    reg_valid_mask = (y_valid.astype(int) == 1) & (amt_valid.astype(float) > 0)
    n_reg_train = int(reg_train_mask.sum())
    n_reg_valid = int(reg_valid_mask.sum())
    if n_reg_train < 30:
        raise ValueError(
            f"Hay solo {n_reg_train} casos positivos para entrenar el regresor condicional. "
            "Aumenta la muestra, amplía el horizonte o revisa la definición de regularización."
        )

    regressor = build_regressor(random_state)
    fit_kwargs: dict[str, Any] = {
        "categorical_feature": categorical_feature,
        "callbacks": _lgbm_callbacks(stopping_rounds=50),
    }
    if n_reg_valid >= 10:
        fit_kwargs["eval_set"] = [(X_valid_lgb.loc[reg_valid_mask], np.log1p(amt_valid.loc[reg_valid_mask]))]
        fit_kwargs["eval_metric"] = "l1"
    else:
        fit_kwargs["callbacks"] = [lgb.log_evaluation(period=0)]

    regressor.fit(
        X_train_lgb.loc[reg_train_mask],
        np.log1p(amt_train.loc[reg_train_mask]),
        **fit_kwargs,
    )

    raw_cal = classifier.predict_proba(X_cal_lgb)[:, 1]
    calibrator = _calibrate_probabilities(raw_cal, y_cal)

    raw_test = classifier.predict_proba(X_test_lgb)[:, 1]
    proba_test = _apply_calibrator(raw_test, calibrator)
    pred_amount_conditional = _predict_amount_conditional(regressor, X_test_lgb)
    pred_amount_capped = _cap_by_current_debt(pred_amount_conditional, data.loc[X_test.index])
    expected_recovery = proba_test * pred_amount_capped

    baseline_proba = _fit_baseline(X_train, y_train, X_test, numeric_cols, categorical_cols, random_state)

    base_ap = float(y_test.mean())
    metrics: dict[str, Any] = {}
    metrics["model_type"] = "LightGBM Hurdle Model: calibrated binary classifier + conditional positive-amount regressor"
    metrics["split_strategy"] = "GroupShuffleSplit by SK_ID_CURR when available"
    metrics["target_transform_amount"] = "log1p on positive recovered amounts only"
    metrics["categorical_strategy"] = "Native pandas categorical features in LightGBM; no OneHotEncoder in production model"
    metrics["numeric_scaling"] = "None; tree-based models do not require StandardScaler"
    metrics["n_rows_train"] = int(len(X_train))
    metrics["n_rows_validation"] = int(len(X_valid))
    metrics["n_rows_calibration"] = int(len(X_cal))
    metrics["n_rows_test"] = int(len(X_test))
    metrics["n_positive_amount_train"] = n_reg_train
    metrics["n_positive_amount_validation"] = n_reg_valid
    metrics["target_rate_train"] = float(y_train.mean())
    metrics["target_rate_validation"] = float(y_valid.mean())
    metrics["target_rate_calibration"] = float(y_cal.mean())
    metrics["target_rate_test"] = base_ap
    metrics["classifier_best_iteration"] = int(getattr(classifier, "best_iteration_", 0) or classifier.n_estimators)
    metrics["regressor_best_iteration"] = int(getattr(regressor, "best_iteration_", 0) or regressor.n_estimators)
    metrics["roc_auc"] = _safe_roc_auc(y_test, proba_test)
    metrics["average_precision"] = _safe_average_precision(y_test, proba_test)
    metrics["baseline_average_precision"] = base_ap
    metrics["baseline_rf_average_precision"] = _safe_average_precision(y_test, baseline_proba)
    metrics["lift_average_precision"] = float(metrics["average_precision"] / base_ap) if base_ap > 0 else float("nan")
    metrics["gini"] = float(2 * metrics["roc_auc"] - 1) if not np.isnan(metrics["roc_auc"]) else float("nan")
    metrics["brier_score"] = float(brier_score_loss(y_test, proba_test))
    metrics.update(_capture_metrics(y_test, proba_test, amt_test))
    metrics.update(_business_threshold_metrics(y_test, proba_test))
    metrics.update(_expected_recovery_metrics(amt_test, expected_recovery, pred_amount_capped))
    metrics[
        "business_metric_note"
    ] = "Para cobranza se prioriza ranking/top-k, calibración y recuperación esperada; no se usa threshold fijo 0.5 como regla comercial."

    holdout_cols = [
        c
        for c in [
            "SK_ID_CURR",
            "SK_ID_PREV",
            "cuota_numero_actual",
            "dias_relativos_vencimiento",
            "monto_vencido_actual",
            "DPD",
            "PAYMENT_RATIO",
            "UNPAID_AMOUNT",
            target_class,
            target_amount,
        ]
        if c in data.columns
    ]
    holdout_predictions = data.loc[X_test.index, holdout_cols].copy()
    holdout_predictions["prob_regulariza_raw"] = raw_test
    holdout_predictions["prob_regulariza"] = proba_test
    holdout_predictions["monto_recuperado_pred"] = pred_amount_conditional
    holdout_predictions["monto_recuperado_pred_acotado"] = pred_amount_capped
    holdout_predictions["recuperacion_esperada_modelo"] = expected_recovery

    return ModelArtifacts(
        classifier=classifier,
        regressor=regressor,
        feature_columns=feature_columns,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        metrics=metrics,
        categorical_levels=categorical_levels,
        probability_calibrator=calibrator,
        holdout_predictions=holdout_predictions,
        amount_target_log_transformed=True,
    )


def _apply_calibrator_to_series(raw: np.ndarray, calibrator: LogisticRegression | None) -> np.ndarray:
    return _apply_calibrator(raw, calibrator)


def score_dataset(df: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    scored = df.copy()
    categorical_levels = artifacts.categorical_levels or _fit_categorical_levels(scored, artifacts.categorical_columns)
    X = _prepare_lgbm_frame(
        scored,
        artifacts.feature_columns,
        artifacts.numeric_columns,
        artifacts.categorical_columns,
        categorical_levels,
    )
    raw = artifacts.classifier.predict_proba(X)[:, 1]
    scored["prob_regulariza_raw"] = raw
    scored["prob_regulariza"] = _apply_calibrator_to_series(raw, artifacts.probability_calibrator)

    pred_amount_conditional = _predict_amount_conditional(artifacts.regressor, X)
    pred_amount_capped = _cap_by_current_debt(pred_amount_conditional, scored)
    scored["monto_recuperado_pred"] = pred_amount_conditional
    scored["monto_recuperado_pred_acotado"] = pred_amount_capped
    scored["recuperacion_esperada_modelo"] = scored["prob_regulariza"].to_numpy() * pred_amount_capped
    return scored


def save_artifacts(artifacts: ModelArtifacts, models_dir) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts.classifier, models_dir / "modelo_regularizacion_lgbm.joblib")
    joblib.dump(artifacts.probability_calibrator, models_dir / "calibrador_probabilidad.joblib")
    joblib.dump(artifacts.regressor, models_dir / "modelo_monto_recuperado_condicional_lgbm.joblib")

    # Aliases para no romper scripts antiguos que esperan estos nombres.
    joblib.dump(artifacts.classifier, models_dir / "modelo_regularizacion.joblib")
    joblib.dump(artifacts.regressor, models_dir / "modelo_monto_recuperado.joblib")

    joblib.dump(
        {
            "feature_columns": artifacts.feature_columns,
            "numeric_columns": artifacts.numeric_columns,
            "categorical_columns": artifacts.categorical_columns,
            "categorical_levels": artifacts.categorical_levels,
            "metrics": artifacts.metrics,
            "model_family": "LightGBM Hurdle Model",
            "amount_target_log_transformed": artifacts.amount_target_log_transformed,
            "has_probability_calibrator": artifacts.probability_calibrator is not None,
            "has_holdout_predictions": artifacts.holdout_predictions is not None,
        },
        models_dir / "metadata_modelos.joblib",
    )
