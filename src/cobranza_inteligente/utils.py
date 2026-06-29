from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def ensure_dirs(paths: Iterable[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv_if_exists(path: Path, usecols: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    if usecols is not None:
        # Lee encabezado primero para evitar error si alguna columna no existe.
        header = pd.read_csv(path, nrows=0)
        existing = [c for c in usecols if c in header.columns]
        missing = sorted(set(usecols) - set(existing))
        if missing:
            warnings.warn(f"Columnas ausentes en {path.name}: {missing}")
        usecols = existing

    df = pd.read_csv(path, usecols=usecols, nrows=nrows)
    return reduce_memory_usage(df)


def reduce_memory_usage(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memoria convirtiendo numéricos a tipos más pequeños.

    En Home Credit hay tablas grandes; esto ayuda a correr el MVP en notebooks o laptops.
    """
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def safe_divide(numerator: pd.Series, denominator: pd.Series, default: float = 0.0) -> pd.Series:
    result = numerator / denominator.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan).fillna(default)


def add_prefix_except_keys(df: pd.DataFrame, prefix: str, keys: list[str]) -> pd.DataFrame:
    rename_map = {col: f"{prefix}{col}" for col in df.columns if col not in keys}
    return df.rename(columns=rename_map)
