"""
Helpers para persistir y cargar artefactos del proyecto.
Centraliza paths y formato.
"""
import json
import joblib
import pandas as pd

from src.config import (MODELS_DIR, PREDICTIONS_DIR, METRICS_DIR,
                         DATA_PROCESSED)


# ============================================================
# Modelos
# ============================================================

def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name: str):
    path = MODELS_DIR / f"{name}.joblib"
    return joblib.load(path)


# ============================================================
# Predicciones
# ============================================================

def save_predictions_df(df: pd.DataFrame, name: str):
    path = PREDICTIONS_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_predictions_df(name: str) -> pd.DataFrame:
    path = PREDICTIONS_DIR / f"{name}.parquet"
    return pd.read_parquet(path)


# ============================================================
# Métricas
# ============================================================

def save_metrics(metrics: dict, name: str):
    path = METRICS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_metrics(name: str) -> dict:
    path = METRICS_DIR / f"{name}.json"
    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# Data procesado
# ============================================================

def save_processed(df: pd.DataFrame, name: str):
    path = DATA_PROCESSED / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def load_processed(name: str) -> pd.DataFrame:
    path = DATA_PROCESSED / f"{name}.parquet"
    return pd.read_parquet(path)
