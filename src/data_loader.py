"""
Carga y limpieza inicial del dataset INTC.
Lee desde data/raw/intc.us.txt, devuelve un DataFrame limpio y ordenado.
"""
import pandas as pd
import numpy as np

from src.config import DATASET_FILE, START_DATE


def load_intc(filter_modern: bool = True) -> pd.DataFrame:
    """
    Carga el dataset de INTC desde data/raw/intc.us.txt.

    Parameters
    ----------
    filter_modern : bool
        Si True, filtra desde START_DATE (1990-01-01) en adelante.

    Returns
    -------
    pd.DataFrame
        Columnas: date, open, high, low, close, volume
        Ordenado cronológicamente, sin NaNs en OHLCV.
    """
    if not DATASET_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {DATASET_FILE}. "
            "Asegúrate de que data/raw/intc.us.txt existe."
        )

    df = pd.read_csv(DATASET_FILE)

    # Normalizar nombres de columnas
    df.columns = [c.strip().lower() for c in df.columns]

    # Convertir fecha
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Asegurar tipos numéricos
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eliminar columna inútil
    if "openint" in df.columns:
        df = df.drop(columns=["openint"])

    # Limpiar nulos en OHLCV
    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    if filter_modern:
        df = df[df["date"] >= pd.Timestamp(START_DATE)].copy()
        df = df.reset_index(drop=True)

    return df


def quick_qa(df: pd.DataFrame) -> dict:
    """
    QA rápido del dataset.

    Returns
    -------
    dict con estadísticas descriptivas y flags de calidad.
    """
    stats = {
        "n_rows": len(df),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "n_duplicates_date": int(df["date"].duplicated().sum()),
        "n_negative_volume": int((df["volume"] < 0).sum()) if "volume" in df else 0,
    }

    # Días hábiles esperados vs presentes
    expected = pd.bdate_range(stats["date_min"], stats["date_max"])
    actual = pd.DatetimeIndex(df["date"].unique())
    missing = expected.difference(actual)
    stats["n_business_days_expected"] = len(expected)
    stats["n_business_days_present"] = len(actual)
    stats["n_business_days_missing"] = len(missing)
    stats["pct_business_days_missing"] = (
        len(missing) / len(expected) * 100 if len(expected) > 0 else 0.0
    )

    # OHLC consistencia
    stats["n_high_lt_low"] = int((df["high"] < df["low"]).sum())
    stats["n_high_lt_open_close"] = int(
        (df["high"] < np.maximum(df["open"], df["close"])).sum()
    )
    stats["n_low_gt_open_close"] = int(
        (df["low"] > np.minimum(df["open"], df["close"])).sum()
    )

    return stats


if __name__ == "__main__":
    df = load_intc()
    qa = quick_qa(df)
    print(f"Dataset: {df.shape}")
    print(f"Período: {qa['date_min']} → {qa['date_max']}")
    print(f"Días hábiles faltantes: {qa['n_business_days_missing']} "
          f"({qa['pct_business_days_missing']:.2f}%)")
