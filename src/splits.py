"""
Particionamiento temporal del dataset.
Train / Validation / Test cronológicos, sin shuffle.
"""
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.config import TRAIN_FRAC, VAL_FRAC, N_CV_SPLITS


def temporal_split(df: pd.DataFrame,
                    train_frac: float = TRAIN_FRAC,
                    val_frac: float = VAL_FRAC):
    """
    Divide cronológicamente en train / val / test (15% por defecto cada una de val/test).

    Parameters
    ----------
    df : DataFrame con columna 'date', ordenado cronológicamente.

    Returns
    -------
    train, val, test : tres DataFrames con índices reseteados.
    """
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    cut_train = int(n * train_frac)
    cut_val = int(n * (train_frac + val_frac))

    train = df.iloc[:cut_train].copy().reset_index(drop=True)
    val = df.iloc[cut_train:cut_val].copy().reset_index(drop=True)
    test = df.iloc[cut_val:].copy().reset_index(drop=True)

    # Verificación crítica
    assert train["date"].max() < val["date"].min(), "Leakage: train→val"
    assert val["date"].max() < test["date"].min(), "Leakage: val→test"

    return train, val, test


def make_tscv(n_splits: int = N_CV_SPLITS) -> TimeSeriesSplit:
    """Devuelve un TimeSeriesSplit con folds expandidos."""
    return TimeSeriesSplit(n_splits=n_splits)


def split_summary(train, val, test) -> str:
    """String con resumen del split, útil para imprimir en notebooks."""
    return (
        f"Train: {len(train):>5} obs | "
        f"{train['date'].min().date()} → {train['date'].max().date()}\n"
        f"Val  : {len(val):>5} obs | "
        f"{val['date'].min().date()} → {val['date'].max().date()}\n"
        f"Test : {len(test):>5} obs | "
        f"{test['date'].min().date()} → {test['date'].max().date()}"
    )


if __name__ == "__main__":
    from src.data_loader import load_intc
    from src.features import build_features
    from src.targets import build_all_targets

    df = load_intc()
    df = build_features(df)
    df = build_all_targets(df)
    df = df.dropna().reset_index(drop=True)

    train, val, test = temporal_split(df)
    print(split_summary(train, val, test))
