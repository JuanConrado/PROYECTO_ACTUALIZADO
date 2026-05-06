"""
Construcción de los targets de regresión y clasificación.
TODOS los targets usan shift NEGATIVO (futuro), pero el shift es CONSISTENTE
con el tamaño de la ventana para evitar overlap.
"""
import pandas as pd

from src.config import HORIZONS, PRIMARY_HORIZON


def build_regression_targets(df: pd.DataFrame,
                              horizons=HORIZONS) -> pd.DataFrame:
    """
    Para cada horizonte h, target_vol_h = vol_h.shift(-h).
    Esto garantiza que la ventana del target [t+1, t+h] no solapa con
    la ventana de vol_h actual [t-h+1, t].
    """
    df = df.copy()
    for h in horizons:
        col_vol = f"vol_{h}"
        if col_vol not in df.columns:
            raise KeyError(
                f"Columna {col_vol} no existe. ¿Llamó a build_features primero?"
            )
        df[f"target_vol_{h}"] = df[col_vol].shift(-h)
    return df


def build_classification_target(df: pd.DataFrame,
                                  horizon: int = PRIMARY_HORIZON) -> pd.DataFrame:
    """
    target_class = 1 si la volatilidad futura supera a la actual; 0 en caso contrario.
    Usa el horizonte primario (7 días).
    """
    df = df.copy()
    col_vol = f"vol_{horizon}"
    col_target = f"target_vol_{horizon}"
    if col_vol not in df.columns or col_target not in df.columns:
        raise KeyError(
            "Faltan columnas vol_h o target_vol_h. Llamó a build_regression_targets primero?"
        )
    df["target_class"] = (df[col_target] > df[col_vol]).astype(int)
    return df


def build_regime_target(df: pd.DataFrame, train_idx,
                         horizon: int = 21,
                         n_regimes: int = 3) -> pd.DataFrame:
    """
    Construye target_regime ∈ {0, 1, 2} basado en cuantiles de vol_horizon
    calculados SOLO sobre train_idx (anti-leakage).

    Parameters
    ----------
    df : DataFrame que ya contiene vol_horizon.
    train_idx : índices del set de entrenamiento.
    horizon : horizonte de la volatilidad (default 21).
    n_regimes : número de regímenes (default 3 = terciles).

    Returns
    -------
    DataFrame con columna 'target_regime' y atributo de borders en metadata.
    """
    df = df.copy()
    col = f"vol_{horizon}"
    if col not in df.columns:
        raise KeyError(f"Columna {col} no existe.")

    # Calcular bordes SOLO con train
    train_values = df.loc[train_idx, col].dropna()
    quantiles = [(i + 1) / n_regimes for i in range(n_regimes - 1)]
    borders = train_values.quantile(quantiles).values

    # Asignar régimen al dataset completo usando bordes fijos
    def _assign_regime(v):
        if pd.isna(v):
            return -1  # missing
        for i, b in enumerate(borders):
            if v <= b:
                return i
        return n_regimes - 1

    df["target_regime"] = df[col].apply(_assign_regime)

    # Guardar borders en attrs
    df.attrs["regime_borders"] = list(borders)

    return df


def build_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Construye todos los targets en orden correcto."""
    df = build_regression_targets(df)
    df = build_classification_target(df)
    return df


if __name__ == "__main__":
    from src.data_loader import load_intc
    from src.features import build_features

    df = load_intc()
    df = build_features(df)
    df = build_all_targets(df)
    print(df[["date", "target_vol_7", "target_vol_14",
              "target_vol_21", "target_class"]].tail(15))
