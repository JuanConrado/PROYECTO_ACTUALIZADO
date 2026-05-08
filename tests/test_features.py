"""
Tests anti-leakage para features.
Verifica que ninguna feature en fecha t use información de fechas > t.
"""
import numpy as np
import pandas as pd
import pytest

from src.data_loader import load_intc
from src.features import build_features, get_feature_columns


@pytest.fixture(scope="module")
def df_with_features():
    df = load_intc()
    df = build_features(df)
    return df


def test_features_have_no_future_information(df_with_features):
    """
    Para cada feature f y fecha t, el valor de f[t] debe poder
    calcularse usando solo el subset df[date <= t].
    Estrategia: compara el valor en t calculado sobre el dataset completo
    contra el valor en t calculado sobre df[:t+1].
    Si difieren, hay leakage.
    """
    df = df_with_features.copy()
    feature_cols = get_feature_columns(df)

    # Probar en 3 puntos: 30%, 60%, 90% del dataset
    n = len(df)
    test_indices = [int(n * 0.3), int(n * 0.6), int(n * 0.9)]

    for t in test_indices:
        # Re-calcular features sobre el subset
        df_raw_partial = load_intc().iloc[:t + 1].copy().reset_index(drop=True)
        df_recalc = build_features(df_raw_partial)

        for col in feature_cols:
            val_full = df.iloc[t][col]
            val_partial = df_recalc.iloc[t][col]
            # Permitir NaN coincidentes
            if pd.isna(val_full) and pd.isna(val_partial):
                continue
            assert np.isclose(val_full, val_partial, equal_nan=True, rtol=1e-9), (
                f"Feature {col} en t={t} difiere entre dataset completo y parcial. "
                f"Esto indica leakage temporal. "
                f"full={val_full}, partial={val_partial}"
            )


def test_no_inf_in_features(df_with_features):
    """Después de build_features, no debe haber inf ni -inf."""
    df = df_with_features
    feature_cols = get_feature_columns(df)
    for col in feature_cols:
        assert not np.isinf(df[col].dropna()).any(), (
            f"Columna {col} contiene infinitos después de build_features."
        )


def test_feature_count_reasonable(df_with_features):
    """Espera entre 40 y 70 features."""
    df = df_with_features
    n_features = len(get_feature_columns(df))
    assert 40 <= n_features <= 70, f"Conteo inesperado de features: {n_features}"
