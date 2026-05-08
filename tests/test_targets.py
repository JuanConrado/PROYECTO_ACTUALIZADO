"""Tests de los targets de regresión y clasificación."""
import numpy as np

from src.data_loader import load_intc
from src.features import build_features
from src.targets import (build_regression_targets, build_classification_target)


def test_target_vol_7_is_shifted_correctly():
    """target_vol_7[t] debe ser igual a vol_7[t+7]."""
    df = load_intc()
    df = build_features(df)
    df = build_regression_targets(df)

    # Tomar un índice intermedio donde ambas existen
    n = len(df)
    t = int(n * 0.5)
    expected = df["vol_7"].iloc[t + 7]
    actual = df["target_vol_7"].iloc[t]

    if np.isnan(expected) and np.isnan(actual):
        return
    assert np.isclose(actual, expected, equal_nan=True), (
        f"target_vol_7[{t}]={actual} != vol_7[{t+7}]={expected}"
    )


def test_target_class_is_binary():
    df = load_intc()
    df = build_features(df)
    df = build_regression_targets(df)
    df = build_classification_target(df)
    valid = df["target_class"].dropna().unique()
    assert set(valid).issubset({0, 1}), f"Valores inesperados en target_class: {valid}"


def test_target_class_balance_reasonable():
    """El target binario debe estar entre 30% y 70% de cada clase."""
    df = load_intc()
    df = build_features(df)
    df = build_regression_targets(df)
    df = build_classification_target(df)
    p1 = df["target_class"].mean()
    assert 0.30 <= p1 <= 0.70, f"Desbalance inesperado: P(clase=1)={p1:.3f}"
