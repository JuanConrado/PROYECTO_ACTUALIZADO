"""Tests del particionamiento temporal."""
from src.data_loader import load_intc
from src.features import build_features
from src.targets import build_all_targets
from src.splits import temporal_split


def test_temporal_order_train_val_test():
    df = load_intc()
    df = build_features(df)
    df = build_all_targets(df)
    df = df.dropna().reset_index(drop=True)
    train, val, test = temporal_split(df)

    assert train["date"].max() < val["date"].min(), \
        "Leakage temporal: train solapa con val"
    assert val["date"].max() < test["date"].min(), \
        "Leakage temporal: val solapa con test"


def test_split_proportions_sane():
    df = load_intc()
    df = build_features(df)
    df = build_all_targets(df)
    df = df.dropna().reset_index(drop=True)
    train, val, test = temporal_split(df)

    n = len(df)
    assert 0.65 <= len(train) / n <= 0.75
    assert 0.10 <= len(val) / n <= 0.20
    assert 0.10 <= len(test) / n <= 0.20
