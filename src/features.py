"""
Feature engineering temporal para INTC.
Todas las features son CAUSALES: usan solo información disponible al tiempo t.

Familias de features (~55 totales):
  - Lags de log-retornos
  - Lags de volatilidades rolling
  - Promedios de retorno absoluto
  - Range OHLC
  - Rolling range
  - Retornos acumulados
  - Ratios de volatilidad (estructura temporal)
  - Momentum (RSI, Stochastic)
  - Bandas de Bollinger
  - ATR (Average True Range)
  - Variables de calendario
"""
import numpy as np
import pandas as pd

from src.config import VOL_WINDOWS


# ------------------------------------------------------------
# Bloques individuales
# ------------------------------------------------------------

def add_log_returns(df: pd.DataFrame) -> pd.DataFrame:
    """log_ret_t = log(close_t / close_{t-1})."""
    df = df.copy()
    df["log_ret"] = np.log(df["close"]).diff()
    return df


def add_rolling_volatilities(df: pd.DataFrame,
                             windows=VOL_WINDOWS) -> pd.DataFrame:
    """vol_w = std rolling de log_ret en ventana w."""
    df = df.copy()
    for w in windows:
        df[f"vol_{w}"] = df["log_ret"].rolling(w).std()
    return df


def add_lag_features(df: pd.DataFrame,
                     log_ret_lags=(1, 2, 3, 5, 7, 10, 14, 21),
                     vol_windows=VOL_WINDOWS,
                     vol_lags=(1, 2, 3, 5, 7, 14, 21)) -> pd.DataFrame:
    """Lags causales de log-retornos y volatilidades."""
    df = df.copy()

    # Lags de log_ret
    for k in log_ret_lags:
        df[f"log_ret_lag{k}"] = df["log_ret"].shift(k)

    # Lags de cada volatilidad rolling
    for w in vol_windows:
        for k in vol_lags:
            df[f"vol_{w}_lag{k}"] = df[f"vol_{w}"].shift(k)

    return df


def add_abs_return_means(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """Media móvil del retorno absoluto: proxy de volatilidad realizada."""
    df = df.copy()
    for w in windows:
        df[f"abs_ret_ma{w}"] = df["log_ret"].abs().rolling(w).mean()
    return df


def add_ohlc_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features OHLCV derivadas."""
    df = df.copy()
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["oc_change"] = (df["close"] - df["open"]) / df["open"]
    df["vol_change"] = df["volume"].pct_change()
    df["vol_ma_10"] = df["volume"].rolling(10).mean()
    df["vol_ratio_10"] = df["volume"] / df["vol_ma_10"]
    return df


def add_rolling_range(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """Promedio rolling de hl_range."""
    df = df.copy()
    for w in windows:
        df[f"hl_range_ma{w}"] = df["hl_range"].rolling(w).mean()
    return df


def add_cumulative_returns(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """Retorno acumulado rolling."""
    df = df.copy()
    for w in windows:
        df[f"cum_ret_{w}"] = df["log_ret"].rolling(w).sum()
    return df


def add_volatility_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Estructura temporal de volatilidad."""
    df = df.copy()
    df["vol_ratio_7_14"] = df["vol_7"] / df["vol_14"]
    df["vol_ratio_7_21"] = df["vol_7"] / df["vol_21"]
    df["vol_ratio_14_28"] = df["vol_14"] / df["vol_28"]
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index (RSI)."""
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Oscilador estocástico %K."""
    df = df.copy()
    low_min = df["low"].rolling(period).min()
    high_max = df["high"].rolling(period).max()
    df[f"stoch_k_{period}"] = (
        100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    )
    return df


def add_bollinger_bandwidth(df: pd.DataFrame, period: int = 20,
                             n_std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bandwidth = (upper - lower) / middle."""
    df = df.copy()
    ma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    df[f"bb_bandwidth_{period}"] = (upper - lower) / ma.replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range (ATR)."""
    df = df.copy()
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df[f"atr_{period}"] = true_range.rolling(period).mean()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Variables de calendario causales."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)
    return df


# ------------------------------------------------------------
# Pipeline principal
# ------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todas las features sobre el dataframe completo.
    Reemplaza inf/-inf por NaN para que el imputer del Pipeline las maneje.

    Parameters
    ----------
    df : DataFrame con columnas date, open, high, low, close, volume.

    Returns
    -------
    DataFrame con ~55 features adicionales.
    """
    df = add_log_returns(df)
    df = add_rolling_volatilities(df)
    df = add_lag_features(df)
    df = add_abs_return_means(df)
    df = add_ohlc_features(df)
    df = add_rolling_range(df)
    df = add_cumulative_returns(df)
    df = add_volatility_ratios(df)
    df = add_rsi(df)
    df = add_stochastic_k(df)
    df = add_bollinger_bandwidth(df)
    df = add_atr(df)
    df = add_calendar_features(df)

    # Reemplazar inf y -inf por NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Devuelve la lista de columnas que son features (no targets ni OHLCV crudos).
    """
    exclude = {
        # Crudos
        "date", "open", "high", "low", "close", "volume",
        # Auxiliares (la versión rolling no se usa como feature)
        "log_ret", "vol_7", "vol_14", "vol_21", "vol_28", "vol_ma_10",
        # Targets
        "target_vol_7", "target_vol_14", "target_vol_21", "target_class",
        "target_regime",
    }
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    from src.data_loader import load_intc
    df = load_intc()
    df_feat = build_features(df)
    cols = get_feature_columns(df_feat)
    print(f"Filas: {len(df_feat)}")
    print(f"Features: {len(cols)}")
    print(cols)
