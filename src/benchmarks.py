"""
Benchmarks econométricos clásicos para predicción de volatilidad.
Comparten la misma interfaz: .fit(y_train) / .predict_path(n_steps).
"""
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ------------------------------------------------------------
# Naive: y_pred[t+h] = y[t]
# ------------------------------------------------------------

class NaiveForecast:
    name = "Naive"

    def fit(self, y_train):
        self.last_value_ = float(y_train.iloc[-1] if hasattr(y_train, "iloc")
                                  else y_train[-1])
        return self

    def predict_path(self, n_steps: int) -> np.ndarray:
        return np.repeat(self.last_value_, n_steps)


# ------------------------------------------------------------
# Rolling Mean: media móvil simple
# ------------------------------------------------------------

class RollingMeanForecast:
    name = "Rolling Mean"

    def __init__(self, window: int = 21):
        self.window = window

    def fit(self, y_train):
        self.history_ = pd.Series(y_train).reset_index(drop=True)
        self.last_mean_ = float(self.history_.iloc[-self.window:].mean())
        return self

    def predict_path(self, n_steps: int) -> np.ndarray:
        return np.repeat(self.last_mean_, n_steps)


# ------------------------------------------------------------
# EWMA: Exponential Weighted Moving Average (RiskMetrics)
# ------------------------------------------------------------

class EWMAForecast:
    name = "EWMA"

    def __init__(self, alpha: float = 0.94):
        self.alpha = alpha

    def fit(self, y_train):
        y = pd.Series(y_train).reset_index(drop=True)
        ewm = y.ewm(alpha=(1 - self.alpha), adjust=False).mean()
        self.last_value_ = float(ewm.iloc[-1])
        return self

    def predict_path(self, n_steps: int) -> np.ndarray:
        return np.repeat(self.last_value_, n_steps)


# ------------------------------------------------------------
# ARIMA (auto-order)
# ------------------------------------------------------------

class ARIMAForecast:
    name = "ARIMA"

    def __init__(self, order=None, max_p=3, max_q=3, d=1):
        self.order = order
        self.max_p = max_p
        self.max_q = max_q
        self.d = d

    def fit(self, y_train):
        try:
            from pmdarima import auto_arima
            if self.order is None:
                model = auto_arima(
                    y_train, seasonal=False,
                    max_p=self.max_p, max_q=self.max_q, d=self.d,
                    suppress_warnings=True, error_action="ignore",
                )
                self.order = model.order
                self.model_ = model
            else:
                from statsmodels.tsa.arima.model import ARIMA
                self.model_ = ARIMA(y_train, order=self.order).fit()
        except ImportError:
            from statsmodels.tsa.arima.model import ARIMA
            self.order = self.order or (1, 1, 1)
            self.model_ = ARIMA(y_train, order=self.order).fit()
        return self

    def predict_path(self, n_steps: int) -> np.ndarray:
        try:
            forecast = self.model_.predict(n_periods=n_steps)
        except TypeError:
            forecast = self.model_.forecast(steps=n_steps)
        return np.asarray(forecast).flatten()


# ------------------------------------------------------------
# GARCH(1,1) — el estándar para volatilidad financiera
# ------------------------------------------------------------

class GARCHForecast:
    """
    GARCH(1,1) sobre log-retornos. Pronostica volatilidad condicional (σ_t).
    Espera que se le pase la SERIE DE LOG-RETORNOS, no de volatilidad.
    """
    name = "GARCH(1,1)"

    def __init__(self, p: int = 1, q: int = 1, dist: str = "normal"):
        self.p = p
        self.q = q
        self.dist = dist

    def fit(self, log_returns):
        from arch import arch_model
        self._scale = 100.0
        y = pd.Series(log_returns).dropna() * self._scale
        self.model_ = arch_model(
            y, vol="GARCH", p=self.p, q=self.q, dist=self.dist
        ).fit(disp="off")
        return self

    def predict_path(self, n_steps: int) -> np.ndarray:
        forecast = self.model_.forecast(horizon=n_steps, reindex=False)
        sigma = np.sqrt(forecast.variance.values[-1, :]) / self._scale
        return sigma


# ------------------------------------------------------------
# Helper: aplicar un benchmark con expanding window forecast
# ------------------------------------------------------------

def expanding_window_forecast(model_class, model_kwargs,
                              y_full, train_size: int,
                              horizon: int = 1,
                              input_is_returns: bool = False,
                              returns_full=None) -> np.ndarray:
    """
    Para cada t desde train_size hasta len(y_full)-horizon:
      - Reentrena el modelo con y_full[:t]
      - Predice y_full[t+horizon-1]
    Devuelve el array de predicciones.

    Si input_is_returns (caso GARCH), el modelo se entrena con returns_full
    pero la predicción se compara contra y_full (volatilidad).
    """
    n = len(y_full)
    preds = np.full(n, np.nan)

    for t in range(train_size, n - horizon + 1):
        train_data = (returns_full.iloc[:t]
                      if input_is_returns else y_full.iloc[:t])
        try:
            mdl = model_class(**model_kwargs).fit(train_data)
            forecast = mdl.predict_path(horizon)
            preds[t + horizon - 1] = forecast[-1]
        except Exception:
            preds[t + horizon - 1] = np.nan

    return preds
