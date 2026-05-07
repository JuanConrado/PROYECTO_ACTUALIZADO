"""
HVRF — Hybrid Volatility Regime Forecaster.
Modelo original que combina:
  Stage 1: Régime Router (clasificador multiclase de régimen de volatilidad)
  Stage 2: Specialists (3 regresores especializados, uno por régimen)
  Stage 3: Mezcla suave (predicción ponderada por probabilidad de régimen)

Cumple sklearn API: fit, predict.
"""
import numpy as np
import pandas as pd
from copy import deepcopy

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBClassifier, XGBRegressor

from src.config import RANDOM_STATE


# ------------------------------------------------------------
# Specialists por defecto (uno por régimen)
# ------------------------------------------------------------

def _default_specialist_low():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
    ])


def _default_specialist_med():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=RANDOM_STATE, verbosity=0, tree_method="hist",
        )),
    ])


def _default_specialist_high():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf", C=2.0, epsilon=0.001)),
    ])


def _default_router():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, verbosity=0,
            tree_method="hist", eval_metric="mlogloss",
        )),
    ])


# ------------------------------------------------------------
# HVRF principal
# ------------------------------------------------------------

class HVRF(BaseEstimator, RegressorMixin):
    """
    Hybrid Volatility Regime Forecaster.

    Parameters
    ----------
    n_regimes : int
        Número de regímenes (default 3 = low/med/high).
    regime_var : str
        Nombre de la columna de volatilidad usada para definir regímenes
        en train. Por defecto 'vol_21'. Se espera que esté en X durante fit.
    router : sklearn estimator
        Modelo clasificador que produce P(régimen | features).
        Por defecto XGBClassifier.
    specialists : dict {regime_id: estimator}
        Regresor especializado para cada régimen.
    mixture : str
        'soft' (default) o 'argmax'.
    fallback_when_empty : str
        'global_ridge' o 'mean'. Qué hacer si un régimen tiene <= n_min muestras.
    n_min_per_regime : int
        Si un régimen tiene menos muestras que esto, no se entrena su specialist
        y se usa el fallback en su lugar.
    """

    def __init__(self,
                 n_regimes: int = 3,
                 regime_var: str = "vol_21",
                 router=None,
                 specialists: dict = None,
                 mixture: str = "soft",
                 fallback_when_empty: str = "global_ridge",
                 n_min_per_regime: int = 50):
        self.n_regimes = n_regimes
        self.regime_var = regime_var
        self.router = router
        self.specialists = specialists
        self.mixture = mixture
        self.fallback_when_empty = fallback_when_empty
        self.n_min_per_regime = n_min_per_regime

    # --- helpers ---

    def _resolve_router(self):
        return deepcopy(self.router) if self.router is not None else _default_router()

    def _resolve_specialists(self):
        if self.specialists is not None:
            return {k: deepcopy(v) for k, v in self.specialists.items()}
        return {
            0: _default_specialist_low(),
            1: _default_specialist_med(),
            2: _default_specialist_high(),
        }

    def _fit_global_fallback(self, X, y):
        fb = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        ])
        fb.fit(X, y)
        return fb

    # --- fit ---

    def fit(self, X, y, regime_labels=None):
        """
        Parameters
        ----------
        X : DataFrame de features (incluyendo regime_var como columna).
        y : array/Series de target de regresión.
        regime_labels : array opcional con etiquetas pre-calculadas.
            Si es None, se calculan como terciles de X[regime_var] sobre train.
        """
        X = pd.DataFrame(X).copy()
        y = np.asarray(y)

        # 1. Calcular bordes de régimen sobre train (si no se pasaron labels)
        if regime_labels is None:
            if self.regime_var not in X.columns:
                raise ValueError(
                    f"Columna {self.regime_var} no encontrada en X. "
                    "Pásala explícitamente vía regime_labels o agrégala a X."
                )
            vol = X[self.regime_var].dropna()
            qs = [(i + 1) / self.n_regimes for i in range(self.n_regimes - 1)]
            self.regime_borders_ = list(vol.quantile(qs).values)
            regime_labels = self._assign_regimes(X[self.regime_var].values)
        else:
            self.regime_borders_ = None
            regime_labels = np.asarray(regime_labels)

        self.regime_labels_train_ = regime_labels

        # 2. Entrenar router
        self.router_ = self._resolve_router()
        # quitar la columna regime_var de las features del router para no chivar
        X_router = X.drop(columns=[self.regime_var], errors="ignore")
        self.feature_names_ = list(X_router.columns)
        valid = ~np.isnan(regime_labels.astype(float)) & (regime_labels >= 0)
        self.router_.fit(X_router.iloc[valid].values, regime_labels[valid])

        # 3. Entrenar specialists
        self.specialists_ = {}
        all_specs = self._resolve_specialists()
        for r in range(self.n_regimes):
            mask = (regime_labels == r) & ~np.isnan(y)
            if mask.sum() < self.n_min_per_regime:
                self.specialists_[r] = None  # se reemplaza por fallback
            else:
                spec = all_specs[r]
                spec.fit(X_router.iloc[mask].values, y[mask])
                self.specialists_[r] = spec

        # 4. Fallback global
        self.global_fallback_ = self._fit_global_fallback(
            X_router.iloc[~np.isnan(y)].values, y[~np.isnan(y)]
        )

        return self

    # --- predict ---

    def _assign_regimes(self, values):
        out = np.full(len(values), -1, dtype=int)
        for i, v in enumerate(values):
            if pd.isna(v):
                out[i] = -1
                continue
            r = 0
            for b in self.regime_borders_:
                if v <= b:
                    break
                r += 1
            out[i] = min(r, self.n_regimes - 1)
        return out

    def predict(self, X):
        X = pd.DataFrame(X).copy()
        X_router = X.drop(columns=[self.regime_var], errors="ignore")
        X_router = X_router[self.feature_names_]  # mismas columnas, mismo orden

        # P(régimen | X)
        proba = self.router_.predict_proba(X_router.values)

        if self.mixture == "argmax":
            chosen = np.argmax(proba, axis=1)
            preds = np.empty(len(X))
            for r in range(self.n_regimes):
                mask = chosen == r
                if mask.sum() == 0:
                    continue
                spec = self.specialists_.get(r)
                if spec is None:
                    preds[mask] = self.global_fallback_.predict(
                        X_router.iloc[mask].values
                    )
                else:
                    preds[mask] = spec.predict(X_router.iloc[mask].values)
            return preds

        # Mezcla suave (default)
        preds_per_regime = np.zeros((len(X), self.n_regimes))
        for r in range(self.n_regimes):
            spec = self.specialists_.get(r)
            if spec is None:
                preds_per_regime[:, r] = self.global_fallback_.predict(
                    X_router.values
                )
            else:
                preds_per_regime[:, r] = spec.predict(X_router.values)

        return np.sum(proba * preds_per_regime, axis=1)

    def predict_proba_regime(self, X):
        """Devuelve la matriz de probabilidades del router."""
        X = pd.DataFrame(X).copy()
        X_router = X.drop(columns=[self.regime_var], errors="ignore")
        X_router = X_router[self.feature_names_]
        return self.router_.predict_proba(X_router.values)
