"""
Pruebas estadísticas para evaluación de modelos.
- Diagnósticos de residuos: White, BDS, Ljung-Box, Jarque-Bera, ACF
- Comparación de modelos:
    * Diebold-Mariano (regresión)
    * DeLong real con matrices de influencia (clasificación)
    * Bootstrap CI para cualquier métrica
- Corrección Bonferroni para comparaciones múltiples
"""
import numpy as np
import pandas as pd
from scipy import stats

import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.tsa.stattools import bds
from statsmodels.stats.stattools import jarque_bera


# ============================================================
# 1. Diagnósticos de residuos
# ============================================================

def white_test(residuals, X_subset, max_features: int = 5) -> dict:
    """
    Test de White para heterocedasticidad.
    Usa solo las primeras `max_features` columnas para evitar costo cuadrático.
    """
    try:
        X = np.asarray(X_subset)[:, :max_features]
        X_const = sm.add_constant(X)
        stat, pval, _, _ = het_white(residuals, X_const)
        return {
            "statistic": float(stat),
            "p_value": float(pval),
            "interpretation": (
                "Heterocedasticidad detectada (p<0.05)"
                if pval < 0.05 else "Homocedasticidad (p>=0.05)"
            ),
        }
    except Exception as e:
        return {"error": str(e), "p_value": np.nan, "statistic": np.nan,
                "interpretation": "no calculable"}


def bds_test(residuals, max_dim: int = 3) -> dict:
    """Test BDS de independencia."""
    try:
        bds_stat, bds_pval = bds(np.asarray(residuals), max_dim=max_dim)
        return {
            "statistics": np.asarray(bds_stat).tolist(),
            "p_values": np.asarray(bds_pval).tolist(),
            "p_value_dim2": float(bds_pval[0]),
            "interpretation": (
                "Dependencia no lineal detectada (p<0.05)"
                if bds_pval[0] < 0.05 else "Residuos independientes (p>=0.05)"
            ),
        }
    except Exception as e:
        return {"error": str(e), "p_value_dim2": np.nan,
                "interpretation": "no calculable"}


def ljung_box_test(residuals, lags: int = 10) -> dict:
    """Test de Ljung-Box para autocorrelación de residuos."""
    try:
        result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
        pval = float(result["lb_pvalue"].iloc[0])
        return {
            "statistic": float(result["lb_stat"].iloc[0]),
            "p_value": pval,
            "lags": lags,
            "interpretation": (
                "Autocorrelación detectada (p<0.05)"
                if pval < 0.05 else "Sin autocorrelación significativa (p>=0.05)"
            ),
        }
    except Exception as e:
        return {"error": str(e), "p_value": np.nan,
                "interpretation": "no calculable"}


def jarque_bera_test(residuals) -> dict:
    """Test de Jarque-Bera para normalidad."""
    try:
        stat, pval, skew, kurt = jarque_bera(np.asarray(residuals))
        return {
            "statistic": float(stat),
            "p_value": float(pval),
            "skew": float(skew),
            "kurtosis": float(kurt),
            "interpretation": (
                "Residuos no normales (p<0.05)"
                if pval < 0.05 else "No se rechaza normalidad (p>=0.05)"
            ),
        }
    except Exception as e:
        return {"error": str(e), "p_value": np.nan,
                "interpretation": "no calculable"}


def full_residual_diagnostics(model_name, y_true, y_pred, X_test) -> dict:
    """Aplica los 4 tests de diagnóstico y devuelve un dict resumen."""
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    return {
        "model": model_name,
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals, ddof=1)),
        "white": white_test(residuals, X_test),
        "bds": bds_test(residuals),
        "ljung_box": ljung_box_test(residuals),
        "jarque_bera": jarque_bera_test(residuals),
    }


# ============================================================
# 2. Diebold-Mariano (regresión)
# ============================================================

def diebold_mariano(e1, e2, h: int = 1, loss: str = "mse") -> tuple:
    """
    Test de Diebold-Mariano (1995) para comparar precisión predictiva.

    H0: ambos modelos tienen la misma precisión predictiva
    H1: tienen precisión diferente

    Parameters
    ----------
    e1, e2 : arrays de errores (y_true - y_pred) de cada modelo.
    h : horizonte de predicción (para corrección de varianza).
    loss : "mse" (cuadrática) o "mae" (absoluta).

    Returns
    -------
    (dm_statistic, p_value)
    """
    e1 = np.asarray(e1)
    e2 = np.asarray(e2)

    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError("loss debe ser 'mse' o 'mae'")

    T = len(d)
    mean_d = np.mean(d)

    # Varianza de d con corrección Newey-West (para h>1)
    var_d = np.var(d, ddof=1)
    if h > 1:
        for k in range(1, h):
            cov_k = np.cov(d[:-k], d[k:], ddof=1)[0, 1]
            var_d += 2 * (1 - k / h) * cov_k
    var_d = var_d / T

    if var_d <= 0:
        return 0.0, 1.0

    dm_stat = mean_d / np.sqrt(var_d)
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def diebold_mariano_matrix(predictions: dict, y_true,
                            loss: str = "mse") -> tuple:
    """
    Aplica DM por pares a un dict {modelo: y_pred}.

    Returns
    -------
    p_values : DataFrame simétrico
    diff_rmse : DataFrame con diferencias de RMSE
    """
    names = list(predictions.keys())
    n = len(names)
    p_values = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    diff_rmse = pd.DataFrame(np.zeros((n, n)), index=names, columns=names)

    y_true = np.asarray(y_true)
    rmses = {nm: np.sqrt(np.mean((y_true - np.asarray(yp))**2))
             for nm, yp in predictions.items()}

    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i >= j:
                continue
            e1 = y_true - np.asarray(predictions[m1])
            e2 = y_true - np.asarray(predictions[m2])
            _, pval = diebold_mariano(e1, e2, loss=loss)
            p_values.loc[m1, m2] = pval
            p_values.loc[m2, m1] = pval
            diff_rmse.loc[m1, m2] = rmses[m1] - rmses[m2]
            diff_rmse.loc[m2, m1] = -(rmses[m1] - rmses[m2])

    return p_values, diff_rmse


# ============================================================
# 3. DeLong real (clasificación) — con matrices de influencia
# ============================================================

def _compute_midrank(x):
    """Ranks de x con manejo de empates (mid-rank), basado en Sun & Xu (2014)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def delong_test(y_true, prob_a, prob_b) -> tuple:
    """
    DeLong test (1988) para comparar dos AUCs correlacionados.
    Implementación basada en Sun & Xu (2014).

    Parameters
    ----------
    y_true : array binario {0,1}
    prob_a, prob_b : arrays de probabilidades del clasificador A y B

    Returns
    -------
    (delta_auc, p_value, var_delta)
    """
    y_true = np.asarray(y_true).astype(int)
    prob_a = np.asarray(prob_a)
    prob_b = np.asarray(prob_b)

    pos = y_true == 1
    neg = y_true == 0
    m = pos.sum()
    n = neg.sum()

    if m == 0 or n == 0:
        return 0.0, 1.0, 0.0

    # AUCs estimados
    def _auc_and_v(scores):
        sp = scores[pos]
        sn = scores[neg]
        # Mid-ranks
        all_scores = np.concatenate([sp, sn])
        ranks = _compute_midrank(all_scores)
        Tp = ranks[:m]
        Tn = ranks[m:]
        # AUC
        auc = (np.sum(Tp) - m * (m + 1) / 2) / (m * n)
        # Componentes de influencia
        Tp_only = _compute_midrank(sp)
        Tn_only = _compute_midrank(sn)
        V10 = (Tp - Tp_only) / n
        V01 = 1.0 - (Tn - Tn_only) / m
        return auc, V10, V01

    auc_a, V10_a, V01_a = _auc_and_v(prob_a)
    auc_b, V10_b, V01_b = _auc_and_v(prob_b)

    # Matrices de covarianza
    s10 = np.cov(V10_a, V10_b, ddof=1)
    s01 = np.cov(V01_a, V01_b, ddof=1)
    cov = s10 / m + s01 / n

    delta = auc_a - auc_b
    var_delta = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]

    if var_delta <= 0:
        return float(delta), 1.0, float(var_delta)

    z = delta / np.sqrt(var_delta)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(delta), float(p_value), float(var_delta)


def delong_test_matrix(probs: dict, y_true) -> tuple:
    """Aplica DeLong por pares a un dict {modelo: y_prob}."""
    names = list(probs.keys())
    n = len(names)
    p_values = pd.DataFrame(np.ones((n, n)), index=names, columns=names)
    diff_auc = pd.DataFrame(np.zeros((n, n)), index=names, columns=names)

    for i, m1 in enumerate(names):
        for j, m2 in enumerate(names):
            if i >= j:
                continue
            delta, pval, _ = delong_test(y_true, probs[m1], probs[m2])
            p_values.loc[m1, m2] = pval
            p_values.loc[m2, m1] = pval
            diff_auc.loc[m1, m2] = delta
            diff_auc.loc[m2, m1] = -delta

    return p_values, diff_auc


# ============================================================
# 4. Bootstrap CI
# ============================================================

def bootstrap_metric(y_true, y_pred, metric_fn,
                      n_boot: int = 1000, ci: float = 0.95,
                      seed: int = 42) -> dict:
    """Bootstrap percentile CI para una métrica."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            scores.append(s)
        except Exception:
            continue
    scores = np.asarray(scores)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return {
        "mean": float(np.mean(scores)),
        "median": float(np.median(scores)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "ci_level": ci,
        "n_boot": len(scores),
    }


# ============================================================
# 5. Corrección de Bonferroni
# ============================================================

def bonferroni_correction(p_values, alpha: float = 0.05):
    """
    Aplica corrección de Bonferroni a un array/Series de p-values.
    Devuelve umbral ajustado y máscara de significancia.
    """
    p_values = np.asarray(p_values).flatten()
    p_values = p_values[~np.isnan(p_values)]
    n_comparisons = len(p_values)
    alpha_bonf = alpha / n_comparisons if n_comparisons > 0 else alpha
    significant = p_values < alpha_bonf
    return {
        "alpha_original": alpha,
        "n_comparisons": n_comparisons,
        "alpha_bonferroni": alpha_bonf,
        "n_significant_uncorrected": int(np.sum(p_values < alpha)),
        "n_significant_bonferroni": int(np.sum(significant)),
    }
