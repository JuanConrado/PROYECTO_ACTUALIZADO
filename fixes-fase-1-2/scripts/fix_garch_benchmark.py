"""
fix_garch_benchmark.py
======================

Corrige el bug del benchmark GARCH(1,1) en el notebook 03.

DIAGNÓSTICO DEL BUG:
La implementación actual entrena GARCH(1,1) sobre log_retornos diarios y
toma el forecast de variance a 1045 pasos (= longitud del test). Pero
GARCH converge a la varianza INCONDICIONAL a largo plazo, que para INTC
es ~0.027 (muy por encima del promedio empírico ~0.014 porque GARCH
asume colas más pesadas). Esto genera un sesgo gigante porque el target
es `target_vol_7` (volatilidad realizada a 7 días, ~0.012).

CORRECCIÓN TEÓRICAMENTE CORRECTA:
target_vol_7 = std de log-retornos en ventana de 7 días.
Si los retornos siguen σ_t variable según GARCH(1,1):
  E[std_7_dias] ≈ sqrt(mean(σ_t²)) para t=1..7
Es decir, el RMS de los primeros 7 forecasts de σ del GARCH.

RESULTADO ESPERADO:
- Antes: RMSE = 0.01452, R² = -4.02 (peor benchmark)
- Después: RMSE ≈ 0.00674, R² ≈ -0.08 (competitivo con Ridge)

USO:
    python fix_garch_benchmark.py

Esto sobrescribe outputs/predictions/benchmarks_test_preds.parquet con
la columna GARCH(1,1) corregida, y actualiza benchmarks_metrics.json.
NO modifica el notebook 03 — solo regenera los artefactos.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent  # subir un nivel desde scripts/
sys.path.insert(0, str(ROOT))

from src.io_utils import (load_processed, load_predictions_df,
                           save_predictions_df, save_metrics)


def garch_vol_h_forecast(log_returns, n_steps, h=7):
    """
    Predice volatilidad realizada a h-días (vol_h) usando GARCH(1,1).
    
    Implementación CORRECTA:
    
    El forecast a multi-step de arch_model converge a la varianza incondicional
    de largo plazo, NO a la varianza esperada en el corto plazo. Para evitar
    este sesgo:
    
    1. Pedimos UN forecast de horizonte h (no mucho más largo).
    2. Tomamos el RMS de esos h sigmas como predicción puntual.
    3. Para n_steps muestras del test, usamos ESE MISMO RMS constante
       (porque sin walk-forward, no tenemos información actualizada).
    
    Para una versión MÁS PRECISA, habría que hacer walk-forward (refit GARCH
    en cada paso del test usando información hasta t). Pero eso requiere
    1043 re-fits = ~10 min de cómputo. Esta versión "best-effort sin walk-forward"
    es razonable como benchmark.
    """
    from arch import arch_model
    
    scale = 100.0
    y = pd.Series(log_returns).dropna() * scale
    model = arch_model(y, vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    
    # Forecast SOLO h pasos (no n_steps)
    forecast = model.forecast(horizon=h, reindex=False)
    variance_path_h = forecast.variance.values[-1, :h]  # primeros h pasos
    sigma_daily_h = np.sqrt(variance_path_h) / scale
    
    # RMS de los h sigmas = predicción puntual de vol_h
    sigma_vol_h = np.sqrt(np.mean(sigma_daily_h**2))
    
    # Usar ese RMS constante como benchmark para todo el test
    # (es como un "Naive GARCH": misma predicción cada día porque no hacemos
    #  walk-forward; es coherente con el tratamiento de los otros benchmarks
    #  como Naive y Rolling Mean que también producen constantes en este NB).
    preds = np.full(n_steps, sigma_vol_h)
    
    return preds, model


def main():
    print("=" * 70)
    print("FIX: corrigiendo benchmark GARCH(1,1) del notebook 03")
    print("=" * 70)
    
    # Cargar datos
    train = load_processed("train_reg")
    val = load_processed("val_reg")
    test = load_processed("test_reg")
    hist = pd.concat([train, val]).reset_index(drop=True)
    
    log_ret_hist = hist["log_ret"].dropna()
    y_test = test["target_vol_7"].values
    n_test = len(y_test)
    
    print(f"\nHistórico log_ret: {len(log_ret_hist)} obs")
    print(f"Test target_vol_7: {n_test} obs")
    
    # Predecir con GARCH corregido
    print("\nEntrenando GARCH(1,1) corregido...")
    yp_garch_fixed, model = garch_vol_h_forecast(log_ret_hist, n_test, h=7)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, yp_garch_fixed)))
    mae = float(mean_absolute_error(y_test, yp_garch_fixed))
    r2 = float(r2_score(y_test, yp_garch_fixed))
    
    print(f"\n📊 GARCH(1,1) CORREGIDO:")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   R²:   {r2:.4f}")
    print(f"   Primeros 5 forecasts: {yp_garch_fixed[:5]}")
    
    # Cargar predicciones existentes y reemplazar columna GARCH
    try:
        bench_preds = load_predictions_df("benchmarks_test_preds")
        print(f"\n✅ Cargado benchmarks_test_preds.parquet: {bench_preds.shape}")
    except FileNotFoundError:
        print("\n❌ No existe outputs/predictions/benchmarks_test_preds.parquet")
        sys.exit(1)
    
    # Comparación antes/después
    if "GARCH(1,1)" in bench_preds.columns:
        yp_old = bench_preds["GARCH(1,1)"].values
        rmse_old = float(np.sqrt(mean_squared_error(y_test, yp_old)))
        r2_old = float(r2_score(y_test, yp_old))
        print(f"\n📊 Comparación ANTES vs DESPUÉS:")
        print(f"   ANTES:    RMSE = {rmse_old:.6f}, R² = {r2_old:.4f}")
        print(f"   DESPUÉS:  RMSE = {rmse:.6f}, R² = {r2:.4f}")
        print(f"   Mejora:   {(rmse_old - rmse) / rmse_old * 100:.2f}%")
    
    # Sobrescribir
    bench_preds["GARCH(1,1)"] = yp_garch_fixed
    save_predictions_df(bench_preds, "benchmarks_test_preds")
    print(f"\n✅ benchmarks_test_preds.parquet actualizado")
    
    # Actualizar métricas
    try:
        import json
        from src.config import METRICS_DIR
        with open(METRICS_DIR / "benchmarks_metrics.json") as f:
            bench_metrics = json.load(f)
    except FileNotFoundError:
        bench_metrics = {}
    
    bench_metrics["GARCH(1,1)"] = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "fit_time_seconds": None,  # ya no lo medimos aquí
        "note": "Corregido: usa RMS de 7 sigmas en lugar de forecast a largo plazo (1045 pasos). Ver fix_garch_benchmark.py",
    }
    save_metrics(bench_metrics, "benchmarks_metrics")
    print(f"✅ benchmarks_metrics.json actualizado")
    
    print()
    print("=" * 70)
    print("✅ Fix aplicado. Ahora re-ejecuta:")
    print("   - Notebook 09: DM ahora incluye GARCH corregido")
    print("   - Notebook 13: tabla maestra refleja el ranking real")
    print("=" * 70)


if __name__ == "__main__":
    main()
