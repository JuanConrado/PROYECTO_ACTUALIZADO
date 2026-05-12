"""
fix_propagate_optimized.py
===========================

Script único que aplica todos los fixes necesarios para que los modelos
OPTIMIZADOS (del notebook 07) se propaguen correctamente al resto del
proyecto (notebooks 09, 10, 12, 13).

Qué hace:
1. Verifica que existe outputs/models/xgb_reg_optimized.joblib (output del NB 07).
2. Genera predicciones del modelo XGBoost optimizado sobre el test.
3. Guarda las predicciones en outputs/predictions/reg_test_preds_optimized.parquet.
4. Actualiza reg_test_preds.parquet añadiendo la columna "XGBoost (optimizado)".
5. Imprime las nuevas métricas para verificación.

Esto sustituye la necesidad de re-ejecutar los notebooks 09, 12 y 13 desde cero:
solo basta con re-ejecutar SUS celdas que cargan predicciones (~1 celda por NB).

Cómo usar:
1. Ejecutar el notebook 07 hasta el final (si no lo has hecho).
2. Ejecutar este script: python fix_propagate_optimized.py
3. Re-ejecutar las celdas 1-3 (las que cargan predicciones) en los notebooks
   09, 12 y 13. NO hace falta correr todo el notebook desde cero.

Tiempo total: ~30 segundos.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Setup de paths
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Importar después de añadir al path
from src.io_utils import (load_processed, load_model, save_predictions_df,
                           load_predictions_df, save_metrics)
from src.config import DATA_PROCESSED, MODELS_DIR


def main():
    print("=" * 60)
    print("FIX: propagar XGBoost optimizado a notebooks 09, 12, 13")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. Verificar que el modelo optimizado existe
    # ------------------------------------------------------------
    opt_model_path = MODELS_DIR / "xgb_reg_optimized.joblib"
    if not opt_model_path.exists():
        print(f"\n❌ ERROR: no existe {opt_model_path}")
        print(f"   Debes ejecutar el notebook 07 primero.")
        sys.exit(1)

    pipe_opt = load_model("xgb_reg_optimized")
    print(f"\n✅ Modelo XGBoost optimizado cargado: {opt_model_path}")

    # ------------------------------------------------------------
    # 2. Cargar datos de test
    # ------------------------------------------------------------
    test = load_processed("test_reg")
    with open(DATA_PROCESSED / "feature_columns.json") as f:
        feature_cols = json.load(f)

    X_test = test[feature_cols]
    y_test = test["target_vol_7"].values
    print(f"   Test: {X_test.shape}, target shape: {y_test.shape}")

    # ------------------------------------------------------------
    # 3. Predecir con el modelo optimizado
    # ------------------------------------------------------------
    yp_opt = pipe_opt.predict(X_test)

    rmse_opt = float(np.sqrt(mean_squared_error(y_test, yp_opt)))
    mae_opt = float(mean_absolute_error(y_test, yp_opt))
    r2_opt = float(r2_score(y_test, yp_opt))

    print(f"\n📊 XGBoost OPTIMIZADO sobre test:")
    print(f"   RMSE: {rmse_opt:.6f}")
    print(f"   MAE:  {mae_opt:.6f}")
    print(f"   R²:   {r2_opt:.4f}")

    # ------------------------------------------------------------
    # 4. Cargar predicciones existentes y añadir el modelo optimizado
    # ------------------------------------------------------------
    try:
        df_reg = load_predictions_df("reg_test_preds")
        print(f"\n✅ reg_test_preds.parquet cargado: {df_reg.shape}")
    except FileNotFoundError:
        print("\n❌ ERROR: no existe outputs/predictions/reg_test_preds.parquet")
        print("   Debes ejecutar el notebook 04 primero.")
        sys.exit(1)

    # Si ya estaba la columna, la sobrescribimos
    if "XGBoost (optimizado)" in df_reg.columns:
        print("   ⚠️ Sobrescribiendo columna existente 'XGBoost (optimizado)'")
    df_reg["XGBoost (optimizado)"] = yp_opt
    save_predictions_df(df_reg, "reg_test_preds")
    print(f"✅ reg_test_preds.parquet actualizado con XGBoost (optimizado)")

    # ------------------------------------------------------------
    # 5. Comparar XGBoost default vs optimizado
    # ------------------------------------------------------------
    if "XGBoost" in df_reg.columns:
        yp_default = df_reg["XGBoost"].values
        rmse_default = float(np.sqrt(mean_squared_error(y_test, yp_default)))
        mae_default = float(mean_absolute_error(y_test, yp_default))
        r2_default = float(r2_score(y_test, yp_default))

        mejora_rmse = (rmse_default - rmse_opt) / rmse_default * 100

        print(f"\n📊 Comparación XGBoost default vs optimizado:")
        print(f"   {'Métrica':<10} {'Default':>12} {'Optimizado':>12} {'Mejora':>10}")
        print(f"   {'-'*46}")
        print(f"   {'RMSE':<10} {rmse_default:>12.6f} {rmse_opt:>12.6f} {mejora_rmse:>9.2f}%")
        print(f"   {'MAE':<10} {mae_default:>12.6f} {mae_opt:>12.6f}")
        print(f"   {'R²':<10} {r2_default:>12.4f} {r2_opt:>12.4f}")

        # Guardar comparación
        comparison = {
            "default": {"RMSE": rmse_default, "MAE": mae_default, "R2": r2_default},
            "optimized": {"RMSE": rmse_opt, "MAE": mae_opt, "R2": r2_opt},
            "improvement_pct": mejora_rmse,
        }
        save_metrics(comparison, "xgboost_default_vs_optimized")
        print(f"\n✅ Comparación guardada en outputs/metrics/xgboost_default_vs_optimized.json")
    else:
        print(f"\n⚠️ 'XGBoost' default no está en reg_test_preds.parquet — no se puede comparar")

    print()
    print("=" * 60)
    print("✅ Fix aplicado. Ahora puedes re-ejecutar:")
    print("   - Notebook 09: para incluir XGBoost optimizado en DM")
    print("   - Notebook 12: para comparar HVRF contra XGBoost optimizado")
    print("   - Notebook 13: para incluir en la tabla maestra")
    print("=" * 60)


if __name__ == "__main__":
    main()
