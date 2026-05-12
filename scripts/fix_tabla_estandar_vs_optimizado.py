"""
fix_tabla_estandar_vs_optimizado.py
====================================

Genera la tabla comparativa "modelo estándar vs modelo optimizado" exigida
por la rúbrica del Dr. Lihki.

Compara para cada modelo donde tiene sentido:
  - Tiempo de entrenamiento
  - Tiempo de predicción
  - RMSE en test
  - Uso de memoria del modelo persistido

Modelos comparados:
  - XGBoost: default (hist tree_method, sin tuning) vs optimizado (params del NB 07)
  - KNN: default (algorithm='auto') vs optimizado (algorithm='ball_tree' explícito)
  - Random Forest: default (n_estimators=200) vs optimizado (n_estimators=200, n_jobs=-1)
  - Ridge: default (solver='auto') vs optimizado (solver='saga' explícito)
  - LinearSVR vs SVR rbf: comparación de costos para SVM

Esta tabla cubre la sección "11. Optimización computacional" de la rúbrica.

Cómo usar: python fix_tabla_estandar_vs_optimizado.py

Salida: outputs/metrics/computational_optimization.json + tabla en pantalla.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.io_utils import load_processed, save_metrics
from src.config import DATA_PROCESSED, RANDOM_STATE


def benchmark_model(name, build_fn, X_tr, y_tr, X_te, y_te):
    """Mide tiempo de fit, predict y RMSE para un modelo."""
    model = build_fn()

    t0 = time.time()
    model.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    t0 = time.time()
    yp = model.predict(X_te)
    predict_time = time.time() - t0

    rmse = float(np.sqrt(mean_squared_error(y_te, yp)))
    n_params = _count_params(model)

    return {
        "name": name,
        "fit_time": round(fit_time, 3),
        "predict_time": round(predict_time, 4),
        "rmse": round(rmse, 6),
        "model_size_est": n_params,
    }


def _count_params(model):
    """Estimación cruda del 'tamaño' del modelo."""
    try:
        import pickle
        return len(pickle.dumps(model))
    except Exception:
        return -1


def main():
    print("=" * 70)
    print("TABLA COMPARATIVA: modelo estándar vs modelo optimizado")
    print("=" * 70)

    train = load_processed("train_reg")
    val = load_processed("val_reg")
    test = load_processed("test_reg")
    trainval = pd.concat([train, val]).reset_index(drop=True)

    with open(DATA_PROCESSED / "feature_columns.json") as f:
        feature_cols = json.load(f)

    X_tr = trainval[feature_cols]
    y_tr = trainval["target_vol_7"].values
    X_te = test[feature_cols]
    y_te = test["target_vol_7"].values

    # Imputar para los modelos que no manejan NaN
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    X_te_imp = imp.transform(X_te)

    results = []

    # ============================================================
    # 1. XGBoost: default vs hist + tuning
    # ============================================================
    print("\n[1/5] XGBoost: default vs hist + tuning...")
    from xgboost import XGBRegressor

    def xgb_default():
        return XGBRegressor(random_state=RANDOM_STATE, verbosity=0)

    def xgb_optimized():
        return XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            random_state=RANDOM_STATE, verbosity=0,
            tree_method="hist",  # optimización clave
        )

    r1 = benchmark_model("XGBoost default", xgb_default, X_tr_imp, y_tr, X_te_imp, y_te)
    r2 = benchmark_model("XGBoost hist + tuned", xgb_optimized, X_tr_imp, y_tr, X_te_imp, y_te)
    results.extend([r1, r2])
    print(f"  default:  fit={r1['fit_time']:.2f}s  rmse={r1['rmse']:.6f}")
    print(f"  tuned:    fit={r2['fit_time']:.2f}s  rmse={r2['rmse']:.6f}")

    # ============================================================
    # 2. KNN: brute force vs Ball Tree
    # ============================================================
    print("\n[2/5] KNN: brute force vs Ball Tree...")
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr_imp)
    X_te_sc = sc.transform(X_te_imp)

    def knn_default():
        return KNeighborsRegressor(n_neighbors=11, algorithm="brute")

    def knn_optimized():
        return KNeighborsRegressor(n_neighbors=11, algorithm="ball_tree", n_jobs=-1)

    r3 = benchmark_model("KNN brute force", knn_default, X_tr_sc, y_tr, X_te_sc, y_te)
    r4 = benchmark_model("KNN ball_tree + n_jobs=-1", knn_optimized, X_tr_sc, y_tr, X_te_sc, y_te)
    results.extend([r3, r4])
    print(f"  brute:    fit={r3['fit_time']:.4f}s  predict={r3['predict_time']:.4f}s  rmse={r3['rmse']:.6f}")
    print(f"  tree:     fit={r4['fit_time']:.4f}s  predict={r4['predict_time']:.4f}s  rmse={r4['rmse']:.6f}")

    # ============================================================
    # 3. Random Forest: sin n_jobs vs paralelo
    # ============================================================
    print("\n[3/5] Random Forest: serial vs paralelo...")
    from sklearn.ensemble import RandomForestRegressor

    def rf_serial():
        return RandomForestRegressor(n_estimators=200, max_depth=10,
                                       random_state=RANDOM_STATE, n_jobs=1)

    def rf_parallel():
        return RandomForestRegressor(n_estimators=200, max_depth=10,
                                       random_state=RANDOM_STATE, n_jobs=-1)

    r5 = benchmark_model("RF serial (n_jobs=1)", rf_serial, X_tr_imp, y_tr, X_te_imp, y_te)
    r6 = benchmark_model("RF parallel (n_jobs=-1)", rf_parallel, X_tr_imp, y_tr, X_te_imp, y_te)
    results.extend([r5, r6])
    print(f"  serial:   fit={r5['fit_time']:.2f}s  rmse={r5['rmse']:.6f}")
    print(f"  parallel: fit={r6['fit_time']:.2f}s  rmse={r6['rmse']:.6f}")

    # ============================================================
    # 4. Ridge: solver default vs SAGA
    # ============================================================
    print("\n[4/5] Ridge: solver auto vs SAGA...")
    from sklearn.linear_model import Ridge

    def ridge_default():
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)

    def ridge_saga():
        return Ridge(alpha=1.0, solver="saga", random_state=RANDOM_STATE,
                      max_iter=10000)

    r7 = benchmark_model("Ridge default solver", ridge_default, X_tr_sc, y_tr, X_te_sc, y_te)
    r8 = benchmark_model("Ridge SAGA solver", ridge_saga, X_tr_sc, y_tr, X_te_sc, y_te)
    results.extend([r7, r8])
    print(f"  default:  fit={r7['fit_time']:.4f}s  rmse={r7['rmse']:.6f}")
    print(f"  saga:     fit={r8['fit_time']:.4f}s  rmse={r8['rmse']:.6f}")

    # ============================================================
    # 5. SVR rbf vs LinearSVR
    # ============================================================
    print("\n[5/5] SVR rbf vs LinearSVR...")
    from sklearn.svm import SVR, LinearSVR

    def svr_rbf():
        return SVR(kernel="rbf", C=1.0, epsilon=0.001)

    def linear_svr():
        return LinearSVR(C=1.0, epsilon=0.001, max_iter=5000,
                          random_state=RANDOM_STATE)

    r9 = benchmark_model("SVR rbf", svr_rbf, X_tr_sc, y_tr, X_te_sc, y_te)
    r10 = benchmark_model("LinearSVR", linear_svr, X_tr_sc, y_tr, X_te_sc, y_te)
    results.extend([r9, r10])
    print(f"  rbf:      fit={r9['fit_time']:.2f}s  rmse={r9['rmse']:.6f}")
    print(f"  linear:   fit={r10['fit_time']:.4f}s  rmse={r10['rmse']:.6f}")

    # ============================================================
    # Tabla resumen
    # ============================================================
    print()
    print("=" * 78)
    print(f"{'Modelo':<28} {'Fit (s)':>10} {'Pred (s)':>10} {'RMSE':>10} {'Size (KB)':>12}")
    print("-" * 78)
    for r in results:
        size_kb = r["model_size_est"] / 1024
        print(f"{r['name']:<28} {r['fit_time']:>10.4f} {r['predict_time']:>10.4f} "
              f"{r['rmse']:>10.6f} {size_kb:>12.1f}")
    print("=" * 78)

    # Guardar
    save_metrics({"benchmarks": results}, "computational_optimization")
    print("\n✅ Tabla guardada en outputs/metrics/computational_optimization.json")


if __name__ == "__main__":
    main()
