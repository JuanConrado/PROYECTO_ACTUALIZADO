# Entrega 3 — Instrucciones de instalación

Este ZIP contiene los **7 notebooks finales** del proyecto + un **fix crítico** para `src/hvrf.py`.

## Qué se entrega

```
INTC-VolForecast-Entrega3/
├── 07_optimizacion_hiperparametros.ipynb
├── 08_residuos_y_diagnosticos.ipynb     ← ejecutado en sandbox ✅
├── 09_comparacion_estadistica.ipynb     ← ejecutado en sandbox ✅
├── 10_interpretabilidad.ipynb
├── 11_modelos_avanzados.ipynb
├── 12_hvrf_modelo_original.ipynb        ← ejecutado en sandbox ✅
├── 13_comparacion_final_y_conclusiones.ipynb  ← ejecutado en sandbox ✅
├── src_fix/
│   └── hvrf.py                           ← ¡APLICAR ESTE FIX PRIMERO!
└── INSTRUCCIONES.md                      ← este archivo
```

## Pasos para instalar (ORDEN CRÍTICO)

### 1. Aplicar el fix al `src/hvrf.py`

**ESTO ES OBLIGATORIO.** Si no aplicas el fix, el notebook 12 (HVRF) va a fallar con:

```
ValueError: operands could not be broadcast together with shapes (1045,3) (1045,2)
```

```powershell
# Reemplazar el src/hvrf.py viejo por el corregido
copy "INTC-VolForecast-Entrega3\src_fix\hvrf.py" "src\hvrf.py"
```

**Qué arregla este fix:** cuando un régimen del HVRF tiene muy pocas muestras en train, el router XGBoost solo ve N-1 clases y `predict_proba` devuelve menos columnas que `n_regimes`. La versión nueva detecta dinámicamente las clases reales y hace el broadcasting correcto. Esto es importante para la ablación con 2 regímenes.

### 2. Copiar los 7 notebooks a `notebooks/`

```powershell
copy "INTC-VolForecast-Entrega3\*.ipynb" "notebooks\"
```

### 3. Verificar que tienes los outputs de Entrega 2

Antes de ejecutar la Entrega 3, asegúrate de tener:

```powershell
ls outputs/predictions/
# Debe mostrar:
#   benchmarks_test_preds.parquet  ← del notebook 03
#   reg_test_preds.parquet         ← del notebook 04
#   clf_test_preds.parquet         ← del notebook 05

ls outputs/metrics/
# Debe mostrar:
#   benchmarks_metrics.json
#   regression_metrics.json
#   classification_metrics.json
#   balancing_metrics.json
```

**Si te falta alguno**, regenéralo re-ejecutando el notebook correspondiente:

```powershell
# Si falta benchmarks_*:
jupyter nbconvert --to notebook --execute notebooks/03_benchmarks_temporales.ipynb --inplace

# Si falta balancing_metrics.json:
jupyter nbconvert --to notebook --execute notebooks/06_balanceo_clases.ipynb --inplace
```

### 4. Ejecutar los notebooks de Entrega 3 en orden

```powershell
# Tiempos estimados en máquina típica:
# 07: ~10-30 min (Optuna 100 trials + DEAP 450 evaluations)
# 08: ~1-2 min
# 09: ~1-2 min
# 10: ~3-5 min (permutation importance + LIME)
# 11: ~5-10 min (LightGBM, CatBoost, MLP, LSTM)
# 12: ~5-10 min (HVRF + 4 ablaciones)
# 13: ~1 min

jupyter nbconvert --to notebook --execute notebooks/07_optimizacion_hiperparametros.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/08_residuos_y_diagnosticos.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/09_comparacion_estadistica.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/10_interpretabilidad.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/11_modelos_avanzados.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/12_hvrf_modelo_original.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/13_comparacion_final_y_conclusiones.ipynb --inplace
```

**Tiempo total estimado: 30-60 minutos** según hardware.

**Alternativa más cómoda:** abre cada notebook en VSCode y dale "Run All" — tarda lo mismo pero ves el progreso visualmente.

## Resultados esperados (validados en mi sandbox con tu dataset)

### Notebook 12 — HVRF
- **HVRF (soft, 3 regímenes):** RMSE ≈ 0.006786
- **HVRF (argmax):** RMSE ≈ 0.006800 (muy similar — la mezcla suave aporta poco)
- **HVRF (2 regímenes):** RMSE ≈ 0.006815 (peor — 3 regímenes captan mejor)
- **HVRF (router uniforme):** RMSE ≈ 0.010328 (MUCHO peor — confirma que el router aporta información)
- **Ridge (referencia):** RMSE ≈ 0.006578
- **Diebold-Mariano HVRF vs Ridge:** p ≈ 0.0122 → Ridge es **estadísticamente superior**

> **Hallazgo honesto:** el HVRF **no supera a Ridge** en este dataset. Las ablaciones validan las decisiones de diseño (la mezcla suave > argmax > 2 regímenes > uniforme), pero la complejidad adicional no compensa frente a un modelo lineal regularizado bien especificado. **Esto es un resultado científico legítimo y se reporta así en el notebook 13.**

## Posibles problemas conocidos

### En notebook 11 (modelos avanzados)
Si te falta alguna librería opcional (`lightgbm`, `catboost`, `tensorflow`):
- El notebook **NO falla**, solo reporta "no disponible" para esa librería.
- Los demás modelos avanzados sí se entrenan.

### En notebook 07 (optimización)
- Optuna y DEAP toman tiempo. **No canceles** durante los 30 min de espera.
- Si Grid Search da timeout en tu máquina, edita la celda y baja la rejilla a 2x2x2.

### En notebook 12 (HVRF)
- Si te sale el error de broadcasting, NO aplicaste el fix de `src/hvrf.py`. Aplícalo y reinicia el kernel.

## Después de ejecutar todo

Tu repo debe quedar con todos estos artefactos:

```
outputs/models/
  └── hvrf_final.joblib
  └── xgb_reg_optimized.joblib
  └── reg_lightgbm.joblib (si LightGBM disponible)
  └── reg_catboost.joblib (si CatBoost disponible)
  └── reg_mlp.joblib

outputs/predictions/
  └── advanced_test_preds.parquet
  └── hvrf_test_preds.parquet

outputs/metrics/
  └── optimization_metrics.json
  └── residual_diagnostics.json
  └── statistical_comparisons.json
  └── interpretation.json
  └── advanced_metrics.json
  └── hvrf_metrics.json
  └── final_master_table.json

outputs/figures/  (si guardaste manualmente con save_fig)
  └── (vacío por defecto, o con figuras que tú agregues)
```
