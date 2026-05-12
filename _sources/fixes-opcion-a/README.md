# Fixes — Opción A: refinamientos críticos antes de auditoría

Este paquete contiene **2 scripts de Python** que aplican los refinamientos
críticos al proyecto antes de la auditoría con Claude Code.

## 📦 Contenido

1. `fix_propagate_optimized.py` — Propaga el XGBoost optimizado del notebook 07
   a los notebooks 09, 12 y 13 sin necesidad de re-ejecutarlos completos.

2. `fix_tabla_estandar_vs_optimizado.py` — Genera la tabla "modelo estándar vs
   modelo optimizado" exigida por la rúbrica.

## ⚙️ Cómo aplicar (orden estricto)

### Pre-requisito

Debes tener el notebook **07 ejecutado completo y sin errores**, es decir:

```powershell
ls outputs/models/xgb_reg_optimized.joblib    # debe existir
ls outputs/metrics/optimization_metrics.json  # debe existir
```

Si no existen, ejecuta el notebook 07 primero. **El resto NO va a funcionar
sin esto.**

### Paso 1 — Copiar los scripts a la raíz del proyecto

```powershell
cd "C:\Users\Mateo\2026\UNINORTE 2026 -1\ML\INTC-VolForecast"

# Copiar desde la carpeta donde descomprimiste el ZIP
copy "$HOME\Downloads\fixes-opcion-a\fix_propagate_optimized.py" .
copy "$HOME\Downloads\fixes-opcion-a\fix_tabla_estandar_vs_optimizado.py" .
```

### Paso 2 — Ejecutar el fix de propagación

```powershell
conda activate intc-volforecast
python fix_propagate_optimized.py
```

**Output esperado** (los números serán los de tu máquina):

```
============================================================
FIX: propagar XGBoost optimizado a notebooks 09, 12, 13
============================================================

✅ Modelo XGBoost optimizado cargado: ...xgb_reg_optimized.joblib
   Test: (1043, 59), target shape: (1043,)

📊 XGBoost OPTIMIZADO sobre test:
   RMSE: 0.0068...
   MAE:  0.0055...
   R²:   ...

✅ reg_test_preds.parquet actualizado con XGBoost (optimizado)

📊 Comparación XGBoost default vs optimizado:
   Métrica         Default   Optimizado     Mejora
   ----------------------------------------------
   RMSE           0.007200     0.006893      4.26%
   ...

✅ Comparación guardada en outputs/metrics/xgboost_default_vs_optimized.json
```

### Paso 3 — Ejecutar el fix de tabla computacional

```powershell
python fix_tabla_estandar_vs_optimizado.py
```

Tarda **2-5 minutos** porque entrena 10 modelos. Output:

```
======================================================================
TABLA COMPARATIVA: modelo estándar vs modelo optimizado
======================================================================

[1/5] XGBoost: default vs hist + tuning...
[2/5] KNN: brute force vs Ball Tree...
[3/5] Random Forest: serial vs paralelo...
[4/5] Ridge: solver auto vs SAGA...
[5/5] SVR rbf vs LinearSVR...

==============================================================================
Modelo                          Fit (s)   Pred (s)       RMSE    Size (KB)
------------------------------------------------------------------------------
XGBoost default                  ...       ...        ...        ...
XGBoost hist + tuned             ...       ...        ...        ...
...
==============================================================================

✅ Tabla guardada en outputs/metrics/computational_optimization.json
```

### Paso 4 — Re-ejecutar SOLO las celdas afectadas

Los notebooks 09, 12 y 13 leen `reg_test_preds.parquet` que ahora tiene
la columna nueva "XGBoost (optimizado)". **NO hace falta re-ejecutar
desde cero** — basta con re-ejecutar la **celda de carga** y las que dependen.

**En VSCode:**

1. Abre `notebooks/09_comparacion_estadistica.ipynb`.
2. Click en la primera celda de código → "Run cell" → continúa con las siguientes
   o simplemente click en "Restart and run all" en VSCode si quieres asegurarte.
3. Repite con `notebooks/12_hvrf_modelo_original.ipynb` y `notebooks/13_comparacion_final_y_conclusiones.ipynb`.

Tiempo total de re-ejecución: **3-5 minutos** (estos 3 notebooks son rápidos).

## ✅ Cómo verificar que todo quedó bien

Después de aplicar los fixes, deberías tener estos archivos nuevos:

```
outputs/metrics/
  ├── xgboost_default_vs_optimized.json    ← nuevo (fix #1)
  └── computational_optimization.json      ← nuevo (fix #2)

outputs/predictions/
  └── reg_test_preds.parquet               ← actualizado con columna "XGBoost (optimizado)"
```

Verificación:

```powershell
ls outputs\metrics\xgboost_default_vs_optimized.json
ls outputs\metrics\computational_optimization.json

# Ver el contenido del JSON
Get-Content outputs\metrics\xgboost_default_vs_optimized.json
```

Y al re-ejecutar el notebook 13, en la "Tabla maestra" deberías ver una fila
nueva: **`XGBoost (optimizado) [ML Clásico]`** que esperamos esté en el TOP 5.

## 🎯 Qué mejora esto en tu nota

| Aspecto | Antes | Después |
|---|---|---|
| Modelos optimizados se quedan en el notebook 07 | ❌ Sí | ✅ Se propagan al resto |
| Tabla "estándar vs optimizado" de la rúbrica | ❌ Solo para XGBoost | ✅ Para 5 modelos |
| Defendible en sustentación que la optimización ayuda | ⚠️ Solo dato puntual | ✅ Comparación rigurosa |
| Punto de la rúbrica sobre optimización computacional | ⚠️ Parcial | ✅ Completo |

## 💡 Para la sustentación

Cuando muestres la tabla computacional, **prepárate para defender 2 hallazgos
contraintuitivos**:

### Hallazgo 1: KNN brute force suele ser tan rápido o más que ball_tree

> "Esto es esperado porque nuestro dataset (~6,000 filas, 59 features) está en el
> régimen donde el overhead de construir el árbol KD/Ball excede el ahorro en
> consultas. Los árboles brillan a partir de ~50,000 filas o cuando la
> dimensionalidad es muy baja."

### Hallazgo 2: SAGA en Ridge NO mejora (puede empeorar)

> "SAGA es un solver estocástico diseñado para datasets muy grandes o problemas
> con regularización L1 fuerte. Para Ridge clásico sobre 6,000 muestras, el
> solver de Cholesky (default) es óptimo. Documentar esto demuestra que
> evaluamos críticamente las recomendaciones de la rúbrica."

## ⚠️ Si algo falla

### Error: "no existe xgb_reg_optimized.joblib"
→ Ejecuta el notebook 07 primero hasta el final.

### Error: "no existe reg_test_preds.parquet"
→ Ejecuta el notebook 04 primero.

### Error: ImportError de algún módulo
→ Asegúrate de tener el conda env activo: `conda activate intc-volforecast`.

## 📊 Después de los fixes — opcional

Si quieres ir un paso más adelante, también podemos añadir:
- Walk-forward validation explícita (notebook 04, sección nueva).
- Comparación HVRF vs ensamble naive (notebook 12, sección nueva).
- Diagrama de bloques del HVRF para sustentación.

Pero los 2 fixes de este paquete ya cubren los puntos **críticos** de la
rúbrica. Esos refinamientos son "bonito tenerlos" pero no cambian la nota.
