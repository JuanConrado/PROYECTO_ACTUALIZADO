# Arquitectura y decisiones de diseño — INTC-VolForecast v2

Este documento es la **fuente de verdad** sobre por qué el proyecto está estructurado como está. Cualquier cambio futuro debe respetar (o cambiar conscientemente) los principios aquí descritos.

---

## 1. Principios no negociables

### 1.1 Tiempo sagrado
- Ningún `train_test_split(shuffle=True)`.
- Ningún `KFold` simple.
- Validación cruzada exclusivamente con `TimeSeriesSplit` o walk-forward.
- Tests automáticos verifican que `train.date.max() < test.date.min()`.

### 1.2 Pipeline o no existe
- Cualquier paso ajustable sobre datos (imputer, scaler, balanceo, selección de features) vive **dentro** de un `sklearn.pipeline.Pipeline` o `imblearn.pipeline.Pipeline`.
- Nunca se ajusta un scaler sobre el dataset completo antes del split.

### 1.3 Notebooks independientes
- Cada notebook lee de disco lo que necesita, escribe a disco lo que produce.
- No hay variables compartidas entre kernels.
- Si quieres correr el notebook 09 sin haber corrido el 04, basta con que existan los archivos en `outputs/predictions/`.

### 1.4 Reproducibilidad determinista
- Una sola fuente de verdad para `RANDOM_STATE` (en `src/config.py`).
- Versiones de librerías congeladas en `requirements.txt`.
- Una receta `make all` reproduce todo desde cero.

### 1.5 Lo que se afirma se mide
- "El modelo A es mejor que B" requiere prueba estadística + IC bootstrap.
- Corrección de Bonferroni para comparaciones múltiples.

---

## 2. Decisiones de diseño con justificación

### 2.1 ¿Por qué split 70/15/15 y no 75/25?

El proyecto v1 usaba 75% train / 25% test. Pasamos a **70% train / 15% validation / 15% test** porque:

1. La rúbrica del Dr. Lihki recomienda evaluación rigurosa con conjunto de validación separado.
2. Permite usar el set de validation para selección de modelos (early stopping, comparación de variantes) **sin contaminar el test final**.
3. Para series financieras de ~7,000 observaciones, 15% (≈1,000 días, ~4 años) es suficiente para evaluación robusta.

### 2.2 ¿Por qué shift sin overlap?

`target_vol_7 = vol_7.shift(-7)` (no `shift(-1)`).

Si usáramos `shift(-1)`:
- La ventana del target abarca de `t+1` a `t+7`.
- La ventana de `vol_7` (feature actual) abarca de `t-6` a `t`.
- Comparten 0 días en `t+1`, pero la **proximidad induce correlación espuria**.

Más grave: usar `shift(-1)` con ventana de 7 días genera **6 días de overlap entre targets consecutivos**, inflando R² hasta valores irreales (~0.71 en v1). El shift de 7 elimina este efecto.

### 2.3 ¿Por qué features causales?

Toda feature en fecha `t` debe poder calcularse usando solo datos de fechas `<= t`. El test `tests/test_features.py` lo verifica automáticamente. Esto incluye:

- Lags: solo `shift(positivo)`.
- Rolling: solo ventana hacia atrás.
- Calendarios: solo basados en `t`.
- Estadísticos: nunca calculados sobre el dataset completo y luego "aplicados".

### 2.4 ¿Por qué Pipeline manual dentro de Optuna/DEAP?

`GridSearchCV` y `RandomizedSearchCV` aceptan un `Pipeline` como estimador y respetan automáticamente la encapsulación en cada fold. Pero **Optuna y DEAP usan funciones objetivo manuales**, donde es fácil olvidar el preprocesamiento y meter leakage por accidente.

Solución: dentro de cada `objective(trial)` y cada `eval_individual(ind)`, se construye explícitamente un `Pipeline` y se ajusta dentro del loop de `TimeSeriesSplit`. Nunca se transforma el dataset completo y luego se itera.

### 2.5 ¿Por qué DeLong real y no bootstrap de AUC?

El test de DeLong (1988) usa la teoría de matrices de influencia para estimar la varianza del AUC asintóticamente. Es **el estándar académico** y es lo que el Dr. Lihki enseña en la sección 9.9 de sus notas.

El proyecto v1 implementaba un "DeLong" que en realidad era bootstrap. Renombrarlo no es suficiente: si el profesor pregunta "¿implementaron DeLong?" y la respuesta honesta es "no, hicimos bootstrap", queda mal. Por eso `src/stats_tests.py` incluye una implementación correcta basada en matrices de influencia.

### 2.6 ¿Por qué GARCH como benchmark?

GARCH(1,1) es el **estándar de oro** para predicción de volatilidad financiera desde Bollerslev (1986). Cualquier modelo de ML que prediga volatilidad **debe** ser comparado contra GARCH para ser defendible.

Si nuestro mejor ML no le gana a GARCH, eso no es un fracaso — es un hallazgo legítimo: las señales no-lineales no aportan más que la dinámica de varianza condicional ya capturada por GARCH. Lo reportamos honestamente.

### 2.7 ¿Por qué tres regímenes en HVRF y no dos o cinco?

- **Dos regímenes** (calma/crisis): no captura el período de transición, donde está la mayor incertidumbre.
- **Cinco regímenes**: en train (~5,000 días) implica ~1,000 días por régimen — insuficiente para entrenar un modelo robusto por categoría, especialmente para los regímenes extremos.
- **Tres regímenes** (low/med/high): balance óptimo entre granularidad y muestras por régimen (~1,700 días por categoría en train).

Los límites se calculan como **terciles de `vol_21` sobre el train**, y se aplican fijos al test. Esto evita leakage temporal en la asignación de regímenes.

### 2.8 ¿Por qué mezcla suave y no argmax?

```
ŷ = Σ P(régimen | X) · ŷ_régimen
```

vs.

```
ŷ = ŷ_argmax(P(régimen | X))
```

La mezcla suave:
1. Es **diferenciable**, lo que evita saltos discontinuos en días de transición de régimen.
2. **Captura la incertidumbre del router**: si el router asigna probabilidades 0.45/0.35/0.20, la predicción mezcla las tres en lugar de descartar dos.
3. Tiene **menor varianza** que argmax en validación.

---

## 3. Trazabilidad de archivos

```
data/raw/intc.us.txt
   ├── 01_eda.ipynb (lectura)
   └── data/interim/intc_clean.parquet (escrita)
        └── 02_features_and_targets.ipynb
             └── data/processed/{features, train_*, test_*}.parquet
                  ├── 03_benchmarks_temporales.ipynb
                  ├── 04_regresion_ml.ipynb
                  │    └── outputs/predictions/reg_test_preds.parquet
                  │    └── outputs/models/{model_name}.joblib
                  ├── 05_clasificacion_ml.ipynb
                  │    └── outputs/predictions/clf_test_probs.parquet
                  ├── 06_balanceo_clases.ipynb
                  ├── 07_optimizacion_hiperparametros.ipynb
                  │    └── outputs/models/{xgb,ridge}_optuna.joblib
                  ├── 08_residuos_y_diagnosticos.ipynb
                  ├── 09_comparacion_estadistica.ipynb
                  ├── 10_interpretabilidad.ipynb
                  ├── 11_modelos_avanzados.ipynb
                  ├── 12_hvrf_modelo_original.ipynb
                  │    └── outputs/models/hvrf_final.joblib
                  └── 13_comparacion_final_y_conclusiones.ipynb
                       └── outputs/metrics/all_metrics.json
```

---

## 4. Convenciones de código

- **Nombres de variables**: snake_case en español o inglés (consistente por archivo).
- **Random state**: siempre `RANDOM_STATE = 42` (importado de `src/config.py`).
- **Fechas**: `pd.Timestamp` o `datetime.date`, nunca strings.
- **Paths**: `pathlib.Path`, nunca strings con barras.
- **DataFrames**: index reseteado tras cualquier filter (`.reset_index(drop=True)`).

---

## 5. Estilo de visualización

- Paleta unificada en `src/viz.py`.
- Estilo `seaborn-v0_8-whitegrid` por defecto.
- Tamaño estándar de figura: `(12, 4)` para series temporales, `(10, 6)` para distribuciones.
- Todos los plots con `plt.tight_layout()` antes de `plt.show()`.
- Exportar versión PNG a `outputs/figures/` cuando sea para sustentación.

---

## 6. Política de error y debugging

- Si un test falla → el `Makefile` aborta el build. No hay book sin tests pasando.
- Si un modelo no converge → se reporta el error en el notebook (no se oculta), se discute en interpretación.
- Si una librería opcional (LSTM, CatBoost) no está disponible → se documenta en el notebook 11 como limitación, se continúa sin ella.

---

## 7. Tipo y trazabilidad de outputs

| Tipo de output | Carpeta | Formato | Quién lo lee |
|---|---|---|---|
| Modelo entrenado | `outputs/models/` | `.joblib` | NB siguientes, sustentación |
| Predicciones por modelo | `outputs/predictions/` | `.parquet` | NB 09, 13 |
| Métricas por modelo | `outputs/metrics/` | `.json` | NB 13 |
| Figura individual | `outputs/figures/` | `.png` (300 dpi) | Sustentación |
| Dataset procesado | `data/processed/` | `.parquet` | NB siguientes |

Todos los nombres de archivo siguen el patrón `{descripcion_clara}.{ext}` (sin espacios, sin caracteres especiales).
