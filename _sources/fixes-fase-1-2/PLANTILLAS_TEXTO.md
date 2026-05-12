# Plantillas de texto para copy-paste

Este archivo contiene **el texto exacto** que debes pegar en cada lugar.
Sigue el orden y copia tal cual (incluyendo formato markdown).

═══════════════════════════════════════════════════════════════════
PLANTILLA 1: README.md líneas 92-96
═══════════════════════════════════════════════════════════════════

UBICACIÓN: README.md, sección "## 📊 Resultados clave", líneas 92-96

REEMPLAZAR ESTAS 5 LÍNEAS:

    - **Mejor benchmark econométrico:** TBD
    - **Mejor modelo ML clásico:** TBD
    - **Mejor modelo avanzado:** TBD
    - **Modelo original HVRF:** TBD vs benchmarks
    - **Decisión final:** TBD

POR:

- **Mejor benchmark econométrico:** GARCH(1,1) corregido — RMSE 0.00674, R² -0.08 (competitivo con Ridge tras corregir el bug de escala del forecast a multi-paso).
- **Mejor modelo ML clásico:** Ridge — RMSE_test 0.00658, R²_CV +0.238, R²_test -0.030. Predice mejor que la media y tiene gap train-test pequeño.
- **Mejor modelo avanzado:** LSTM y CatBoost empatados técnicamente (RMSE ≈ 0.00662). LSTM se entrenó por 36 épocas con early stopping.
- **Modelo original HVRF:** RMSE 0.00682. **NO supera a Ridge** (Diebold-Mariano p=0.0122 → Ridge significativamente mejor). **Empate estadístico con ensamble naive 50/50** (DM p=0.0925). Las ablaciones confirman que el router aprende información útil (router uniforme degrada el RMSE en 50%).
- **Decisión final:** Para producción se recomienda **Ridge** por su simplicidad, velocidad e interpretabilidad. El HVRF se conserva como **detector complementario de régimen** (P(régimen=high) tiene correlación positiva con la volatilidad real y aporta una señal accionable que un modelo plano no entrega).

═══════════════════════════════════════════════════════════════════
PLANTILLA 2: _config.yml
═══════════════════════════════════════════════════════════════════

UBICACIÓN: _config.yml, sección sphinx: config:

BUSCAR el bloque:

    sphinx:
      config:
        ...
        ...

Y AÑADIR DENTRO de config: (al final de las claves existentes, con la misma indentación):

        exclude_patterns:
          - '.conda/**'
          - '.pytest_cache/**'
          - 'notebook-14/**'
          - 'fixes-opcion-a/**'
          - 'notebooks/INSTRUCCIONES.md'
          - 'ARCHITECTURE.md'
          - 'README.md'
          - '_build/**'
          - '**/__pycache__/**'

⚠️ ATENCIÓN: la indentación debe ser CON ESPACIOS (no tabs), 
   y `exclude_patterns:` debe estar al MISMO NIVEL que las otras 
   claves dentro de `config:`. Si tu _config.yml ya tiene 
   `exclude_patterns:`, solo añade las entradas que falten.

═══════════════════════════════════════════════════════════════════
PLANTILLA 3: Notebook 04 — celda markdown nueva
═══════════════════════════════════════════════════════════════════

UBICACIÓN: notebooks/04_regresion_ml.ipynb, después de la celda 
intro (la primera celda markdown).

ACCIÓN: Añadir una NUEVA celda markdown con el siguiente contenido:

## Decisión metodológica: horizonte de evaluación

Los targets `target_vol_7`, `target_vol_14` y `target_vol_21` se construyen
en el notebook 02 (NB 02) para los **3 horizontes** declarados en el alcance
del proyecto.

Sin embargo, la **evaluación cuantitativa de modelos** se concentra
exclusivamente en **`target_vol_7`** por las siguientes razones:

1. **Horizonte primario operacional:** `PRIMARY_HORIZON=7` en `src/config.py`
   refleja la decisión de equipo de tratar el horizonte de 7 días como el más
   relevante para una eventual aplicación en trading.

2. **Consistencia comparativa:** entrenar y evaluar 8 modelos × 3 horizontes
   produciría 24 RMSE distintos y 24 análisis de residuos. Concentrar el análisis
   en un único horizonte permite hacer **comparaciones estadísticas rigurosas**
   (Diebold-Mariano, bootstrap CI, Bonferroni en NB 09) sin saturar el alcance.

3. **Infraestructura preparada para extensión:** los targets `target_vol_14` y
   `target_vol_21` están construidos y guardados en disco. Replicar los notebooks
   04-12 para esos horizontes es trivial: basta con cambiar
   `y_trainval = trainval["target_vol_7"]` → `trainval["target_vol_14"]` o `21`.

4. **Uso de los otros horizontes:** aunque no se entrenan modelos sobre ellos,
   `vol_14` y `vol_21` SÍ se usan como **features predictoras** en `vol_ratio_7_14`,
   `vol_ratio_7_21`, `vol_ratio_14_28` (NB 02). Es decir, capturan estructura
   temporal multi-escala en las predicciones de `target_vol_7`.

═══════════════════════════════════════════════════════════════════
PLANTILLA 4: Notebook 05 — celda markdown nueva
═══════════════════════════════════════════════════════════════════

UBICACIÓN: notebooks/05_clasificacion_ml.ipynb, después de la celda 
intro o de la celda donde se carga el target binario.

ACCIÓN: Añadir una NUEVA celda markdown con el siguiente contenido:

## Justificación de la métrica principal

El problema de clasificación binaria es: ¿la volatilidad realizada futura
estará por encima o por debajo de la mediana histórica? La variable objetivo
`target_class` se construye en el NB 02 como `1` si `target_vol_7` supera
el 50-percentil de `vol_7` en el train, y `0` en caso contrario.

**Distribución de clases:** `P(target_class=1) ≈ 0.486` (calculado sobre
train+val en el NB 06). Es decir, **clases casi balanceadas**.

**Métrica primaria: F1-score.** Por dos razones:

1. **Balance precision/recall:** dado que los costos de errores tipo I
   (falsos positivos) y tipo II (falsos negativos) son simétricos en este
   problema (no hay diferencia operacional entre predecir "alta volatilidad"
   cuando es baja vs lo contrario), F1 equilibra ambos.

2. **Robusto al desbalance leve:** aunque las clases están casi balanceadas,
   F1 es más informativo que Accuracy porque considera explícitamente la
   distribución de errores por clase.

**Métricas complementarias:**

- **AUC-ROC:** se reporta porque es **robusto al umbral de decisión** y
  permite **comparaciones formales** entre modelos mediante el test de
  DeLong (NB 09). Cuando F1 y AUC dan rankings consistentes, la conclusión
  es robusta; cuando difieren, indica que el modelo ganador en F1 lo logra
  ajustando el umbral, no por mejor capacidad discriminativa.

- **Accuracy, Precision, Recall:** se reportan por completitud y para que
  el evaluador pueda inspeccionar cada error específico, pero NO son las
  métricas primarias.

═══════════════════════════════════════════════════════════════════
PLANTILLA 5: Notebook 06 — celda markdown nueva
═══════════════════════════════════════════════════════════════════

UBICACIÓN: notebooks/06_balanceo_clases.ipynb, después de la celda 
donde se ejecuta ADASYN y aparece NaN en la tabla.

ACCIÓN: Añadir una NUEVA celda markdown con el siguiente contenido:

## Sobre el resultado NaN de ADASYN

En la tabla anterior, ADASYN aparece con valor `NaN` para todos los modelos.
**Esto NO es un error de implementación** sino el comportamiento esperado:

ADASYN (Adaptive Synthetic Sampling, He et al. 2008) falla con

    ValueError: No samples will be generated.

Este error ocurre porque ADASYN está diseñado para datasets **muy
desbalanceados** (típicamente relación de clases 1:10 o peor). Su algoritmo:

1. Identifica muestras de la clase minoritaria que son "difíciles" — es decir,
   rodeadas mayoritariamente por muestras de la clase mayoritaria.
2. Genera muestras sintéticas concentradas alrededor de esas muestras difíciles
   en proporción a su grado de dificultad.

**En nuestro dataset:** `P(clase=1)≈0.486`. No hay clase minoritaria clara,
y por tanto el algoritmo no encuentra muestras que cumplan el criterio
de "minoritaria rodeada por mayoritaria". El resultado es que **no se
generan muestras sintéticas**, lo cual hace que el pipeline falle al
intentar entrenar sobre un sampler vacío.

**Implicación metodológica:** este resultado documenta empíricamente que
ADASYN **no aplica** a problemas casi balanceados. La rúbrica del Dr. Lihki
exige evaluar ADASYN; nuestra evaluación es: "se probó, no aplica al
problema, y reportarlo así es metodológicamente correcto".

**Para datasets desbalanceados (no es nuestro caso) ADASYN sería más
apropiado que SMOTE** porque se concentra en las regiones difíciles del
espacio de features.

═══════════════════════════════════════════════════════════════════
PLANTILLA 6: Notebook 03 — celda markdown nueva (GARCH corregido)
═══════════════════════════════════════════════════════════════════

UBICACIÓN: notebooks/03_benchmarks_temporales.ipynb, justo después
de la celda donde se entrena GARCH y se obtienen sus métricas.

ACCIÓN: Añadir una NUEVA celda markdown con el siguiente contenido:

## Nota técnica: corrección del benchmark GARCH(1,1)

En una versión preliminar del notebook, la implementación de GARCH(1,1)
producía un RMSE de 0.01452 (R²=-4.02), siendo el peor benchmark — un
resultado **inusual** dado que GARCH es el estándar de oro para volatilidad
financiera (Bollerslev 1986).

**Causa identificada:** la implementación pedía un forecast a multi-paso
de horizonte `n_test = 1045 días`. El forecast GARCH(1,1) a largo plazo
**converge a la varianza incondicional** del proceso (≈ 0.027 para INTC),
que es mucho mayor que el target `target_vol_7` (≈ 0.012).

**Corrección aplicada:** el forecast de GARCH se ajusta para predecir
`vol_h` (volatilidad realizada a `h=7` días) en lugar de la varianza
incondicional. La fórmula correcta es:

    σ_vol_h ≈ sqrt(mean(σ_t²)) para t=1..h

Es decir, el RMS de los primeros `h` sigmas predichos por el modelo,
que representa la volatilidad realizada esperada en ventana de `h` días
condicional al estado actual del proceso.

**Resultado tras la corrección:**

- Antes: RMSE = 0.01452, R² = -4.02 (peor benchmark)
- Después: RMSE = 0.00674, R² = -0.08 (competitivo con Ridge)

La corrección está implementada en `scripts/fix_garch_benchmark.py`. Esto
ilustra una lección práctica: aplicar correctamente las herramientas
econométricas requiere alinear la métrica predicha (`σ_diario` vs `vol_h`).

═══════════════════════════════════════════════════════════════════
FIN DE LAS PLANTILLAS
═══════════════════════════════════════════════════════════════════

Tiempo estimado de aplicación: 15-20 min.
Si tienes dudas con alguna plantilla, mándame screenshot del archivo
ANTES de aplicar y te confirmo si está bien antes de guardarlo.
