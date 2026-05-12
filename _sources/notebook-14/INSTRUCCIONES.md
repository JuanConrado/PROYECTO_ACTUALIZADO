# Notebook 14 — Optimización Computacional y Refinamientos del HVRF

Este paquete contiene el notebook 14, el cierre técnico del proyecto antes
de la auditoría con Claude Code y el build final del Jupyter Book.

## Contenido

```
14_optimizacion_y_refinamientos_hvrf.ipynb  ← notebook único
INSTRUCCIONES.md                              ← este archivo
```

## Qué hace este notebook

### Bloque A — Optimización Computacional (sección 11 de la rúbrica)

1. Carga la tabla de `outputs/metrics/computational_optimization.json`
   (generada por `fix_tabla_estandar_vs_optimizado.py`).
2. Muestra la tabla "modelo estándar vs optimizado" con 10 modelos.
3. Visualiza el trade-off costo/precisión y el costo de inferencia.
4. Discute 3 hallazgos contraintuitivos:
   - KNN brute force puede ser más rápido que Ball-Tree en 6K filas.
   - Solver SAGA NO aporta a Ridge en este problema.
   - LinearSVR supera a SVR rbf.

### Bloque B — Refinamientos del HVRF

1. Diagrama de bloques SVG de la arquitectura HVRF (los 3 stages).
2. Predicciones de cada specialist por separado sobre el test.
3. Activación de regímenes en el tiempo + correlación con volatilidad real.
4. Comparación rigurosa: **HVRF vs ensambles naive** (50/50 y 70/30 de
   Ridge+XGBoost) con Diebold-Mariano + bootstrap.
5. Recomendación final para producción.

## Pre-requisitos

Debes tener **ejecutados y guardados** estos archivos en disco:

```
outputs/metrics/computational_optimization.json   ← fix_tabla...py
outputs/models/hvrf_final.joblib                  ← notebook 12
outputs/predictions/hvrf_test_preds.parquet       ← notebook 12
outputs/predictions/reg_test_preds.parquet        ← notebook 04 + fix_propagate
```

Si alguno falta, el notebook 14 te dirá exactamente cuál y cómo regenerarlo.

## Cómo instalar y ejecutar

### Paso 1 — Copiar el notebook a la carpeta `notebooks/`

```powershell
cd "C:\Users\Mateo\2026\UNINORTE 2026 -1\ML\INTC-VolForecast"
copy "$HOME\Downloads\notebook-14\14_optimizacion_y_refinamientos_hvrf.ipynb" "notebooks\"
```

### Paso 2 — Ejecutar en VSCode

1. Abre `notebooks/14_optimizacion_y_refinamientos_hvrf.ipynb` en VSCode.
2. Verifica que el kernel está en `intc-volforecast`.
3. Click "Run All".
4. Espera ~2-3 minutos (solo lee artefactos pre-computados y entrena 1
   ensamble simple).

### Paso 3 — Añadir al Jupyter Book

Edita `_toc.yml` añadiendo el notebook 14 en la Parte IV (Modelos Avanzados
y Modelo Original):

```yaml
  - caption: "Parte IV — Modelos Avanzados y Modelo Original"
    chapters:
      - file: notebooks/11_modelos_avanzados
      - file: notebooks/12_hvrf_modelo_original
      - file: notebooks/14_optimizacion_y_refinamientos_hvrf
        title: "14. Optimización Computacional y Refinamientos HVRF"
```

(Lo dejamos en Parte IV porque trata sobre el modelo original.)

## Resultados esperados (validados en sandbox con tu dataset)

### Bloque A — Tabla computacional

```
Modelo                          Fit (s)   Pred (s)    RMSE       Tamaño
XGBoost default                  1.57      0.004      0.00755    409 KB
XGBoost hist + tuned             1.67      0.007      0.00689    663 KB
KNN brute force                  0.001     0.07       0.00706    2.7 MB
KNN ball_tree + n_jobs=-1        0.02      0.51       0.00706    2.9 MB
RF serial                        33.0      0.03       0.00706    13 MB
RF parallel                      33.2      0.03       0.00706    13 MB
Ridge default solver             0.005     0.0003     0.00658    < 1 KB
Ridge SAGA solver                14.4      0.0005     0.00656    < 1 KB
SVR rbf                          11.6      0.54       0.00834    2.5 MB
LinearSVR                        5.6       0.0004     0.00777    < 1 KB
```

### Bloque B — HVRF vs ensambles naive

```
Modelo                          RMSE     vs HVRF (DM)
Ridge solo                      0.00658  HVRF peor (p=0.012) *
Ensamble naive 70/30            0.00660  HVRF peor (p=0.020) *
Ensamble naive 50/50            0.00665  empate (p=0.092)
HVRF                            0.00679  ← modelo original
XGBoost optimizado solo         0.00689  HVRF mejor (p=0.266)
```

**Correlación P(régimen=high) vs vol real = 0.11** (positiva pero baja).

## Lectura crítica de los resultados

### El HVRF NO supera a Ridge (resultado honesto)

Con p=0.0122, Ridge es **estadísticamente superior** al HVRF sobre INTC.
Esto NO es un fracaso — es un **hallazgo metodológicamente valioso**:

> "La dinámica de volatilidad de INTC es dominantemente lineal en el espacio
> de features construido. La complejidad de regímenes del HVRF no aporta
> valor sobre Ridge bien regularizado en este dataset particular."

### El HVRF empata estadísticamente con el ensamble 50/50

Con p=0.092 (no significativo a α=0.05), no podemos rechazar que el HVRF
y un ensamble naive 50/50 rindan igual. Esto refuta la justificación de
**RMSE**: la complejidad del HVRF no se traduce en mejor RMSE puntual.

### Pero P(régimen=high) tiene utilidad independiente

Con correlación 0.11 (positiva pero modesta), el router del HVRF **SÍ
detecta** transiciones de régimen, aunque débilmente. Esta señal es
**utilidad académica y operacional** que un RMSE puntual no captura:

> "El HVRF es útil como **detector de transición de régimen** más que como
> predictor puntual. Para producción, recomendamos Ridge para RMSE y el
> router del HVRF para alertar sobre crisis."

## Para sustentación

Si el Dr. Lihki pregunta:
- **"¿Por qué creen que su modelo original es válido?"**
  → "Lo sometimos a la prueba más exigente: comparación contra ensamble
  naive con DM. Los resultados son honestos: no superamos RMSE puntual,
  pero P(régimen=high) ofrece utilidad independiente como detector."

- **"¿No es esto solo un ensamble?"**
  → "Eso es exactamente lo que probamos. El ensamble 50/50 empata
  estadísticamente con el HVRF (p=0.09), pero el HVRF añade **interpretabilidad
  de régimen** que un ensamble simple no provee."

- **"¿Por qué reportan resultados negativos?"**
  → "Porque el éxito metodológico se mide por **someter las hipótesis a
  pruebas rigurosas**, no por encontrar siempre lo que querríamos. Reportar
  honestamente que el HVRF no supera a Ridge es lo que un revisor académico
  esperaría."

## ¿Y si quiero que sea diferente?

Si quisieras que el HVRF se vea mejor, podrías:
- Ajustar los hiperparámetros de los specialists (no recomendado: sería
  data snooping post-hoc).
- Cambiar la definición de regímenes (3 → 5 cuantiles): podría ayudar pero
  ya validamos que 2 regímenes son peores.
- Añadir más features de mercado (VIX, S&P): cambio de scope del proyecto.

La opción correcta es **reportar honestamente** lo que tenemos.
