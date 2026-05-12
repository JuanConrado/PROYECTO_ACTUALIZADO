# Fixes Fase 1 + Fase 2 — Auditoría INTC-VolForecast

Este paquete aplica las correcciones identificadas por la auditoría de Claude
Code para subir la nota de **4.2/5.0 → 4.7-4.8/5.0**.

## 📦 Contenido

```
fixes-fase-1-2/
├── apply_all_fixes.py              ← script automático (Fase 1)
├── scripts/
│   └── fix_garch_benchmark.py      ← corrección del bug GARCH (descubrimiento valioso!)
├── PLANTILLAS_TEXTO.md             ← texto exacto a pegar en archivos (Fase 2)
└── INSTRUCCIONES.md                ← este archivo
```

## 🎯 Cambios que aplica

### Fase 1: automática (5 min, lo hace `apply_all_fixes.py`)

| # | Cambio | Estado |
|---|---|---|
| 1 | Crea carpeta `scripts/` y mueve `fix_*.py` allí | Auto ✅ |
| 2 | Borra carpetas duplicadas `notebook-14/` y `fixes-opcion-a/` | Auto ✅ |
| 3 | Actualiza `.gitignore` para evitar reaparición | Auto ✅ |
| 4 | **Corrige bug del GARCH** (RMSE 0.01452 → 0.00674) | Auto ✅ |

### Fase 2: manual (15-20 min, usando `PLANTILLAS_TEXTO.md`)

| # | Cambio | Tiempo |
|---|---|---|
| 5 | Rellenar 5 TBDs en `README.md` | 3 min |
| 6 | Añadir `exclude_patterns` al `_config.yml` | 2 min |
| 7 | Celda markdown en NB 04 (horizonte único) | 3 min |
| 8 | Celda markdown en NB 05 (justificación F1) | 3 min |
| 9 | Celda markdown en NB 06 (NaN ADASYN) | 3 min |
| 10 | Celda markdown en NB 03 (corrección GARCH) | 2 min |
| 11 | Re-ejecutar última celda NB 07 (crash kernel) | 2 min |
| 12 | Re-ejecutar NB 09 y NB 13 (para incluir GARCH corregido) | 5 min |
| 13 | `jupyter-book build .` | 5 min |
| 14 | `git add + commit + push` | 2 min |

## 🚀 Cómo aplicar (orden estricto)

### Paso 1: Descomprimir y posicionarse

```powershell
cd "C:\Users\Mateo\2026\UNINORTE 2026 -1\ML\INTC-VolForecast"

# Descomprimir el ZIP
Expand-Archive "$HOME\Downloads\INTC-fixes-fase-1-2.zip" -DestinationPath . -Force
```

### Paso 2: Ejecutar el script automático

```powershell
conda activate intc-volforecast

# El script crea scripts/, mueve archivos, borra duplicados, fix GARCH
python fixes-fase-1-2/apply_all_fixes.py
```

**Output esperado:**

```
======================================================================
APLICANDO CORRECCIONES DE LA AUDITORÍA
======================================================================

Paso 1: setup de directorio scripts/
✅ Directorio scripts/ listo
✅ Movido: fix_propagate_optimized.py → scripts/
✅ Movido: fix_tabla_estandar_vs_optimizado.py → scripts/

Paso 2: limpieza de directorios duplicados
✅ Borrado: notebook-14/
✅ Borrado: fixes-opcion-a/

Paso 3: actualizar .gitignore
✅ Añadidas 7 entradas a .gitignore

Paso 4: aplicar fix del GARCH (corrección de bug)
   ...
✅ GARCH CORREGIDO: RMSE 0.014520 → 0.006737 (mejora 53.7%)
```

⚠️ **El script automático también copia `scripts/fix_garch_benchmark.py`
a tu carpeta `scripts/`.** Verifica con `ls scripts/`.

### Paso 3: Aplicar las 5 plantillas de texto manual

Abre `PLANTILLAS_TEXTO.md` en cualquier visor (VSCode o Notepad).

Sigue las 5 plantillas en orden:

1. **PLANTILLA 1:** README.md líneas 92-96 (TBDs)
2. **PLANTILLA 2:** _config.yml (exclude_patterns)
3. **PLANTILLA 3:** NB 04 celda markdown (horizonte único)
4. **PLANTILLA 4:** NB 05 celda markdown (justificación F1)
5. **PLANTILLA 5:** NB 06 celda markdown (NaN ADASYN)
6. **PLANTILLA 6:** NB 03 celda markdown (corrección GARCH)

Cada plantilla te dice **exactamente dónde** pegar el texto y **qué texto pegar**.

### Paso 4: Re-ejecutar última celda del NB 07

Abre `notebooks/07_optimizacion_hiperparametros.ipynb` en VSCode.
Ve a la ÚLTIMA celda de código.
`Shift+Enter` para ejecutarla.
`Ctrl+S` para guardar (esto borra el traceback rojo del crash anterior).

### Paso 5: Re-ejecutar NB 09 y NB 13

Con el GARCH corregido, los notebooks 09 y 13 deben actualizar:
- NB 09: tabla Diebold-Mariano con GARCH corregido
- NB 13: tabla maestra con GARCH compitiendo con Ridge

```powershell
# Abrir cada uno en VSCode y "Run All", o:
jupyter nbconvert --to notebook --execute notebooks/09_comparacion_estadistica.ipynb --inplace
jupyter nbconvert --to notebook --execute notebooks/13_comparacion_final_y_conclusiones.ipynb --inplace
```

### Paso 6: Re-build del Jupyter Book

```powershell
jupyter-book build .
```

Output esperado: `build succeeded, X warnings` donde X < 5 (antes eran 12).

### Paso 7: Commit final y push

```powershell
git status   # ver qué archivos cambiaron
git add .
git commit -m "Aplicar correcciones de auditoría: fix GARCH, limpieza, docs"
git push
```

## 📊 Resultados esperados después de aplicar

### Ranking final (NB 13 tabla maestra)

```
Modelo                          RMSE       Categoría
─────────────────────────────────────────────────────────
EWMA                            0.006481   Benchmark
Ridge                           0.006578   ML Clásico
CatBoost                        0.006620   Avanzado
LSTM                            0.006619   Avanzado
GARCH(1,1) CORREGIDO            0.006737   Benchmark   ← antes era 0.01452
HVRF                            0.006816   Original
Lasso                           0.006766   ML Clásico
XGBoost (optimizado)            0.006893   ML Clásico
...
```

### Conclusiones que mejoran

Antes: "GARCH es el peor benchmark, sospechoso."
Después: "GARCH es competitivo con Ridge, confirma que el modelo econométrico clásico captura la mayor parte de la dinámica de varianza condicional, y los modelos ML ofrecen mejoras marginales."

## 🎤 Para sustentación

Si el Dr. Lihki pregunta sobre el GARCH:

**Pregunta:** "¿Por qué GARCH es el peor benchmark en la versión anterior?"

**Respuesta corta:** "Era un bug: pedíamos forecast a 1045 pasos y GARCH converge a la varianza incondicional. Lo corregimos pidiendo solo el horizonte target (7 días) y tomando el RMS. Ahora compite con Ridge."

**Respuesta larga:** Ver "PLANTILLA 6" del archivo PLANTILLAS_TEXTO.md.

## 🆘 Si algo falla

### Error: "No module named 'src'" al correr fix_garch_benchmark.py

→ Asegúrate de ejecutarlo desde la RAÍZ del proyecto, no desde dentro de scripts/:
```powershell
python scripts/fix_garch_benchmark.py  # ✅ correcto
cd scripts && python fix_garch_benchmark.py  # ❌ incorrecto
```

### Error: el _config.yml no buildea después de añadir exclude_patterns

→ Probable error de indentación. Verifica que cada línea bajo `exclude_patterns:` tenga `      - 'pattern'` (6 espacios + `- ` + ruta entre comillas).

### Error: jupyter-book build da más warnings que antes

→ Probablemente el `exclude_patterns` no quedó bien aplicado. Compara con la PLANTILLA 2.

### El NB 09 o NB 13 no muestra GARCH corregido al re-ejecutar

→ Verifica que `outputs/predictions/benchmarks_test_preds.parquet` se actualizó:
```powershell
Get-Item outputs\predictions\benchmarks_test_preds.parquet | Select LastWriteTime
```
Debe tener la fecha de HOY (después de correr apply_all_fixes.py).

## ⏱️ Tiempo total estimado

| Bloque | Tiempo |
|---|---|
| Script automático (Fase 1) | 2 min |
| Plantillas manuales (Fase 2) | 15-20 min |
| Re-ejecutar NBs 09 y 13 | 5 min |
| Build del Jupyter Book | 5 min |
| Commit + push | 2 min |
| **TOTAL** | **30-35 min** |

## 🎯 Después de aplicar

Tu proyecto debería:
- ✅ Pasar de 4.2 → 4.7-4.8 estimado.
- ✅ GARCH ya no es el peor benchmark.
- ✅ README completo sin TBDs.
- ✅ Build limpio sin warnings.
- ✅ Cada decisión metodológica documentada.

Después de esto, lo que queda es **preparar la sustentación de 10 minutos**
con el material ya generado (NB 14 tiene el diagrama HVRF, ranking maestro,
ablaciones, etc.).
