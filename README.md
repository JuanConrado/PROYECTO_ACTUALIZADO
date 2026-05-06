# INTC-VolForecast

**Predicción de volatilidad realizada de Intel Corporation (INTC) mediante Machine Learning supervisado y benchmarks econométricos**

> Proyecto académico — Curso de Machine Learning, Pregrado en Ciencia de Datos, Universidad del Norte.
> Docente: Dr. Lihki Rubio.

---

## 👥 Autores

| Nombre | Rol |
|---|---|
| Juan Camilo Conrado | Data scientist |
| Sergio Cadavid | Data scientist |
| Mateo Chang | Data scientist |

---

## 🎯 Objetivo

Construir un sistema reproducible que prediga la **volatilidad realizada futura** del stock INTC en horizontes de 7, 14 y 21 días, comparando:

1. **Benchmarks econométricos** clásicos (Naive, Rolling Mean, EWMA, ARIMA, GARCH).
2. **Modelos de Machine Learning** vistos en clase (KNN, Ridge, Lasso, Logistic Regression L1/L2, Naive Bayes, Decision Tree, Random Forest, SVM/SVR, XGBoost).
3. **Modelos avanzados** (LightGBM, CatBoost, MLP, LSTM).
4. Un **modelo original propuesto** llamado **HVRF — Hybrid Volatility Regime Forecaster**, que combina clasificación de régimen + regresores especializados + ensamble suave.

Todas las comparaciones se validan estadísticamente con **Diebold-Mariano**, **DeLong** y **bootstrap CI**, con corrección **Bonferroni** para comparaciones múltiples.

---

## 🗂️ Estructura del repositorio

```
INTC-VolForecast/
├── data/
│   ├── raw/                  # Dataset original (commitido)
│   ├── interim/              # Datos limpios (gitignored)
│   └── processed/            # Datos con features (gitignored)
├── src/                      # Código importable
├── notebooks/                # 14 notebooks numerados
├── outputs/
│   ├── models/               # Modelos persistidos
│   ├── predictions/          # Predicciones por modelo
│   ├── metrics/              # Métricas en JSON
│   └── figures/              # Gráficos exportados
├── tests/                    # Tests anti-leakage
├── _config.yml               # Jupyter Book
├── _toc.yml                  # Tabla de contenidos
└── requirements.txt          # Versiones congeladas
```

---

## 🚀 Reproducir el proyecto

### Opción 1: con `make` (recomendado)

```bash
git clone https://github.com/JuanConrado/PROYECTO_ACTUALIZADO.git
cd PROYECTO_ACTUALIZADO
git checkout v2

# 1. Crear entorno
make install

# 2. Verificar tests anti-leakage
make test

# 3. Construir el Jupyter Book completo
make build

# 4. Ver el book renderizado
open _build/html/index.html
```

### Opción 2: paso a paso

```bash
pip install -r requirements.txt
pytest tests/
jupyter-book build .
```

---

## 📊 Resultados clave

> **Nota:** los números finales se completan tras ejecutar todos los notebooks. Esta sección se actualiza al final.

- **Mejor benchmark econométrico:** TBD
- **Mejor modelo ML clásico:** TBD
- **Mejor modelo avanzado:** TBD
- **Modelo original HVRF:** TBD vs benchmarks
- **Decisión final:** TBD

---

## 📚 Notebooks en orden

| # | Notebook | Contenido |
|---|---|---|
| 00 | `00_setup` | Verificación de entorno y tests |
| 01 | `01_eda` | Análisis exploratorio enfocado en INTC |
| 02 | `02_features_and_targets` | 55 features temporales + 4 targets |
| 03 | `03_benchmarks_temporales` | Naive, Rolling, EWMA, ARIMA, GARCH |
| 04 | `04_regresion_ml` | 8 modelos de regresión |
| 05 | `05_clasificacion_ml` | 8 modelos de clasificación |
| 06 | `06_balanceo_clases` | SMOTE, ADASYN, class_weight |
| 07 | `07_optimizacion_hiperparametros` | Grid, Random, Optuna, DEAP |
| 08 | `08_residuos_y_diagnosticos` | White, BDS, Ljung-Box, ACF, JB |
| 09 | `09_comparacion_estadistica` | DM, DeLong real, Bootstrap, Bonferroni |
| 10 | `10_interpretabilidad` | Feature importance, LIME |
| 11 | `11_modelos_avanzados` | LightGBM, CatBoost, MLP, LSTM |
| 12 | `12_hvrf_modelo_original` | HVRF — diseño completo y ablaciones |
| 13 | `13_comparacion_final_y_conclusiones` | Tabla maestra, ranking, conclusiones |

---

## ⚖️ Licencia

MIT License. Ver [LICENSE](LICENSE).

---

## 📞 Contacto

Para preguntas técnicas sobre el proyecto, ver el archivo [ARCHITECTURE.md](ARCHITECTURE.md) que documenta cada decisión metodológica.
