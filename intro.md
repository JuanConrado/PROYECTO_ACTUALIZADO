# Proyecto Final: Predicción de Volatilidad — INTC

**Machine Learning**

---

👥 **Integrantes**

| Nombre |
|---|
| Juan Camilo Conrado |
| Sergio Cadavid |
| Mateo Chang |

---

---

## Descripción del Proyecto

Este proyecto desarrolla un sistema de predicción de volatilidad futura del stock de **Intel Corporation (INTC)** utilizando técnicas de Machine Learning supervisado para regresión y clasificación.

El dataset utilizado proviene de [Price Volume Data for all US Stocks & ETFs (Kaggle)](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs), con datos históricos desde 1972.

---

## Estructura del Proyecto

| Capítulo | Contenido |
|---|---|
| **1. EDA** | Análisis exploratorio: estadísticos, distribuciones, patrones de velas, QA |
| **2. Datos & Features** | Carga, limpieza, retornos logarítmicos, volatilidad rolling, 52 features |
| **3. Regresión** | Benchmark de modelos, métricas, residuos, tests White y BDS |
| **4. Clasificación** | Benchmark, métricas, matrices de confusión, ROC, balanceo de clases |
| **5. Optimización** | Grid Search, Random Search, Optuna (Bayesiana), DEAP (Genético), optimización computacional |
| **6. Interpretabilidad** | LIME, importancia de características XGBoost, DeLong, Diebold-Mariano, Bootstrap |
| **7. Conclusiones** | Síntesis de resultados y decisiones metodológicas |

---

## Problema

Se predicen **3 horizontes de volatilidad futura** (7, 14 y 21 días) usando shifts sin overlap para evitar data leakage:

$$\text{target\_vol\_7} = \text{vol\_7}.shift(-7)$$

$$\text{target\_vol\_14} = \text{vol\_14}.shift(-14)$$

$$\text{target\_vol\_21} = \text{vol\_21}.shift(-21)$$

Para clasificación, el target binario indica si la volatilidad futura **sube (1) o baja (0)** respecto al valor actual.

---

## Reproducibilidad

```
pip install -r requirements.txt
jupyter-book build .
```

El dataset se puede descargar con:

```
python data/download_dataset.py
```
