# INTC-VolForecast

**Predicción de volatilidad realizada de Intel Corporation (INTC)
mediante Machine Learning supervisado y benchmarks econométricos**

---

👥 **Equipo**

| Nombre |
|---|
| Juan Camilo Conrado |
| Sergio Cadavid |
| Mateo Chang |

🎓 **Curso:** Machine Learning · Pregrado en Ciencia de Datos · Universidad del Norte
👨‍🏫 **Docente:** Dr. Lihki Rubio

---

## Resumen ejecutivo

Este libro presenta un sistema reproducible para predecir la **volatilidad realizada futura** del stock de **Intel Corporation (INTC)** en los horizontes de 7, 14 y 21 días.

El proyecto integra cuatro capas metodológicas:

1. **Benchmarks econométricos clásicos** (Naive, Rolling Mean, EWMA, ARIMA, GARCH).
2. **Modelos de Machine Learning supervisado** vistos en clase (KNN, Ridge, Lasso, Logistic Regression L1/L2, Naive Bayes, Decision Tree, Random Forest, SVM/SVR, XGBoost).
3. **Modelos avanzados** (LightGBM, CatBoost, MLP, LSTM).
4. Un **modelo original propuesto**: el **Hybrid Volatility Regime Forecaster (HVRF)**.

Todas las comparaciones entre modelos se validan estadísticamente con pruebas formales (Diebold-Mariano para regresión, DeLong para AUC en clasificación, bootstrap para intervalos de confianza), aplicando corrección de Bonferroni para múltiples comparaciones.

---

## Datos

| Fuente | [Price Volume Data for all US Stocks & ETFs (Kaggle)](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) |
|---|---|
| Ticker | INTC (Intel Corporation) |
| Período usado | 1990-01-01 a 2017-11-10 |
| Frecuencia | Diaria |
| Variables originales | OHLCV |
| Observaciones efectivas | ~7,000 días |

Se filtra el período pre-1990 porque los precios extremadamente bajos de INTC en los años 70 y 80 generan retornos logarítmicos artefactuales tras los splits accionarios.

---

## Definición operativa del problema

### Targets de regresión (sin overlap)

$$
\text{target\_vol\_7}_t = \text{vol\_7}_{t+7}, \quad \text{con} \quad \text{vol\_7}_t = \mathrm{Std}(\log r_{t-6:t})
$$

Análogamente para horizontes de 14 y 21 días. El shift sin overlap es **crítico** para evitar que el target comparta información con la feature `vol_7` actual, lo cual inflaría artificialmente el R².

### Target de clasificación

$$
\text{target\_class}_t = \mathbb{1}\{\text{target\_vol\_7}_t > \text{vol\_7}_t\}
$$

Pregunta económica: *¿el régimen de volatilidad va a aumentar?* Útil para gestión de riesgo (decidir reducir tamaño de posición, ajustar stops, comprar protección).

---

## Particionamiento temporal

| Bloque | Proporción | Aprox. años |
|---|---|---|
| Train | 70% | 1990 – 2009 |
| Validation | 15% | 2009 – 2013 |
| Test | 15% | 2013 – 2017 |

**Validación cruzada interna:** `TimeSeriesSplit` con 5 folds expandidos sobre train+validation.

---

## Reproducibilidad

```bash
git clone https://github.com/JuanConrado/PROYECTO_ACTUALIZADO.git
cd PROYECTO_ACTUALIZADO
git checkout v2
make install
make test
make build
```

Para una descripción completa de las decisiones metodológicas, ver el archivo `ARCHITECTURE.md` en la raíz del repositorio.

---

## Estructura del libro

El libro se organiza en cinco partes que reflejan el flujo de un proyecto de Machine Learning aplicado a series financieras:

- **Parte I — Datos y Exploración**: EDA, ingeniería de features, definición de targets.
- **Parte II — Benchmarks y Modelos Base**: comparación contra benchmarks econométricos y aplicación de los 8 modelos de ML clásicos para regresión y clasificación.
- **Parte III — Optimización y Diagnósticos**: tuning de hiperparámetros (4 métodos), análisis de residuos, comparación estadística rigurosa, interpretabilidad.
- **Parte IV — Modelos Avanzados y Original**: LightGBM, CatBoost, MLP, LSTM y la propuesta original HVRF.
- **Parte V — Síntesis**: tabla comparativa final, ranking, conclusiones y limitaciones honestas.
