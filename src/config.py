"""
Configuración global del proyecto.
Una sola fuente de verdad para paths, semillas y constantes.
"""
from pathlib import Path

# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_INTERIM = DATA_DIR / "interim"
DATA_PROCESSED = DATA_DIR / "processed"

OUTPUTS_DIR = ROOT_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
METRICS_DIR = OUTPUTS_DIR / "metrics"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Crear directorios si no existen
for _d in [DATA_INTERIM, DATA_PROCESSED, MODELS_DIR,
           PREDICTIONS_DIR, METRICS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Constantes globales
# ============================================================

# Semilla determinista para todos los modelos
RANDOM_STATE = 42

# Período de filtrado del dataset
START_DATE = "1990-01-01"
END_DATE = "2017-11-10"  # último día disponible

# Particionamiento temporal (proporciones)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15
assert abs(TRAIN_FRAC + VAL_FRAC + TEST_FRAC - 1.0) < 1e-9

# Validación cruzada interna
N_CV_SPLITS = 5

# Bootstrap
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# Horizontes de predicción
HORIZONS = [7, 14, 21]
PRIMARY_HORIZON = 7  # el horizonte principal de evaluación

# Ventanas de volatilidad rolling
VOL_WINDOWS = [7, 14, 21, 28]

# Ticker
TICKER = "INTC"
DATASET_FILE = DATA_RAW / "intc.us.txt"

# ============================================================
# Configuración HVRF
# ============================================================

HVRF_N_REGIMES = 3  # 3 = low / medium / high volatility
HVRF_REGIME_LABELS = {0: "low", 1: "medium", 2: "high"}

# ============================================================
# Estilo de visualización
# ============================================================

PLT_STYLE = "seaborn-v0_8-whitegrid"
FIG_SIZE_TS = (12, 4)         # series temporales
FIG_SIZE_DIST = (10, 6)       # distribuciones
FIG_SIZE_GRID = (16, 10)      # subplots múltiples
DPI_EXPORT = 300

# Paleta consistente
PALETTE = {
    "ridge": "#2196F3",
    "lasso": "#03A9F4",
    "elasticnet": "#00BCD4",
    "knn": "#9C27B0",
    "tree": "#FF9800",
    "rf": "#FF5722",
    "xgb": "#4CAF50",
    "lgbm": "#8BC34A",
    "catboost": "#CDDC39",
    "svm": "#E91E63",
    "svr": "#E91E63",
    "logreg_l1": "#673AB7",
    "logreg_l2": "#3F51B5",
    "nb": "#FFC107",
    "mlp": "#795548",
    "lstm": "#607D8B",
    "naive": "#9E9E9E",
    "rolling": "#9E9E9E",
    "ewma": "#9E9E9E",
    "arima": "#212121",
    "garch": "#000000",
    "hvrf": "#FF1744",
}
