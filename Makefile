# ============================================================
# INTC-VolForecast — Makefile
# ============================================================
# Uso:
#   make install   - instala dependencias
#   make test      - corre tests anti-leakage
#   make data      - reconstruye data/processed/
#   make build     - construye Jupyter Book completo
#   make clean     - borra build y outputs
#   make all       - install + test + build
# ============================================================

.PHONY: install test data build clean all help

PYTHON ?= python
PIP ?= pip

help:
	@echo "Comandos disponibles:"
	@echo "  make install   - Instala dependencias del requirements.txt"
	@echo "  make test      - Corre tests con pytest"
	@echo "  make data      - Genera data/processed/ desde data/raw/"
	@echo "  make build     - Construye el Jupyter Book"
	@echo "  make clean     - Limpia _build/ y outputs/"
	@echo "  make all       - install + test + build"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

data:
	$(PYTHON) -c "from src.data_loader import load_intc; from src.features import build_features; from src.targets import build_all_targets; df = load_intc(); df = build_features(df); df = build_all_targets(df); df.to_parquet('data/processed/features.parquet'); print('Dataset procesado guardado:', df.shape)"

build:
	jupyter-book build .

clean:
	rm -rf _build/
	rm -rf data/interim/* data/processed/*
	rm -rf outputs/models/* outputs/predictions/* outputs/metrics/* outputs/figures/*
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true

all: install test build
	@echo "✅ Build completo. Abre _build/html/index.html"
