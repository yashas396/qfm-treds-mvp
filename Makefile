# =============================================================================
# QGAI Quantum Financial Modeling - TReDS MVP
# Makefile for Common Development Tasks
# =============================================================================

.PHONY: help install install-dev test lint format clean run docs

# Default target
help:
	@echo "QGAI TReDS Fraud Detection MVP - Available Commands"
	@echo "====================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install      Install production dependencies"
	@echo "  make install-dev  Install development dependencies"
	@echo "  make venv         Create virtual environment"
	@echo ""
	@echo "Running:"
	@echo "  make run          Run full pipeline"
	@echo "  make run-train    Run training only"
	@echo "  make run-predict  Run prediction only"
	@echo ""
	@echo "Testing:"
	@echo "  make test         Run all tests"
	@echo "  make test-cov     Run tests with coverage"
	@echo "  make test-fast    Run tests (skip slow)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint         Run linters (flake8, mypy)"
	@echo "  make format       Format code (black, isort)"
	@echo "  make check        Run all quality checks"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs         Build documentation"
	@echo "  make docs-serve   Serve documentation locally"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove generated files"
	@echo "  make clean-all    Remove all artifacts including venv"

# =============================================================================
# Setup Commands
# =============================================================================

venv:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  Windows: .\\venv\\Scripts\\activate"
	@echo "  Linux/Mac: source venv/bin/activate"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e ".[dev,docs]"

# =============================================================================
# Running Commands
# =============================================================================

run:
	python main.py --mode full

run-train:
	python main.py --mode train

run-predict:
	python main.py --mode predict

run-validate:
	python main.py --mode validate

run-verbose:
	python main.py --mode full --verbose

# =============================================================================
# Testing Commands
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

test-quantum:
	pytest tests/ -v -m "quantum"

test-integration:
	pytest tests/ -v -m "integration"

# =============================================================================
# Code Quality Commands
# =============================================================================

lint:
	flake8 src/ tests/ config/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ config/ main.py
	isort src/ tests/ config/ main.py

check: lint
	black --check src/ tests/ config/ main.py
	isort --check-only src/ tests/ config/ main.py

# =============================================================================
# Documentation Commands
# =============================================================================

docs:
	mkdocs build

docs-serve:
	mkdocs serve

# =============================================================================
# Cleanup Commands
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

clean-data:
	rm -f data/generated/*.csv data/generated/*.json data/generated/*.pkl 2>/dev/null || true
	rm -f data/outputs/*.csv data/outputs/*.json data/outputs/*.pkl 2>/dev/null || true

clean-models:
	rm -f models/*.pkl models/*.joblib 2>/dev/null || true

clean-reports:
	rm -f reports/*.html reports/*.pdf 2>/dev/null || true

clean-all: clean clean-data clean-models clean-reports
	rm -rf venv/ .venv/ 2>/dev/null || true
	rm -rf logs/ 2>/dev/null || true

# =============================================================================
# Development Helpers
# =============================================================================

.PHONY: tree
tree:
	@echo "Project Structure:"
	@echo "=================="
	@find . -type f -name "*.py" | grep -v __pycache__ | grep -v venv | sort

.PHONY: stats
stats:
	@echo "Code Statistics:"
	@echo "================"
	@find . -name "*.py" -not -path "./venv/*" -not -name "__pycache__" | xargs wc -l | tail -1

.PHONY: config-check
config-check:
	python -c "from config import get_config; c = get_config(); print('Config OK')"
