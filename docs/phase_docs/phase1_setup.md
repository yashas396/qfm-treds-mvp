# Phase 1: Project Setup & Environment Configuration

**Document Version:** 1.0.0
**Status:** Complete
**Date:** January 7, 2026
**Author:** QGAI Quantum Financial Modeling Team

---

## 1. Overview

### 1.1 Purpose

Phase 1 establishes the foundational project structure, configuration management, and development environment for the Hybrid Classical-Quantum TReDS Invoice Fraud Detection System.

### 1.2 Objectives

| Objective | Description | Status |
|-----------|-------------|--------|
| Project Structure | Create organized directory hierarchy | ✅ Complete |
| Dependencies | Define all required packages | ✅ Complete |
| Configuration | Centralized parameter management | ✅ Complete |
| Constants | Immutable system values | ✅ Complete |
| Utilities | Common helper functions | ✅ Complete |
| Documentation | Initial project documentation | ✅ Complete |

### 1.3 Deliverables

1. ✅ `requirements.txt` - All dependencies with version constraints
2. ✅ `config/config.py` - Main configuration class with dataclasses
3. ✅ `config/constants.py` - System constants and enumerations
4. ✅ `README.md` - Project overview and quickstart guide
5. ✅ `.gitignore` - Version control exclusions
6. ✅ `main.py` - Main entry point
7. ✅ `setup.py` / `pyproject.toml` - Package configuration
8. ✅ `src/utils/` - Logging and helper utilities

---

## 2. Project Structure

### 2.1 Directory Layout

```
qfm_treds_mvp/
├── config/                         # Configuration management
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Main configuration class (dataclasses)
│   └── constants.py               # System constants & enums
│
├── src/                           # Source code
│   ├── __init__.py               # Package root
│   ├── data_generation/          # Phase 2: Synthetic data
│   │   └── __init__.py
│   ├── feature_engineering/      # Phase 3: Feature extraction
│   │   └── __init__.py
│   ├── classical/                # Phase 4: Random Forest
│   │   └── __init__.py
│   ├── quantum/                  # Phase 5: QUBO/Ring detection
│   │   └── __init__.py
│   ├── pipeline/                 # Phase 6: Integration
│   │   └── __init__.py
│   ├── explainability/           # Phase 7: SHAP/Reports
│   │   └── __init__.py
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── logger.py             # Logging configuration
│       └── helpers.py            # Common helper functions
│
├── tests/                        # Test suite
│   └── __init__.py
│
├── data/                         # Data storage
│   ├── generated/               # Synthetic data output
│   │   └── .gitkeep
│   └── outputs/                 # Model predictions
│       └── .gitkeep
│
├── models/                       # Saved model artifacts
│   └── .gitkeep
│
├── reports/                      # Generated reports
│   └── .gitkeep
│
├── docs/                         # Documentation
│   └── phase_docs/              # Phase-wise documentation
│       └── phase1_setup.md      # This document
│
├── notebooks/                    # Jupyter notebooks
│   └── .gitkeep
│
├── main.py                       # Main entry point
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup script
├── pyproject.toml               # Project configuration
├── .gitignore                   # Git ignore rules
└── README.md                    # Project README
```

### 2.2 Module Responsibilities

| Module | Responsibility | Phase |
|--------|---------------|-------|
| `config/` | Configuration management | 1 |
| `src/data_generation/` | Synthetic data creation | 2 |
| `src/feature_engineering/` | Feature extraction | 3 |
| `src/classical/` | Default prediction model | 4 |
| `src/quantum/` | QUBO ring detection | 5 |
| `src/pipeline/` | Hybrid integration | 6 |
| `src/explainability/` | Model interpretation | 7 |
| `tests/` | Unit & integration tests | 8 |
| `docs/` | Documentation | 9 |

---

## 3. Dependencies

### 3.1 Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical computing |
| pandas | >=1.3.0 | Data manipulation |
| scipy | >=1.7.0 | Scientific computing |
| scikit-learn | >=1.0.0 | Classical ML |
| networkx | >=2.6.0 | Graph analysis |
| dimod | >=0.12.0 | QUBO modeling |
| dwave-samplers | >=1.0.0 | Simulated annealing |
| shap | >=0.41.0 | Explainability |

### 3.2 Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | >=3.5.0 | Static plots |
| seaborn | >=0.11.0 | Statistical visualization |
| plotly | >=5.0.0 | Interactive plots |

### 3.3 Development

| Package | Version | Purpose |
|---------|---------|---------|
| pytest | >=7.0.0 | Testing framework |
| pytest-cov | >=3.0.0 | Coverage reporting |
| black | >=23.0.0 | Code formatting |
| isort | >=5.10.0 | Import sorting |
| mypy | >=1.0.0 | Type checking |

### 3.4 Installation Commands

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

---

## 4. Configuration System

### 4.1 Configuration Architecture

```
Config (Main)
├── PathConfig           # File system paths
├── DataGenerationConfig # Synthetic data parameters
├── ClassicalModelConfig # Random Forest settings
├── QUBOConfig           # Quantum/QUBO parameters
├── FeatureConfig        # Feature engineering
├── RiskScoringConfig    # Risk categorization
├── ExplainabilityConfig # SHAP settings
└── LoggingConfig        # Logging configuration
```

### 4.2 Usage Examples

```python
from config import get_config

# Get global configuration
config = get_config()

# Access nested parameters
n_estimators = config.classical.n_estimators  # 100
k_communities = config.quantum.k_communities  # 5
n_buyers = config.data_generation.n_buyers    # 100

# Check MVP targets
target_auc = config.classical.target_auc_roc  # 0.75

# Validate configuration
from config.config import validate_config
errors = validate_config(config)
```

### 4.3 Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `classical.n_estimators` | 100 | Random Forest trees |
| `classical.max_depth` | 10 | Maximum tree depth |
| `quantum.k_communities` | 5 | Communities to detect |
| `quantum.num_reads` | 1000 | Annealing iterations |
| `data_generation.n_buyers` | 100 | Synthetic buyers |
| `data_generation.n_rings` | 3 | Fraud rings to inject |

---

## 5. Constants System

### 5.1 Enumeration Classes

```python
from config.constants import (
    EntityType,      # BUYER, SUPPLIER, DUAL
    CreditRating,    # AAA, AA, A, BBB, BB, B, NR
    RiskCategory,    # CRITICAL, HIGH, MODERATE, LOW
    TargetingTier,   # TIER_1, TIER_2, TIER_3, TIER_4
)
```

### 5.2 Success Criteria Constants

```python
from config.constants import SuccessCriteria

# MVP targets
SuccessCriteria.TARGET_AUC_ROC           # 0.75
SuccessCriteria.TARGET_MODULARITY         # 0.30
SuccessCriteria.TARGET_RING_RECOVERY_RATE # 0.70
```

---

## 6. Utility Functions

### 6.1 Logger

```python
from src.utils.logger import setup_logger, get_logger

# Setup with custom configuration
logger = setup_logger("my_module", level="DEBUG")
logger.info("Processing started")

# Get logger for module
logger = get_logger(__name__)
```

### 6.2 Helper Functions

```python
from src.utils.helpers import (
    ensure_directory,      # Create directory if not exists
    timestamp_filename,    # Generate timestamped filename
    load_json, save_json, # JSON I/O
    load_pickle, save_pickle,  # Pickle I/O
    format_currency,       # INR formatting
    format_percentage,     # Percentage formatting
    Timer,                # Context manager for timing
)

# Example usage
with Timer("Model training") as t:
    model.fit(X, y)
print(t)  # "Model training: 5.23 seconds"
```

---

## 7. Entry Point

### 7.1 Command Line Interface

```bash
# Full pipeline
python main.py

# Train only
python main.py --mode train

# Predict only
python main.py --mode predict

# With verbose output
python main.py --verbose

# Skip ring detection
python main.py --no-rings

# Custom seed
python main.py --seed 123
```

### 7.2 Execution Flow

```
main.py
    │
    ├── Phase 2: Data Generation
    │   └── EntityGenerator, InvoiceGenerator
    │
    ├── Phase 3: Feature Engineering
    │   └── FeatureEngineer, GraphBuilder
    │
    ├── Phase 4: Classical Training
    │   └── DefaultPredictor, ModelEvaluator
    │
    ├── Phase 5: Quantum Detection
    │   └── QUBOBuilder, CommunityDetector, RingScorer
    │
    ├── Phase 6: Integration
    │   └── HybridPipeline, RiskScorer, TargetingEngine
    │
    ├── Phase 7: Explainability
    │   └── ShapExplainer, RingExplainer, ReportGenerator
    │
    └── Phase 8: Validation
        └── Success criteria checks
```

---

## 8. Validation

### 8.1 Structure Verification

```bash
# List project structure
ls -la qfm_treds_mvp/

# Verify configuration loads
python -c "from config import get_config; print(get_config())"

# Run import tests
python -c "import src; print(src.__version__)"
```

### 8.2 Expected Output

```
Configuration loaded successfully!
Version: 1.0.0
Environment: development
Project root: D:\QGAI RND\financialModeling(T&P)\qfm_treds_mvp
Configuration is valid.
```

---

## 9. Phase 1 Checklist

| Item | Status |
|------|--------|
| Directory structure created | ✅ |
| requirements.txt complete | ✅ |
| config/config.py implemented | ✅ |
| config/constants.py implemented | ✅ |
| README.md created | ✅ |
| .gitignore configured | ✅ |
| main.py entry point | ✅ |
| setup.py / pyproject.toml | ✅ |
| Logger utility | ✅ |
| Helper functions | ✅ |
| All __init__.py files | ✅ |
| .gitkeep for empty dirs | ✅ |
| Phase 1 documentation | ✅ |

---

## 10. Next Steps

**Phase 2: Synthetic Data Generation** will implement:

1. `src/data_generation/entity_generator.py`
   - Generate buyers, suppliers, ring members
   - Apply credit rating distributions
   - Create ring entity clusters

2. `src/data_generation/invoice_generator.py`
   - Generate legitimate invoices
   - Inject ring/fraud invoices
   - Apply realistic amount distributions

3. `src/data_generation/data_validator.py`
   - Validate data quality
   - Check referential integrity
   - Verify ground truth labels

---

## 11. References

1. PRD: `prd_hybrid_quantum_treds.md`
2. D-Wave Documentation: https://docs.dwavesys.com
3. scikit-learn Documentation: https://scikit-learn.org
4. SHAP Documentation: https://shap.readthedocs.io

---

*Document Classification: Internal - Confidential*
*QGAI Quantum Financial Modeling Team*
