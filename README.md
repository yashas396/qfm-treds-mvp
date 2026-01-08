# Hybrid Classical-Quantum TReDS Invoice Fraud Detection System

**QGAI Quantum Financial Modeling | Targeting & Prediction**

[![Quantum Ready](https://img.shields.io/badge/Quantum-Ready-blueviolet?style=for-the-badge)](https://github.com/yashas396/qfm-treds-mvp)
[![D-Wave](https://img.shields.io/badge/D--Wave-Compatible-00A4E4?style=for-the-badge)](https://github.com/yashas396/qfm-treds-mvp)
[![Status](https://img.shields.io/badge/Status-MVP%20Complete-success?style=for-the-badge)](https://github.com/yashas396/qfm-treds-mvp)

[![Python](https://img.shields.io/badge/python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sklearn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-green?style=flat-square)](https://shap.readthedocs.io)

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://qfm-treds-mvp-yashas396.streamlit.app/)

**Try the live application:** https://qfm-treds-mvp-yashas396.streamlit.app/

Features available in the demo:
- Risk Overview Dashboard with real-time metrics
- Invoice Queue with risk assessment and PDF export
- Ring Analysis with fraud network visualization
- Decision Log for audit trail

---

## Overview

This MVP implements a **Hybrid Classical-Quantum System** for TReDS (Trade Receivables Discounting System) invoice fraud detection. The system combines:

| Component | Method | Purpose |
|-----------|--------|---------|
| **Classical** | Random Forest | Invoice default prediction |
| **Quantum** | QUBO Optimization | Fraud ring detection via community detection |

### Why Hybrid?

- **Default Prediction**: Supervised learning problem with labeled data - classical ML excels
- **Ring Detection**: NP-hard combinatorial optimization - maps naturally to quantum annealing

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CLASSICAL TRACK                    QUANTUM TRACK                       │
│  ───────────────                    ─────────────                       │
│                                                                         │
│  Invoice Features ──▶ Random     Transaction Graph ──▶ QUBO            │
│                       Forest                           Formulation      │
│        │                │               │                   │           │
│        ▼                ▼               ▼                   ▼           │
│  P(default)        Feature       Modularity          Simulated          │
│                   Importance      Matrix             Annealing          │
│        │                                                    │           │
│        └──────────────────┬─────────────────────────────────┘           │
│                           │                                             │
│                           ▼                                             │
│                   COMPOSITE RISK SCORE                                  │
│                   + Ring Membership                                     │
│                   + Targeting List                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd qfm_treds_mvp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

```bash
# Run complete pipeline
python main.py

# Run with custom configuration
python main.py --config custom_config.yaml
```

### 3. Run Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Project Structure

```
qfm_treds_mvp/
├── config/                     # Configuration management
│   ├── __init__.py
│   ├── config.py              # Main configuration class
│   └── constants.py           # System constants
│
├── src/                       # Source code
│   ├── data_generation/       # Synthetic data generation
│   │   ├── __init__.py
│   │   ├── entity_generator.py
│   │   ├── invoice_generator.py
│   │   └── data_validator.py
│   │
│   ├── feature_engineering/   # Feature extraction
│   │   ├── __init__.py
│   │   ├── invoice_features.py
│   │   ├── entity_features.py
│   │   └── graph_builder.py
│   │
│   ├── classical/             # Classical ML component
│   │   ├── __init__.py
│   │   ├── default_predictor.py
│   │   └── model_evaluator.py
│   │
│   ├── quantum/               # Quantum/QUBO component
│   │   ├── __init__.py
│   │   ├── qubo_builder.py
│   │   ├── community_detector.py
│   │   └── ring_scorer.py
│   │
│   ├── pipeline/              # Integration pipeline
│   │   ├── __init__.py
│   │   ├── hybrid_pipeline.py
│   │   ├── risk_scorer.py
│   │   └── targeting.py
│   │
│   ├── explainability/        # Model interpretation
│   │   ├── __init__.py
│   │   ├── shap_explainer.py
│   │   ├── ring_explainer.py
│   │   └── report_generator.py
│   │
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       ├── logger.py
│       └── helpers.py
│
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_data_generation.py
│   ├── test_classical.py
│   ├── test_quantum.py
│   └── test_pipeline.py
│
├── data/                      # Data storage
│   ├── generated/             # Generated synthetic data
│   └── outputs/               # Model outputs
│
├── models/                    # Saved model artifacts
├── reports/                   # Generated reports
├── docs/                      # Documentation
│   └── phase_docs/            # Phase-wise documentation
│
├── notebooks/                 # Jupyter notebooks
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## MVP Success Criteria

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| Default Model AUC-ROC | ≥ 0.75 | **0.8389** | ✅ PASSED |
| Ring Detection Modularity | ≥ 0.30 | **0.45+** | ✅ PASSED |
| Ring Recovery Rate | ≥ 70% | **85%+** | ✅ PASSED |
| Explainability Coverage | 100% | **100%** | ✅ PASSED |
| Quantum Readiness | Full | **D-Wave Ready** | ✅ PASSED |

---

## Components

### 1. Classical Component: Default Prediction

Random Forest classifier predicting invoice default probability based on:
- Invoice-level features (amount, timing, acceptance delay)
- Buyer-level features (credit rating, payment history)
- Relationship features (trading history, frequency)

### 2. Quantum Component: Ring Detection

QUBO-based community detection for fraud ring identification:
- Transaction graph construction
- Modularity matrix formulation
- Simulated annealing solver (D-Wave ready)
- Ring probability scoring

### 3. Composite Risk Scoring

```
Composite Score = 0.5 × P(default) + 0.5 × P(ring)
```

Risk categories:
- **Critical** (≥0.7): Immediate investigation
- **High** (≥0.5): Priority review
- **Moderate** (≥0.3): Enhanced monitoring
- **Low** (<0.3): Standard process

---

## Configuration

Configuration is managed through `config/config.py`:

```python
from config import get_config

config = get_config()

# Access parameters
print(config.classical.n_estimators)  # 100
print(config.quantum.k_communities)   # 5
```

---

## Sample Output

```json
{
    "invoice_id": "INV003421",
    "buyer_id": "R0103",
    "supplier_id": "R0104",
    "amount": 2500000.00,
    "default_probability": 0.32,
    "ring_probability": 0.78,
    "composite_risk_score": 0.55,
    "risk_category": "High",
    "community_id": 2,
    "recommended_action": "Escalate to fraud investigation team"
}
```

---

## Future Roadmap

1. **D-Wave Integration**: Replace simulated annealing with D-Wave Advantage
2. **QAOA Implementation**: Gate-based quantum alternative
3. **Real-time API**: Production REST API
4. **Dashboard**: Interactive visualization interface
5. **GST Integration**: e-Invoice verification layer

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Classical ML | scikit-learn |
| Quantum/QUBO | dimod, dwave-samplers |
| Graph Analysis | NetworkX |
| Explainability | SHAP |
| Visualization | matplotlib, seaborn, plotly |

---

## References

1. Negre, C. F. A., et al. (2020). "Detecting multiple communities using quantum annealing." *PLOS ONE*
2. D-Wave Systems. "Community Detection with QUBO." *D-Wave Documentation*
3. AWS Quantum Blog (2023). "Hybrid Quantum-Classical Optimization"

---

## Team

**QGAI Quantum Financial Modeling Team**

---

## License

Proprietary - QGAI Internal Use Only

---

*Document Version: 1.0.0 | Last Updated: January 2026*
