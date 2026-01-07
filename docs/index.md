# QGAI TReDS Fraud Detection System

## Hybrid Classical-Quantum Invoice Fraud Detection

Welcome to the documentation for the **QGAI Quantum Financial Modeling TReDS MVP**.

---

## Overview

This system implements a hybrid classical-quantum approach for detecting invoice fraud in the Trade Receivables Discounting System (TReDS).

### Key Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Classical | Random Forest | Invoice default prediction |
| Quantum | QUBO Optimization | Fraud ring detection |

---

## Quick Navigation

### Getting Started
- [Installation Guide](getting_started/installation.md)
- [Quick Start](getting_started/quickstart.md)
- [Configuration](getting_started/configuration.md)

### Phase Documentation
- [Phase 1: Project Setup](phase_docs/phase1_setup.md)
- [Phase 2: Data Generation](phase_docs/phase2_data.md)
- [Phase 3: Feature Engineering](phase_docs/phase3_features.md)
- [Phase 4: Classical Model](phase_docs/phase4_classical.md)
- [Phase 5: Quantum Detection](phase_docs/phase5_quantum.md)
- [Phase 6: Integration](phase_docs/phase6_integration.md)
- [Phase 7: Explainability](phase_docs/phase7_explain.md)
- [Phase 8: Testing](phase_docs/phase8_testing.md)
- [Phase 9: Documentation](phase_docs/phase9_docs.md)

### API Reference
- [Configuration API](api/config.md)
- [Data Generation API](api/data_generation.md)
- [Classical Model API](api/classical.md)
- [Quantum Model API](api/quantum.md)
- [Pipeline API](api/pipeline.md)

### Technical Guides
- [QUBO Formulation](guides/qubo.md)
- [Feature Engineering](guides/features.md)
- [Model Explainability](guides/explainability.md)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID SYSTEM                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Invoice Data ──► Feature        Transaction               │
│                    Engineering    Graph                     │
│                         │              │                    │
│                         ▼              ▼                    │
│                    Random        QUBO                       │
│                    Forest        Community                  │
│                         │        Detection                  │
│                         ▼              │                    │
│                    P(default)    Community                  │
│                         │        Assignments                │
│                         │              │                    │
│                         └──────┬───────┘                    │
│                                │                            │
│                                ▼                            │
│                        Composite Risk                       │
│                        Score + Targeting                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## MVP Success Criteria

| Metric | Target |
|--------|--------|
| Default Model AUC-ROC | ≥ 0.75 |
| Ring Detection Modularity | ≥ 0.30 |
| Ring Recovery Rate | ≥ 70% |
| Explainability Coverage | 100% |

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | January 2026 | Initial MVP release |

---

*QGAI Quantum Financial Modeling Team*
