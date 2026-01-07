"""
QGAI Quantum Financial Modeling - TReDS MVP
Explainability Module

This module provides model interpretation and reporting:
- SHAPExplainer: SHAP-based feature importance for default predictions
- Local and global explanations for model interpretability
- Human-readable explanation generation

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from .shap_explainer import (
    SHAPExplainer,
    ExplanationResult,
    GlobalExplanation,
    LocalExplanation,
    FeatureContribution,
    explain_predictions
)


__all__ = [
    # SHAP Explainer
    "SHAPExplainer",
    "ExplanationResult",
    "GlobalExplanation",
    "LocalExplanation",
    "FeatureContribution",
    "explain_predictions",
]
