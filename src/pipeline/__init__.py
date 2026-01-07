"""
QGAI Quantum Financial Modeling - TReDS MVP
Pipeline Integration Module

This module provides the hybrid pipeline integration:
- HybridPipeline: Main pipeline combining classical and quantum components
- RiskScorer: Composite risk score calculation
- Convenience functions for end-to-end execution

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from .risk_scorer import (
    RiskScorer,
    RiskScore,
    RiskScoringResult,
    compute_risk_scores
)

from .hybrid_pipeline import (
    HybridPipeline,
    HybridPipelineResult,
    PipelineStage,
    run_hybrid_pipeline
)


__all__ = [
    # Risk Scoring
    "RiskScorer",
    "RiskScore",
    "RiskScoringResult",
    "compute_risk_scores",

    # Hybrid Pipeline
    "HybridPipeline",
    "HybridPipelineResult",
    "PipelineStage",
    "run_hybrid_pipeline",
]
