"""
QGAI Quantum Financial Modeling - TReDS MVP
Quantum/QUBO Module

This module provides QUBO-based optimization for ring detection:
- QUBOFormulator: Constructs QUBO formulation for modularity maximization
- RingDetector: Solves QUBO using simulated annealing and scores communities

The architecture is designed for easy swap-in of D-Wave quantum hardware.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from .qubo_formulator import (
    QUBOFormulator,
    QUBOFormulationResult,
    formulate_qubo
)

from .ring_detector import (
    RingDetector,
    RingDetectionResult,
    Community,
    detect_rings
)


__all__ = [
    # QUBO Formulation
    "QUBOFormulator",
    "QUBOFormulationResult",
    "formulate_qubo",

    # Ring Detection
    "RingDetector",
    "RingDetectionResult",
    "Community",
    "detect_rings",
]
