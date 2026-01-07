"""
QGAI Quantum Financial Modeling - TReDS MVP
Feature Engineering Module

This module provides feature extraction and transformation:
- FeatureEngineer: Main feature engineering pipeline
- InvoiceFeatureExtractor: Invoice-level feature extraction
- EntityFeatureExtractor: Buyer/Supplier feature aggregation
- TransactionGraphBuilder: Transaction graph construction for QUBO

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from .feature_engineer import (
    FeatureEngineer,
    FeatureEngineeringResult,
    engineer_features
)

from .invoice_features import (
    InvoiceFeatureExtractor,
    InvoiceFeatureResult,
    extract_invoice_features
)

from .entity_features import (
    EntityFeatureExtractor,
    EntityFeatureResult,
    extract_entity_features
)

from .graph_builder import (
    TransactionGraphBuilder,
    GraphBuildResult,
    build_transaction_graph
)


__all__ = [
    # Main pipeline
    "FeatureEngineer",
    "FeatureEngineeringResult",
    "engineer_features",

    # Invoice features
    "InvoiceFeatureExtractor",
    "InvoiceFeatureResult",
    "extract_invoice_features",

    # Entity features
    "EntityFeatureExtractor",
    "EntityFeatureResult",
    "extract_entity_features",

    # Graph building
    "TransactionGraphBuilder",
    "GraphBuildResult",
    "build_transaction_graph",
]
