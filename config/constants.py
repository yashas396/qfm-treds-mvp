"""
QGAI Quantum Financial Modeling - TReDS MVP
System Constants

This module defines immutable constants used throughout the system.
These values should not be changed during runtime.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from enum import Enum, auto
from typing import Dict, List, Tuple


# =============================================================================
# ENTITY TYPE CONSTANTS
# =============================================================================

class EntityType(Enum):
    """Entity types in TReDS ecosystem."""
    BUYER = "buyer"
    SUPPLIER = "supplier"
    DUAL = "dual"  # Can act as both buyer and supplier (ring entity)


class CreditRating(Enum):
    """Credit rating categories."""
    AAA = "AAA"
    AA = "AA"
    A = "A"
    BBB = "BBB"
    BB = "BB"
    B = "B"
    NR = "NR"  # Not Rated


class IndustryType(Enum):
    """Industry sector categories."""
    MANUFACTURING = "Manufacturing"
    IT = "IT"
    PHARMA = "Pharma"
    AUTO = "Auto"
    FMCG = "FMCG"


class RiskCategory(Enum):
    """Risk categorization levels."""
    CRITICAL = "Critical"
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"


class TargetingTier(Enum):
    """Investigation priority tiers."""
    TIER_1 = "TIER_1"  # Immediate Investigation
    TIER_2 = "TIER_2"  # Ring Investigation
    TIER_3 = "TIER_3"  # Credit Review
    TIER_4 = "TIER_4"  # Standard Process


# =============================================================================
# MODEL CONSTANTS
# =============================================================================

class ModelConstants:
    """Constants for ML models."""

    # Random Forest defaults
    RF_N_ESTIMATORS = 100
    RF_MAX_DEPTH = 10
    RF_MIN_SAMPLES_SPLIT = 10
    RF_MIN_SAMPLES_LEAF = 5

    # Cross-validation
    CV_FOLDS = 5

    # Feature count
    TOTAL_FEATURES = 17

    # Model file names
    DEFAULT_MODEL_FILENAME = "default_predictor.pkl"
    RING_MODEL_FILENAME = "ring_detector.pkl"
    PIPELINE_FILENAME = "hybrid_pipeline.pkl"


# =============================================================================
# QUBO CONSTANTS
# =============================================================================

class QUBOConstants:
    """Constants for QUBO optimization."""

    # Default parameters
    DEFAULT_K_COMMUNITIES = 5
    DEFAULT_PENALTY_WEIGHT = 1.0
    DEFAULT_NUM_READS = 1000
    DEFAULT_NUM_SWEEPS = 1000

    # Beta range for simulated annealing
    BETA_MIN = 0.1
    BETA_MAX = 3.0

    # Ring detection thresholds
    MIN_COMMUNITY_SIZE = 3
    MAX_COMMUNITY_SIZE = 15
    DEFAULT_RING_THRESHOLD = 0.5

    # Variable naming pattern
    VAR_NAME_PATTERN = "x_{node}_{community}"

    # Solver types
    SOLVER_SIMULATED_ANNEALING = "simulated_annealing"
    SOLVER_DWAVE_HYBRID = "dwave_hybrid"
    SOLVER_EXACT = "exact"


# =============================================================================
# DATA GENERATION CONSTANTS
# =============================================================================

class DataGenerationConstants:
    """Constants for synthetic data generation."""

    # Entity ID prefixes
    BUYER_PREFIX = "B"
    SUPPLIER_PREFIX = "S"
    RING_PREFIX = "R"

    # Invoice ID prefix
    INVOICE_PREFIX = "INV"

    # Ring ID prefix
    RING_ID_PREFIX = "RING_"

    # Default counts (MVP scale)
    DEFAULT_N_BUYERS = 100
    DEFAULT_N_SUPPLIERS = 200
    DEFAULT_N_RINGS = 3
    DEFAULT_RING_SIZES = [5, 7, 8]
    DEFAULT_N_LEGITIMATE = 5000
    DEFAULT_N_RING_INVOICES = 500

    # Amount parameters (in INR)
    MIN_INVOICE_AMOUNT = 10000
    MAX_INVOICE_AMOUNT = 100000000  # 10 Cr

    # Turnover parameters (in Cr)
    BUYER_TURNOVER_MEAN = 6.0
    BUYER_TURNOVER_STD = 1.0
    SUPPLIER_TURNOVER_MEAN = 2.0
    SUPPLIER_TURNOVER_STD = 1.5
    RING_TURNOVER_MEAN = 3.0
    RING_TURNOVER_STD = 0.3

    # Date ranges
    DEFAULT_START_DATE = "2024-01-01"
    DEFAULT_END_DATE = "2024-12-31"
    RING_START_DATE = "2024-06-01"

    # Payment terms (days)
    PAYMENT_TERMS = [30, 45, 60, 90]


# =============================================================================
# FEATURE CONSTANTS
# =============================================================================

class FeatureConstants:
    """Constants for feature engineering."""

    # Feature names (invoice-level)
    FEATURE_AMOUNT_LOG = "amount_log"
    FEATURE_DAYS_TO_DUE = "days_to_due"
    FEATURE_ACCEPTANCE_DELAY = "acceptance_delay_days"
    FEATURE_AMOUNT_ZSCORE = "amount_zscore_buyer"
    FEATURE_IS_ROUND_AMOUNT = "is_round_amount"
    FEATURE_IS_MONTH_END = "is_month_end"

    # Feature names (buyer-level)
    FEATURE_BUYER_AVG_AMOUNT = "buyer_avg_invoice_amount"
    FEATURE_BUYER_INVOICE_COUNT = "buyer_invoice_count"
    FEATURE_BUYER_DEFAULT_RATE = "buyer_default_rate"
    FEATURE_BUYER_CREDIT_ENCODED = "buyer_credit_rating_encoded"

    # Feature names (relationship-level)
    FEATURE_REL_AGE_DAYS = "relationship_age_days"
    FEATURE_REL_COUNT = "relationship_count"
    FEATURE_IS_NEW_REL = "is_new_relationship"

    # All features list
    ALL_FEATURES = [
        "amount_log",
        "days_to_due",
        "acceptance_delay_days",
        "amount_zscore_buyer",
        "is_round_amount",
        "is_month_end",
        "buyer_avg_amount",
        "buyer_invoice_count",
        "buyer_default_rate",
        "buyer_credit_rating_encoded",
        "relationship_age_days",
        "relationship_count",
        "is_new_relationship",
    ]

    # Thresholds
    NEW_RELATIONSHIP_DAYS = 30
    MONTH_END_DAY = 25
    ROUND_AMOUNT_MODULO = 10000

    # Rolling window
    ROLLING_WINDOW_DAYS = 30


# =============================================================================
# RISK THRESHOLDS
# =============================================================================

class RiskThresholds:
    """Thresholds for risk categorization."""

    # Composite risk thresholds
    CRITICAL = 0.7
    HIGH = 0.5
    MODERATE = 0.3

    # Targeting matrix thresholds
    HIGH_DEFAULT_PROB = 0.5
    HIGH_RING_PROB = 0.3

    # Alert thresholds
    IMMEDIATE_ALERT = 0.8
    REVIEW_REQUIRED = 0.6
    MONITOR = 0.4


# =============================================================================
# OUTPUT SCHEMA CONSTANTS
# =============================================================================

class OutputSchema:
    """Constants for output data schemas."""

    # Invoice prediction columns
    INVOICE_OUTPUT_COLUMNS = [
        "invoice_id",
        "buyer_id",
        "supplier_id",
        "amount",
        "default_probability",
        "ring_probability",
        "composite_risk_score",
        "risk_category",
        "community_id",
    ]

    # Entity risk columns
    ENTITY_OUTPUT_COLUMNS = [
        "entity_id",
        "entity_type",
        "community_id",
        "ring_probability",
        "total_invoices",
        "total_amount",
    ]

    # Ring detection columns
    RING_OUTPUT_COLUMNS = [
        "community_id",
        "ring_probability",
        "member_count",
        "members",
        "total_internal_amount",
        "density",
        "reciprocity",
    ]

    # Explanation columns
    EXPLANATION_COLUMNS = [
        "feature",
        "value",
        "shap_value",
        "direction",
        "importance_rank",
    ]


# =============================================================================
# API CONSTANTS
# =============================================================================

class APIConstants:
    """Constants for API responses (future use)."""

    # Response status codes
    STATUS_SUCCESS = "success"
    STATUS_ERROR = "error"
    STATUS_WARNING = "warning"

    # Pagination
    DEFAULT_PAGE_SIZE = 100
    MAX_PAGE_SIZE = 1000

    # Rate limiting
    REQUESTS_PER_MINUTE = 60


# =============================================================================
# VISUALIZATION CONSTANTS
# =============================================================================

class VisualizationConstants:
    """Constants for visualizations."""

    # Color palette for risk categories
    RISK_COLORS = {
        "Critical": "#FF0000",  # Red
        "High": "#FF8C00",      # Dark Orange
        "Moderate": "#FFD700",  # Gold
        "Low": "#32CD32",       # Lime Green
    }

    # Community colors (for graph visualization)
    COMMUNITY_COLORS = [
        "#1f77b4",  # Blue
        "#ff7f0e",  # Orange
        "#2ca02c",  # Green
        "#d62728",  # Red
        "#9467bd",  # Purple
        "#8c564b",  # Brown
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
        "#bcbd22",  # Yellow-green
        "#17becf",  # Cyan
    ]

    # Figure sizes
    FIGURE_SIZE_SMALL = (8, 6)
    FIGURE_SIZE_MEDIUM = (12, 8)
    FIGURE_SIZE_LARGE = (16, 10)

    # Font sizes
    TITLE_FONT_SIZE = 14
    LABEL_FONT_SIZE = 12
    TICK_FONT_SIZE = 10


# =============================================================================
# SUCCESS CRITERIA CONSTANTS (MVP)
# =============================================================================

class SuccessCriteria:
    """MVP success criteria thresholds."""

    # Classical model
    TARGET_AUC_ROC = 0.75
    TARGET_AVERAGE_PRECISION = 0.30
    TARGET_RECALL_AT_10_FPR = 0.50
    TARGET_F1_SCORE = 0.40

    # QUBO ring detection
    TARGET_MODULARITY = 0.30
    TARGET_RING_RECOVERY_RATE = 0.70

    # Explainability
    TARGET_EXPLAINABILITY_COVERAGE = 1.00  # 100%

    # System
    TARGET_QUANTUM_READINESS = True


if __name__ == "__main__":
    # Print all constants for verification
    print("=== Model Constants ===")
    print(f"RF N_ESTIMATORS: {ModelConstants.RF_N_ESTIMATORS}")

    print("\n=== QUBO Constants ===")
    print(f"K_COMMUNITIES: {QUBOConstants.DEFAULT_K_COMMUNITIES}")

    print("\n=== Risk Thresholds ===")
    print(f"CRITICAL: {RiskThresholds.CRITICAL}")
    print(f"HIGH: {RiskThresholds.HIGH}")

    print("\n=== Success Criteria ===")
    print(f"TARGET_AUC_ROC: {SuccessCriteria.TARGET_AUC_ROC}")
    print(f"TARGET_MODULARITY: {SuccessCriteria.TARGET_MODULARITY}")
