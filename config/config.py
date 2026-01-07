"""
QGAI Quantum Financial Modeling - TReDS MVP
Main Configuration File

This module contains all configurable parameters for the Hybrid Classical-Quantum
TReDS Invoice Fraud Detection System. Parameters are organized by component.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026

Usage:
    from config import Config, get_config

    # Get default configuration
    config = get_config()

    # Access parameters
    n_estimators = config.classical.n_estimators
    k_communities = config.quantum.k_communities
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path
import os


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

@dataclass
class PathConfig:
    """File system path configuration."""

    # Base project directory
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # Data directories
    data_dir: Path = field(init=False)
    generated_data_dir: Path = field(init=False)
    output_data_dir: Path = field(init=False)

    # Model directory
    models_dir: Path = field(init=False)

    # Reports directory
    reports_dir: Path = field(init=False)

    # Logs directory
    logs_dir: Path = field(init=False)

    def __post_init__(self):
        """Initialize derived paths after dataclass initialization."""
        self.data_dir = self.project_root / "data"
        self.generated_data_dir = self.data_dir / "generated"
        self.output_data_dir = self.data_dir / "outputs"
        self.models_dir = self.project_root / "models"
        self.reports_dir = self.project_root / "reports"
        self.logs_dir = self.project_root / "logs"

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        directories = [
            self.data_dir,
            self.generated_data_dir,
            self.output_data_dir,
            self.models_dir,
            self.reports_dir,
            self.logs_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# DATA GENERATION CONFIGURATION
# =============================================================================

@dataclass
class DataGenerationConfig:
    """Configuration for synthetic data generation."""

    # Entity generation
    n_buyers: int = 100
    n_suppliers: int = 200
    n_rings: int = 3
    ring_sizes: List[int] = field(default_factory=lambda: [5, 7, 8])

    # Invoice generation
    n_legitimate_invoices: int = 5000
    n_ring_invoices: int = 500

    # Date range
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"

    # Ring entity registration window (days)
    ring_registration_window: int = 14

    # Default probabilities by credit rating
    default_probabilities: Dict[str, float] = field(default_factory=lambda: {
        'AAA': 0.005,
        'AA': 0.01,
        'A': 0.02,
        'BBB': 0.05,
        'BB': 0.10,
        'B': 0.15,
        'NR': 0.20
    })

    # Ring default probability (higher for fraud)
    ring_default_probability: float = 0.35

    # Credit rating distributions
    buyer_credit_ratings: Dict[str, float] = field(default_factory=lambda: {
        'AAA': 0.10,
        'AA': 0.25,
        'A': 0.35,
        'BBB': 0.20,
        'BB': 0.10
    })

    supplier_credit_ratings: Dict[str, float] = field(default_factory=lambda: {
        'A': 0.10,
        'BBB': 0.20,
        'BB': 0.30,
        'B': 0.30,
        'NR': 0.10
    })

    # Industry sectors
    industry_sectors: List[str] = field(default_factory=lambda: [
        'Manufacturing', 'IT', 'Pharma', 'Auto', 'FMCG'
    ])

    # Payment terms (days)
    payment_terms: List[int] = field(default_factory=lambda: [30, 45, 60, 90])

    # Round amounts for ring invoices
    ring_round_amounts: List[float] = field(default_factory=lambda: [
        100000, 500000, 1000000, 2500000
    ])

    # Random seed for reproducibility
    random_seed: int = 42


# =============================================================================
# CLASSICAL MODEL CONFIGURATION
# =============================================================================

@dataclass
class ClassicalModelConfig:
    """Configuration for Random Forest default prediction model."""

    # Random Forest hyperparameters
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    class_weight: str = 'balanced'
    n_jobs: int = -1

    # Cross-validation
    cv_folds: int = 5
    cv_shuffle: bool = True

    # Evaluation metrics
    primary_metric: str = 'roc_auc'
    secondary_metrics: List[str] = field(default_factory=lambda: [
        'average_precision', 'f1', 'precision', 'recall'
    ])

    # Target performance (MVP criteria)
    target_auc_roc: float = 0.75
    target_average_precision: float = 0.30
    target_recall_at_10_fpr: float = 0.50
    target_f1_score: float = 0.40

    # Classification threshold
    default_threshold: float = 0.5

    # Random seed
    random_seed: int = 42


# =============================================================================
# QUANTUM/QUBO CONFIGURATION
# =============================================================================

@dataclass
class QUBOConfig:
    """Configuration for QUBO-based community detection."""

    # Community detection parameters
    k_communities: int = 20  # Higher k creates smaller groups for ring detection
    penalty_weight: float = 1.0

    # Simulated annealing parameters
    num_reads: int = 1000
    num_sweeps: int = 1000
    beta_range: List[float] = field(default_factory=lambda: [0.1, 3.0])

    # Ring scoring thresholds
    ring_detection_threshold: float = 0.15  # Lower threshold for better recall
    min_ring_size: int = 3
    max_ring_size: int = 15

    # Ring scoring weights
    density_weight: float = 0.35
    reciprocity_weight: float = 0.35
    size_weight: float = 0.15
    regularity_weight: float = 0.15

    # Target performance (MVP criteria)
    target_modularity: float = 0.3
    target_ring_recovery_rate: float = 0.70

    # Random seed
    random_seed: int = 42

    # D-Wave configuration (for future production)
    use_dwave_hardware: bool = False
    dwave_solver: str = "hybrid_binary_quadratic_model_version2"
    dwave_time_limit: int = 10  # seconds


# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # Invoice-level features
    invoice_features: List[str] = field(default_factory=lambda: [
        'amount_log',
        'days_to_due',
        'acceptance_delay_days',
        'amount_zscore_buyer',
        'is_round_amount',
        'is_month_end',
        'time_since_last_invoice'
    ])

    # Buyer-level features
    buyer_features: List[str] = field(default_factory=lambda: [
        'buyer_total_invoices_30d',
        'buyer_unique_suppliers_30d',
        'buyer_avg_invoice_amount',
        'buyer_std_invoice_amount',
        'buyer_acceptance_rate',
        'buyer_avg_acceptance_delay',
        'buyer_default_rate_historical',
        'buyer_credit_rating_encoded'
    ])

    # Supplier-level features
    supplier_features: List[str] = field(default_factory=lambda: [
        'supplier_age_days',
        'supplier_unique_buyers_30d',
        'supplier_avg_invoice_amount',
        'supplier_total_invoices_30d'
    ])

    # Relationship features
    relationship_features: List[str] = field(default_factory=lambda: [
        'relationship_age_days',
        'relationship_invoice_count',
        'relationship_total_amount',
        'is_new_relationship'
    ])

    # Feature engineering parameters
    rolling_window_days: int = 30
    new_relationship_threshold_days: int = 30
    month_end_day_threshold: int = 25
    round_amount_modulo: int = 10000

    # Credit rating encoding
    credit_rating_encoding: Dict[str, int] = field(default_factory=lambda: {
        'AAA': 6,
        'AA': 5,
        'A': 4,
        'BBB': 3,
        'BB': 2,
        'B': 1,
        'NR': 0
    })


# =============================================================================
# RISK SCORING CONFIGURATION
# =============================================================================

@dataclass
class RiskScoringConfig:
    """Configuration for composite risk scoring."""

    # Composite score weights
    default_weight: float = 0.5
    ring_weight: float = 0.5
    interaction_weight: float = 0.3

    # Risk category thresholds
    critical_threshold: float = 0.7
    high_threshold: float = 0.5
    high_risk_threshold: float = 0.6
    moderate_threshold: float = 0.3

    # Risk categories
    risk_categories: Dict[str, str] = field(default_factory=lambda: {
        'critical': 'Critical',
        'high': 'High',
        'moderate': 'Moderate',
        'low': 'Low'
    })

    # Targeting priority matrix
    tier_1_conditions: Dict[str, float] = field(default_factory=lambda: {
        'min_default_prob': 0.5,
        'min_ring_prob': 0.3
    })

    tier_2_conditions: Dict[str, float] = field(default_factory=lambda: {
        'max_default_prob': 0.5,
        'min_ring_prob': 0.3
    })

    tier_3_conditions: Dict[str, float] = field(default_factory=lambda: {
        'min_default_prob': 0.5,
        'max_ring_prob': 0.3
    })


# =============================================================================
# EXPLAINABILITY CONFIGURATION
# =============================================================================

@dataclass
class ExplainabilityConfig:
    """Configuration for model explainability."""

    # SHAP configuration
    shap_max_samples: int = 100
    shap_check_additivity: bool = False

    # Top features to show in explanations
    top_risk_factors: int = 3
    top_protective_factors: int = 3

    # Ring explanation parameters
    show_circular_patterns: bool = True
    max_cycles_to_show: int = 5

    # Report generation
    include_visualizations: bool = True
    report_format: str = 'json'  # 'json', 'html', 'pdf'


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LoggingConfig:
    """Configuration for logging."""

    # Log level
    level: str = "INFO"

    # Log format
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"

    # Log to file
    log_to_file: bool = True
    log_filename: str = "treds_fraud_detection.log"

    # Log rotation
    rotation: str = "10 MB"
    retention: str = "1 week"


# =============================================================================
# MAIN CONFIGURATION CLASS
# =============================================================================

@dataclass
class Config:
    """
    Main configuration class combining all sub-configurations.

    This class provides a single access point for all system configuration
    parameters organized by component.

    Attributes:
        paths: File system path configuration
        data_generation: Synthetic data generation parameters
        classical: Random Forest model parameters
        quantum: QUBO community detection parameters
        features: Feature engineering parameters
        risk_scoring: Composite risk scoring parameters
        explainability: Model explainability parameters
        logging: Logging configuration
    """

    paths: PathConfig = field(default_factory=PathConfig)
    data_generation: DataGenerationConfig = field(default_factory=DataGenerationConfig)
    classical: ClassicalModelConfig = field(default_factory=ClassicalModelConfig)
    quantum: QUBOConfig = field(default_factory=QUBOConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    risk_scoring: RiskScoringConfig = field(default_factory=RiskScoringConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # System metadata
    version: str = "1.0.0"
    environment: str = "development"  # development, staging, production

    def __post_init__(self):
        """Ensure directories exist after initialization."""
        self.paths.ensure_directories()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create configuration from dictionary."""
        return cls(
            paths=PathConfig(**config_dict.get('paths', {})),
            data_generation=DataGenerationConfig(**config_dict.get('data_generation', {})),
            classical=ClassicalModelConfig(**config_dict.get('classical', {})),
            quantum=QUBOConfig(**config_dict.get('quantum', {})),
            features=FeatureConfig(**config_dict.get('features', {})),
            risk_scoring=RiskScoringConfig(**config_dict.get('risk_scoring', {})),
            explainability=ExplainabilityConfig(**config_dict.get('explainability', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
        )


# =============================================================================
# CONFIGURATION SINGLETON
# =============================================================================

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config: The singleton configuration instance.

    Example:
        >>> config = get_config()
        >>> print(config.classical.n_estimators)
        100
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config() -> None:
    """Reset the configuration singleton (useful for testing)."""
    global _config_instance
    _config_instance = None


def set_config(config: Config) -> None:
    """
    Set a custom configuration instance.

    Args:
        config: Custom Config instance to use.
    """
    global _config_instance
    _config_instance = config


# =============================================================================
# CONFIGURATION VALIDATION
# =============================================================================

def validate_config(config: Config) -> List[str]:
    """
    Validate configuration parameters.

    Args:
        config: Configuration to validate.

    Returns:
        List of validation error messages (empty if valid).
    """
    errors = []

    # Validate data generation
    if config.data_generation.n_rings != len(config.data_generation.ring_sizes):
        errors.append(
            f"n_rings ({config.data_generation.n_rings}) must match "
            f"length of ring_sizes ({len(config.data_generation.ring_sizes)})"
        )

    # Validate classical model
    if config.classical.n_estimators < 1:
        errors.append("n_estimators must be at least 1")

    if config.classical.max_depth < 1:
        errors.append("max_depth must be at least 1")

    # Validate QUBO
    if config.quantum.k_communities < 2:
        errors.append("k_communities must be at least 2")

    if config.quantum.penalty_weight <= 0:
        errors.append("penalty_weight must be positive")

    # Validate risk scoring
    weights_sum = config.risk_scoring.default_weight + config.risk_scoring.ring_weight
    if abs(weights_sum - 1.0) > 0.001:
        errors.append(f"Risk weights must sum to 1.0, got {weights_sum}")

    return errors


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print(f"Configuration loaded successfully!")
    print(f"Version: {config.version}")
    print(f"Environment: {config.environment}")
    print(f"Project root: {config.paths.project_root}")

    # Validate
    errors = validate_config(config)
    if errors:
        print(f"Validation errors: {errors}")
    else:
        print("Configuration is valid.")
