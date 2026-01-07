"""
QGAI Quantum Financial Modeling - TReDS MVP
Phase 1: Setup Validation Tests

These tests verify that the Phase 1 setup is complete and correct.

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class TestProjectStructure:
    """Test that project structure is correct."""

    def test_project_root_exists(self):
        """Verify project root directory exists."""
        assert PROJECT_ROOT.exists()

    def test_required_directories_exist(self):
        """Verify all required directories exist."""
        required_dirs = [
            "config",
            "src",
            "src/data_generation",
            "src/feature_engineering",
            "src/classical",
            "src/quantum",
            "src/pipeline",
            "src/explainability",
            "src/utils",
            "tests",
            "data",
            "data/generated",
            "data/outputs",
            "models",
            "reports",
            "docs",
            "docs/phase_docs",
            "notebooks",
        ]
        for dir_name in required_dirs:
            dir_path = PROJECT_ROOT / dir_name
            assert dir_path.exists(), f"Directory missing: {dir_name}"

    def test_required_files_exist(self):
        """Verify all required files exist."""
        required_files = [
            "requirements.txt",
            "README.md",
            ".gitignore",
            "main.py",
            "setup.py",
            "pyproject.toml",
            "mkdocs.yml",
            "Makefile",
            "config/__init__.py",
            "config/config.py",
            "config/constants.py",
            "src/__init__.py",
            "src/utils/__init__.py",
            "src/utils/logger.py",
            "src/utils/helpers.py",
            "tests/__init__.py",
            "docs/phase_docs/phase1_setup.md",
        ]
        for file_name in required_files:
            file_path = PROJECT_ROOT / file_name
            assert file_path.exists(), f"File missing: {file_name}"


class TestConfiguration:
    """Test configuration module."""

    def test_config_imports(self):
        """Verify config module can be imported."""
        from config import Config, get_config

    def test_get_config_returns_config(self):
        """Verify get_config returns a Config instance."""
        from config import Config, get_config
        config = get_config()
        assert isinstance(config, Config)

    def test_config_has_required_attributes(self):
        """Verify config has all required sub-configurations."""
        from config import get_config
        config = get_config()

        assert hasattr(config, 'paths')
        assert hasattr(config, 'data_generation')
        assert hasattr(config, 'classical')
        assert hasattr(config, 'quantum')
        assert hasattr(config, 'features')
        assert hasattr(config, 'risk_scoring')
        assert hasattr(config, 'explainability')
        assert hasattr(config, 'logging')

    def test_config_default_values(self):
        """Verify config has expected default values."""
        from config import get_config
        config = get_config()

        # Classical model defaults
        assert config.classical.n_estimators == 100
        assert config.classical.max_depth == 10
        assert config.classical.target_auc_roc == 0.75

        # QUBO defaults
        assert config.quantum.k_communities == 5
        assert config.quantum.num_reads == 1000
        assert config.quantum.target_modularity == 0.3

        # Data generation defaults
        assert config.data_generation.n_buyers == 100
        assert config.data_generation.n_suppliers == 200
        assert config.data_generation.n_rings == 3

    def test_config_validation(self):
        """Verify config validation works."""
        from config.config import validate_config, get_config
        config = get_config()
        errors = validate_config(config)
        assert len(errors) == 0, f"Config validation errors: {errors}"


class TestConstants:
    """Test constants module."""

    def test_constants_imports(self):
        """Verify constants can be imported."""
        from config.constants import (
            ModelConstants,
            QUBOConstants,
            DataGenerationConstants,
            FeatureConstants,
            RiskThresholds,
            SuccessCriteria,
        )

    def test_entity_type_enum(self):
        """Verify EntityType enum values."""
        from config.constants import EntityType
        assert EntityType.BUYER.value == "buyer"
        assert EntityType.SUPPLIER.value == "supplier"
        assert EntityType.DUAL.value == "dual"

    def test_risk_category_enum(self):
        """Verify RiskCategory enum values."""
        from config.constants import RiskCategory
        assert RiskCategory.CRITICAL.value == "Critical"
        assert RiskCategory.HIGH.value == "High"
        assert RiskCategory.MODERATE.value == "Moderate"
        assert RiskCategory.LOW.value == "Low"

    def test_success_criteria_values(self):
        """Verify MVP success criteria are defined."""
        from config.constants import SuccessCriteria
        assert SuccessCriteria.TARGET_AUC_ROC == 0.75
        assert SuccessCriteria.TARGET_MODULARITY == 0.30
        assert SuccessCriteria.TARGET_RING_RECOVERY_RATE == 0.70


class TestUtilities:
    """Test utility modules."""

    def test_logger_imports(self):
        """Verify logger can be imported."""
        from src.utils.logger import setup_logger, get_logger

    def test_logger_works(self):
        """Verify logger can be created and used."""
        from src.utils.logger import setup_logger
        logger = setup_logger("test", log_to_file=False)
        # Should not raise
        logger.info("Test message")

    def test_helpers_imports(self):
        """Verify helpers can be imported."""
        from src.utils.helpers import (
            ensure_directory,
            timestamp_filename,
            load_json,
            save_json,
            format_currency,
            format_percentage,
            Timer,
        )

    def test_timestamp_filename(self):
        """Verify timestamp filename generation."""
        from src.utils.helpers import timestamp_filename
        filename = timestamp_filename("test", "csv")
        assert filename.startswith("test_")
        assert filename.endswith(".csv")

    def test_format_currency(self):
        """Verify Indian currency formatting."""
        from src.utils.helpers import format_currency
        result = format_currency(1234567.89)
        assert "12,34,567.89" in result
        assert result.startswith("₹")

    def test_format_percentage(self):
        """Verify percentage formatting."""
        from src.utils.helpers import format_percentage
        assert format_percentage(0.75) == "75.00%"
        assert format_percentage(0.333, 1) == "33.3%"


class TestSourceModules:
    """Test that source modules can be imported."""

    def test_src_init(self):
        """Verify src package can be imported."""
        import src
        assert hasattr(src, '__version__')

    def test_data_generation_init(self):
        """Verify data_generation module can be imported."""
        from src import data_generation

    def test_feature_engineering_init(self):
        """Verify feature_engineering module can be imported."""
        from src import feature_engineering

    def test_classical_init(self):
        """Verify classical module can be imported."""
        from src import classical

    def test_quantum_init(self):
        """Verify quantum module can be imported."""
        from src import quantum

    def test_pipeline_init(self):
        """Verify pipeline module can be imported."""
        from src import pipeline

    def test_explainability_init(self):
        """Verify explainability module can be imported."""
        from src import explainability


def run_validation():
    """Run all validation tests and print results."""
    print("\n" + "=" * 60)
    print("PHASE 1 VALIDATION")
    print("=" * 60)

    # Run tests
    exit_code = pytest.main([__file__, "-v", "--tb=short"])

    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✓ PHASE 1 VALIDATION PASSED")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ PHASE 1 VALIDATION FAILED")
        print("=" * 60)

    return exit_code


if __name__ == "__main__":
    run_validation()
