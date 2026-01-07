"""
QGAI Quantum Financial Modeling - TReDS MVP
Test Suite

This package contains all unit and integration tests for the
TReDS fraud detection system.

Test modules:
- test_data_generation: Tests for synthetic data generation
- test_feature_engineering: Tests for feature extraction
- test_classical: Tests for default prediction model
- test_quantum: Tests for QUBO ring detection
- test_pipeline: Tests for hybrid pipeline integration
- test_validation: Tests for success criteria validation

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026

Usage:
    # Run all tests
    pytest tests/

    # Run with coverage
    pytest tests/ --cov=src --cov-report=html

    # Run specific test module
    pytest tests/test_classical.py -v

    # Run tests matching pattern
    pytest tests/ -k "test_default"
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
