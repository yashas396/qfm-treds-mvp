"""
QGAI Quantum Financial Modeling - TReDS MVP
Data Generation Module

This module provides synthetic data generation for the TReDS fraud detection system:
- EntityGenerator: Creates synthetic buyers, suppliers, and ring members
- InvoiceGenerator: Creates synthetic invoices including fraud patterns
- DataValidator: Validates generated data quality

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026

Example:
    >>> from src.data_generation import EntityGenerator, InvoiceGenerator, DataValidator
    >>>
    >>> # Generate entities
    >>> entity_gen = EntityGenerator()
    >>> entities_df = entity_gen.generate()
    >>>
    >>> # Generate invoices
    >>> invoice_gen = InvoiceGenerator()
    >>> invoices_df = invoice_gen.generate(entities_df)
    >>>
    >>> # Validate data
    >>> validator = DataValidator()
    >>> result = validator.validate(entities_df, invoices_df)
    >>> print(f"Valid: {result.is_valid}")
"""

from .entity_generator import (
    EntityGenerator,
    EntityGenerationResult,
    generate_entities,
)

from .invoice_generator import (
    InvoiceGenerator,
    InvoiceGenerationResult,
    generate_invoices,
)

from .data_validator import (
    DataValidator,
    ValidationResult,
    ValidationIssue,
    ValidationSeverity,
    validate_data,
)

__all__ = [
    # Entity generation
    "EntityGenerator",
    "EntityGenerationResult",
    "generate_entities",

    # Invoice generation
    "InvoiceGenerator",
    "InvoiceGenerationResult",
    "generate_invoices",

    # Validation
    "DataValidator",
    "ValidationResult",
    "ValidationIssue",
    "ValidationSeverity",
    "validate_data",
]
