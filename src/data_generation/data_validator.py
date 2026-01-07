"""
QGAI Quantum Financial Modeling - TReDS MVP
Data Validator Module

This module validates generated synthetic data for:
- Schema correctness
- Referential integrity
- Business logic constraints
- Statistical properties

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.constants import DataGenerationConstants as DGC


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    ERROR = "error"      # Critical issue, data is invalid
    WARNING = "warning"  # Potential issue, data may be usable
    INFO = "info"        # Informational note


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    affected_records: int = 0
    details: Optional[Dict] = None


@dataclass
class ValidationResult:
    """Result container for data validation."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    entities_validated: int = 0
    invoices_validated: int = 0
    validation_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get only error-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]

    @property
    def warnings(self) -> List[ValidationIssue]:
        """Get only warning-level issues."""
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]

    @property
    def n_errors(self) -> int:
        """Count of errors."""
        return len(self.errors)

    @property
    def n_warnings(self) -> int:
        """Count of warnings."""
        return len(self.warnings)

    def summary(self) -> str:
        """Generate validation summary string."""
        status = "PASSED" if self.is_valid else "FAILED"
        return (
            f"Validation {status}\n"
            f"  Entities: {self.entities_validated}\n"
            f"  Invoices: {self.invoices_validated}\n"
            f"  Errors: {self.n_errors}\n"
            f"  Warnings: {self.n_warnings}"
        )


class DataValidator:
    """
    Validate generated synthetic TReDS data.

    This class performs comprehensive validation including:
    - Schema validation (required columns, data types)
    - Referential integrity (buyer/supplier IDs exist)
    - Business logic (date ordering, amount ranges)
    - Statistical properties (default rates, ring patterns)

    Example:
        >>> validator = DataValidator()
        >>> result = validator.validate(entities_df, invoices_df)
        >>> if result.is_valid:
        ...     print("Data is valid!")
        ... else:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
    """

    def __init__(self, strict: bool = True):
        """
        Initialize DataValidator.

        Args:
            strict: If True, any error makes data invalid.
                   If False, only critical errors invalidate data.
        """
        self.strict = strict
        self._issues: List[ValidationIssue] = []

    def validate(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> ValidationResult:
        """
        Perform comprehensive validation of entities and invoices.

        Args:
            entities_df: DataFrame of entities
            invoices_df: DataFrame of invoices

        Returns:
            ValidationResult: Comprehensive validation result
        """
        self._issues = []

        # Schema validation
        self._validate_entities_schema(entities_df)
        self._validate_invoices_schema(invoices_df)

        # Stop early if schema validation found critical errors (missing columns)
        schema_errors = [i for i in self._issues if i.category == 'schema' and i.severity == ValidationSeverity.ERROR]
        if schema_errors:
            # Cannot proceed with further validation if schema is invalid
            is_valid = False
            return ValidationResult(
                is_valid=is_valid,
                issues=self._issues.copy(),
                entities_validated=len(entities_df),
                invoices_validated=len(invoices_df),
                validation_timestamp=datetime.now()
            )

        # Referential integrity
        self._validate_referential_integrity(entities_df, invoices_df)

        # Business logic
        self._validate_entity_logic(entities_df)
        self._validate_invoice_logic(invoices_df)

        # Statistical properties
        self._validate_statistics(entities_df, invoices_df)

        # Ring structure
        self._validate_ring_structure(entities_df, invoices_df)

        # Determine overall validity
        is_valid = len(self.errors) == 0 if self.strict else self._has_critical_errors()

        return ValidationResult(
            is_valid=is_valid,
            issues=self._issues.copy(),
            entities_validated=len(entities_df),
            invoices_validated=len(invoices_df),
            validation_timestamp=datetime.now()
        )

    @property
    def errors(self) -> List[ValidationIssue]:
        """Get error-level issues."""
        return [i for i in self._issues if i.severity == ValidationSeverity.ERROR]

    def _has_critical_errors(self) -> bool:
        """Check for critical errors that make data unusable."""
        critical_categories = ['schema', 'referential_integrity']
        return any(
            i.severity == ValidationSeverity.ERROR and i.category in critical_categories
            for i in self._issues
        )

    def _add_issue(
        self,
        severity: ValidationSeverity,
        category: str,
        message: str,
        affected_records: int = 0,
        details: Optional[Dict] = None
    ) -> None:
        """Add a validation issue."""
        self._issues.append(ValidationIssue(
            severity=severity,
            category=category,
            message=message,
            affected_records=affected_records,
            details=details
        ))

    # =========================================================================
    # Schema Validation
    # =========================================================================

    def _validate_entities_schema(self, df: pd.DataFrame) -> None:
        """Validate entities DataFrame schema."""
        required_columns = [
            'entity_id', 'entity_type', 'registration_date', 'turnover_cr',
            'credit_rating', 'industry_sector', 'is_ring_member', 'ring_id'
        ]

        # Check required columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self._add_issue(
                ValidationSeverity.ERROR,
                'schema',
                f"Entities missing required columns: {missing}",
                details={'missing_columns': missing}
            )
            return  # Can't proceed without required columns

        # Check for null values in critical columns
        critical_cols = ['entity_id', 'entity_type', 'registration_date']
        for col in critical_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    'schema',
                    f"Entities column '{col}' has {null_count} null values",
                    affected_records=null_count
                )

        # Check unique entity_id
        duplicates = df['entity_id'].duplicated().sum()
        if duplicates > 0:
            self._add_issue(
                ValidationSeverity.ERROR,
                'schema',
                f"Entities has {duplicates} duplicate entity_ids",
                affected_records=duplicates
            )

    def _validate_invoices_schema(self, df: pd.DataFrame) -> None:
        """Validate invoices DataFrame schema."""
        required_columns = [
            'invoice_id', 'buyer_id', 'supplier_id', 'invoice_date',
            'due_date', 'acceptance_date', 'amount', 'is_defaulted',
            'is_in_ring', 'ring_id'
        ]

        # Check required columns
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            self._add_issue(
                ValidationSeverity.ERROR,
                'schema',
                f"Invoices missing required columns: {missing}",
                details={'missing_columns': missing}
            )
            return

        # Check for null values
        critical_cols = ['invoice_id', 'buyer_id', 'supplier_id', 'amount']
        for col in critical_cols:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    'schema',
                    f"Invoices column '{col}' has {null_count} null values",
                    affected_records=null_count
                )

        # Check unique invoice_id
        duplicates = df['invoice_id'].duplicated().sum()
        if duplicates > 0:
            self._add_issue(
                ValidationSeverity.ERROR,
                'schema',
                f"Invoices has {duplicates} duplicate invoice_ids",
                affected_records=duplicates
            )

    # =========================================================================
    # Referential Integrity
    # =========================================================================

    def _validate_referential_integrity(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> None:
        """Validate referential integrity between entities and invoices."""
        entity_ids = set(entities_df['entity_id'].unique())

        # Check buyer IDs
        invalid_buyers = ~invoices_df['buyer_id'].isin(entity_ids)
        if invalid_buyers.any():
            count = invalid_buyers.sum()
            invalid_ids = invoices_df[invalid_buyers]['buyer_id'].unique()[:5]
            self._add_issue(
                ValidationSeverity.ERROR,
                'referential_integrity',
                f"{count} invoices have invalid buyer_id",
                affected_records=count,
                details={'sample_invalid_ids': list(invalid_ids)}
            )

        # Check supplier IDs
        invalid_suppliers = ~invoices_df['supplier_id'].isin(entity_ids)
        if invalid_suppliers.any():
            count = invalid_suppliers.sum()
            invalid_ids = invoices_df[invalid_suppliers]['supplier_id'].unique()[:5]
            self._add_issue(
                ValidationSeverity.ERROR,
                'referential_integrity',
                f"{count} invoices have invalid supplier_id",
                affected_records=count,
                details={'sample_invalid_ids': list(invalid_ids)}
            )

        # Check ring_id consistency
        entity_rings = set(entities_df[entities_df['is_ring_member']]['ring_id'].unique())
        invoice_rings = set(invoices_df[invoices_df['is_in_ring']]['ring_id'].unique())

        if invoice_rings - entity_rings:
            orphan_rings = invoice_rings - entity_rings
            self._add_issue(
                ValidationSeverity.ERROR,
                'referential_integrity',
                f"Invoices reference non-existent rings: {orphan_rings}",
                details={'orphan_rings': list(orphan_rings)}
            )

    # =========================================================================
    # Business Logic Validation
    # =========================================================================

    def _validate_entity_logic(self, df: pd.DataFrame) -> None:
        """Validate entity business logic."""
        # Check turnover range
        negative_turnover = df['turnover_cr'] < 0
        if negative_turnover.any():
            self._add_issue(
                ValidationSeverity.ERROR,
                'business_logic',
                f"{negative_turnover.sum()} entities have negative turnover",
                affected_records=negative_turnover.sum()
            )

        # Check valid credit ratings
        valid_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'NR']
        invalid_ratings = ~df['credit_rating'].isin(valid_ratings)
        if invalid_ratings.any():
            invalid = df[invalid_ratings]['credit_rating'].unique()
            self._add_issue(
                ValidationSeverity.WARNING,
                'business_logic',
                f"Entities have invalid credit ratings: {list(invalid)}",
                affected_records=invalid_ratings.sum()
            )

        # Check entity types
        valid_types = ['buyer', 'supplier', 'dual']
        invalid_types = ~df['entity_type'].isin(valid_types)
        if invalid_types.any():
            self._add_issue(
                ValidationSeverity.ERROR,
                'business_logic',
                f"{invalid_types.sum()} entities have invalid entity_type",
                affected_records=invalid_types.sum()
            )

    def _validate_invoice_logic(self, df: pd.DataFrame) -> None:
        """Validate invoice business logic."""
        # Check amount range
        invalid_amounts = (df['amount'] < DGC.MIN_INVOICE_AMOUNT) | (df['amount'] > DGC.MAX_INVOICE_AMOUNT)
        if invalid_amounts.any():
            self._add_issue(
                ValidationSeverity.WARNING,
                'business_logic',
                f"{invalid_amounts.sum()} invoices have amounts outside valid range",
                affected_records=invalid_amounts.sum()
            )

        # Check date ordering: invoice_date <= acceptance_date <= due_date
        date_issues = (
            (df['acceptance_date'] < df['invoice_date']) |
            (df['due_date'] < df['invoice_date'])
        )
        if date_issues.any():
            self._add_issue(
                ValidationSeverity.ERROR,
                'business_logic',
                f"{date_issues.sum()} invoices have invalid date ordering",
                affected_records=date_issues.sum()
            )

        # Check buyer != supplier
        self_invoices = df['buyer_id'] == df['supplier_id']
        if self_invoices.any():
            self._add_issue(
                ValidationSeverity.ERROR,
                'business_logic',
                f"{self_invoices.sum()} invoices have same buyer and supplier",
                affected_records=self_invoices.sum()
            )

    # =========================================================================
    # Statistical Validation
    # =========================================================================

    def _validate_statistics(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> None:
        """Validate statistical properties of generated data."""
        # Overall default rate should be reasonable (1-10%)
        default_rate = invoices_df['is_defaulted'].mean()
        if default_rate < 0.01 or default_rate > 0.15:
            self._add_issue(
                ValidationSeverity.WARNING,
                'statistics',
                f"Overall default rate {default_rate:.2%} outside expected range (1-15%)",
                details={'default_rate': default_rate}
            )

        # Ring invoices should have higher default rate
        legit_default = invoices_df[~invoices_df['is_in_ring']]['is_defaulted'].mean()
        ring_default = invoices_df[invoices_df['is_in_ring']]['is_defaulted'].mean()

        if ring_default <= legit_default:
            self._add_issue(
                ValidationSeverity.WARNING,
                'statistics',
                f"Ring default rate ({ring_default:.2%}) should be higher than legitimate ({legit_default:.2%})",
                details={'ring_default': ring_default, 'legit_default': legit_default}
            )

        # Check entity type distribution
        n_buyers = len(entities_df[entities_df['entity_type'] == 'buyer'])
        n_suppliers = len(entities_df[entities_df['entity_type'] == 'supplier'])

        if n_buyers == 0:
            self._add_issue(
                ValidationSeverity.ERROR,
                'statistics',
                "No buyer entities generated"
            )
        if n_suppliers == 0:
            self._add_issue(
                ValidationSeverity.ERROR,
                'statistics',
                "No supplier entities generated"
            )

    # =========================================================================
    # Ring Structure Validation
    # =========================================================================

    def _validate_ring_structure(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> None:
        """Validate fraud ring structure and patterns."""
        ring_entities = entities_df[entities_df['is_ring_member']]
        ring_invoices = invoices_df[invoices_df['is_in_ring']]

        if len(ring_entities) == 0:
            self._add_issue(
                ValidationSeverity.INFO,
                'ring_structure',
                "No ring entities in data (may be intentional)"
            )
            return

        # Check each ring has minimum size
        ring_sizes = ring_entities.groupby('ring_id').size()
        small_rings = ring_sizes[ring_sizes < 3]
        if len(small_rings) > 0:
            self._add_issue(
                ValidationSeverity.WARNING,
                'ring_structure',
                f"{len(small_rings)} rings have fewer than 3 members",
                details={'small_rings': small_rings.to_dict()}
            )

        # Check ring invoices involve ring members
        ring_invoice_buyers = set(ring_invoices['buyer_id'].unique())
        ring_invoice_suppliers = set(ring_invoices['supplier_id'].unique())
        ring_member_ids = set(ring_entities['entity_id'].unique())

        non_member_buyers = ring_invoice_buyers - ring_member_ids
        if non_member_buyers:
            self._add_issue(
                ValidationSeverity.ERROR,
                'ring_structure',
                f"Ring invoices have non-ring buyers: {list(non_member_buyers)[:5]}",
                details={'non_member_buyers': list(non_member_buyers)}
            )

        non_member_suppliers = ring_invoice_suppliers - ring_member_ids
        if non_member_suppliers:
            self._add_issue(
                ValidationSeverity.ERROR,
                'ring_structure',
                f"Ring invoices have non-ring suppliers: {list(non_member_suppliers)[:5]}",
                details={'non_member_suppliers': list(non_member_suppliers)}
            )

        # Check for circular patterns (A→B, B→C, C→A)
        for ring_id in ring_entities['ring_id'].unique():
            ring_inv = ring_invoices[ring_invoices['ring_id'] == ring_id]
            if len(ring_inv) > 0:
                edges = set(zip(ring_inv['buyer_id'], ring_inv['supplier_id']))
                nodes = set(ring_inv['buyer_id'].tolist() + ring_inv['supplier_id'].tolist())

                # Each node should have at least one outgoing edge
                buyers_in_ring = set(ring_inv['buyer_id'].unique())
                suppliers_in_ring = set(ring_inv['supplier_id'].unique())

                # Circular pattern: each entity should be both buyer and supplier
                dual_role = buyers_in_ring & suppliers_in_ring
                if len(dual_role) < len(nodes) * 0.5:
                    self._add_issue(
                        ValidationSeverity.INFO,
                        'ring_structure',
                        f"Ring {ring_id}: Only {len(dual_role)}/{len(nodes)} entities have dual roles",
                        details={'ring_id': ring_id, 'dual_role_count': len(dual_role)}
                    )


def validate_data(
    entities_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    strict: bool = True
) -> ValidationResult:
    """
    Convenience function to validate data.

    Args:
        entities_df: DataFrame of entities
        invoices_df: DataFrame of invoices
        strict: If True, any error makes data invalid

    Returns:
        ValidationResult: Validation result
    """
    validator = DataValidator(strict=strict)
    return validator.validate(entities_df, invoices_df)


if __name__ == "__main__":
    # Test data validation
    print("=" * 60)
    print("DATA VALIDATION TEST")
    print("=" * 60)

    # Generate test data
    from entity_generator import EntityGenerator
    from invoice_generator import InvoiceGenerator

    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    # Validate
    validator = DataValidator()
    result = validator.validate(entities_df, invoices_df)

    print(f"\n{result.summary()}")

    if result.issues:
        print("\nIssues Found:")
        for issue in result.issues:
            icon = "❌" if issue.severity == ValidationSeverity.ERROR else "⚠️" if issue.severity == ValidationSeverity.WARNING else "ℹ️"
            print(f"  {icon} [{issue.category}] {issue.message}")

    print(f"\nValidation {'PASSED' if result.is_valid else 'FAILED'}")
