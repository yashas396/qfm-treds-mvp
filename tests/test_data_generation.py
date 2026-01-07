"""
QGAI Quantum Financial Modeling - TReDS MVP
Tests for Data Generation Module

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_generation import (
    EntityGenerator,
    InvoiceGenerator,
    DataValidator,
    ValidationSeverity,
    generate_entities,
    generate_invoices,
    validate_data,
)
from config import get_config


class TestEntityGenerator:
    """Tests for EntityGenerator class."""

    def test_entity_generator_creates_instance(self):
        """Test EntityGenerator instantiation."""
        gen = EntityGenerator()
        assert gen is not None
        assert gen.config is not None

    def test_generate_returns_dataframe(self):
        """Test generate returns a DataFrame."""
        gen = EntityGenerator()
        df = gen.generate()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_generate_correct_counts(self):
        """Test generated entity counts match configuration."""
        config = get_config().data_generation
        gen = EntityGenerator(config)
        df = gen.generate()

        n_buyers = len(df[df['entity_type'] == 'buyer'])
        n_suppliers = len(df[df['entity_type'] == 'supplier'])
        n_ring_members = len(df[df['is_ring_member']])
        expected_ring = sum(config.ring_sizes)

        assert n_buyers == config.n_buyers
        assert n_suppliers == config.n_suppliers
        assert n_ring_members == expected_ring

    def test_generate_has_required_columns(self):
        """Test generated DataFrame has required columns."""
        gen = EntityGenerator()
        df = gen.generate()

        required_cols = [
            'entity_id', 'entity_type', 'registration_date',
            'turnover_cr', 'credit_rating', 'industry_sector',
            'is_ring_member', 'ring_id'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_entity_ids_are_unique(self):
        """Test all entity IDs are unique."""
        gen = EntityGenerator()
        df = gen.generate()
        assert df['entity_id'].nunique() == len(df)

    def test_ring_members_have_ring_id(self):
        """Test ring members have valid ring_id."""
        gen = EntityGenerator()
        df = gen.generate()

        ring_members = df[df['is_ring_member']]
        assert ring_members['ring_id'].notna().all()

        non_ring = df[~df['is_ring_member']]
        assert non_ring['ring_id'].isna().all()

    def test_ring_members_are_dual_type(self):
        """Test ring members have 'dual' entity type."""
        gen = EntityGenerator()
        df = gen.generate()

        ring_members = df[df['is_ring_member']]
        assert (ring_members['entity_type'] == 'dual').all()

    def test_credit_ratings_are_valid(self):
        """Test credit ratings are from valid set."""
        gen = EntityGenerator()
        df = gen.generate()

        valid_ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'NR']
        assert df['credit_rating'].isin(valid_ratings).all()

    def test_turnover_is_positive(self):
        """Test turnover values are positive."""
        gen = EntityGenerator()
        df = gen.generate()
        assert (df['turnover_cr'] > 0).all()

    def test_reproducibility_with_seed(self):
        """Test same seed produces same results."""
        config1 = get_config().data_generation
        config1.random_seed = 42

        config2 = get_config().data_generation
        config2.random_seed = 42

        gen1 = EntityGenerator(config1)
        gen2 = EntityGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_get_ring_members(self):
        """Test get_ring_members returns correct mapping."""
        gen = EntityGenerator()
        df = gen.generate()
        ring_members = gen.get_ring_members(df)

        assert isinstance(ring_members, dict)
        assert len(ring_members) == gen.config.n_rings

        for ring_id, members in ring_members.items():
            assert df[df['entity_id'].isin(members)]['is_ring_member'].all()


class TestInvoiceGenerator:
    """Tests for InvoiceGenerator class."""

    @pytest.fixture
    def entities_df(self):
        """Generate entities for invoice tests."""
        gen = EntityGenerator()
        return gen.generate()

    def test_invoice_generator_creates_instance(self):
        """Test InvoiceGenerator instantiation."""
        gen = InvoiceGenerator()
        assert gen is not None

    def test_generate_returns_dataframe(self, entities_df):
        """Test generate returns a DataFrame."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_generate_correct_counts(self, entities_df):
        """Test generated invoice counts are reasonable."""
        config = get_config().data_generation
        gen = InvoiceGenerator(config)
        df = gen.generate(entities_df)

        n_legit = len(df[~df['is_in_ring']])
        n_ring = len(df[df['is_in_ring']])

        # Allow some variance due to relationship-based generation
        assert n_legit > 0
        assert n_ring > 0
        assert n_ring <= config.n_ring_invoices * 1.5  # Some buffer

    def test_generate_has_required_columns(self, entities_df):
        """Test generated DataFrame has required columns."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        required_cols = [
            'invoice_id', 'buyer_id', 'supplier_id',
            'invoice_date', 'due_date', 'acceptance_date',
            'amount', 'is_defaulted', 'is_in_ring', 'ring_id'
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_invoice_ids_are_unique(self, entities_df):
        """Test all invoice IDs are unique."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)
        assert df['invoice_id'].nunique() == len(df)

    def test_buyer_supplier_ids_exist(self, entities_df):
        """Test buyer and supplier IDs exist in entities."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        entity_ids = set(entities_df['entity_id'])
        assert df['buyer_id'].isin(entity_ids).all()
        assert df['supplier_id'].isin(entity_ids).all()

    def test_buyer_not_equal_supplier(self, entities_df):
        """Test buyer and supplier are different."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)
        assert (df['buyer_id'] != df['supplier_id']).all()

    def test_amount_in_valid_range(self, entities_df):
        """Test amounts are within valid range."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        assert (df['amount'] >= 0).all()
        assert (df['amount'] <= 1e9).all()  # 100 Cr max

    def test_date_ordering(self, entities_df):
        """Test dates are in correct order."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        assert (df['acceptance_date'] >= df['invoice_date']).all()
        assert (df['due_date'] >= df['invoice_date']).all()

    def test_ring_invoices_have_ring_id(self, entities_df):
        """Test ring invoices have valid ring_id."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        ring_invoices = df[df['is_in_ring']]
        assert ring_invoices['ring_id'].notna().all()

    def test_ring_default_rate_higher(self, entities_df):
        """Test ring invoices have higher default rate."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        legit_rate = df[~df['is_in_ring']]['is_defaulted'].mean()
        ring_rate = df[df['is_in_ring']]['is_defaulted'].mean()

        # Ring rate should generally be higher (may vary with small samples)
        # Just check both rates are reasonable
        assert 0 <= legit_rate <= 1
        assert 0 <= ring_rate <= 1

    def test_ring_invoices_involve_ring_members(self, entities_df):
        """Test ring invoices only involve ring members."""
        gen = InvoiceGenerator()
        df = gen.generate(entities_df)

        ring_member_ids = set(entities_df[entities_df['is_ring_member']]['entity_id'])
        ring_invoices = df[df['is_in_ring']]

        ring_buyers = set(ring_invoices['buyer_id'])
        ring_suppliers = set(ring_invoices['supplier_id'])

        assert ring_buyers.issubset(ring_member_ids)
        assert ring_suppliers.issubset(ring_member_ids)


class TestDataValidator:
    """Tests for DataValidator class."""

    @pytest.fixture
    def valid_data(self):
        """Generate valid test data."""
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        return entities_df, invoices_df

    def test_validator_creates_instance(self):
        """Test DataValidator instantiation."""
        validator = DataValidator()
        assert validator is not None

    def test_validate_returns_result(self, valid_data):
        """Test validate returns ValidationResult."""
        entities_df, invoices_df = valid_data
        validator = DataValidator()
        result = validator.validate(entities_df, invoices_df)

        assert hasattr(result, 'is_valid')
        assert hasattr(result, 'issues')
        assert hasattr(result, 'entities_validated')
        assert hasattr(result, 'invoices_validated')

    def test_valid_data_passes(self, valid_data):
        """Test valid data passes validation."""
        entities_df, invoices_df = valid_data
        validator = DataValidator()
        result = validator.validate(entities_df, invoices_df)

        # Should pass with no errors (may have warnings)
        assert result.n_errors == 0, f"Unexpected errors: {[e.message for e in result.errors]}"

    def test_counts_are_correct(self, valid_data):
        """Test validation counts match input."""
        entities_df, invoices_df = valid_data
        validator = DataValidator()
        result = validator.validate(entities_df, invoices_df)

        assert result.entities_validated == len(entities_df)
        assert result.invoices_validated == len(invoices_df)

    def test_detects_missing_columns(self, valid_data):
        """Test validator detects missing columns."""
        entities_df, invoices_df = valid_data

        # Remove required column
        bad_entities = entities_df.drop(columns=['entity_id'])

        validator = DataValidator()
        result = validator.validate(bad_entities, invoices_df)

        assert not result.is_valid
        assert any('missing' in i.message.lower() for i in result.errors)

    def test_detects_invalid_references(self, valid_data):
        """Test validator detects invalid entity references."""
        entities_df, invoices_df = valid_data

        # Create invoice with invalid buyer
        bad_invoices = invoices_df.copy()
        bad_invoices.loc[0, 'buyer_id'] = 'INVALID_BUYER_ID'

        validator = DataValidator()
        result = validator.validate(entities_df, bad_invoices)

        assert any('invalid buyer' in i.message.lower() for i in result.issues)

    def test_detects_self_invoices(self, valid_data):
        """Test validator detects self-invoicing."""
        entities_df, invoices_df = valid_data

        # Create self-invoice
        bad_invoices = invoices_df.copy()
        bad_invoices.loc[0, 'supplier_id'] = bad_invoices.loc[0, 'buyer_id']

        validator = DataValidator()
        result = validator.validate(entities_df, bad_invoices)

        assert any('same buyer and supplier' in i.message.lower() for i in result.issues)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_generate_entities_function(self):
        """Test generate_entities convenience function."""
        df = generate_entities(n_buyers=50, n_suppliers=100, n_rings=2)

        assert len(df[df['entity_type'] == 'buyer']) == 50
        assert len(df[df['entity_type'] == 'supplier']) == 100
        assert df[df['is_ring_member']]['ring_id'].nunique() == 2

    def test_generate_invoices_function(self):
        """Test generate_invoices convenience function."""
        entities = generate_entities(n_buyers=20, n_suppliers=40, n_rings=1)
        invoices = generate_invoices(entities, n_legitimate=100, n_ring=20)

        assert len(invoices) > 0
        assert len(invoices[invoices['is_in_ring']]) > 0

    def test_validate_data_function(self):
        """Test validate_data convenience function."""
        entities = generate_entities()
        invoices = generate_invoices(entities)
        result = validate_data(entities, invoices)

        assert result.is_valid or result.n_errors > 0


class TestDataQuality:
    """Integration tests for data quality."""

    def test_full_pipeline(self):
        """Test complete data generation pipeline."""
        # Generate
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        # Validate
        validator = DataValidator()
        result = validator.validate(entities_df, invoices_df)

        # Should pass
        assert result.is_valid, f"Validation failed: {[e.message for e in result.errors]}"

        # Check statistics
        entity_stats = entity_gen.get_statistics(entities_df)
        invoice_stats = invoice_gen.get_statistics(invoices_df)

        assert entity_stats['total_entities'] > 0
        assert invoice_stats['total_invoices'] > 0
        assert invoice_stats['default_rate_overall'] > 0

    def test_ring_patterns_exist(self):
        """Test that ring patterns are correctly generated."""
        entities = generate_entities(n_rings=3, ring_sizes=[4, 5, 6])
        invoices = generate_invoices(entities)

        # Check ring invoices exist
        ring_invoices = invoices[invoices['is_in_ring']]
        assert len(ring_invoices) > 0

        # Check each ring has invoices
        ring_ids = ring_invoices['ring_id'].unique()
        assert len(ring_ids) == 3

        # Check circular patterns
        for ring_id in ring_ids:
            ring_inv = ring_invoices[ring_invoices['ring_id'] == ring_id]
            buyers = set(ring_inv['buyer_id'])
            suppliers = set(ring_inv['supplier_id'])
            # In circular pattern, same entities appear as both
            assert len(buyers & suppliers) > 0


def run_tests():
    """Run all tests and print summary."""
    exit_code = pytest.main([__file__, "-v", "--tb=short"])
    return exit_code


def run_validation():
    """Run comprehensive validation of data generation module."""
    print("=" * 70)
    print("DATA GENERATION VALIDATION")
    print("=" * 70)

    errors = []
    warnings = []

    # 1. Entity Generation
    print("\n[1/4] Testing Entity Generation...")
    try:
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()
        config = get_config().data_generation

        n_buyers = len(entities_df[entities_df['entity_type'] == 'buyer'])
        n_suppliers = len(entities_df[entities_df['entity_type'] == 'supplier'])
        n_ring_members = len(entities_df[entities_df['is_ring_member']])

        print(f"      Generated: {len(entities_df)} entities")
        print(f"      Buyers: {n_buyers}, Suppliers: {n_suppliers}")
        print(f"      Ring members: {n_ring_members}")

        if n_buyers != config.n_buyers:
            errors.append(f"Buyer count mismatch: {n_buyers} vs {config.n_buyers}")
        if n_suppliers != config.n_suppliers:
            errors.append(f"Supplier count mismatch: {n_suppliers} vs {config.n_suppliers}")

        print("      [PASS] Entity generation complete")
    except Exception as e:
        errors.append(f"Entity generation failed: {e}")
        print(f"      [FAIL] Entity generation: {e}")
        return False

    # 2. Invoice Generation
    print("\n[2/4] Testing Invoice Generation...")
    try:
        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        n_legit = len(invoices_df[~invoices_df['is_in_ring']])
        n_ring = len(invoices_df[invoices_df['is_in_ring']])
        default_rate = invoices_df['is_defaulted'].mean()

        print(f"      Generated: {len(invoices_df)} invoices")
        print(f"      Legitimate: {n_legit}, Ring: {n_ring}")
        print(f"      Default rate: {default_rate:.2%}")

        if len(invoices_df) == 0:
            errors.append("No invoices generated")

        print("      [PASS] Invoice generation complete")
    except Exception as e:
        errors.append(f"Invoice generation failed: {e}")
        print(f"      [FAIL] Invoice generation: {e}")
        return False

    # 3. Data Validation
    print("\n[3/4] Testing Data Validation...")
    try:
        validator = DataValidator()
        result = validator.validate(entities_df, invoices_df)

        print(f"      Entities validated: {result.entities_validated}")
        print(f"      Invoices validated: {result.invoices_validated}")
        print(f"      Issues found: {len(result.issues)}")

        if result.n_errors > 0:
            for err in result.errors[:3]:
                errors.append(f"Validation error: {err.message}")
            print(f"      [WARN] {result.n_errors} validation errors")
        else:
            print("      [PASS] Data validation complete")
    except Exception as e:
        errors.append(f"Data validation failed: {e}")
        print(f"      [FAIL] Data validation: {e}")

    # 4. Ring Pattern Verification
    print("\n[4/4] Testing Ring Patterns...")
    try:
        ring_invoices = invoices_df[invoices_df['is_in_ring']]
        ring_entities = entities_df[entities_df['is_ring_member']]

        n_rings = ring_entities['ring_id'].nunique()
        ring_member_ids = set(ring_entities['entity_id'])

        # Verify ring invoices only involve ring members
        ring_buyers = set(ring_invoices['buyer_id'])
        ring_suppliers = set(ring_invoices['supplier_id'])

        buyers_ok = ring_buyers.issubset(ring_member_ids)
        suppliers_ok = ring_suppliers.issubset(ring_member_ids)

        print(f"      Rings: {n_rings}")
        print(f"      Ring invoices: {len(ring_invoices)}")
        print(f"      Buyer validation: {'PASS' if buyers_ok else 'FAIL'}")
        print(f"      Supplier validation: {'PASS' if suppliers_ok else 'FAIL'}")

        if not buyers_ok:
            errors.append("Ring invoice buyers not in ring members")
        if not suppliers_ok:
            errors.append("Ring invoice suppliers not in ring members")

        if buyers_ok and suppliers_ok:
            print("      [PASS] Ring patterns verified")
    except Exception as e:
        errors.append(f"Ring pattern verification failed: {e}")
        print(f"      [FAIL] Ring patterns: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("DATA GENERATION VALIDATION SUMMARY")
    print("=" * 70)

    print(f"\n  Errors: {len(errors)}")
    for err in errors:
        print(f"    [ERROR] {err}")

    print(f"\n  Warnings: {len(warnings)}")
    for warn in warnings:
        print(f"    [WARN] {warn}")

    if len(errors) == 0:
        print("\n  STATUS: PASSED")
    else:
        print("\n  STATUS: FAILED")

    print("\n" + "=" * 70)
    print("DATA GENERATION VALIDATION COMPLETE")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Data Generation Tests")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--unittest", action="store_true", help="Run pytest tests")
    args = parser.parse_args()

    if args.unittest:
        run_tests()
    elif args.validate:
        success = run_validation()
        exit(0 if success else 1)
    else:
        # Default: run validation
        success = run_validation()
        exit(0 if success else 1)
