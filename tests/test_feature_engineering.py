"""
QGAI Quantum Financial Modeling - TReDS MVP
Feature Engineering Test Suite

This module tests the feature engineering pipeline:
- Invoice feature extraction
- Entity feature extraction
- Graph building
- Complete feature engineering pipeline

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import unittest

from config.config import get_config, DataGenerationConfig
from src.data_generation import EntityGenerator, InvoiceGenerator
from src.feature_engineering import (
    FeatureEngineer,
    InvoiceFeatureExtractor,
    EntityFeatureExtractor,
    TransactionGraphBuilder,
    engineer_features
)


class TestInvoiceFeatureExtractor(unittest.TestCase):
    """Test invoice-level feature extraction."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

    def test_extractor_initialization(self):
        """Test extractor can be initialized."""
        extractor = InvoiceFeatureExtractor()
        self.assertIsNotNone(extractor)
        self.assertIsNotNone(extractor.config)

    def test_extract_features(self):
        """Test feature extraction produces expected columns."""
        extractor = InvoiceFeatureExtractor()
        features_df = extractor.extract(self.invoices_df)

        # Check expected features exist
        expected_features = ['amount_log', 'amount_sqrt', 'is_round_amount', 'is_month_end']
        for feat in expected_features:
            self.assertIn(feat, features_df.columns, f"Missing feature: {feat}")

    def test_amount_log_positive(self):
        """Test log transform produces non-negative values."""
        extractor = InvoiceFeatureExtractor()
        features_df = extractor.extract(self.invoices_df)

        self.assertTrue((features_df['amount_log'] >= 0).all())
        self.assertFalse(features_df['amount_log'].isnull().any())

    def test_pattern_features_binary(self):
        """Test pattern features are binary (0 or 1)."""
        extractor = InvoiceFeatureExtractor()
        features_df = extractor.extract(self.invoices_df)

        binary_features = ['is_round_amount', 'is_month_end', 'is_quarter_end']
        for feat in binary_features:
            if feat in features_df.columns:
                unique_vals = features_df[feat].unique()
                self.assertTrue(set(unique_vals).issubset({0, 1}),
                                f"{feat} should be binary")

    def test_extract_with_metadata(self):
        """Test extraction with metadata."""
        extractor = InvoiceFeatureExtractor()
        result = extractor.extract_with_metadata(self.invoices_df)

        self.assertEqual(result.n_records, len(self.invoices_df))
        self.assertGreater(result.n_features, 0)
        self.assertIsInstance(result.feature_names, list)


class TestEntityFeatureExtractor(unittest.TestCase):
    """Test entity-level feature extraction."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

    def test_extractor_initialization(self):
        """Test extractor can be initialized."""
        extractor = EntityFeatureExtractor()
        self.assertIsNotNone(extractor)

    def test_extract_returns_three_dataframes(self):
        """Test extraction returns buyer, supplier, and relationship features."""
        extractor = EntityFeatureExtractor()
        buyer_feats, supplier_feats, rel_feats = extractor.extract(
            self.entities_df, self.invoices_df
        )

        self.assertIsInstance(buyer_feats, pd.DataFrame)
        self.assertIsInstance(supplier_feats, pd.DataFrame)
        self.assertIsInstance(rel_feats, pd.DataFrame)

    def test_buyer_features_columns(self):
        """Test buyer features have expected columns."""
        extractor = EntityFeatureExtractor()
        buyer_feats, _, _ = extractor.extract(self.entities_df, self.invoices_df)

        expected_cols = ['buyer_id', 'buyer_total_invoices', 'buyer_avg_invoice_amount']
        for col in expected_cols:
            self.assertIn(col, buyer_feats.columns, f"Missing: {col}")

    def test_supplier_features_columns(self):
        """Test supplier features have expected columns."""
        extractor = EntityFeatureExtractor()
        _, supplier_feats, _ = extractor.extract(self.entities_df, self.invoices_df)

        expected_cols = ['supplier_id', 'supplier_total_invoices', 'supplier_avg_invoice_amount']
        for col in expected_cols:
            self.assertIn(col, supplier_feats.columns, f"Missing: {col}")

    def test_relationship_features_columns(self):
        """Test relationship features have expected columns."""
        extractor = EntityFeatureExtractor()
        _, _, rel_feats = extractor.extract(self.entities_df, self.invoices_df)

        expected_cols = ['buyer_id', 'supplier_id', 'relationship_invoice_count']
        for col in expected_cols:
            self.assertIn(col, rel_feats.columns, f"Missing: {col}")

    def test_default_rate_in_range(self):
        """Test default rate is between 0 and 1."""
        extractor = EntityFeatureExtractor()
        buyer_feats, _, _ = extractor.extract(self.entities_df, self.invoices_df)

        if 'buyer_default_rate' in buyer_feats.columns:
            self.assertTrue((buyer_feats['buyer_default_rate'] >= 0).all())
            self.assertTrue((buyer_feats['buyer_default_rate'] <= 1).all())

    def test_extract_with_metadata(self):
        """Test extraction with metadata."""
        extractor = EntityFeatureExtractor()
        result = extractor.extract_with_metadata(self.entities_df, self.invoices_df)

        self.assertGreater(result.n_buyers, 0)
        self.assertGreater(result.n_suppliers, 0)
        self.assertGreater(result.n_relationships, 0)


class TestTransactionGraphBuilder(unittest.TestCase):
    """Test transaction graph building."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

    def test_builder_initialization(self):
        """Test builder can be initialized."""
        builder = TransactionGraphBuilder()
        self.assertIsNotNone(builder)

    def test_build_returns_result(self):
        """Test build returns GraphBuildResult."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.graph)
        self.assertIsNotNone(result.undirected_graph)

    def test_graph_has_nodes_and_edges(self):
        """Test graph has nodes and edges."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df)

        self.assertGreater(result.n_nodes, 0)
        self.assertGreater(result.n_edges, 0)

    def test_adjacency_matrix_shape(self):
        """Test adjacency matrix has correct shape."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df)

        expected_shape = (result.n_nodes, result.n_nodes)
        self.assertEqual(result.adjacency_matrix.shape, expected_shape)

    def test_modularity_matrix_computed(self):
        """Test modularity matrix is computed."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df, compute_modularity=True)

        self.assertIsNotNone(result.modularity_matrix)
        self.assertEqual(result.modularity_matrix.shape, result.adjacency_matrix.shape)

    def test_modularity_matrix_optional(self):
        """Test modularity matrix can be skipped."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df, compute_modularity=False)

        self.assertIsNone(result.modularity_matrix)

    def test_graph_statistics(self):
        """Test graph statistics computation."""
        builder = TransactionGraphBuilder()
        result = builder.build(self.invoices_df)
        stats = builder.get_graph_statistics(result)

        self.assertIn('n_nodes', stats)
        self.assertIn('n_edges', stats)
        self.assertIn('density', stats)
        self.assertIn('avg_degree', stats)

    def test_weight_by_amount(self):
        """Test graph building with amount weighting."""
        builder = TransactionGraphBuilder(weight_by='amount')
        result = builder.build(self.invoices_df)
        self.assertGreater(result.total_edge_weight, 0)

    def test_weight_by_count(self):
        """Test graph building with count weighting."""
        builder = TransactionGraphBuilder(weight_by='count')
        result = builder.build(self.invoices_df)
        self.assertGreater(result.total_edge_weight, 0)


class TestFeatureEngineer(unittest.TestCase):
    """Test main feature engineering pipeline."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

    def test_engineer_initialization(self):
        """Test feature engineer can be initialized."""
        engineer = FeatureEngineer()
        self.assertIsNotNone(engineer)
        self.assertFalse(engineer._is_fitted)

    def test_fit_transform_returns_result(self):
        """Test fit_transform returns FeatureEngineeringResult."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.features_df)
        self.assertTrue(engineer._is_fitted)

    def test_result_has_expected_fields(self):
        """Test result has all expected fields."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)

        self.assertIsInstance(result.features_df, pd.DataFrame)
        self.assertIsInstance(result.feature_names, list)
        self.assertGreater(result.n_samples, 0)
        self.assertGreater(result.n_features, 0)
        self.assertIsInstance(result.feature_statistics, dict)

    def test_model_features_present(self):
        """Test MODEL_FEATURES are present in output."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)

        # At least some model features should be present
        present_features = [f for f in engineer.MODEL_FEATURES
                           if f in result.features_df.columns]
        self.assertGreater(len(present_features), 5)

    def test_graph_result_included(self):
        """Test graph result is included when requested."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(
            self.entities_df, self.invoices_df, build_graph=True
        )

        self.assertIsNotNone(result.graph_result)
        self.assertGreater(result.graph_result.n_nodes, 0)

    def test_graph_result_optional(self):
        """Test graph building can be skipped."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(
            self.entities_df, self.invoices_df, build_graph=False
        )

        self.assertIsNone(result.graph_result)

    def test_get_feature_matrix(self):
        """Test feature matrix extraction."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)
        X, y = engineer.get_feature_matrix(result)

        self.assertEqual(X.shape[0], result.n_samples)
        self.assertEqual(X.shape[1], result.n_features)
        self.assertEqual(len(y), result.n_samples)

    def test_target_is_binary(self):
        """Test target variable is binary."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)
        _, y = engineer.get_feature_matrix(result)

        unique_vals = set(np.unique(y))
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_no_null_in_features(self):
        """Test features have no null values after processing."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)
        X, _ = engineer.get_feature_matrix(result)

        self.assertFalse(np.isnan(X).any())

    def test_transform_without_fit_raises(self):
        """Test transform before fit raises error."""
        engineer = FeatureEngineer()

        with self.assertRaises(ValueError):
            engineer.transform(self.entities_df, self.invoices_df)

    def test_convenience_function(self):
        """Test engineer_features convenience function."""
        result = engineer_features(self.entities_df, self.invoices_df)

        self.assertIsNotNone(result)
        self.assertGreater(result.n_samples, 0)

    def test_feature_statistics_computed(self):
        """Test feature statistics are computed."""
        engineer = FeatureEngineer()
        result = engineer.fit_transform(self.entities_df, self.invoices_df)

        self.assertGreater(len(result.feature_statistics), 0)

        # Check structure of statistics
        for feat, stats in result.feature_statistics.items():
            self.assertIn('mean', stats)
            self.assertIn('std', stats)
            self.assertIn('min', stats)
            self.assertIn('max', stats)


class TestFeatureEngineeringIntegration(unittest.TestCase):
    """Integration tests for complete feature engineering pipeline."""

    def test_full_pipeline_small_data(self):
        """Test full pipeline with small dataset."""
        # Generate small dataset
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        # Run full pipeline
        engineer = FeatureEngineer()
        result = engineer.fit_transform(entities_df, invoices_df)

        # Basic sanity checks
        self.assertGreater(result.n_samples, 100)
        self.assertGreater(result.n_features, 5)
        self.assertIsNotNone(result.graph_result)

    def test_pipeline_preserves_invoice_ids(self):
        """Test pipeline preserves invoice IDs."""
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(entities_df, invoices_df)

        self.assertIn('invoice_id', result.features_df.columns)
        self.assertEqual(len(result.features_df), len(invoices_df))

    def test_default_rate_variance(self):
        """Test that default rate shows variance (ring vs legitimate)."""
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(entities_df, invoices_df)

        _, y = engineer.get_feature_matrix(result)

        # Should have both defaulted and non-defaulted
        default_rate = y.mean()
        self.assertGreater(default_rate, 0.01, "Should have some defaults")
        self.assertLess(default_rate, 0.50, "Should not be mostly defaults")


def run_validation():
    """Run comprehensive validation of feature engineering."""
    print("=" * 70)
    print("FEATURE ENGINEERING VALIDATION")
    print("=" * 70)

    # Generate data
    print("\n[1/5] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"      Entities: {len(entities_df)}")
    print(f"      Invoices: {len(invoices_df)}")

    # Test invoice features
    print("\n[2/5] Testing invoice feature extraction...")
    invoice_extractor = InvoiceFeatureExtractor()
    invoice_result = invoice_extractor.extract_with_metadata(invoices_df)
    print(f"      Features: {invoice_result.n_features}")
    print(f"      Records: {invoice_result.n_records}")
    print(f"      Feature names: {invoice_result.feature_names[:5]}...")

    # Test entity features
    print("\n[3/5] Testing entity feature extraction...")
    entity_extractor = EntityFeatureExtractor()
    entity_result = entity_extractor.extract_with_metadata(entities_df, invoices_df)
    print(f"      Buyers: {entity_result.n_buyers}")
    print(f"      Suppliers: {entity_result.n_suppliers}")
    print(f"      Relationships: {entity_result.n_relationships}")

    # Test graph building
    print("\n[4/5] Testing graph building...")
    graph_builder = TransactionGraphBuilder()
    graph_result = graph_builder.build(invoices_df)
    graph_stats = graph_builder.get_graph_statistics(graph_result)
    print(f"      Nodes: {graph_result.n_nodes}")
    print(f"      Edges: {graph_result.n_edges}")
    print(f"      Density: {graph_stats['density']:.4f}")
    print(f"      Avg degree: {graph_stats['avg_degree']:.2f}")
    print(f"      Components: {graph_stats['n_components']}")

    # Test full pipeline
    print("\n[5/5] Testing complete feature engineering pipeline...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df)

    print(f"      Samples: {result.n_samples}")
    print(f"      Features: {result.n_features}")
    print(f"      Feature names: {result.feature_names}")

    # Get feature matrix
    X, y = engineer.get_feature_matrix(result)
    print(f"\n      Feature matrix shape: {X.shape}")
    print(f"      Target vector shape: {y.shape}")
    print(f"      Default rate: {y.mean():.2%}")
    print(f"      Null values in X: {np.isnan(X).sum()}")

    # Feature statistics summary
    print("\n      Feature Statistics (sample):")
    for i, (feat, stats) in enumerate(list(result.feature_statistics.items())[:5]):
        print(f"        {feat}:")
        print(f"          mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"          range=[{stats['min']:.4f}, {stats['max']:.4f}]")

    # Validation summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    errors = []
    warnings = []

    # Check feature count
    if result.n_features < 10:
        errors.append(f"Too few features: {result.n_features} (expected >= 10)")

    # Check null values
    if np.isnan(X).any():
        errors.append("Feature matrix contains null values")

    # Check default rate
    if y.mean() < 0.01:
        warnings.append(f"Very low default rate: {y.mean():.2%}")
    elif y.mean() > 0.50:
        warnings.append(f"Very high default rate: {y.mean():.2%}")

    # Check graph connectivity
    if graph_stats['n_components'] > graph_result.n_nodes // 2:
        warnings.append(f"Graph is highly fragmented: {graph_stats['n_components']} components")

    # Check modularity matrix
    if graph_result.modularity_matrix is None:
        errors.append("Modularity matrix not computed")
    else:
        if graph_result.modularity_matrix.shape != graph_result.adjacency_matrix.shape:
            errors.append("Modularity matrix shape mismatch")

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
    print("FEATURE ENGINEERING VALIDATION COMPLETE")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Feature Engineering Tests")
    parser.add_argument("--validate", action="store_true", help="Run validation only")
    parser.add_argument("--unittest", action="store_true", help="Run unit tests")
    args = parser.parse_args()

    if args.unittest:
        unittest.main(argv=[''], exit=False, verbosity=2)
    elif args.validate:
        success = run_validation()
        exit(0 if success else 1)
    else:
        # Default: run validation
        success = run_validation()
        exit(0 if success else 1)
