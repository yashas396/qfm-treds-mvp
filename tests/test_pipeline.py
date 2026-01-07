"""
QGAI Quantum Financial Modeling - TReDS MVP
Pipeline Integration Test Suite

This module tests the hybrid pipeline integration:
- RiskScorer
- HybridPipeline
- End-to-end execution
- MVP criteria validation

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

from src.data_generation import EntityGenerator, InvoiceGenerator
from src.feature_engineering import FeatureEngineer
from src.classical import DefaultPredictor
from src.quantum import RingDetector
from src.pipeline import (
    RiskScorer,
    HybridPipeline,
    run_hybrid_pipeline
)


class TestRiskScorer(unittest.TestCase):
    """Test RiskScorer class."""

    def setUp(self):
        """Set up test data."""
        # Create simple test data
        self.invoices_df = pd.DataFrame({
            'invoice_id': ['INV001', 'INV002', 'INV003'],
            'buyer_id': ['B001', 'B002', 'B001'],
            'supplier_id': ['S001', 'S002', 'S003'],
            'amount': [100000, 200000, 150000]
        })

        self.default_probs = np.array([0.1, 0.8, 0.3])
        self.ring_scores = {
            'B001': 0.7,
            'B002': 0.1,
            'S001': 0.5,
            'S002': 0.2,
            'S003': 0.3
        }

    def test_scorer_initialization(self):
        """Test scorer can be initialized."""
        scorer = RiskScorer()
        self.assertIsNotNone(scorer)

    def test_score_returns_result(self):
        """Test scoring returns RiskScoringResult."""
        scorer = RiskScorer()
        result = scorer.score(self.invoices_df, self.default_probs, self.ring_scores)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.scores)
        self.assertEqual(len(result.scores), 3)

    def test_composite_scores_in_range(self):
        """Test composite scores are in [0, 1]."""
        scorer = RiskScorer()
        result = scorer.score(self.invoices_df, self.default_probs, self.ring_scores)

        for score in result.scores:
            self.assertGreaterEqual(score.composite_score, 0.0)
            self.assertLessEqual(score.composite_score, 1.0)

    def test_risk_levels_assigned(self):
        """Test risk levels are assigned."""
        scorer = RiskScorer()
        result = scorer.score(self.invoices_df, self.default_probs, self.ring_scores)

        valid_levels = {'CRITICAL', 'HIGH', 'MEDIUM', 'LOW'}
        for score in result.scores:
            self.assertIn(score.risk_level, valid_levels)

    def test_targeting_list(self):
        """Test targeting list generation."""
        scorer = RiskScorer()
        result = scorer.score(self.invoices_df, self.default_probs, self.ring_scores)

        targeting = scorer.get_targeting_list(result, top_n=2)

        self.assertEqual(len(targeting), 2)
        # Should be sorted by composite score
        self.assertGreaterEqual(
            targeting.iloc[0]['composite_score'],
            targeting.iloc[1]['composite_score']
        )

    def test_risk_summary(self):
        """Test risk summary generation."""
        scorer = RiskScorer()
        result = scorer.score(self.invoices_df, self.default_probs, self.ring_scores)

        summary = scorer.get_risk_summary(result)

        self.assertIn('total_invoices', summary)
        self.assertIn('n_high_risk', summary)
        self.assertIn('avg_composite_score', summary)


class TestHybridPipeline(unittest.TestCase):
    """Test HybridPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline can be initialized."""
        pipeline = HybridPipeline()
        self.assertIsNotNone(pipeline)

    def test_pipeline_run(self):
        """Test pipeline can run end-to-end."""
        pipeline = HybridPipeline(k_communities=3)
        result = pipeline.run(verbose=False)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.entities_df)
        self.assertIsNotNone(result.invoices_df)
        self.assertIsNotNone(result.classical_result)
        self.assertIsNotNone(result.quantum_result)
        self.assertIsNotNone(result.risk_result)

    def test_pipeline_with_provided_data(self):
        """Test pipeline with provided data."""
        # Generate data externally
        entity_gen = EntityGenerator()
        entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        invoices_df = invoice_gen.generate(entities_df)

        pipeline = HybridPipeline(k_communities=3)
        result = pipeline.run(
            entities_df=entities_df,
            invoices_df=invoices_df,
            generate_data=False,
            verbose=False
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result.entities_df), len(entities_df))

    def test_stage_summary(self):
        """Test stage summary generation."""
        pipeline = HybridPipeline(k_communities=3)
        result = pipeline.run(verbose=False)

        summary = pipeline.get_stage_summary()

        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn('stage', summary.columns)
        self.assertIn('status', summary.columns)


class TestMVPCriteria(unittest.TestCase):
    """Test that pipeline meets MVP criteria."""

    @classmethod
    def setUpClass(cls):
        """Run pipeline once for all tests."""
        pipeline = HybridPipeline(k_communities=5)
        cls.result = pipeline.run(verbose=False)
        cls.pipeline = pipeline

    def test_classical_auc_target(self):
        """Test that classical AUC-ROC meets target >= 0.75."""
        target = 0.75
        actual = self.result.classical_result.test_evaluation.auc_roc

        self.assertGreaterEqual(
            actual, target,
            f"AUC-ROC {actual:.4f} is below target {target}"
        )

    def test_pipeline_completes(self):
        """Test that pipeline completes all stages."""
        summary = self.pipeline.get_stage_summary()
        all_completed = (summary['status'] == 'completed').all()

        self.assertTrue(all_completed, "Not all pipeline stages completed")

    def test_risk_scores_generated(self):
        """Test that risk scores are generated for all invoices."""
        n_invoices = len(self.result.invoices_df)
        n_scores = len(self.result.risk_result.scores)

        self.assertEqual(n_invoices, n_scores)


def run_validation():
    """Run comprehensive validation of hybrid pipeline."""
    print("=" * 70)
    print("HYBRID PIPELINE VALIDATION")
    print("=" * 70)

    # Run pipeline
    print("\n[1/3] Running hybrid pipeline...")
    pipeline = HybridPipeline(k_communities=5)
    result = pipeline.run(verbose=True)

    # Stage summary
    print("\n[2/3] Stage Summary:")
    summary = pipeline.get_stage_summary()
    for _, row in summary.iterrows():
        runtime = f"{row['runtime_seconds']:.2f}s" if row['runtime_seconds'] else "N/A"
        print(f"   {row['stage']}: {row['status']} ({runtime})")

    # Risk summary
    print("\n[3/3] Risk Summary:")
    risk_summary = pipeline.scorer.get_risk_summary(result.risk_result)

    print(f"   Total invoices: {risk_summary['total_invoices']}")
    print(f"   High risk: {risk_summary['n_high_risk']} ({risk_summary['pct_high_risk']:.2f}%)")
    print(f"   Ring associated: {risk_summary['n_ring_associated']} ({risk_summary['pct_ring_associated']:.2f}%)")
    print(f"   Avg composite score: {risk_summary['avg_composite_score']:.4f}")

    print(f"\n   Risk level distribution:")
    for level, count in risk_summary['risk_level_counts'].items():
        print(f"     {level}: {count}")

    # Top high-risk invoices
    targeting = pipeline.scorer.get_targeting_list(result.risk_result, top_n=5)
    print(f"\n   Top 5 high-risk invoices:")
    print(targeting[['invoice_id', 'default_probability', 'ring_score', 'composite_score', 'risk_level']].to_string(index=False))

    # MVP Criteria Check
    print("\n" + "=" * 70)
    print("MVP CRITERIA CHECK")
    print("=" * 70)

    errors = []
    warnings = []

    # Classical AUC-ROC
    target_auc = 0.75
    actual_auc = result.classical_result.test_evaluation.auc_roc
    if actual_auc >= target_auc:
        print(f"\n  [PASS] Classical AUC-ROC >= {target_auc}: {actual_auc:.4f}")
    else:
        errors.append(f"Classical AUC-ROC {actual_auc:.4f} is below target {target_auc}")
        print(f"\n  [FAIL] Classical AUC-ROC >= {target_auc}: {actual_auc:.4f}")

    # Pipeline completion
    all_completed = (summary['status'] == 'completed').all()
    if all_completed:
        print(f"  [PASS] All pipeline stages completed")
    else:
        errors.append("Some pipeline stages did not complete")
        print(f"  [FAIL] Some pipeline stages did not complete")

    # Risk scores generated
    n_invoices = len(result.invoices_df)
    n_scores = len(result.risk_result.scores)
    if n_invoices == n_scores:
        print(f"  [PASS] Risk scores generated for all invoices")
    else:
        errors.append(f"Missing risk scores: {n_invoices - n_scores} invoices")
        print(f"  [FAIL] Missing risk scores: {n_invoices - n_scores} invoices")

    # Runtime check
    if result.total_runtime_seconds < 300:  # 5 minutes
        print(f"  [PASS] Pipeline runtime: {result.total_runtime_seconds:.2f}s")
    else:
        warnings.append(f"Pipeline runtime ({result.total_runtime_seconds:.2f}s) is slow")
        print(f"  [WARN] Pipeline runtime: {result.total_runtime_seconds:.2f}s")

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
    print("HYBRID PIPELINE VALIDATION COMPLETE")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Integration Tests")
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
