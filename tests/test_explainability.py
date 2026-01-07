"""
QGAI Quantum Financial Modeling - TReDS MVP
Explainability Test Suite

This module tests the explainability framework:
- SHAP explainer
- Global and local explanations
- Human-readable explanation generation

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
from src.explainability import (
    SHAPExplainer,
    ExplanationResult,
    explain_predictions
)


class TestSHAPExplainer(unittest.TestCase):
    """Test SHAPExplainer class."""

    @classmethod
    def setUpClass(cls):
        """Generate test data and train model."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(cls.entities_df, cls.invoices_df, build_graph=False)

        cls.X, cls.y = engineer.get_feature_matrix(result)
        cls.feature_names = result.feature_names

        predictor = DefaultPredictor()
        predictor.fit(cls.X, cls.y, cls.feature_names)
        cls.model = predictor.model

    def test_explainer_initialization(self):
        """Test explainer can be initialized."""
        explainer = SHAPExplainer()
        self.assertIsNotNone(explainer)
        self.assertFalse(explainer._is_fitted)

    def test_fit_explainer(self):
        """Test explainer can be fitted."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        self.assertTrue(explainer._is_fitted)
        self.assertIsNotNone(explainer.explainer)

    def test_explain_returns_result(self):
        """Test explain returns ExplanationResult."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:50])

        self.assertIsNotNone(result)
        self.assertIsInstance(result, ExplanationResult)

    def test_shap_values_shape(self):
        """Test SHAP values have correct shape."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        n_samples = 50
        result = explainer.explain(self.X[:n_samples])

        self.assertEqual(result.shap_values.shape[0], n_samples)
        self.assertEqual(result.shap_values.shape[1], len(self.feature_names))

    def test_global_explanation(self):
        """Test global explanation generation."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:50])

        self.assertIsNotNone(result.global_explanation)
        self.assertEqual(
            len(result.global_explanation.feature_importances),
            len(self.feature_names)
        )
        self.assertEqual(
            len(result.global_explanation.feature_ranking),
            len(self.feature_names)
        )

    def test_local_explanations(self):
        """Test local explanation generation."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        n_samples = 50
        result = explainer.explain(self.X[:n_samples], generate_local=True)

        self.assertIsNotNone(result.local_explanations)
        self.assertEqual(len(result.local_explanations), n_samples)

    def test_local_explanation_structure(self):
        """Test local explanation has required fields."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:10])
        local = result.local_explanations[0]

        self.assertIsNotNone(local.invoice_id)
        self.assertIsNotNone(local.prediction)
        self.assertIsNotNone(local.contributions)
        self.assertIsNotNone(local.explanation_text)

    def test_explanation_text_not_empty(self):
        """Test explanation text is generated."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:10])
        local = result.local_explanations[0]

        self.assertGreater(len(local.explanation_text), 0)

    def test_feature_importance_df(self):
        """Test feature importance DataFrame generation."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:50])
        df = explainer.get_feature_importance_df(result)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('feature', df.columns)
        self.assertIn('importance', df.columns)

    def test_explanation_summary(self):
        """Test explanation summary generation."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:50])
        summary = explainer.get_explanation_summary(result)

        self.assertIn('n_samples_explained', summary)
        self.assertIn('top_features', summary)

    def test_explain_before_fit_raises(self):
        """Test explain before fit raises error."""
        explainer = SHAPExplainer()

        with self.assertRaises(ValueError):
            explainer.explain(self.X[:10])


class TestExplainabilityIntegration(unittest.TestCase):
    """Integration tests for explainability."""

    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(cls.entities_df, cls.invoices_df, build_graph=False)

        cls.X, cls.y = engineer.get_feature_matrix(result)
        cls.feature_names = result.feature_names

        predictor = DefaultPredictor()
        predictor.fit(cls.X, cls.y, cls.feature_names)
        cls.model = predictor.model

    def test_convenience_function(self):
        """Test explain_predictions convenience function."""
        result = explain_predictions(
            self.model,
            self.X[:50],
            self.feature_names
        )

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.global_explanation)

    def test_all_features_explained(self):
        """Test all features are in explanation."""
        explainer = SHAPExplainer()
        explainer.fit(self.model, self.feature_names)

        result = explainer.explain(self.X[:50])

        explained_features = set(result.global_explanation.feature_importances.keys())
        expected_features = set(self.feature_names)

        self.assertEqual(explained_features, expected_features)


def run_validation():
    """Run comprehensive validation of explainability."""
    print("=" * 70)
    print("EXPLAINABILITY VALIDATION")
    print("=" * 70)

    # Generate data
    print("\n[1/4] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"      Invoices: {len(invoices_df)}")

    # Feature engineering
    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df, build_graph=False)
    X, y = engineer.get_feature_matrix(result)

    print(f"      Features: {X.shape[1]}")

    # Train model
    print("\n[3/4] Training model...")
    predictor = DefaultPredictor()
    predictor.fit(X, y, result.feature_names)

    # Generate explanations
    print("\n[4/4] Generating SHAP explanations...")
    explainer = SHAPExplainer()
    explainer.fit(predictor.model, result.feature_names)

    n_explain = min(100, len(X))
    explanation_result = explainer.explain(X[:n_explain])

    print(f"\n      Samples explained: {explanation_result.n_samples}")
    print(f"      Expected value (base rate): {explanation_result.expected_value:.4f}")

    print(f"\n      Top 5 Important Features (Global):")
    for feat, imp in explanation_result.global_explanation.feature_ranking[:5]:
        direction = "increases" if explanation_result.global_explanation.mean_shap_values[feat] > 0 else "decreases"
        print(f"        {feat}: {imp:.4f} ({direction} risk)")

    # Sample local explanations
    print(f"\n      Sample Local Explanations:")
    for i in [0, 1, 2]:
        if i < len(explanation_result.local_explanations):
            local = explanation_result.local_explanations[i]
            print(f"\n      --- Explanation {i+1} ---")
            print(f"      Invoice: {local.invoice_id}")
            print(f"      Prediction: {local.prediction:.4f}")
            print(f"      Risk Factors: {local.top_risk_factors[:2]}")
            print(f"      Protective Factors: {local.top_protective_factors[:2]}")

    # Feature importance DataFrame
    importance_df = explainer.get_feature_importance_df(explanation_result)
    print(f"\n      Feature Importance DataFrame:")
    print(importance_df.head().to_string())

    # MVP Criteria Check
    print("\n" + "=" * 70)
    print("MVP CRITERIA CHECK")
    print("=" * 70)

    errors = []
    warnings = []

    # All features explained
    n_explained = len(explanation_result.global_explanation.feature_importances)
    n_features = len(result.feature_names)
    if n_explained == n_features:
        print(f"\n  [PASS] All features explained: {n_explained}/{n_features}")
    else:
        errors.append(f"Missing feature explanations: {n_features - n_explained}")
        print(f"\n  [FAIL] Missing feature explanations: {n_features - n_explained}")

    # Local explanations generated
    if explanation_result.local_explanations:
        print(f"  [PASS] Local explanations generated: {len(explanation_result.local_explanations)}")
    else:
        errors.append("Local explanations not generated")
        print(f"  [FAIL] Local explanations not generated")

    # Explanation text generated
    if explanation_result.local_explanations and explanation_result.local_explanations[0].explanation_text:
        print(f"  [PASS] Human-readable explanations generated")
    else:
        warnings.append("Explanation text is empty")
        print(f"  [WARN] Explanation text is empty")

    # SHAP values computed
    if explanation_result.shap_values is not None:
        print(f"  [PASS] SHAP values computed: shape {explanation_result.shap_values.shape}")
    else:
        errors.append("SHAP values not computed")
        print(f"  [FAIL] SHAP values not computed")

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
    print("EXPLAINABILITY VALIDATION COMPLETE")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explainability Tests")
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
