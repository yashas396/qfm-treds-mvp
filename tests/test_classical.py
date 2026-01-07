"""
QGAI Quantum Financial Modeling - TReDS MVP
Classical Model Test Suite

This module tests the classical default prediction model:
- DefaultPredictor training and evaluation
- ModelTrainer pipeline
- Model persistence
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
import tempfile
import os

from src.data_generation import EntityGenerator, InvoiceGenerator
from src.feature_engineering import FeatureEngineer
from src.classical import (
    DefaultPredictor,
    ModelTrainer,
    train_default_predictor,
    train_and_evaluate
)


class TestDefaultPredictor(unittest.TestCase):
    """Test DefaultPredictor class."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(cls.entities_df, cls.invoices_df, build_graph=False)

        cls.X, cls.y = engineer.get_feature_matrix(result)
        cls.feature_names = result.feature_names

    def test_predictor_initialization(self):
        """Test predictor can be initialized."""
        predictor = DefaultPredictor()
        self.assertIsNotNone(predictor)
        self.assertFalse(predictor._is_fitted)

    def test_fit_returns_result(self):
        """Test fit returns ModelTrainingResult."""
        predictor = DefaultPredictor()
        result = predictor.fit(self.X, self.y, self.feature_names)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.model)
        self.assertTrue(predictor._is_fitted)

    def test_cross_validation_scores(self):
        """Test cross-validation is performed."""
        predictor = DefaultPredictor()
        result = predictor.fit(self.X, self.y, self.feature_names, validate=True)

        self.assertIsNotNone(result.evaluation.cv_scores)
        self.assertIsNotNone(result.evaluation.cv_mean)
        self.assertEqual(len(result.evaluation.cv_scores), 5)

    def test_predict_proba(self):
        """Test probability predictions."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        proba = predictor.predict_proba(self.X)

        self.assertEqual(len(proba), len(self.X))
        self.assertTrue((proba >= 0).all())
        self.assertTrue((proba <= 1).all())

    def test_predict_binary(self):
        """Test binary predictions."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        result = predictor.predict(self.X)

        self.assertEqual(len(result.predictions), len(self.X))
        self.assertTrue(set(result.predictions).issubset({0, 1}))

    def test_feature_importances(self):
        """Test feature importances extraction."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        importances = predictor.get_feature_importances()

        self.assertEqual(len(importances), len(self.feature_names))
        self.assertTrue(all(v >= 0 for v in importances.values()))

    def test_top_features(self):
        """Test top features extraction."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        top5 = predictor.get_top_features(5)

        self.assertEqual(len(top5), 5)
        # Should be sorted by importance (descending)
        importances = [imp for _, imp in top5]
        self.assertEqual(importances, sorted(importances, reverse=True))

    def test_evaluate(self):
        """Test evaluation method."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        evaluation = predictor.evaluate(self.X, self.y)

        self.assertIsNotNone(evaluation)
        self.assertGreater(evaluation.auc_roc, 0)
        self.assertIsNotNone(evaluation.confusion_matrix)

    def test_calibration(self):
        """Test probability calibration."""
        predictor = DefaultPredictor()
        result = predictor.fit(self.X, self.y, self.feature_names, calibrate=True)

        self.assertTrue(result.is_calibrated)
        self.assertIsNotNone(result.calibrated_model)

    def test_save_load_model(self):
        """Test model persistence."""
        predictor = DefaultPredictor()
        predictor.fit(self.X, self.y, self.feature_names)

        original_proba = predictor.predict_proba(self.X[:10])

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            filepath = f.name

        try:
            predictor.save_model(filepath)

            new_predictor = DefaultPredictor()
            new_predictor.load_model(filepath)

            loaded_proba = new_predictor.predict_proba(self.X[:10])

            np.testing.assert_array_almost_equal(original_proba, loaded_proba)
        finally:
            os.unlink(filepath)

    def test_predict_before_fit_raises(self):
        """Test that predicting before fit raises error."""
        predictor = DefaultPredictor()

        with self.assertRaises(ValueError):
            predictor.predict_proba(self.X)


class TestModelTrainer(unittest.TestCase):
    """Test ModelTrainer class."""

    @classmethod
    def setUpClass(cls):
        """Generate test data once for all tests."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(cls.entities_df, cls.invoices_df, build_graph=False)

        cls.X, cls.y = engineer.get_feature_matrix(result)
        cls.feature_names = result.feature_names

    def test_trainer_initialization(self):
        """Test trainer can be initialized."""
        trainer = ModelTrainer()
        self.assertIsNotNone(trainer)

    def test_data_split(self):
        """Test data splitting."""
        trainer = ModelTrainer()
        data_split = trainer.split_data(self.X, self.y, self.feature_names)

        self.assertIsNotNone(data_split)
        total = data_split.train_size + data_split.val_size + data_split.test_size
        self.assertEqual(total, len(self.X))

    def test_stratified_split(self):
        """Test that splits are stratified."""
        trainer = ModelTrainer()
        data_split = trainer.split_data(self.X, self.y, self.feature_names)

        original_rate = self.y.mean()
        train_rate = data_split.y_train.mean()
        test_rate = data_split.y_test.mean()

        # Rates should be similar (within 5 percentage points)
        self.assertAlmostEqual(train_rate, original_rate, delta=0.05)
        self.assertAlmostEqual(test_rate, original_rate, delta=0.05)

    def test_train_pipeline(self):
        """Test complete training pipeline."""
        trainer = ModelTrainer()
        result = trainer.train(self.X, self.y, self.feature_names)

        self.assertIsNotNone(result)
        self.assertIsNotNone(result.model)
        self.assertIsNotNone(result.training_result)
        self.assertIsNotNone(result.test_evaluation)

    def test_test_evaluation(self):
        """Test evaluation on test set."""
        trainer = ModelTrainer()
        result = trainer.train(self.X, self.y, self.feature_names)

        # Test evaluation should have all metrics
        self.assertIsNotNone(result.test_evaluation.auc_roc)
        self.assertIsNotNone(result.test_evaluation.precision)
        self.assertIsNotNone(result.test_evaluation.recall)
        self.assertIsNotNone(result.test_evaluation.f1)


class TestMVPCriteria(unittest.TestCase):
    """Test that model meets MVP criteria."""

    @classmethod
    def setUpClass(cls):
        """Generate test data and train model."""
        entity_gen = EntityGenerator()
        cls.entities_df = entity_gen.generate()

        invoice_gen = InvoiceGenerator()
        cls.invoices_df = invoice_gen.generate(cls.entities_df)

        engineer = FeatureEngineer()
        result = engineer.fit_transform(cls.entities_df, cls.invoices_df, build_graph=False)

        X, y = engineer.get_feature_matrix(result)

        trainer = ModelTrainer()
        cls.result = trainer.train(X, y, result.feature_names)

    def test_auc_roc_target(self):
        """Test that AUC-ROC meets target >= 0.75."""
        target_auc = 0.75
        actual_auc = self.result.test_evaluation.auc_roc

        self.assertGreaterEqual(
            actual_auc,
            target_auc,
            f"AUC-ROC {actual_auc:.4f} is below target {target_auc}"
        )

    def test_model_generalizes(self):
        """Test that model generalizes (CV scores close to test)."""
        cv_mean = self.result.training_result.evaluation.cv_mean
        test_auc = self.result.test_evaluation.auc_roc

        # Should be within 0.1 of each other (not overfitting)
        self.assertAlmostEqual(cv_mean, test_auc, delta=0.1)


def run_validation():
    """Run comprehensive validation of classical model."""
    print("=" * 70)
    print("CLASSICAL MODEL VALIDATION")
    print("=" * 70)

    # Generate data
    print("\n[1/5] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"      Entities: {len(entities_df)}")
    print(f"      Invoices: {len(invoices_df)}")

    # Feature engineering
    print("\n[2/5] Engineering features...")
    engineer = FeatureEngineer()
    fe_result = engineer.fit_transform(entities_df, invoices_df, build_graph=False)

    X, y = engineer.get_feature_matrix(fe_result)
    print(f"      Features: {X.shape[1]}")
    print(f"      Samples: {X.shape[0]}")
    print(f"      Default rate: {y.mean():.2%}")

    # Train model
    print("\n[3/5] Training model...")
    trainer = ModelTrainer()
    result = trainer.train(X, y, fe_result.feature_names)

    print(f"\n      Data split:")
    print(f"        Train: {result.data_split.train_size}")
    print(f"        Val:   {result.data_split.val_size}")
    print(f"        Test:  {result.data_split.test_size}")

    # Cross-validation results
    print(f"\n[4/5] Cross-validation results:")
    print(f"      CV AUC-ROC: {result.training_result.evaluation.cv_mean:.4f} (+/- {result.training_result.evaluation.cv_std:.4f})")
    print(f"      CV Scores: {[f'{s:.4f}' for s in result.training_result.evaluation.cv_scores]}")

    # Test set evaluation
    print(f"\n[5/5] Test set evaluation:")
    print(f"      AUC-ROC:          {result.test_evaluation.auc_roc:.4f}")
    print(f"      Precision:        {result.test_evaluation.precision:.4f}")
    print(f"      Recall:           {result.test_evaluation.recall:.4f}")
    print(f"      F1 Score:         {result.test_evaluation.f1:.4f}")
    print(f"      Avg Precision:    {result.test_evaluation.average_precision:.4f}")

    print(f"\n      Confusion Matrix:")
    cm = result.test_evaluation.confusion_matrix
    print(f"        TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
    print(f"        FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    print(f"\n      Top 5 Feature Importances:")
    for feat, imp in result.model.get_top_features(5):
        print(f"        {feat}: {imp:.4f}")

    # MVP Criteria Check
    print("\n" + "=" * 70)
    print("MVP CRITERIA CHECK")
    print("=" * 70)

    errors = []
    warnings = []

    # AUC-ROC check
    target_auc = 0.75
    actual_auc = result.test_evaluation.auc_roc
    if actual_auc >= target_auc:
        print(f"\n  [PASS] AUC-ROC >= {target_auc}: {actual_auc:.4f}")
    else:
        errors.append(f"AUC-ROC {actual_auc:.4f} is below target {target_auc}")
        print(f"\n  [FAIL] AUC-ROC >= {target_auc}: {actual_auc:.4f}")

    # Overfitting check
    cv_mean = result.training_result.evaluation.cv_mean
    if abs(cv_mean - actual_auc) > 0.1:
        warnings.append(f"Possible overfitting: CV={cv_mean:.4f} vs Test={actual_auc:.4f}")
        print(f"  [WARN] Generalization gap: CV={cv_mean:.4f} vs Test={actual_auc:.4f}")
    else:
        print(f"  [PASS] Good generalization: CV={cv_mean:.4f} vs Test={actual_auc:.4f}")

    # Feature importance sanity
    top_features = result.model.get_top_features(3)
    expected_important = ['buyer_default_rate', 'buyer_credit_rating_encoded', 'acceptance_delay_days']
    top_feature_names = [f for f, _ in top_features]

    important_found = any(f in top_feature_names for f in expected_important)
    if important_found:
        print(f"  [PASS] Expected important features in top 3: {top_feature_names}")
    else:
        warnings.append(f"Expected features not in top 3: {top_feature_names}")
        print(f"  [WARN] Expected features not in top 3: {top_feature_names}")

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
    print("CLASSICAL MODEL VALIDATION COMPLETE")
    print("=" * 70)

    return len(errors) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classical Model Tests")
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
