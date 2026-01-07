"""
QGAI Quantum Financial Modeling - TReDS MVP
Default Predictor Module

This module implements the classical Random Forest model for invoice default prediction:
- Model training with class balancing
- Cross-validation evaluation
- Prediction with probability calibration
- Feature importance extraction

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
from sklearn.calibration import CalibratedClassifierCV
import joblib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import ClassicalModelConfig, get_config


@dataclass
class PredictionResult:
    """Result container for predictions."""
    predictions: np.ndarray
    probabilities: np.ndarray
    threshold: float
    n_samples: int


@dataclass
class EvaluationResult:
    """Result container for model evaluation."""
    auc_roc: float
    precision: float
    recall: float
    f1: float
    average_precision: float
    confusion_matrix: np.ndarray
    classification_report: str
    cv_scores: Optional[np.ndarray] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    feature_importances: Optional[Dict[str, float]] = None
    evaluation_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelTrainingResult:
    """Result container for model training."""
    model: RandomForestClassifier
    calibrated_model: Optional[CalibratedClassifierCV]
    evaluation: EvaluationResult
    feature_names: List[str]
    is_calibrated: bool
    training_timestamp: datetime = field(default_factory=datetime.now)


class DefaultPredictor:
    """
    Random Forest classifier for invoice default prediction.

    This class implements:
    - Model training with class weight balancing
    - Cross-validation for robust evaluation
    - Probability calibration (optional)
    - Feature importance extraction

    The model predicts P(default) for each invoice, which is used
    in the hybrid system alongside QUBO ring detection scores.

    Attributes:
        config: ClassicalModelConfig with model parameters
        model: Trained RandomForestClassifier
        calibrated_model: Calibrated version for better probabilities
        feature_names: List of feature names used in training

    Example:
        >>> predictor = DefaultPredictor()
        >>> result = predictor.fit(X_train, y_train, feature_names)
        >>> predictions = predictor.predict_proba(X_test)
    """

    def __init__(self, config: Optional[ClassicalModelConfig] = None):
        """
        Initialize DefaultPredictor.

        Args:
            config: ClassicalModelConfig instance. If None, uses default config.
        """
        self.config = config or get_config().classical
        self.model: Optional[RandomForestClassifier] = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_names: List[str] = []
        self._is_fitted = False
        self._is_calibrated = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        calibrate: bool = True,
        validate: bool = True
    ) -> ModelTrainingResult:
        """
        Train the Random Forest model.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            y: Target vector of shape (n_samples,)
            feature_names: List of feature names
            calibrate: Whether to calibrate probabilities
            validate: Whether to run cross-validation

        Returns:
            ModelTrainingResult: Training result with evaluation metrics
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight=self.config.class_weight,
            random_state=self.config.random_seed,
            n_jobs=-1
        )

        # Cross-validation (if requested)
        cv_scores = None
        if validate:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.config.random_seed)
            cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='roc_auc')

        # Fit the model
        self.model.fit(X, y)
        self._is_fitted = True

        # Calibrate probabilities (if requested)
        if calibrate:
            self.calibrated_model = CalibratedClassifierCV(
                self.model,
                method='isotonic',
                cv=3
            )
            self.calibrated_model.fit(X, y)
            self._is_calibrated = True

        # Evaluate on training data (for feature importance and baseline metrics)
        evaluation = self._evaluate(X, y, cv_scores)

        return ModelTrainingResult(
            model=self.model,
            calibrated_model=self.calibrated_model,
            evaluation=evaluation,
            feature_names=self.feature_names,
            is_calibrated=self._is_calibrated
        )

    def predict(
        self,
        X: np.ndarray,
        threshold: Optional[float] = None
    ) -> PredictionResult:
        """
        Make binary predictions.

        Args:
            X: Feature matrix of shape (n_samples, n_features)
            threshold: Classification threshold. If None, uses config default.

        Returns:
            PredictionResult: Predictions and probabilities
        """
        self._check_fitted()

        if threshold is None:
            threshold = self.config.default_threshold

        # Get probabilities
        probabilities = self.predict_proba(X)

        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)

        return PredictionResult(
            predictions=predictions,
            probabilities=probabilities,
            threshold=threshold,
            n_samples=len(X)
        )

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict default probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features)

        Returns:
            np.ndarray: Probability of default for each sample
        """
        self._check_fitted()

        if self._is_calibrated and self.calibrated_model is not None:
            # Use calibrated model for better probability estimates
            return self.calibrated_model.predict_proba(X)[:, 1]
        else:
            return self.model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        threshold: Optional[float] = None
    ) -> EvaluationResult:
        """
        Evaluate model on test data.

        Args:
            X: Feature matrix
            y: True labels
            threshold: Classification threshold

        Returns:
            EvaluationResult: Comprehensive evaluation metrics
        """
        self._check_fitted()

        if threshold is None:
            threshold = self.config.default_threshold

        return self._evaluate(X, y, threshold=threshold)

    def _evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_scores: Optional[np.ndarray] = None,
        threshold: Optional[float] = None
    ) -> EvaluationResult:
        """Internal evaluation method."""
        if threshold is None:
            threshold = self.config.default_threshold

        # Get predictions and probabilities
        proba = self.predict_proba(X)
        pred = (proba >= threshold).astype(int)

        # Calculate metrics
        auc_roc = roc_auc_score(y, proba)
        precision = precision_score(y, pred, zero_division=0)
        recall = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        avg_precision = average_precision_score(y, proba)

        # Confusion matrix
        cm = confusion_matrix(y, pred)

        # Classification report
        report = classification_report(y, pred, target_names=['No Default', 'Default'])

        # Feature importances
        feature_importances = None
        if self.model is not None:
            importances = self.model.feature_importances_
            feature_importances = dict(zip(self.feature_names, importances))

        return EvaluationResult(
            auc_roc=auc_roc,
            precision=precision,
            recall=recall,
            f1=f1,
            average_precision=avg_precision,
            confusion_matrix=cm,
            classification_report=report,
            cv_scores=cv_scores,
            cv_mean=cv_scores.mean() if cv_scores is not None else None,
            cv_std=cv_scores.std() if cv_scores is not None else None,
            feature_importances=feature_importances
        )

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get feature importances from the trained model.

        Returns:
            Dict mapping feature names to importance scores
        """
        self._check_fitted()

        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True
        ))

    def get_top_features(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N most important features.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_name, importance) tuples
        """
        importances = self.get_feature_importances()
        return list(importances.items())[:n]

    def find_optimal_threshold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric: str = 'f1'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Find optimal classification threshold.

        Args:
            X: Feature matrix
            y: True labels
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        self._check_fitted()

        proba = self.predict_proba(X)
        precisions, recalls, thresholds = precision_recall_curve(y, proba)

        # Calculate F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

        if metric == 'f1':
            best_idx = np.argmax(f1_scores)
        elif metric == 'precision':
            # Find highest precision with recall >= 0.3
            valid_idx = recalls >= 0.3
            if valid_idx.any():
                best_idx = np.where(valid_idx)[0][np.argmax(precisions[valid_idx])]
            else:
                best_idx = np.argmax(precisions)
        elif metric == 'recall':
            # Find highest recall with precision >= 0.3
            valid_idx = precisions >= 0.3
            if valid_idx.any():
                best_idx = np.where(valid_idx)[0][np.argmax(recalls[valid_idx])]
            else:
                best_idx = np.argmax(recalls)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # Handle edge case for last threshold
        if best_idx >= len(thresholds):
            best_idx = len(thresholds) - 1

        optimal_threshold = thresholds[best_idx]

        # Calculate metrics at optimal threshold
        pred = (proba >= optimal_threshold).astype(int)
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0)
        }

        return optimal_threshold, metrics

    def save_model(self, filepath: str) -> None:
        """
        Save model to disk.

        Args:
            filepath: Path to save the model
        """
        self._check_fitted()

        model_data = {
            'model': self.model,
            'calibrated_model': self.calibrated_model,
            'feature_names': self.feature_names,
            'config': self.config,
            'is_calibrated': self._is_calibrated
        }
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str) -> None:
        """
        Load model from disk.

        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)

        self.model = model_data['model']
        self.calibrated_model = model_data.get('calibrated_model')
        self.feature_names = model_data['feature_names']
        self.config = model_data.get('config', self.config)
        self._is_calibrated = model_data.get('is_calibrated', False)
        self._is_fitted = True

    def _check_fitted(self) -> None:
        """Check if model is fitted."""
        if not self._is_fitted or self.model is None:
            raise ValueError("Model must be fitted before making predictions")


def train_default_predictor(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[List[str]] = None,
    calibrate: bool = True
) -> ModelTrainingResult:
    """
    Convenience function to train default predictor.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        calibrate: Whether to calibrate probabilities

    Returns:
        ModelTrainingResult: Training result
    """
    predictor = DefaultPredictor()
    return predictor.fit(X, y, feature_names, calibrate=calibrate)


if __name__ == "__main__":
    # Test default predictor
    print("=" * 60)
    print("DEFAULT PREDICTOR TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator
    from src.feature_engineering import FeatureEngineer

    print("\n[1/4] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)
    print(f"      Generated {len(invoices_df)} invoices")

    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df, build_graph=False)

    X, y = engineer.get_feature_matrix(result)
    print(f"      Feature matrix: {X.shape}")
    print(f"      Default rate: {y.mean():.2%}")

    print("\n[3/4] Training model...")
    predictor = DefaultPredictor()
    training_result = predictor.fit(X, y, result.feature_names)

    print(f"\n      Cross-validation AUC-ROC:")
    print(f"        Mean: {training_result.evaluation.cv_mean:.4f}")
    print(f"        Std:  {training_result.evaluation.cv_std:.4f}")

    print(f"\n      Training metrics:")
    print(f"        AUC-ROC:    {training_result.evaluation.auc_roc:.4f}")
    print(f"        Precision:  {training_result.evaluation.precision:.4f}")
    print(f"        Recall:     {training_result.evaluation.recall:.4f}")
    print(f"        F1 Score:   {training_result.evaluation.f1:.4f}")

    print("\n[4/4] Feature importances (top 5):")
    for feat, imp in predictor.get_top_features(5):
        print(f"      {feat}: {imp:.4f}")

    # Find optimal threshold
    opt_threshold, metrics = predictor.find_optimal_threshold(X, y)
    print(f"\n      Optimal threshold: {opt_threshold:.4f}")
    print(f"      F1 at optimal: {metrics['f1']:.4f}")

    print("\n" + "=" * 60)
    print("DEFAULT PREDICTOR TEST COMPLETE")
    print("=" * 60)
