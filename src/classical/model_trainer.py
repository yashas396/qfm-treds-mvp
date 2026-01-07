"""
QGAI Quantum Financial Modeling - TReDS MVP
Model Trainer Module

This module provides the complete training workflow:
- Data splitting (train/validation/test)
- Model training with hyperparameter tuning
- Evaluation on held-out test data
- Model persistence and versioning

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import ClassicalModelConfig, get_config
from .default_predictor import DefaultPredictor, ModelTrainingResult, EvaluationResult


@dataclass
class DataSplit:
    """Container for train/validation/test splits."""
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    train_size: int
    val_size: int
    test_size: int


@dataclass
class TrainingPipelineResult:
    """Complete training pipeline result."""
    model: DefaultPredictor
    training_result: ModelTrainingResult
    test_evaluation: EvaluationResult
    data_split: DataSplit
    best_params: Optional[Dict[str, Any]]
    training_history: Dict
    model_version: str
    training_timestamp: datetime = field(default_factory=datetime.now)


class ModelTrainer:
    """
    Complete training pipeline for default prediction model.

    This class handles:
    - Data splitting with stratification
    - Optional hyperparameter tuning
    - Model training and validation
    - Final evaluation on held-out test set
    - Model persistence

    Attributes:
        config: ClassicalModelConfig with model parameters
        predictor: DefaultPredictor instance
        data_split: DataSplit with train/val/test data

    Example:
        >>> trainer = ModelTrainer()
        >>> result = trainer.train(X, y, feature_names)
        >>> trainer.save("models/default_model.joblib")
    """

    def __init__(self, config: Optional[ClassicalModelConfig] = None):
        """
        Initialize ModelTrainer.

        Args:
            config: ClassicalModelConfig instance. If None, uses default config.
        """
        self.config = config or get_config().classical
        self.predictor: Optional[DefaultPredictor] = None
        self.data_split: Optional[DataSplit] = None
        self._model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: Optional[int] = None
    ) -> DataSplit:
        """
        Split data into train/validation/test sets.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            test_size: Fraction for test set
            val_size: Fraction for validation set (from remaining after test)
            random_state: Random seed for reproducibility

        Returns:
            DataSplit: Container with all splits
        """
        if random_state is None:
            random_state = self.config.random_seed

        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        # Second split: train vs val
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction,
            stratify=y_trainval,
            random_state=random_state
        )

        self.data_split = DataSplit(
            X_train=X_train,
            X_val=X_val,
            X_test=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            train_size=len(X_train),
            val_size=len(X_val),
            test_size=len(X_test)
        )

        return self.data_split

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        tune_hyperparameters: bool = False,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> TrainingPipelineResult:
        """
        Complete training pipeline.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            tune_hyperparameters: Whether to perform grid search
            test_size: Fraction for test set
            val_size: Fraction for validation set

        Returns:
            TrainingPipelineResult: Complete training result
        """
        training_history = {
            'steps': [],
            'metrics': {}
        }

        # Step 1: Split data
        training_history['steps'].append('data_split')
        data_split = self.split_data(X, y, feature_names, test_size, val_size)

        print(f"Data split:")
        print(f"  Train: {data_split.train_size} samples ({data_split.y_train.mean():.2%} default)")
        print(f"  Val:   {data_split.val_size} samples ({data_split.y_val.mean():.2%} default)")
        print(f"  Test:  {data_split.test_size} samples ({data_split.y_test.mean():.2%} default)")

        # Step 2: Hyperparameter tuning (optional)
        best_params = None
        if tune_hyperparameters:
            training_history['steps'].append('hyperparameter_tuning')
            best_params = self._tune_hyperparameters(
                data_split.X_train, data_split.y_train
            )
            print(f"\nBest hyperparameters: {best_params}")

            # Update config with best params
            if 'n_estimators' in best_params:
                self.config.n_estimators = best_params['n_estimators']
            if 'max_depth' in best_params:
                self.config.max_depth = best_params['max_depth']
            if 'min_samples_split' in best_params:
                self.config.min_samples_split = best_params['min_samples_split']

        # Step 3: Train model
        training_history['steps'].append('model_training')
        self.predictor = DefaultPredictor(self.config)

        # Train on train + validation data
        X_train_full = np.vstack([data_split.X_train, data_split.X_val])
        y_train_full = np.concatenate([data_split.y_train, data_split.y_val])

        training_result = self.predictor.fit(
            X_train_full,
            y_train_full,
            feature_names,
            calibrate=True,
            validate=True
        )

        training_history['metrics']['train'] = {
            'auc_roc': training_result.evaluation.auc_roc,
            'f1': training_result.evaluation.f1,
            'cv_mean': training_result.evaluation.cv_mean,
            'cv_std': training_result.evaluation.cv_std
        }

        # Step 4: Evaluate on test set
        training_history['steps'].append('test_evaluation')
        test_evaluation = self.predictor.evaluate(
            data_split.X_test,
            data_split.y_test
        )

        training_history['metrics']['test'] = {
            'auc_roc': test_evaluation.auc_roc,
            'precision': test_evaluation.precision,
            'recall': test_evaluation.recall,
            'f1': test_evaluation.f1
        }

        return TrainingPipelineResult(
            model=self.predictor,
            training_result=training_result,
            test_evaluation=test_evaluation,
            data_split=data_split,
            best_params=best_params,
            training_history=training_history,
            model_version=self._model_version
        )

    def _tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning using grid search.

        Args:
            X: Training features
            y: Training labels

        Returns:
            Dict with best hyperparameters
        """
        from sklearn.ensemble import RandomForestClassifier

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        base_model = RandomForestClassifier(
            class_weight='balanced',
            random_state=self.config.random_seed,
            n_jobs=-1
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.random_seed)

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X, y)

        return grid_search.best_params_

    def train_with_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str]
    ) -> Tuple[ModelTrainingResult, EvaluationResult]:
        """
        Train model and evaluate on validation set.

        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names

        Returns:
            Tuple of (training_result, validation_evaluation)
        """
        # Split data
        data_split = self.split_data(X, y, feature_names)

        # Train on training set only
        self.predictor = DefaultPredictor(self.config)
        training_result = self.predictor.fit(
            data_split.X_train,
            data_split.y_train,
            feature_names
        )

        # Evaluate on validation set
        val_evaluation = self.predictor.evaluate(
            data_split.X_val,
            data_split.y_val
        )

        return training_result, val_evaluation

    def evaluate_on_test(self) -> EvaluationResult:
        """
        Evaluate trained model on test set.

        Returns:
            EvaluationResult: Test set evaluation
        """
        if self.predictor is None:
            raise ValueError("Model must be trained before evaluation")
        if self.data_split is None:
            raise ValueError("Data must be split before evaluation")

        return self.predictor.evaluate(
            self.data_split.X_test,
            self.data_split.y_test
        )

    def save(self, filepath: str) -> None:
        """
        Save trained model and metadata.

        Args:
            filepath: Path to save the model
        """
        if self.predictor is None:
            raise ValueError("Model must be trained before saving")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        self.predictor.save_model(filepath)

    def load(self, filepath: str) -> None:
        """
        Load trained model.

        Args:
            filepath: Path to the saved model
        """
        self.predictor = DefaultPredictor(self.config)
        self.predictor.load_model(filepath)

    def get_model(self) -> DefaultPredictor:
        """Get the trained predictor."""
        if self.predictor is None:
            raise ValueError("Model must be trained first")
        return self.predictor


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    tune_hyperparameters: bool = False
) -> TrainingPipelineResult:
    """
    Convenience function for complete training pipeline.

    Args:
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names
        tune_hyperparameters: Whether to tune hyperparameters

    Returns:
        TrainingPipelineResult: Complete training result
    """
    trainer = ModelTrainer()
    return trainer.train(X, y, feature_names, tune_hyperparameters)


if __name__ == "__main__":
    # Test model trainer
    print("=" * 60)
    print("MODEL TRAINER TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator
    from src.feature_engineering import FeatureEngineer

    print("\n[1/4] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df, build_graph=False)

    X, y = engineer.get_feature_matrix(result)
    print(f"      Total samples: {len(X)}")
    print(f"      Features: {X.shape[1]}")
    print(f"      Default rate: {y.mean():.2%}")

    print("\n[3/4] Training model...")
    trainer = ModelTrainer()
    pipeline_result = trainer.train(X, y, result.feature_names)

    print(f"\n      Training Metrics:")
    print(f"        CV AUC-ROC: {pipeline_result.training_result.evaluation.cv_mean:.4f} (+/- {pipeline_result.training_result.evaluation.cv_std:.4f})")

    print(f"\n      Test Set Metrics:")
    print(f"        AUC-ROC:   {pipeline_result.test_evaluation.auc_roc:.4f}")
    print(f"        Precision: {pipeline_result.test_evaluation.precision:.4f}")
    print(f"        Recall:    {pipeline_result.test_evaluation.recall:.4f}")
    print(f"        F1 Score:  {pipeline_result.test_evaluation.f1:.4f}")

    print(f"\n      Confusion Matrix:")
    cm = pipeline_result.test_evaluation.confusion_matrix
    print(f"        TN: {cm[0,0]:4d}  FP: {cm[0,1]:4d}")
    print(f"        FN: {cm[1,0]:4d}  TP: {cm[1,1]:4d}")

    print("\n[4/4] Top feature importances:")
    for feat, imp in pipeline_result.model.get_top_features(5):
        print(f"      {feat}: {imp:.4f}")

    # Check if meets MVP criteria
    print("\n" + "-" * 40)
    print("MVP CRITERIA CHECK:")
    target_auc = 0.75
    actual_auc = pipeline_result.test_evaluation.auc_roc
    status = "PASS" if actual_auc >= target_auc else "FAIL"
    print(f"  AUC-ROC >= {target_auc}: {actual_auc:.4f} [{status}]")

    print("\n" + "=" * 60)
    print("MODEL TRAINER TEST COMPLETE")
    print("=" * 60)
