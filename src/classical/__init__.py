"""
QGAI Quantum Financial Modeling - TReDS MVP
Classical ML Module

This module provides classical machine learning components:
- DefaultPredictor: Random Forest model for invoice default prediction
- ModelTrainer: Complete training pipeline with train/val/test splits
- Evaluation utilities and result containers

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

from .default_predictor import (
    DefaultPredictor,
    PredictionResult,
    EvaluationResult,
    ModelTrainingResult,
    train_default_predictor
)

from .model_trainer import (
    ModelTrainer,
    DataSplit,
    TrainingPipelineResult,
    train_and_evaluate
)


__all__ = [
    # Predictor
    "DefaultPredictor",
    "PredictionResult",
    "EvaluationResult",
    "ModelTrainingResult",
    "train_default_predictor",

    # Trainer
    "ModelTrainer",
    "DataSplit",
    "TrainingPipelineResult",
    "train_and_evaluate",
]
