"""
QGAI Quantum Financial Modeling - TReDS MVP
SHAP Explainer Module

This module provides SHAP-based explanations for model predictions:
- Global feature importance via SHAP
- Local (instance-level) explanations
- Explanation summaries and visualizations

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import ExplainabilityConfig, get_config


@dataclass
class FeatureContribution:
    """Single feature contribution to a prediction."""
    feature_name: str
    feature_value: float
    shap_value: float
    contribution_direction: str  # 'increases_risk' or 'decreases_risk'


@dataclass
class LocalExplanation:
    """Explanation for a single prediction."""
    invoice_id: str
    prediction: float
    base_value: float
    contributions: List[FeatureContribution]
    top_risk_factors: List[Tuple[str, float]]
    top_protective_factors: List[Tuple[str, float]]
    explanation_text: str


@dataclass
class GlobalExplanation:
    """Global model explanation."""
    feature_importances: Dict[str, float]
    mean_shap_values: Dict[str, float]
    feature_ranking: List[Tuple[str, float]]
    n_samples_used: int


@dataclass
class ExplanationResult:
    """Complete explanation result."""
    global_explanation: GlobalExplanation
    local_explanations: Optional[List[LocalExplanation]]
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    n_samples: int
    explanation_timestamp: datetime = field(default_factory=datetime.now)


class SHAPExplainer:
    """
    SHAP-based explainer for default prediction model.

    This class provides interpretable explanations:
    - Global feature importance (which features matter most overall)
    - Local explanations (why a specific invoice was flagged)
    - Natural language explanation generation

    Uses TreeExplainer for Random Forest (fast and exact).

    Attributes:
        config: ExplainabilityConfig with parameters
        explainer: SHAP TreeExplainer instance
        feature_names: List of feature names

    Example:
        >>> explainer = SHAPExplainer()
        >>> explainer.fit(model, feature_names)
        >>> result = explainer.explain(X)
        >>> print(result.global_explanation.feature_ranking[:5])
    """

    def __init__(self, config: Optional[ExplainabilityConfig] = None):
        """
        Initialize SHAPExplainer.

        Args:
            config: ExplainabilityConfig instance
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required for explanations. Install with: pip install shap")

        self.config = config or get_config().explainability
        self.explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        model,
        feature_names: List[str],
        background_data: Optional[np.ndarray] = None
    ) -> None:
        """
        Fit explainer to model.

        Args:
            model: Trained model (RandomForest)
            feature_names: List of feature names
            background_data: Optional background data for explainer
        """
        self.feature_names = feature_names

        # Use TreeExplainer for tree-based models (efficient)
        self.explainer = shap.TreeExplainer(model)
        self._is_fitted = True

    def explain(
        self,
        X: np.ndarray,
        invoice_ids: Optional[List[str]] = None,
        n_samples: Optional[int] = None,
        generate_local: bool = True
    ) -> ExplanationResult:
        """
        Generate explanations for predictions.

        Args:
            X: Feature matrix
            invoice_ids: Optional list of invoice IDs
            n_samples: Optional limit on samples to explain
            generate_local: Whether to generate local explanations

        Returns:
            ExplanationResult: Complete explanation result
        """
        self._check_fitted()

        # Limit samples if specified
        if n_samples is not None and n_samples < len(X):
            X = X[:n_samples]
            if invoice_ids:
                invoice_ids = invoice_ids[:n_samples]

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # List of arrays (one per class) - use positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif isinstance(shap_values, np.ndarray):
            # 3D array: (samples, features, classes) - use positive class
            if shap_values.ndim == 3:
                shap_values = shap_values[:, :, 1] if shap_values.shape[2] > 1 else shap_values[:, :, 0]
            # 2D array: already in correct format (samples, features)

        # Ensure 2D array
        if shap_values.ndim == 1:
            shap_values = shap_values.reshape(1, -1)

        # Get expected value
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            if len(expected_value) > 1:
                expected_value = expected_value[1]
            else:
                expected_value = expected_value[0]
        expected_value = float(expected_value)

        # Generate global explanation
        global_explanation = self._generate_global_explanation(shap_values)

        # Generate local explanations
        local_explanations = None
        if generate_local:
            local_explanations = self._generate_local_explanations(
                X, shap_values, expected_value, invoice_ids
            )

        return ExplanationResult(
            global_explanation=global_explanation,
            local_explanations=local_explanations,
            shap_values=shap_values,
            expected_value=expected_value,
            feature_names=self.feature_names,
            n_samples=len(X)
        )

    def _generate_global_explanation(
        self,
        shap_values: np.ndarray
    ) -> GlobalExplanation:
        """Generate global feature importance explanation."""
        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Mean SHAP values (direction of effect)
        mean_shap = shap_values.mean(axis=0)

        # Convert to Python floats to avoid numpy array comparison issues
        feature_importances = {
            name: float(val) for name, val in zip(self.feature_names, mean_abs_shap)
        }
        mean_shap_values = {
            name: float(val) for name, val in zip(self.feature_names, mean_shap)
        }

        # Rank features
        feature_ranking = sorted(
            feature_importances.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return GlobalExplanation(
            feature_importances=feature_importances,
            mean_shap_values=mean_shap_values,
            feature_ranking=feature_ranking,
            n_samples_used=len(shap_values)
        )

    def _generate_local_explanations(
        self,
        X: np.ndarray,
        shap_values: np.ndarray,
        expected_value: float,
        invoice_ids: Optional[List[str]]
    ) -> List[LocalExplanation]:
        """Generate local explanations for each sample."""
        local_explanations = []

        for i in range(len(X)):
            invoice_id = invoice_ids[i] if invoice_ids else f"sample_{i}"

            # Get prediction
            prediction = expected_value + shap_values[i].sum()

            # Create feature contributions
            contributions = []
            for j, (fname, fval, sval) in enumerate(
                zip(self.feature_names, X[i], shap_values[i])
            ):
                direction = 'increases_risk' if sval > 0 else 'decreases_risk'
                contributions.append(FeatureContribution(
                    feature_name=fname,
                    feature_value=float(fval),
                    shap_value=float(sval),
                    contribution_direction=direction
                ))

            # Sort by absolute SHAP value
            contributions.sort(key=lambda c: abs(c.shap_value), reverse=True)

            # Top risk and protective factors
            top_risk = [
                (c.feature_name, c.shap_value)
                for c in contributions
                if c.shap_value > 0
            ][:self.config.top_risk_factors]

            top_protective = [
                (c.feature_name, c.shap_value)
                for c in contributions
                if c.shap_value < 0
            ][:self.config.top_protective_factors]

            # Generate explanation text
            explanation_text = self._generate_explanation_text(
                invoice_id, prediction, top_risk, top_protective
            )

            local_explanations.append(LocalExplanation(
                invoice_id=str(invoice_id),
                prediction=float(prediction),
                base_value=float(expected_value),
                contributions=contributions,
                top_risk_factors=top_risk,
                top_protective_factors=top_protective,
                explanation_text=explanation_text
            ))

        return local_explanations

    def _generate_explanation_text(
        self,
        invoice_id: str,
        prediction: float,
        risk_factors: List[Tuple[str, float]],
        protective_factors: List[Tuple[str, float]]
    ) -> str:
        """Generate human-readable explanation text."""
        risk_level = 'HIGH' if prediction > 0.5 else 'MODERATE' if prediction > 0.2 else 'LOW'

        text = f"Invoice {invoice_id} - Risk Level: {risk_level} (Score: {prediction:.2%})\n\n"

        if risk_factors:
            text += "Key Risk Factors:\n"
            for factor, value in risk_factors[:3]:
                readable_name = self._format_feature_name(factor)
                text += f"  - {readable_name} (contribution: +{value:.3f})\n"

        if protective_factors:
            text += "\nProtective Factors:\n"
            for factor, value in protective_factors[:3]:
                readable_name = self._format_feature_name(factor)
                text += f"  - {readable_name} (contribution: {value:.3f})\n"

        return text

    def _format_feature_name(self, name: str) -> str:
        """Format feature name for human readability."""
        # Replace underscores with spaces, capitalize
        readable = name.replace('_', ' ').title()

        # Custom formatting for known features
        formatting = {
            'buyer default rate': 'Buyer\'s Historical Default Rate',
            'amount log': 'Invoice Amount (Log Scaled)',
            'days to due': 'Days Until Due Date',
            'acceptance delay days': 'Acceptance Processing Delay',
            'buyer credit rating encoded': 'Buyer Credit Rating',
            'relationship invoice count': 'Transaction History with Supplier',
            'buyer age days': 'Buyer Platform Tenure',
            'is round amount': 'Round Amount (Suspicious Pattern)',
            'is month end': 'Month-End Transaction',
            'is new relationship': 'New Buyer-Supplier Relationship'
        }

        lower_name = name.lower().replace('_', ' ')
        return formatting.get(lower_name, readable)

    def get_feature_importance_df(
        self,
        result: ExplanationResult
    ) -> pd.DataFrame:
        """
        Get feature importance as DataFrame.

        Args:
            result: ExplanationResult from explain()

        Returns:
            DataFrame with feature importance
        """
        records = [
            {
                'feature': name,
                'importance': imp,
                'mean_effect': result.global_explanation.mean_shap_values[name],
                'direction': 'increases_risk' if result.global_explanation.mean_shap_values[name] > 0 else 'decreases_risk'
            }
            for name, imp in result.global_explanation.feature_ranking
        ]
        return pd.DataFrame(records)

    def get_explanation_summary(
        self,
        result: ExplanationResult,
        top_n: int = 5
    ) -> Dict[str, Any]:
        """
        Get summary of explanations.

        Args:
            result: ExplanationResult from explain()
            top_n: Number of top features to include

        Returns:
            Dict with summary information
        """
        return {
            'n_samples_explained': result.n_samples,
            'expected_value': result.expected_value,
            'top_features': result.global_explanation.feature_ranking[:top_n],
            'feature_count': len(result.feature_names),
            'explanation_timestamp': result.explanation_timestamp.isoformat()
        }

    def _check_fitted(self) -> None:
        """Check if explainer is fitted."""
        if not self._is_fitted or self.explainer is None:
            raise ValueError("Explainer must be fitted before generating explanations")


def explain_predictions(
    model,
    X: np.ndarray,
    feature_names: List[str],
    invoice_ids: Optional[List[str]] = None
) -> ExplanationResult:
    """
    Convenience function for generating explanations.

    Args:
        model: Trained model
        X: Feature matrix
        feature_names: List of feature names
        invoice_ids: Optional invoice IDs

    Returns:
        ExplanationResult
    """
    explainer = SHAPExplainer()
    explainer.fit(model, feature_names)
    return explainer.explain(X, invoice_ids)


if __name__ == "__main__":
    # Test SHAP explainer
    print("=" * 60)
    print("SHAP EXPLAINER TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator
    from src.feature_engineering import FeatureEngineer
    from src.classical import DefaultPredictor

    print("\n[1/4] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print("\n[2/4] Engineering features...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df, build_graph=False)
    X, y = engineer.get_feature_matrix(result)

    print("\n[3/4] Training model...")
    predictor = DefaultPredictor()
    predictor.fit(X, y, result.feature_names)

    print("\n[4/4] Generating explanations...")
    explainer = SHAPExplainer()
    explainer.fit(predictor.model, result.feature_names)

    # Explain a sample of predictions
    explanation_result = explainer.explain(X[:100], n_samples=100)

    print(f"\n      Samples explained: {explanation_result.n_samples}")
    print(f"      Expected value: {explanation_result.expected_value:.4f}")

    print(f"\n      Top 5 Important Features:")
    for feat, imp in explanation_result.global_explanation.feature_ranking[:5]:
        direction = '+' if explanation_result.global_explanation.mean_shap_values[feat] > 0 else '-'
        print(f"        {feat}: {imp:.4f} ({direction})")

    # Sample local explanation
    if explanation_result.local_explanations:
        print(f"\n      Sample Local Explanation:")
        local = explanation_result.local_explanations[0]
        print(f"        {local.explanation_text}")

    print("\n" + "=" * 60)
    print("SHAP EXPLAINER TEST COMPLETE")
    print("=" * 60)
