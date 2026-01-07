"""
QGAI Quantum Financial Modeling - TReDS MVP
Risk Scorer Module

This module computes composite risk scores combining:
- Classical default probability from Random Forest
- Quantum ring score from QUBO community detection

The composite score formula:
    CompositeRisk = w_default * P(default) + w_ring * RingScore + w_interaction * (P(default) * RingScore)

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import RiskScoringConfig, get_config


@dataclass
class RiskScore:
    """Individual risk score for an invoice."""
    invoice_id: str
    default_probability: float
    ring_score: float
    composite_score: float
    risk_level: str
    is_high_risk: bool
    is_ring_associated: bool
    buyer_id: str
    supplier_id: str


@dataclass
class RiskScoringResult:
    """Result container for risk scoring."""
    scores: List[RiskScore]
    scores_df: pd.DataFrame
    n_high_risk: int
    n_ring_associated: int
    default_weight: float
    ring_weight: float
    threshold: float
    scoring_timestamp: datetime = field(default_factory=datetime.now)


class RiskScorer:
    """
    Compute composite risk scores combining classical and quantum components.

    This class implements the hybrid scoring approach:
    1. Takes default probabilities from Random Forest
    2. Takes ring scores from QUBO detection
    3. Combines them with configurable weights
    4. Classifies invoices by risk level

    Risk Levels:
    - CRITICAL: Composite score >= 0.8
    - HIGH: Composite score >= 0.6
    - MEDIUM: Composite score >= 0.4
    - LOW: Composite score < 0.4

    Attributes:
        config: RiskScoringConfig with weights and thresholds
        default_weight: Weight for default probability
        ring_weight: Weight for ring score
        interaction_weight: Weight for interaction term

    Example:
        >>> scorer = RiskScorer()
        >>> result = scorer.score(invoice_df, default_probs, ring_scores)
        >>> high_risk = result.scores_df[result.scores_df['is_high_risk']]
    """

    RISK_LEVELS = {
        'CRITICAL': (0.8, 1.0),
        'HIGH': (0.6, 0.8),
        'MEDIUM': (0.4, 0.6),
        'LOW': (0.0, 0.4)
    }

    def __init__(
        self,
        default_weight: Optional[float] = None,
        ring_weight: Optional[float] = None,
        interaction_weight: Optional[float] = None,
        config: Optional[RiskScoringConfig] = None
    ):
        """
        Initialize RiskScorer.

        Args:
            default_weight: Weight for default probability
            ring_weight: Weight for ring score
            interaction_weight: Weight for interaction term
            config: RiskScoringConfig instance
        """
        self.config = config or get_config().risk_scoring
        self.default_weight = default_weight or self.config.default_weight
        self.ring_weight = ring_weight or self.config.ring_weight
        self.interaction_weight = interaction_weight or self.config.interaction_weight
        self.threshold = self.config.high_risk_threshold

    def score(
        self,
        invoices_df: pd.DataFrame,
        default_probabilities: np.ndarray,
        ring_scores: Dict[str, float],
        entity_column: str = 'buyer_id'
    ) -> RiskScoringResult:
        """
        Compute composite risk scores for invoices.

        Args:
            invoices_df: DataFrame with invoice data
            default_probabilities: Array of P(default) for each invoice
            ring_scores: Dict mapping entity_id to ring score
            entity_column: Column to use for ring score lookup

        Returns:
            RiskScoringResult: Scoring result with all scores
        """
        scores = []

        for i, row in invoices_df.iterrows():
            invoice_id = row['invoice_id']
            buyer_id = row['buyer_id']
            supplier_id = row['supplier_id']

            # Get default probability
            default_prob = default_probabilities[i] if i < len(default_probabilities) else 0.0

            # Get ring scores for buyer and supplier
            buyer_ring_score = ring_scores.get(buyer_id, 0.0)
            supplier_ring_score = ring_scores.get(supplier_id, 0.0)

            # Use max ring score (invoice is risky if either party is in a ring)
            ring_score = max(buyer_ring_score, supplier_ring_score)

            # Compute composite score
            composite = self._compute_composite_score(default_prob, ring_score)

            # Determine risk level
            risk_level = self._get_risk_level(composite)
            is_high_risk = composite >= self.threshold
            is_ring_associated = ring_score > 0.3

            scores.append(RiskScore(
                invoice_id=str(invoice_id),
                default_probability=default_prob,
                ring_score=ring_score,
                composite_score=composite,
                risk_level=risk_level,
                is_high_risk=is_high_risk,
                is_ring_associated=is_ring_associated,
                buyer_id=str(buyer_id),
                supplier_id=str(supplier_id)
            ))

        # Create DataFrame
        scores_df = pd.DataFrame([
            {
                'invoice_id': s.invoice_id,
                'buyer_id': s.buyer_id,
                'supplier_id': s.supplier_id,
                'default_probability': s.default_probability,
                'ring_score': s.ring_score,
                'composite_score': s.composite_score,
                'risk_level': s.risk_level,
                'is_high_risk': s.is_high_risk,
                'is_ring_associated': s.is_ring_associated
            }
            for s in scores
        ])

        n_high_risk = sum(1 for s in scores if s.is_high_risk)
        n_ring_associated = sum(1 for s in scores if s.is_ring_associated)

        return RiskScoringResult(
            scores=scores,
            scores_df=scores_df,
            n_high_risk=n_high_risk,
            n_ring_associated=n_ring_associated,
            default_weight=self.default_weight,
            ring_weight=self.ring_weight,
            threshold=self.threshold
        )

    def _compute_composite_score(
        self,
        default_prob: float,
        ring_score: float
    ) -> float:
        """
        Compute composite risk score.

        Formula:
            Composite = w_d * P(default) + w_r * RingScore + w_i * P(default) * RingScore

        The interaction term captures that ring-associated defaults are especially risky.
        """
        composite = (
            self.default_weight * default_prob +
            self.ring_weight * ring_score +
            self.interaction_weight * default_prob * ring_score
        )

        # Normalize to [0, 1]
        max_possible = self.default_weight + self.ring_weight + self.interaction_weight
        composite = min(1.0, composite / max_possible) if max_possible > 0 else 0.0

        return composite

    def _get_risk_level(self, score: float) -> str:
        """Get risk level for a given score."""
        for level, (low, high) in self.RISK_LEVELS.items():
            if low <= score < high:
                return level
        return 'CRITICAL' if score >= 0.8 else 'LOW'

    def get_targeting_list(
        self,
        result: RiskScoringResult,
        top_n: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get prioritized targeting list for investigation.

        Args:
            result: RiskScoringResult from score()
            top_n: Return top N invoices
            min_score: Minimum composite score

        Returns:
            DataFrame sorted by composite_score descending
        """
        df = result.scores_df.copy()

        # Filter by minimum score if specified
        if min_score is not None:
            df = df[df['composite_score'] >= min_score]

        # Sort by composite score
        df = df.sort_values('composite_score', ascending=False)

        # Limit to top N if specified
        if top_n is not None:
            df = df.head(top_n)

        return df.reset_index(drop=True)

    def get_risk_summary(self, result: RiskScoringResult) -> Dict[str, Any]:
        """
        Get summary statistics for risk scores.

        Args:
            result: RiskScoringResult from score()

        Returns:
            Dict with summary statistics
        """
        df = result.scores_df

        return {
            'total_invoices': len(df),
            'n_high_risk': result.n_high_risk,
            'n_ring_associated': result.n_ring_associated,
            'pct_high_risk': result.n_high_risk / len(df) * 100 if len(df) > 0 else 0,
            'pct_ring_associated': result.n_ring_associated / len(df) * 100 if len(df) > 0 else 0,
            'risk_level_counts': df['risk_level'].value_counts().to_dict(),
            'avg_default_prob': df['default_probability'].mean(),
            'avg_ring_score': df['ring_score'].mean(),
            'avg_composite_score': df['composite_score'].mean(),
            'max_composite_score': df['composite_score'].max(),
            'default_weight': result.default_weight,
            'ring_weight': result.ring_weight,
            'threshold': result.threshold
        }


def compute_risk_scores(
    invoices_df: pd.DataFrame,
    default_probabilities: np.ndarray,
    ring_scores: Dict[str, float]
) -> RiskScoringResult:
    """
    Convenience function for risk scoring.

    Args:
        invoices_df: DataFrame with invoice data
        default_probabilities: Array of P(default)
        ring_scores: Dict mapping entity_id to ring score

    Returns:
        RiskScoringResult
    """
    scorer = RiskScorer()
    return scorer.score(invoices_df, default_probabilities, ring_scores)


if __name__ == "__main__":
    # Test risk scorer
    print("=" * 60)
    print("RISK SCORER TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator
    from src.feature_engineering import FeatureEngineer
    from src.classical import DefaultPredictor
    from src.quantum import RingDetector

    print("\n[1/5] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"      Invoices: {len(invoices_df)}")

    print("\n[2/5] Engineering features...")
    engineer = FeatureEngineer()
    fe_result = engineer.fit_transform(entities_df, invoices_df)
    X, y = engineer.get_feature_matrix(fe_result)

    print("\n[3/5] Training default predictor...")
    predictor = DefaultPredictor()
    predictor.fit(X, y, fe_result.feature_names)
    default_probs = predictor.predict_proba(X)

    print(f"      Mean P(default): {default_probs.mean():.4f}")

    print("\n[4/5] Detecting rings...")
    detector = RingDetector(k_communities=5)
    ring_result = detector.detect(
        fe_result.graph_result.modularity_matrix,
        fe_result.graph_result.node_list,
        fe_result.graph_result.adjacency_matrix,
        num_reads=100,
        num_sweeps=500
    )
    ring_scores = detector.get_ring_scores(ring_result)

    print(f"      Entities with ring scores: {len(ring_scores)}")

    print("\n[5/5] Computing risk scores...")
    scorer = RiskScorer()
    result = scorer.score(invoices_df, default_probs, ring_scores)

    summary = scorer.get_risk_summary(result)
    print(f"\n      Total invoices: {summary['total_invoices']}")
    print(f"      High risk: {summary['n_high_risk']} ({summary['pct_high_risk']:.2f}%)")
    print(f"      Ring associated: {summary['n_ring_associated']} ({summary['pct_ring_associated']:.2f}%)")
    print(f"      Avg composite score: {summary['avg_composite_score']:.4f}")
    print(f"      Max composite score: {summary['max_composite_score']:.4f}")

    print(f"\n      Risk level distribution:")
    for level, count in summary['risk_level_counts'].items():
        print(f"        {level}: {count}")

    # Get targeting list
    targeting = scorer.get_targeting_list(result, top_n=10)
    print(f"\n      Top 10 high-risk invoices:")
    print(targeting[['invoice_id', 'composite_score', 'risk_level']].to_string(index=False))

    print("\n" + "=" * 60)
    print("RISK SCORER TEST COMPLETE")
    print("=" * 60)
