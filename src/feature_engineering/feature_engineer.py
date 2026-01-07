"""
QGAI Quantum Financial Modeling - TReDS MVP
Feature Engineer Module

This module provides the main feature engineering pipeline that combines:
- Invoice-level features
- Entity-level features (buyer, supplier)
- Relationship features
- Z-score normalization

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import FeatureConfig, get_config
from .invoice_features import InvoiceFeatureExtractor
from .entity_features import EntityFeatureExtractor
from .graph_builder import TransactionGraphBuilder, GraphBuildResult


@dataclass
class FeatureEngineeringResult:
    """Result container for feature engineering."""
    features_df: pd.DataFrame
    feature_names: List[str]
    graph_result: Optional[GraphBuildResult]
    n_samples: int
    n_features: int
    feature_statistics: Dict
    engineering_timestamp: datetime = field(default_factory=datetime.now)


class FeatureEngineer:
    """
    Main feature engineering pipeline for default prediction.

    This class orchestrates the complete feature engineering process:
    1. Extract invoice-level features
    2. Extract and merge buyer features
    3. Extract and merge supplier features
    4. Extract and merge relationship features
    5. Compute derived features (z-scores, interactions)
    6. Build transaction graph for QUBO

    Attributes:
        config: FeatureConfig with feature parameters
        invoice_extractor: InvoiceFeatureExtractor instance
        entity_extractor: EntityFeatureExtractor instance
        graph_builder: TransactionGraphBuilder instance

    Example:
        >>> engineer = FeatureEngineer()
        >>> result = engineer.fit_transform(entities_df, invoices_df)
        >>> X = result.features_df[result.feature_names]
        >>> y = result.features_df['is_defaulted']
    """

    # Final feature set for model training
    MODEL_FEATURES = [
        # Invoice features
        'amount_log',
        'days_to_due',
        'acceptance_delay_days',
        'amount_zscore_buyer',
        'is_round_amount',
        'is_month_end',

        # Buyer features
        'buyer_avg_invoice_amount',
        'buyer_total_invoices',
        'buyer_default_rate',
        'buyer_credit_rating_encoded',
        'buyer_age_days',

        # Relationship features
        'relationship_age_days',
        'relationship_invoice_count',
        'is_new_relationship',
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: FeatureConfig instance. If None, uses default config.
        """
        self.config = config or get_config().features
        self.invoice_extractor = InvoiceFeatureExtractor(self.config)
        self.entity_extractor = EntityFeatureExtractor(self.config)
        self.graph_builder = TransactionGraphBuilder()

        # Store statistics for transform
        self._buyer_stats: Optional[pd.DataFrame] = None
        self._supplier_stats: Optional[pd.DataFrame] = None
        self._is_fitted = False

    def fit_transform(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        build_graph: bool = True
    ) -> FeatureEngineeringResult:
        """
        Fit feature statistics and transform data.

        Args:
            entities_df: DataFrame with entity data
            invoices_df: DataFrame with invoice data
            build_graph: Whether to build transaction graph

        Returns:
            FeatureEngineeringResult: Complete feature engineering result
        """
        # Step 1: Extract invoice-level features
        features_df = self.invoice_extractor.extract(invoices_df.copy())

        # Step 2: Extract entity-level features
        buyer_features, supplier_features, rel_features = self.entity_extractor.extract(
            entities_df, invoices_df
        )

        # Store for later transform
        self._buyer_stats = buyer_features.copy()
        self._supplier_stats = supplier_features.copy()

        # Step 3: Merge buyer features
        features_df = features_df.merge(
            buyer_features,
            on='buyer_id',
            how='left'
        )

        # Step 4: Merge supplier features
        features_df = features_df.merge(
            supplier_features,
            on='supplier_id',
            how='left'
        )

        # Step 5: Merge relationship features
        features_df = features_df.merge(
            rel_features,
            on=['buyer_id', 'supplier_id'],
            how='left'
        )

        # Step 6: Compute derived features
        features_df = self._compute_derived_features(features_df)

        # Step 7: Fill missing values
        features_df = self._fill_missing_values(features_df)

        # Step 8: Build transaction graph
        graph_result = None
        if build_graph:
            graph_result = self.graph_builder.build(invoices_df)

        self._is_fitted = True

        # Get feature names that exist in dataframe
        feature_names = [f for f in self.MODEL_FEATURES if f in features_df.columns]

        # Compute statistics
        feature_statistics = self._compute_statistics(features_df, feature_names)

        return FeatureEngineeringResult(
            features_df=features_df,
            feature_names=feature_names,
            graph_result=graph_result,
            n_samples=len(features_df),
            n_features=len(feature_names),
            feature_statistics=feature_statistics
        )

    def transform(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Transform new data using fitted statistics.

        Args:
            entities_df: DataFrame with entity data
            invoices_df: DataFrame with invoice data

        Returns:
            pd.DataFrame: Transformed features

        Raises:
            ValueError: If not fitted
        """
        if not self._is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")

        # Extract invoice features
        features_df = self.invoice_extractor.extract(invoices_df.copy())

        # Merge stored entity features
        if self._buyer_stats is not None:
            features_df = features_df.merge(
                self._buyer_stats,
                on='buyer_id',
                how='left'
            )

        if self._supplier_stats is not None:
            features_df = features_df.merge(
                self._supplier_stats,
                on='supplier_id',
                how='left'
            )

        # Compute derived features
        features_df = self._compute_derived_features(features_df)

        # Fill missing values
        features_df = self._fill_missing_values(features_df)

        return features_df

    def _compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived/interaction features."""
        # Amount z-score relative to buyer average
        if 'buyer_avg_invoice_amount' in df.columns and 'buyer_std_invoice_amount' in df.columns:
            std = df['buyer_std_invoice_amount'].replace(0, 1)
            df['amount_zscore_buyer'] = (
                (df['amount'] - df['buyer_avg_invoice_amount']) / std
            )
        else:
            df['amount_zscore_buyer'] = 0

        # Acceptance delay relative to buyer average
        if 'buyer_avg_acceptance_delay' in df.columns:
            df['acceptance_delay_ratio'] = (
                df['acceptance_delay_days'] / df['buyer_avg_acceptance_delay'].replace(0, 1)
            )

        # Amount to turnover ratio
        if 'buyer_turnover_cr' in df.columns:
            turnover_inr = df['buyer_turnover_cr'] * 1e7  # Convert Cr to INR
            df['amount_to_turnover_ratio'] = df['amount'] / turnover_inr.replace(0, 1)

        return df

    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values with appropriate defaults."""
        # Numeric columns: fill with 0 or median
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().any():
                # Use 0 for most features
                df[col] = df[col].fillna(0)

        return df

    def _compute_statistics(
        self,
        df: pd.DataFrame,
        feature_names: List[str]
    ) -> Dict:
        """Compute feature statistics."""
        stats = {}

        for feature in feature_names:
            if feature in df.columns:
                col = df[feature]
                stats[feature] = {
                    'mean': col.mean(),
                    'std': col.std(),
                    'min': col.min(),
                    'max': col.max(),
                    'null_count': col.isnull().sum(),
                    'null_pct': col.isnull().mean() * 100
                }

        return stats

    def get_feature_matrix(
        self,
        result: FeatureEngineeringResult
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature matrix X and target vector y.

        Args:
            result: FeatureEngineeringResult from fit_transform

        Returns:
            Tuple of (X, y) numpy arrays
        """
        X = result.features_df[result.feature_names].values
        y = result.features_df['is_defaulted'].astype(int).values

        return X, y

    def get_feature_names(self) -> List[str]:
        """Get list of model feature names."""
        return self.MODEL_FEATURES.copy()

    @property
    def feature_names(self) -> List[str]:
        """Property alias for get_feature_names."""
        return self.MODEL_FEATURES.copy()


def engineer_features(
    entities_df: pd.DataFrame,
    invoices_df: pd.DataFrame,
    build_graph: bool = True
) -> FeatureEngineeringResult:
    """
    Convenience function for feature engineering.

    Args:
        entities_df: DataFrame with entity data
        invoices_df: DataFrame with invoice data
        build_graph: Whether to build transaction graph

    Returns:
        FeatureEngineeringResult: Complete feature engineering result
    """
    engineer = FeatureEngineer()
    return engineer.fit_transform(entities_df, invoices_df, build_graph)


if __name__ == "__main__":
    # Test feature engineering pipeline
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator

    print("\n[1/3] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)
    print(f"      Generated {len(entities_df)} entities, {len(invoices_df)} invoices")

    # Run feature engineering
    print("\n[2/3] Engineering features...")
    engineer = FeatureEngineer()
    result = engineer.fit_transform(entities_df, invoices_df)

    print(f"      Samples: {result.n_samples}")
    print(f"      Features: {result.n_features}")
    print(f"      Feature names: {result.feature_names}")

    # Get feature matrix
    print("\n[3/3] Getting feature matrix...")
    X, y = engineer.get_feature_matrix(result)
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Default rate: {y.mean():.2%}")

    # Graph statistics
    if result.graph_result:
        print(f"\n      Graph nodes: {result.graph_result.n_nodes}")
        print(f"      Graph edges: {result.graph_result.n_edges}")

    # Feature statistics
    print("\nFeature Statistics:")
    for feature, stats in list(result.feature_statistics.items())[:5]:
        print(f"  {feature}:")
        print(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}")
        print(f"    min={stats['min']:.4f}, max={stats['max']:.4f}")

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
