"""
QGAI Quantum Financial Modeling - TReDS MVP
Invoice Features Module

This module extracts invoice-level features for default prediction:
- Amount features (log transform, z-score)
- Timing features (days to due, acceptance delay)
- Pattern features (round amount, month end)

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import FeatureConfig, get_config
from config.constants import FeatureConstants as FC


@dataclass
class InvoiceFeatureResult:
    """Result container for invoice feature extraction."""
    features_df: pd.DataFrame
    feature_names: List[str]
    n_features: int
    n_records: int


class InvoiceFeatureExtractor:
    """
    Extract invoice-level features for default prediction.

    This class computes features directly from invoice attributes:
    - Amount-based features
    - Timing-based features
    - Pattern-based features

    Attributes:
        config: FeatureConfig with feature parameters

    Example:
        >>> extractor = InvoiceFeatureExtractor()
        >>> features_df = extractor.extract(invoices_df)
    """

    # Feature names
    FEATURE_NAMES = [
        'amount_log',
        'amount_sqrt',
        'days_to_due',
        'acceptance_delay_days',
        'is_round_amount',
        'is_month_end',
        'is_quarter_end',
        'invoice_day_of_week',
        'invoice_day_of_month',
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize InvoiceFeatureExtractor.

        Args:
            config: FeatureConfig instance. If None, uses default config.
        """
        self.config = config or get_config().features

    def extract(self, invoices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all invoice-level features.

        Args:
            invoices_df: DataFrame with invoice data

        Returns:
            pd.DataFrame: DataFrame with extracted features
        """
        df = invoices_df.copy()

        # Amount features
        df = self._extract_amount_features(df)

        # Timing features
        df = self._extract_timing_features(df)

        # Pattern features
        df = self._extract_pattern_features(df)

        return df

    def extract_with_metadata(self, invoices_df: pd.DataFrame) -> InvoiceFeatureResult:
        """
        Extract features with metadata.

        Args:
            invoices_df: DataFrame with invoice data

        Returns:
            InvoiceFeatureResult: Result with features and metadata
        """
        features_df = self.extract(invoices_df)
        feature_names = [col for col in features_df.columns if col in self.FEATURE_NAMES]

        return InvoiceFeatureResult(
            features_df=features_df,
            feature_names=feature_names,
            n_features=len(feature_names),
            n_records=len(features_df)
        )

    def _extract_amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amount-based features."""
        # Log transform (handles skewed distribution)
        df['amount_log'] = np.log1p(df['amount'])

        # Square root transform (alternative scaling)
        df['amount_sqrt'] = np.sqrt(df['amount'])

        return df

    def _extract_timing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract timing-based features."""
        # Ensure datetime types
        for col in ['invoice_date', 'due_date', 'acceptance_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])

        # Days to due (payment term)
        if 'days_to_due' not in df.columns:
            df['days_to_due'] = (df['due_date'] - df['invoice_date']).dt.days

        # Acceptance delay
        if 'acceptance_delay_days' not in df.columns:
            df['acceptance_delay_days'] = (
                df['acceptance_date'] - df['invoice_date']
            ).dt.days

        # Day of week (0=Monday, 6=Sunday)
        df['invoice_day_of_week'] = df['invoice_date'].dt.dayofweek

        # Day of month
        df['invoice_day_of_month'] = df['invoice_date'].dt.day

        return df

    def _extract_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract pattern-based features."""
        # Is round amount (suspicious if too many)
        round_modulo = self.config.round_amount_modulo
        df['is_round_amount'] = (df['amount'] % round_modulo == 0).astype(int)

        # Is month end (day >= 25)
        month_end_threshold = self.config.month_end_day_threshold
        df['is_month_end'] = (df['invoice_date'].dt.day >= month_end_threshold).astype(int)

        # Is quarter end
        df['is_quarter_end'] = df['invoice_date'].dt.month.isin([3, 6, 9, 12]).astype(int)

        return df

    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces."""
        return self.FEATURE_NAMES.copy()


def extract_invoice_features(invoices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to extract invoice features.

    Args:
        invoices_df: DataFrame with invoice data

    Returns:
        pd.DataFrame: DataFrame with extracted features
    """
    extractor = InvoiceFeatureExtractor()
    return extractor.extract(invoices_df)


if __name__ == "__main__":
    # Test invoice feature extraction
    print("=" * 60)
    print("INVOICE FEATURE EXTRACTION TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator

    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    # Extract features
    extractor = InvoiceFeatureExtractor()
    result = extractor.extract_with_metadata(invoices_df)

    print(f"\nExtracted {result.n_features} invoice features for {result.n_records} records")
    print(f"Features: {result.feature_names}")

    print("\nSample features:")
    print(result.features_df[result.feature_names].head().to_string())

    print("\nFeature statistics:")
    print(result.features_df[result.feature_names].describe().to_string())
