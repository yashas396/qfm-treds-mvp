"""
QGAI Quantum Financial Modeling - TReDS MVP
Entity Features Module

This module extracts entity-level (buyer/supplier) aggregated features:
- Buyer features: transaction patterns, credit rating, default history
- Supplier features: platform tenure, customer diversity
- Relationship features: trading history, frequency

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import FeatureConfig, get_config
from config.constants import FeatureConstants as FC


@dataclass
class EntityFeatureResult:
    """Result container for entity feature extraction."""
    buyer_features: pd.DataFrame
    supplier_features: pd.DataFrame
    relationship_features: pd.DataFrame
    feature_names: Dict[str, List[str]]
    n_buyers: int
    n_suppliers: int
    n_relationships: int


class EntityFeatureExtractor:
    """
    Extract entity-level aggregated features.

    This class computes features by aggregating invoice data:
    - Buyer-level statistics
    - Supplier-level statistics
    - Buyer-Supplier relationship features

    Attributes:
        config: FeatureConfig with feature parameters

    Example:
        >>> extractor = EntityFeatureExtractor()
        >>> buyer_features, supplier_features, rel_features = extractor.extract(
        ...     entities_df, invoices_df
        ... )
    """

    # Buyer feature names
    BUYER_FEATURES = [
        'buyer_total_invoices',
        'buyer_total_amount',
        'buyer_avg_invoice_amount',
        'buyer_std_invoice_amount',
        'buyer_unique_suppliers',
        'buyer_avg_acceptance_delay',
        'buyer_default_rate',
        'buyer_default_count',
        'buyer_credit_rating_encoded',
        'buyer_turnover_cr',
        'buyer_age_days',
    ]

    # Supplier feature names
    SUPPLIER_FEATURES = [
        'supplier_total_invoices',
        'supplier_total_amount',
        'supplier_avg_invoice_amount',
        'supplier_unique_buyers',
        'supplier_age_days',
        'supplier_turnover_cr',
    ]

    # Relationship feature names
    RELATIONSHIP_FEATURES = [
        'relationship_invoice_count',
        'relationship_total_amount',
        'relationship_avg_amount',
        'relationship_age_days',
        'relationship_default_rate',
        'is_new_relationship',
    ]

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize EntityFeatureExtractor.

        Args:
            config: FeatureConfig instance. If None, uses default config.
        """
        self.config = config or get_config().features

    def extract(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract all entity-level features.

        Args:
            entities_df: DataFrame with entity data
            invoices_df: DataFrame with invoice data
            reference_date: Reference date for age calculations

        Returns:
            Tuple of (buyer_features, supplier_features, relationship_features)
        """
        if reference_date is None:
            reference_date = pd.Timestamp.now()

        # Extract buyer features
        buyer_features = self._extract_buyer_features(
            entities_df, invoices_df, reference_date
        )

        # Extract supplier features
        supplier_features = self._extract_supplier_features(
            entities_df, invoices_df, reference_date
        )

        # Extract relationship features
        relationship_features = self._extract_relationship_features(
            invoices_df, reference_date
        )

        return buyer_features, supplier_features, relationship_features

    def extract_with_metadata(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None
    ) -> EntityFeatureResult:
        """
        Extract features with metadata.

        Args:
            entities_df: DataFrame with entity data
            invoices_df: DataFrame with invoice data
            reference_date: Reference date for age calculations

        Returns:
            EntityFeatureResult: Result with features and metadata
        """
        buyer_features, supplier_features, relationship_features = self.extract(
            entities_df, invoices_df, reference_date
        )

        return EntityFeatureResult(
            buyer_features=buyer_features,
            supplier_features=supplier_features,
            relationship_features=relationship_features,
            feature_names={
                'buyer': self.BUYER_FEATURES,
                'supplier': self.SUPPLIER_FEATURES,
                'relationship': self.RELATIONSHIP_FEATURES
            },
            n_buyers=len(buyer_features),
            n_suppliers=len(supplier_features),
            n_relationships=len(relationship_features)
        )

    def _extract_buyer_features(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Extract buyer-level aggregated features."""
        # Get all entities that act as buyers
        buyer_ids = invoices_df['buyer_id'].unique()

        # Aggregate invoice statistics by buyer
        buyer_stats = invoices_df.groupby('buyer_id').agg({
            'invoice_id': 'count',
            'amount': ['sum', 'mean', 'std'],
            'supplier_id': 'nunique',
            'acceptance_delay_days': 'mean',
            'is_defaulted': ['sum', 'mean'],
            'invoice_date': 'min'
        }).reset_index()

        # Flatten column names
        buyer_stats.columns = [
            'buyer_id',
            'buyer_total_invoices',
            'buyer_total_amount',
            'buyer_avg_invoice_amount',
            'buyer_std_invoice_amount',
            'buyer_unique_suppliers',
            'buyer_avg_acceptance_delay',
            'buyer_default_count',
            'buyer_default_rate',
            'buyer_first_invoice_date'
        ]

        # Fill NaN std with 0
        buyer_stats['buyer_std_invoice_amount'] = buyer_stats['buyer_std_invoice_amount'].fillna(0)

        # Merge with entity data
        buyer_entities = entities_df[
            entities_df['entity_id'].isin(buyer_ids)
        ][['entity_id', 'credit_rating', 'turnover_cr', 'registration_date']].copy()

        buyer_entities.columns = ['buyer_id', 'credit_rating', 'buyer_turnover_cr', 'buyer_registration_date']

        buyer_features = buyer_stats.merge(buyer_entities, on='buyer_id', how='left')

        # Credit rating encoding
        credit_encoding = self.config.credit_rating_encoding
        buyer_features['buyer_credit_rating_encoded'] = buyer_features['credit_rating'].map(
            credit_encoding
        ).fillna(0)

        # Buyer age in days
        buyer_features['buyer_registration_date'] = pd.to_datetime(buyer_features['buyer_registration_date'])
        buyer_features['buyer_age_days'] = (
            reference_date - buyer_features['buyer_registration_date']
        ).dt.days

        # Select final columns
        final_cols = ['buyer_id'] + [c for c in self.BUYER_FEATURES if c in buyer_features.columns]
        buyer_features = buyer_features[final_cols]

        return buyer_features

    def _extract_supplier_features(
        self,
        entities_df: pd.DataFrame,
        invoices_df: pd.DataFrame,
        reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Extract supplier-level aggregated features."""
        # Get all entities that act as suppliers
        supplier_ids = invoices_df['supplier_id'].unique()

        # Aggregate invoice statistics by supplier
        supplier_stats = invoices_df.groupby('supplier_id').agg({
            'invoice_id': 'count',
            'amount': ['sum', 'mean'],
            'buyer_id': 'nunique',
        }).reset_index()

        # Flatten column names
        supplier_stats.columns = [
            'supplier_id',
            'supplier_total_invoices',
            'supplier_total_amount',
            'supplier_avg_invoice_amount',
            'supplier_unique_buyers',
        ]

        # Merge with entity data
        supplier_entities = entities_df[
            entities_df['entity_id'].isin(supplier_ids)
        ][['entity_id', 'turnover_cr', 'registration_date']].copy()

        supplier_entities.columns = ['supplier_id', 'supplier_turnover_cr', 'supplier_registration_date']

        supplier_features = supplier_stats.merge(supplier_entities, on='supplier_id', how='left')

        # Supplier age in days
        supplier_features['supplier_registration_date'] = pd.to_datetime(
            supplier_features['supplier_registration_date']
        )
        supplier_features['supplier_age_days'] = (
            reference_date - supplier_features['supplier_registration_date']
        ).dt.days

        # Select final columns
        final_cols = ['supplier_id'] + [c for c in self.SUPPLIER_FEATURES if c in supplier_features.columns]
        supplier_features = supplier_features[final_cols]

        return supplier_features

    def _extract_relationship_features(
        self,
        invoices_df: pd.DataFrame,
        reference_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Extract buyer-supplier relationship features."""
        # Aggregate by buyer-supplier pair
        rel_stats = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
            'invoice_id': 'count',
            'amount': ['sum', 'mean'],
            'invoice_date': ['min', 'max'],
            'is_defaulted': 'mean'
        }).reset_index()

        # Flatten column names
        rel_stats.columns = [
            'buyer_id',
            'supplier_id',
            'relationship_invoice_count',
            'relationship_total_amount',
            'relationship_avg_amount',
            'relationship_first_invoice',
            'relationship_last_invoice',
            'relationship_default_rate'
        ]

        # Relationship age in days (from first invoice)
        rel_stats['relationship_first_invoice'] = pd.to_datetime(rel_stats['relationship_first_invoice'])
        rel_stats['relationship_age_days'] = (
            reference_date - rel_stats['relationship_first_invoice']
        ).dt.days

        # Is new relationship (< 30 days)
        new_threshold = self.config.new_relationship_threshold_days
        rel_stats['is_new_relationship'] = (
            rel_stats['relationship_age_days'] < new_threshold
        ).astype(int)

        # Select final columns
        final_cols = ['buyer_id', 'supplier_id'] + [
            c for c in self.RELATIONSHIP_FEATURES if c in rel_stats.columns
        ]
        relationship_features = rel_stats[final_cols]

        return relationship_features

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get dictionary of feature names by entity type."""
        return {
            'buyer': self.BUYER_FEATURES.copy(),
            'supplier': self.SUPPLIER_FEATURES.copy(),
            'relationship': self.RELATIONSHIP_FEATURES.copy()
        }


def extract_entity_features(
    entities_df: pd.DataFrame,
    invoices_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to extract entity features.

    Args:
        entities_df: DataFrame with entity data
        invoices_df: DataFrame with invoice data

    Returns:
        Tuple of (buyer_features, supplier_features, relationship_features)
    """
    extractor = EntityFeatureExtractor()
    return extractor.extract(entities_df, invoices_df)


if __name__ == "__main__":
    # Test entity feature extraction
    print("=" * 60)
    print("ENTITY FEATURE EXTRACTION TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator

    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    # Extract features
    extractor = EntityFeatureExtractor()
    result = extractor.extract_with_metadata(entities_df, invoices_df)

    print(f"\nBuyer features: {result.n_buyers} buyers")
    print(f"Supplier features: {result.n_suppliers} suppliers")
    print(f"Relationship features: {result.n_relationships} relationships")

    print("\nBuyer feature names:", result.feature_names['buyer'])
    print("\nSample buyer features:")
    print(result.buyer_features.head().to_string())

    print("\nSample relationship features:")
    print(result.relationship_features.head().to_string())
