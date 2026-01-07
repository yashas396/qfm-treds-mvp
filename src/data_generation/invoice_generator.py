"""
QGAI Quantum Financial Modeling - TReDS MVP
Invoice Generator Module

This module generates synthetic TReDS invoices including:
- Legitimate invoices (normal business transactions)
- Ring invoices (fraudulent circular transactions)

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import DataGenerationConfig, get_config
from config.constants import (
    EntityType,
    DataGenerationConstants as DGC,
)


@dataclass
class InvoiceGenerationResult:
    """Result container for invoice generation."""
    invoices_df: pd.DataFrame
    n_legitimate: int
    n_ring: int
    default_rate_legitimate: float
    default_rate_ring: float
    generation_timestamp: datetime


class InvoiceGenerator:
    """
    Generate synthetic TReDS invoices for the fraud detection system.

    This class creates realistic synthetic invoices with:
    - Legitimate transactions between buyers and suppliers
    - Fraudulent circular transactions within rings
    - Realistic default patterns based on credit ratings

    Attributes:
        config: DataGenerationConfig with generation parameters
        random_state: numpy RandomState for reproducibility

    Example:
        >>> from config import get_config
        >>> config = get_config().data_generation
        >>> generator = InvoiceGenerator(config)
        >>> invoices_df = generator.generate(entities_df)
    """

    def __init__(self, config: Optional[DataGenerationConfig] = None):
        """
        Initialize InvoiceGenerator.

        Args:
            config: DataGenerationConfig instance. If None, uses default config.
        """
        self.config = config or get_config().data_generation
        self.random_state = np.random.RandomState(self.config.random_seed)
        self._invoices: List[Dict] = []
        self._invoice_counter = 0

    def generate(self, entities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all invoices and return as DataFrame.

        Args:
            entities_df: DataFrame of entities from EntityGenerator

        Returns:
            pd.DataFrame: DataFrame containing all generated invoices

        Raises:
            ValueError: If entities_df is invalid
        """
        self._validate_entities(entities_df)
        self._invoices = []
        self._invoice_counter = 0

        # Separate entity types
        buyers = entities_df[entities_df['entity_type'].isin(['buyer', 'dual'])]
        suppliers = entities_df[entities_df['entity_type'].isin(['supplier', 'dual'])]
        legit_buyers = buyers[~buyers['is_ring_member']]
        legit_suppliers = suppliers[~suppliers['is_ring_member']]
        ring_entities = entities_df[entities_df['is_ring_member']]

        # Generate legitimate invoices
        self._generate_legitimate_invoices(legit_buyers, legit_suppliers)

        # Generate ring invoices
        self._generate_ring_invoices(ring_entities)

        # Create DataFrame
        invoices_df = pd.DataFrame(self._invoices)

        # Add derived columns
        invoices_df = self._add_derived_columns(invoices_df)

        # Shuffle to mix legitimate and ring invoices
        invoices_df = invoices_df.sample(frac=1, random_state=self.config.random_seed).reset_index(drop=True)

        return invoices_df

    def generate_with_metadata(self, entities_df: pd.DataFrame) -> InvoiceGenerationResult:
        """
        Generate invoices with generation metadata.

        Args:
            entities_df: DataFrame of entities

        Returns:
            InvoiceGenerationResult: Result container with invoices and metadata
        """
        invoices_df = self.generate(entities_df)

        legit = invoices_df[~invoices_df['is_in_ring']]
        ring = invoices_df[invoices_df['is_in_ring']]

        return InvoiceGenerationResult(
            invoices_df=invoices_df,
            n_legitimate=len(legit),
            n_ring=len(ring),
            default_rate_legitimate=legit['is_defaulted'].mean() if len(legit) > 0 else 0,
            default_rate_ring=ring['is_defaulted'].mean() if len(ring) > 0 else 0,
            generation_timestamp=datetime.now()
        )

    def _validate_entities(self, entities_df: pd.DataFrame) -> None:
        """Validate entities DataFrame."""
        required_cols = ['entity_id', 'entity_type', 'credit_rating', 'turnover_cr', 'is_ring_member']
        missing = [col for col in required_cols if col not in entities_df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(entities_df) == 0:
            raise ValueError("Entities DataFrame is empty")

    def _generate_legitimate_invoices(
        self,
        buyers: pd.DataFrame,
        suppliers: pd.DataFrame
    ) -> None:
        """
        Generate legitimate business invoices.

        Creates realistic invoices with:
        - Amount based on buyer turnover
        - Default probability based on credit rating
        - Realistic payment terms and acceptance delays
        """
        start_date = pd.Timestamp(self.config.start_date)
        end_date = pd.Timestamp(self.config.end_date)
        date_range_days = (end_date - start_date).days

        # Create buyer-supplier relationships (not all pairs transact)
        # Each buyer transacts with 5-20 suppliers on average
        relationships = self._create_relationships(buyers, suppliers)

        invoices_per_relationship = self.config.n_legitimate_invoices // len(relationships)

        for (buyer_id, supplier_id), rel_info in relationships.items():
            buyer = buyers[buyers['entity_id'] == buyer_id].iloc[0]

            # Number of invoices for this relationship (with variation)
            n_invoices = max(1, int(invoices_per_relationship * self.random_state.uniform(0.5, 1.5)))

            for _ in range(n_invoices):
                if self._invoice_counter >= self.config.n_legitimate_invoices:
                    return

                invoice = self._create_legitimate_invoice(
                    buyer_id, supplier_id, buyer, start_date, date_range_days
                )
                self._invoices.append(invoice)
                self._invoice_counter += 1

    def _create_relationships(
        self,
        buyers: pd.DataFrame,
        suppliers: pd.DataFrame
    ) -> Dict[Tuple[str, str], Dict]:
        """Create buyer-supplier trading relationships."""
        relationships = {}
        buyer_ids = buyers['entity_id'].tolist()
        supplier_ids = suppliers['entity_id'].tolist()

        for buyer_id in buyer_ids:
            # Each buyer transacts with 5-15 suppliers
            n_suppliers = min(len(supplier_ids), self.random_state.randint(5, 16))
            selected_suppliers = self.random_state.choice(
                supplier_ids, n_suppliers, replace=False
            )

            for supplier_id in selected_suppliers:
                relationships[(buyer_id, supplier_id)] = {
                    'relationship_start': self.random_state.randint(0, 180)
                }

        return relationships

    def _create_legitimate_invoice(
        self,
        buyer_id: str,
        supplier_id: str,
        buyer: pd.Series,
        start_date: pd.Timestamp,
        date_range_days: int
    ) -> Dict:
        """Create a single legitimate invoice."""
        # Invoice date
        invoice_offset = int(self.random_state.randint(0, date_range_days))
        invoice_date = start_date + timedelta(days=invoice_offset)

        # Payment term
        payment_term = int(self.random_state.choice(self.config.payment_terms))
        due_date = invoice_date + timedelta(days=payment_term)

        # Acceptance delay (legitimate: 1-5 days typically)
        acceptance_delay = int(self.random_state.randint(1, 6))
        acceptance_date = invoice_date + timedelta(days=acceptance_delay)

        # Amount based on buyer turnover (with variation)
        base_amount = buyer['turnover_cr'] * 1e5  # Convert Cr to typical invoice
        amount = base_amount * self.random_state.lognormal(0, 0.7)
        amount = np.clip(amount, DGC.MIN_INVOICE_AMOUNT, DGC.MAX_INVOICE_AMOUNT)

        # Round amount flag (legitimate: sometimes round)
        is_round = self.random_state.random() < 0.15  # 15% chance of round amount
        if is_round:
            amount = round(amount / 10000) * 10000

        # Default based on credit rating
        default_prob = self.config.default_probabilities.get(
            buyer['credit_rating'], 0.1
        )
        is_defaulted = self.random_state.random() < default_prob

        return {
            'invoice_id': f"{DGC.INVOICE_PREFIX}{self._invoice_counter:06d}",
            'buyer_id': buyer_id,
            'supplier_id': supplier_id,
            'invoice_date': invoice_date,
            'due_date': due_date,
            'acceptance_date': acceptance_date,
            'amount': round(amount, 2),
            'payment_term_days': payment_term,
            'is_defaulted': is_defaulted,
            'is_in_ring': False,
            'ring_id': None,
            'discount_rate': round(self.random_state.uniform(0.02, 0.04), 4),
            'invoice_status': 'defaulted' if is_defaulted else 'settled',
        }

    def _generate_ring_invoices(self, ring_entities: pd.DataFrame) -> None:
        """
        Generate fraudulent ring invoices.

        Ring characteristics:
        - Circular patterns (A→B, B→C, C→A)
        - Round amounts (suspicious)
        - Instant acceptance (coordinated)
        - Higher default rate
        """
        if len(ring_entities) == 0:
            return

        ring_start_date = pd.Timestamp(DGC.RING_START_DATE)
        ring_groups = ring_entities.groupby('ring_id')

        invoices_per_ring = self.config.n_ring_invoices // len(ring_groups)

        for ring_id, members in ring_groups:
            member_list = members['entity_id'].tolist()
            n_members = len(member_list)

            for j in range(invoices_per_ring):
                invoice = self._create_ring_invoice(
                    ring_id, member_list, n_members, j, ring_start_date
                )
                self._invoices.append(invoice)
                self._invoice_counter += 1

    def _create_ring_invoice(
        self,
        ring_id: str,
        member_list: List[str],
        n_members: int,
        index: int,
        ring_start_date: pd.Timestamp
    ) -> Dict:
        """Create a single ring (fraudulent) invoice."""
        # Circular pattern: A→B, B→C, C→D, ..., N→A
        buyer_idx = index % n_members
        supplier_idx = (index + 1) % n_members
        buyer_id = member_list[buyer_idx]
        supplier_id = member_list[supplier_idx]

        # Invoice date (concentrated period)
        invoice_offset = int(self.random_state.randint(0, 180))
        invoice_date = ring_start_date + timedelta(days=invoice_offset)

        # Fixed payment term for rings (suspicious uniformity)
        payment_term = 45
        due_date = invoice_date + timedelta(days=payment_term)

        # Instant acceptance (suspicious - coordinated)
        acceptance_delay = int(self.random_state.choice([0, 1]))  # 0-1 days
        acceptance_date = invoice_date + timedelta(days=acceptance_delay)

        # Round amounts (suspicious)
        amount = self.random_state.choice(self.config.ring_round_amounts)

        # Higher default rate for rings
        is_defaulted = self.random_state.random() < self.config.ring_default_probability

        return {
            'invoice_id': f"{DGC.INVOICE_PREFIX}{self._invoice_counter:06d}",
            'buyer_id': buyer_id,
            'supplier_id': supplier_id,
            'invoice_date': invoice_date,
            'due_date': due_date,
            'acceptance_date': acceptance_date,
            'amount': amount,
            'payment_term_days': payment_term,
            'is_defaulted': is_defaulted,
            'is_in_ring': True,
            'ring_id': ring_id,
            'discount_rate': round(self.random_state.uniform(0.025, 0.035), 4),
            'invoice_status': 'defaulted' if is_defaulted else 'settled',
        }

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to invoices DataFrame."""
        # Acceptance delay
        df['acceptance_delay_days'] = (
            df['acceptance_date'] - df['invoice_date']
        ).dt.days

        # Days to due
        df['days_to_due'] = (
            df['due_date'] - df['invoice_date']
        ).dt.days

        # Amount category
        df['amount_category'] = pd.cut(
            df['amount'],
            bins=[0, 100000, 500000, 1000000, 5000000, float('inf')],
            labels=['<1L', '1-5L', '5-10L', '10-50L', '>50L']
        )

        # Is round amount
        df['is_round_amount'] = (df['amount'] % 10000 == 0).astype(int)

        # Month of invoice
        df['invoice_month'] = df['invoice_date'].dt.month

        # Is month end
        df['is_month_end'] = (df['invoice_date'].dt.day >= 25).astype(int)

        # Quarter
        df['invoice_quarter'] = df['invoice_date'].dt.quarter

        return df

    def get_statistics(self, invoices_df: pd.DataFrame) -> Dict:
        """
        Get generation statistics.

        Args:
            invoices_df: Generated invoices DataFrame

        Returns:
            Dict with generation statistics
        """
        legit = invoices_df[~invoices_df['is_in_ring']]
        ring = invoices_df[invoices_df['is_in_ring']]

        return {
            'total_invoices': len(invoices_df),
            'n_legitimate': len(legit),
            'n_ring': len(ring),
            'total_amount': invoices_df['amount'].sum(),
            'avg_amount': invoices_df['amount'].mean(),
            'default_rate_overall': invoices_df['is_defaulted'].mean(),
            'default_rate_legitimate': legit['is_defaulted'].mean() if len(legit) > 0 else 0,
            'default_rate_ring': ring['is_defaulted'].mean() if len(ring) > 0 else 0,
            'unique_buyers': invoices_df['buyer_id'].nunique(),
            'unique_suppliers': invoices_df['supplier_id'].nunique(),
            'unique_relationships': invoices_df.groupby(['buyer_id', 'supplier_id']).ngroups,
            'avg_acceptance_delay_legit': legit['acceptance_delay_days'].mean() if len(legit) > 0 else 0,
            'avg_acceptance_delay_ring': ring['acceptance_delay_days'].mean() if len(ring) > 0 else 0,
            'amount_distribution': invoices_df['amount_category'].value_counts().to_dict(),
        }

    def get_ring_transaction_patterns(self, invoices_df: pd.DataFrame) -> Dict:
        """
        Analyze ring transaction patterns.

        Args:
            invoices_df: Generated invoices DataFrame

        Returns:
            Dict with ring transaction patterns
        """
        ring_invoices = invoices_df[invoices_df['is_in_ring']]
        patterns = {}

        for ring_id in ring_invoices['ring_id'].unique():
            ring_data = ring_invoices[ring_invoices['ring_id'] == ring_id]
            patterns[ring_id] = {
                'n_invoices': len(ring_data),
                'total_amount': ring_data['amount'].sum(),
                'unique_entities': set(ring_data['buyer_id'].tolist() + ring_data['supplier_id'].tolist()),
                'n_entities': len(set(ring_data['buyer_id'].tolist() + ring_data['supplier_id'].tolist())),
                'default_rate': ring_data['is_defaulted'].mean(),
                'circular_edges': list(zip(ring_data['buyer_id'], ring_data['supplier_id'])),
            }

        return patterns


def generate_invoices(
    entities_df: pd.DataFrame,
    n_legitimate: int = 5000,
    n_ring: int = 500,
    seed: int = 42
) -> pd.DataFrame:
    """
    Convenience function to generate invoices with custom parameters.

    Args:
        entities_df: DataFrame of entities
        n_legitimate: Number of legitimate invoices
        n_ring: Number of ring invoices
        seed: Random seed

    Returns:
        pd.DataFrame: Generated invoices
    """
    config = get_config().data_generation
    config.n_legitimate_invoices = n_legitimate
    config.n_ring_invoices = n_ring
    config.random_seed = seed

    generator = InvoiceGenerator(config)
    return generator.generate(entities_df)


if __name__ == "__main__":
    # Test invoice generation
    print("=" * 60)
    print("INVOICE GENERATION TEST")
    print("=" * 60)

    # First generate entities
    from entity_generator import EntityGenerator

    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()
    print(f"\nGenerated {len(entities_df)} entities")

    # Then generate invoices
    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"\nGenerated {len(invoices_df)} invoices:")

    stats = invoice_gen.get_statistics(invoices_df)
    print(f"  - Legitimate: {stats['n_legitimate']}")
    print(f"  - Ring-related: {stats['n_ring']}")
    print(f"  - Default rate (legit): {stats['default_rate_legitimate']:.2%}")
    print(f"  - Default rate (ring): {stats['default_rate_ring']:.2%}")
    print(f"  - Avg acceptance delay (legit): {stats['avg_acceptance_delay_legit']:.1f} days")
    print(f"  - Avg acceptance delay (ring): {stats['avg_acceptance_delay_ring']:.1f} days")

    print("\nRing Patterns:")
    patterns = invoice_gen.get_ring_transaction_patterns(invoices_df)
    for ring_id, pattern in patterns.items():
        print(f"  {ring_id}: {pattern['n_invoices']} invoices, {pattern['n_entities']} entities")

    print("\nSample Invoices:")
    print(invoices_df[['invoice_id', 'buyer_id', 'supplier_id', 'amount', 'is_in_ring', 'is_defaulted']].head(5).to_string())
