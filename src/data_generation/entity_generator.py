"""
QGAI Quantum Financial Modeling - TReDS MVP
Entity Generator Module

This module generates synthetic TReDS entities including:
- Buyers (Corporate entities)
- Suppliers (MSME entities)
- Ring Members (Dual-role fraud entities)

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
    CreditRating,
    DataGenerationConstants as DGC,
)


@dataclass
class EntityGenerationResult:
    """Result container for entity generation."""
    entities_df: pd.DataFrame
    n_buyers: int
    n_suppliers: int
    n_ring_members: int
    n_rings: int
    generation_timestamp: datetime


class EntityGenerator:
    """
    Generate synthetic TReDS entities for the fraud detection system.

    This class creates realistic synthetic data for:
    - Corporate buyers with credit ratings
    - MSME suppliers
    - Fraud ring members (dual-role entities)

    Attributes:
        config: DataGenerationConfig with generation parameters
        random_state: numpy RandomState for reproducibility

    Example:
        >>> from config import get_config
        >>> config = get_config().data_generation
        >>> generator = EntityGenerator(config)
        >>> entities_df = generator.generate()
        >>> print(f"Generated {len(entities_df)} entities")
    """

    def __init__(self, config: Optional[DataGenerationConfig] = None):
        """
        Initialize EntityGenerator.

        Args:
            config: DataGenerationConfig instance. If None, uses default config.
        """
        self.config = config or get_config().data_generation
        self.random_state = np.random.RandomState(self.config.random_seed)
        self._entities: List[Dict] = []

    def generate(self) -> pd.DataFrame:
        """
        Generate all entity types and return as DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing all generated entities

        Raises:
            ValueError: If configuration parameters are invalid
        """
        self._validate_config()
        self._entities = []

        # Generate each entity type
        self._generate_buyers()
        self._generate_suppliers()
        self._generate_ring_members()

        # Create DataFrame
        entities_df = pd.DataFrame(self._entities)

        # Add derived columns
        entities_df = self._add_derived_columns(entities_df)

        return entities_df

    def generate_with_metadata(self) -> EntityGenerationResult:
        """
        Generate entities with generation metadata.

        Returns:
            EntityGenerationResult: Result container with entities and metadata
        """
        entities_df = self.generate()

        return EntityGenerationResult(
            entities_df=entities_df,
            n_buyers=len(entities_df[entities_df['entity_type'] == EntityType.BUYER.value]),
            n_suppliers=len(entities_df[entities_df['entity_type'] == EntityType.SUPPLIER.value]),
            n_ring_members=len(entities_df[entities_df['is_ring_member']]),
            n_rings=self.config.n_rings,
            generation_timestamp=datetime.now()
        )

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.n_buyers < 1:
            raise ValueError("n_buyers must be at least 1")
        if self.config.n_suppliers < 1:
            raise ValueError("n_suppliers must be at least 1")
        if self.config.n_rings != len(self.config.ring_sizes):
            raise ValueError("n_rings must match length of ring_sizes")

    def _generate_buyers(self) -> None:
        """Generate corporate buyer entities."""
        base_date = pd.Timestamp(self.config.start_date)

        # Credit rating probabilities
        ratings = list(self.config.buyer_credit_ratings.keys())
        probs = list(self.config.buyer_credit_ratings.values())

        for i in range(self.config.n_buyers):
            # Registration date spread over the year before start_date
            reg_offset = int(self.random_state.randint(0, 365))
            registration_date = base_date - timedelta(days=reg_offset)

            # Turnover (log-normal distribution)
            turnover = self.random_state.lognormal(
                mean=DGC.BUYER_TURNOVER_MEAN,
                sigma=DGC.BUYER_TURNOVER_STD
            )

            # Credit rating based on distribution
            credit_rating = self.random_state.choice(ratings, p=probs)

            # Industry sector
            sector = self.random_state.choice(self.config.industry_sectors)

            entity = {
                'entity_id': f"{DGC.BUYER_PREFIX}{i:04d}",
                'entity_type': EntityType.BUYER.value,
                'registration_date': registration_date,
                'turnover_cr': round(turnover, 2),
                'credit_rating': credit_rating,
                'industry_sector': sector,
                'is_ring_member': False,
                'ring_id': None,
                'gstin': self._generate_gstin(sector),
                'city': self._generate_city(),
                'employee_count': self._generate_employee_count('buyer'),
            }
            self._entities.append(entity)

    def _generate_suppliers(self) -> None:
        """Generate MSME supplier entities."""
        base_date = pd.Timestamp(self.config.start_date)

        # Credit rating probabilities
        ratings = list(self.config.supplier_credit_ratings.keys())
        probs = list(self.config.supplier_credit_ratings.values())

        for i in range(self.config.n_suppliers):
            # Registration date
            reg_offset = int(self.random_state.randint(0, 365))
            registration_date = base_date - timedelta(days=reg_offset)

            # Turnover (smaller than buyers, log-normal)
            turnover = self.random_state.lognormal(
                mean=DGC.SUPPLIER_TURNOVER_MEAN,
                sigma=DGC.SUPPLIER_TURNOVER_STD
            )

            # Credit rating
            credit_rating = self.random_state.choice(ratings, p=probs)

            # Industry sector
            sector = self.random_state.choice(self.config.industry_sectors)

            entity = {
                'entity_id': f"{DGC.SUPPLIER_PREFIX}{i:04d}",
                'entity_type': EntityType.SUPPLIER.value,
                'registration_date': registration_date,
                'turnover_cr': round(turnover, 2),
                'credit_rating': credit_rating,
                'industry_sector': sector,
                'is_ring_member': False,
                'ring_id': None,
                'gstin': self._generate_gstin(sector),
                'city': self._generate_city(),
                'employee_count': self._generate_employee_count('supplier'),
            }
            self._entities.append(entity)

    def _generate_ring_members(self) -> None:
        """
        Generate fraud ring member entities.

        Ring characteristics:
        - Members registered within a short time window (suspicious)
        - Similar turnover values (coordinated)
        - Same industry sector
        - Dual role (can act as both buyer and supplier)
        """
        ring_base_date = pd.Timestamp(DGC.RING_START_DATE)

        for ring_idx in range(self.config.n_rings):
            ring_id = f"{DGC.RING_ID_PREFIX}{ring_idx:02d}"
            ring_size = self.config.ring_sizes[ring_idx]

            # Ring-specific base registration date
            ring_start = ring_base_date + timedelta(
                days=int(self.random_state.randint(0, 60))
            )

            # Shared characteristics for this ring (suspicious similarity)
            ring_turnover_base = self.random_state.lognormal(
                mean=DGC.RING_TURNOVER_MEAN,
                sigma=DGC.RING_TURNOVER_STD
            )
            ring_sector = self.random_state.choice(self.config.industry_sectors)
            ring_city = self._generate_city()

            for j in range(ring_size):
                # Registration within narrow window (suspicious)
                reg_offset = int(self.random_state.randint(
                    0, self.config.ring_registration_window
                ))
                registration_date = ring_start + timedelta(days=reg_offset)

                # Similar turnover (suspicious coordination)
                turnover = ring_turnover_base * self.random_state.uniform(0.8, 1.2)

                # Lower credit ratings typical for fraud
                credit_rating = self.random_state.choice(
                    ['BBB', 'BB', 'B'],
                    p=[0.3, 0.4, 0.3]
                )

                entity = {
                    'entity_id': f"{DGC.RING_PREFIX}{ring_idx}{j:02d}",
                    'entity_type': EntityType.DUAL.value,
                    'registration_date': registration_date,
                    'turnover_cr': round(turnover, 2),
                    'credit_rating': credit_rating,
                    'industry_sector': ring_sector,  # Same sector in ring
                    'is_ring_member': True,
                    'ring_id': ring_id,
                    'gstin': self._generate_gstin(ring_sector),
                    'city': ring_city,  # Same city (often)
                    'employee_count': self._generate_employee_count('ring'),
                }
                self._entities.append(entity)

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns to entities DataFrame."""
        # Entity age in days (from registration to analysis date)
        analysis_date = pd.Timestamp(self.config.end_date)
        df['entity_age_days'] = (analysis_date - df['registration_date']).dt.days

        # Turnover category
        df['turnover_category'] = pd.cut(
            df['turnover_cr'],
            bins=[0, 5, 25, 100, 500, float('inf')],
            labels=['Micro', 'Small', 'Medium', 'Large', 'Enterprise']
        )

        # Credit score (numeric encoding)
        credit_score_map = {
            'AAA': 6, 'AA': 5, 'A': 4, 'BBB': 3, 'BB': 2, 'B': 1, 'NR': 0
        }
        df['credit_score'] = df['credit_rating'].map(credit_score_map)

        return df

    def _generate_gstin(self, sector: str) -> str:
        """Generate a realistic-looking GSTIN."""
        # GSTIN format: 2 digit state + 10 char PAN + 1 digit entity + Z + checksum
        state_code = self.random_state.choice(['27', '29', '06', '09', '33'])
        pan_chars = ''.join(self.random_state.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))
        pan_nums = ''.join(self.random_state.choice(list('0123456789'), 4))
        entity_code = self.random_state.choice(list('0123456789'))
        return f"{state_code}{pan_chars}{pan_nums}{pan_chars[0]}{entity_code}Z{self.random_state.choice(list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'))}"

    def _generate_city(self) -> str:
        """Generate a random Indian city."""
        cities = [
            'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad',
            'Pune', 'Ahmedabad', 'Kolkata', 'Jaipur', 'Surat',
            'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane'
        ]
        return self.random_state.choice(cities)

    def _generate_employee_count(self, entity_type: str) -> int:
        """Generate employee count based on entity type."""
        if entity_type == 'buyer':
            return int(self.random_state.lognormal(6, 1))  # Larger companies
        elif entity_type == 'supplier':
            return int(self.random_state.lognormal(3, 1))  # MSMEs
        else:  # ring
            return int(self.random_state.lognormal(2, 0.5))  # Smaller, similar sizes

    def get_ring_members(self, entities_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Get mapping of ring_id to member entity_ids.

        Args:
            entities_df: Generated entities DataFrame

        Returns:
            Dict mapping ring_id to list of member entity_ids
        """
        ring_members = {}
        ring_df = entities_df[entities_df['is_ring_member']]

        for ring_id in ring_df['ring_id'].unique():
            members = ring_df[ring_df['ring_id'] == ring_id]['entity_id'].tolist()
            ring_members[ring_id] = members

        return ring_members

    def get_statistics(self, entities_df: pd.DataFrame) -> Dict:
        """
        Get generation statistics.

        Args:
            entities_df: Generated entities DataFrame

        Returns:
            Dict with generation statistics
        """
        return {
            'total_entities': len(entities_df),
            'n_buyers': len(entities_df[entities_df['entity_type'] == 'buyer']),
            'n_suppliers': len(entities_df[entities_df['entity_type'] == 'supplier']),
            'n_ring_members': len(entities_df[entities_df['is_ring_member']]),
            'n_rings': entities_df[entities_df['is_ring_member']]['ring_id'].nunique(),
            'credit_rating_distribution': entities_df['credit_rating'].value_counts().to_dict(),
            'industry_distribution': entities_df['industry_sector'].value_counts().to_dict(),
            'avg_turnover_buyer': entities_df[entities_df['entity_type'] == 'buyer']['turnover_cr'].mean(),
            'avg_turnover_supplier': entities_df[entities_df['entity_type'] == 'supplier']['turnover_cr'].mean(),
        }


def generate_entities(
    n_buyers: int = 100,
    n_suppliers: int = 200,
    n_rings: int = 3,
    ring_sizes: List[int] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    Convenience function to generate entities with custom parameters.

    Args:
        n_buyers: Number of buyer entities
        n_suppliers: Number of supplier entities
        n_rings: Number of fraud rings
        ring_sizes: Size of each ring (default: [5, 7, 8])
        seed: Random seed

    Returns:
        pd.DataFrame: Generated entities
    """
    if ring_sizes is None:
        ring_sizes = [5, 7, 8][:n_rings]

    config = get_config().data_generation
    config.n_buyers = n_buyers
    config.n_suppliers = n_suppliers
    config.n_rings = n_rings
    config.ring_sizes = ring_sizes
    config.random_seed = seed

    generator = EntityGenerator(config)
    return generator.generate()


if __name__ == "__main__":
    # Test entity generation
    print("=" * 60)
    print("ENTITY GENERATION TEST")
    print("=" * 60)

    generator = EntityGenerator()
    entities_df = generator.generate()

    print(f"\nGenerated {len(entities_df)} entities:")
    print(f"  - Buyers: {len(entities_df[entities_df['entity_type'] == 'buyer'])}")
    print(f"  - Suppliers: {len(entities_df[entities_df['entity_type'] == 'supplier'])}")
    print(f"  - Ring Members: {len(entities_df[entities_df['is_ring_member']])}")

    print("\nRing Membership:")
    ring_members = generator.get_ring_members(entities_df)
    for ring_id, members in ring_members.items():
        print(f"  {ring_id}: {len(members)} members - {members[:3]}...")

    print("\nSample Entities:")
    print(entities_df.head(3).to_string())

    print("\nStatistics:")
    stats = generator.get_statistics(entities_df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
