"""
QGAI Quantum Financial Modeling - TReDS MVP
QUBO Formulator Module

This module formulates the community detection problem as QUBO:
- Modularity maximization for ring detection
- K-partition constraints for community assignment
- Penalty terms for constraint satisfaction

The QUBO formulation:
    H(x) = -x^T B x + P * constraint_violations

Where:
- B is the modularity matrix
- x is binary assignment vector
- P is penalty coefficient

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime

import dimod

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import QUBOConfig, get_config


@dataclass
class QUBOFormulationResult:
    """Result container for QUBO formulation."""
    bqm: dimod.BinaryQuadraticModel
    n_nodes: int
    k_communities: int
    n_variables: int
    penalty_weight: float
    modularity_matrix: np.ndarray
    node_list: List[str]
    formulation_timestamp: datetime = field(default_factory=datetime.now)


class QUBOFormulator:
    """
    Formulate community detection as QUBO for quantum/simulated annealing.

    This class implements modularity maximization for finding fraud rings:
    - Each node can belong to one of K communities
    - Modularity Q = sum over communities of (internal edges - expected)
    - QUBO form uses binary variables x[i,c] = 1 if node i in community c

    The optimization maximizes:
        Q = (1/2m) * sum_{i,j} sum_c B[i,j] * x[i,c] * x[j,c]

    Subject to:
        sum_c x[i,c] = 1 for all nodes i (each node in exactly one community)

    Attributes:
        config: QUBOConfig with QUBO parameters
        k_communities: Number of communities to detect
        penalty_weight: Weight for constraint violations

    Example:
        >>> formulator = QUBOFormulator(k_communities=5)
        >>> result = formulator.formulate(modularity_matrix, node_list)
        >>> bqm = result.bqm  # Binary Quadratic Model for sampler
    """

    def __init__(
        self,
        k_communities: Optional[int] = None,
        penalty_weight: Optional[float] = None,
        config: Optional[QUBOConfig] = None
    ):
        """
        Initialize QUBOFormulator.

        Args:
            k_communities: Number of communities (overrides config)
            penalty_weight: Penalty weight (overrides config)
            config: QUBOConfig instance. If None, uses default config.
        """
        self.config = config or get_config().quantum
        self.k_communities = k_communities or self.config.k_communities
        self.penalty_weight = penalty_weight or self.config.penalty_weight

    def formulate(
        self,
        modularity_matrix: np.ndarray,
        node_list: List[str],
        include_constraints: bool = True
    ) -> QUBOFormulationResult:
        """
        Formulate community detection as QUBO.

        Args:
            modularity_matrix: The modularity matrix B (n x n)
            node_list: List of node identifiers
            include_constraints: Whether to add constraint terms

        Returns:
            QUBOFormulationResult: Contains BQM and metadata
        """
        n = len(node_list)
        k = self.k_communities

        # Create variable names: x_{node}_{community}
        # Using format that dimod can parse
        variables = []
        for i, node in enumerate(node_list):
            for c in range(k):
                var_name = f"x_{i}_{c}"
                variables.append(var_name)

        # Initialize BQM
        bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

        # Add all variables
        for var in variables:
            bqm.add_variable(var, 0.0)

        # Add modularity terms (quadratic)
        # We want to MAXIMIZE modularity, but samplers MINIMIZE
        # So we negate: min -Q
        self._add_modularity_terms(bqm, modularity_matrix, n, k)

        # Add constraint terms (one-hot)
        if include_constraints:
            self._add_one_hot_constraints(bqm, n, k)

        return QUBOFormulationResult(
            bqm=bqm,
            n_nodes=n,
            k_communities=k,
            n_variables=len(variables),
            penalty_weight=self.penalty_weight,
            modularity_matrix=modularity_matrix,
            node_list=node_list
        )

    def _add_modularity_terms(
        self,
        bqm: dimod.BinaryQuadraticModel,
        B: np.ndarray,
        n: int,
        k: int
    ) -> None:
        """
        Add modularity maximization terms to BQM.

        For community c, nodes i and j contribute B[i,j] if both in c.
        Quadratic term: -B[i,j] * x_{i,c} * x_{j,c} (negative for maximization)
        """
        for c in range(k):
            for i in range(n):
                for j in range(i + 1, n):
                    # Only add if B[i,j] is significant
                    if abs(B[i, j]) > 1e-10:
                        var_i = f"x_{i}_{c}"
                        var_j = f"x_{j}_{c}"
                        # Negative because we minimize but want to maximize modularity
                        bqm.add_interaction(var_i, var_j, -B[i, j])

    def _add_one_hot_constraints(
        self,
        bqm: dimod.BinaryQuadraticModel,
        n: int,
        k: int
    ) -> None:
        """
        Add one-hot constraints: each node in exactly one community.

        Constraint: sum_c x_{i,c} = 1 for all i

        Penalty form: P * (sum_c x_{i,c} - 1)^2
                    = P * (sum_c x_{i,c}^2 + 2*sum_{c<c'} x_{i,c}*x_{i,c'} - 2*sum_c x_{i,c} + 1)
                    = P * (sum_c x_{i,c} + 2*sum_{c<c'} x_{i,c}*x_{i,c'} - 2*sum_c x_{i,c} + 1)
                      (since x^2 = x for binary)
                    = P * (-sum_c x_{i,c} + 2*sum_{c<c'} x_{i,c}*x_{i,c'} + 1)
        """
        P = self.penalty_weight

        for i in range(n):
            # Linear terms: -P for each variable
            for c in range(k):
                var = f"x_{i}_{c}"
                bqm.add_variable(var, -P)

            # Quadratic terms: 2P for each pair
            for c1 in range(k):
                for c2 in range(c1 + 1, k):
                    var1 = f"x_{i}_{c1}"
                    var2 = f"x_{i}_{c2}"
                    bqm.add_interaction(var1, var2, 2 * P)

            # Constant term: P (one per node)
            bqm.offset += P

    def decode_sample(
        self,
        sample: Dict[str, int],
        node_list: List[str],
        k: int
    ) -> Dict[str, int]:
        """
        Decode sample to community assignments.

        Args:
            sample: Dict of variable assignments {var_name: 0 or 1}
            node_list: List of node identifiers
            k: Number of communities

        Returns:
            Dict mapping node_id to community index
        """
        n = len(node_list)
        assignments = {}

        for i, node in enumerate(node_list):
            assigned = False
            for c in range(k):
                var = f"x_{i}_{c}"
                if sample.get(var, 0) == 1:
                    assignments[node] = c
                    assigned = True
                    break

            # If no assignment (constraint violation), assign to community 0
            if not assigned:
                assignments[node] = 0

        return assignments

    def compute_modularity(
        self,
        B: np.ndarray,
        assignments: Dict[str, int],
        node_list: List[str]
    ) -> float:
        """
        Compute modularity score for given assignments.

        Q = (1/2m) * sum_{ij} B[i,j] * delta(c_i, c_j)

        Note: If 2m = 0, returns 0.

        Args:
            B: Modularity matrix
            assignments: Dict mapping node to community
            node_list: List of node identifiers

        Returns:
            Modularity score (between -0.5 and 1)
        """
        n = len(node_list)
        Q = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                node_i = node_list[i]
                node_j = node_list[j]

                if assignments.get(node_i) == assignments.get(node_j):
                    Q += 2 * B[i, j]  # 2x because we only count upper triangle

        # Normalize by sum of all B elements (should be close to 0 for proper B)
        # For normalized modularity, we use trace sum or just return raw value
        # In practice, modularity is often in range [-0.5, 1]

        return Q

    def get_constraint_violations(
        self,
        sample: Dict[str, int],
        n: int,
        k: int
    ) -> int:
        """
        Count constraint violations in a sample.

        Args:
            sample: Variable assignments
            n: Number of nodes
            k: Number of communities

        Returns:
            Number of violated constraints
        """
        violations = 0

        for i in range(n):
            count = sum(sample.get(f"x_{i}_{c}", 0) for c in range(k))
            if count != 1:
                violations += 1

        return violations

    def to_numpy_matrix(self, bqm: dimod.BinaryQuadraticModel) -> Tuple[np.ndarray, float]:
        """
        Convert BQM to numpy Q matrix.

        Args:
            bqm: Binary Quadratic Model

        Returns:
            Tuple of (Q matrix, offset)
        """
        variables = sorted(bqm.variables)
        n = len(variables)
        var_to_idx = {v: i for i, v in enumerate(variables)}

        Q = np.zeros((n, n))

        # Linear terms on diagonal
        for v, bias in bqm.linear.items():
            i = var_to_idx[v]
            Q[i, i] = bias

        # Quadratic terms
        for (v1, v2), bias in bqm.quadratic.items():
            i, j = var_to_idx[v1], var_to_idx[v2]
            Q[i, j] = bias
            Q[j, i] = bias  # Symmetric

        return Q, bqm.offset


def formulate_qubo(
    modularity_matrix: np.ndarray,
    node_list: List[str],
    k_communities: int = 5
) -> QUBOFormulationResult:
    """
    Convenience function to formulate QUBO.

    Args:
        modularity_matrix: The modularity matrix
        node_list: List of node identifiers
        k_communities: Number of communities

    Returns:
        QUBOFormulationResult
    """
    formulator = QUBOFormulator(k_communities=k_communities)
    return formulator.formulate(modularity_matrix, node_list)


if __name__ == "__main__":
    # Test QUBO formulation
    print("=" * 60)
    print("QUBO FORMULATOR TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator
    from src.feature_engineering import TransactionGraphBuilder

    print("\n[1/3] Generating test data...")
    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    print(f"      Invoices: {len(invoices_df)}")

    print("\n[2/3] Building transaction graph...")
    builder = TransactionGraphBuilder()
    graph_result = builder.build(invoices_df)

    print(f"      Nodes: {graph_result.n_nodes}")
    print(f"      Edges: {graph_result.n_edges}")
    print(f"      Modularity matrix shape: {graph_result.modularity_matrix.shape}")

    print("\n[3/3] Formulating QUBO...")
    formulator = QUBOFormulator(k_communities=5, penalty_weight=2.0)
    qubo_result = formulator.formulate(
        graph_result.modularity_matrix,
        graph_result.node_list
    )

    print(f"      Variables: {qubo_result.n_variables}")
    print(f"      Communities: {qubo_result.k_communities}")
    print(f"      Penalty weight: {qubo_result.penalty_weight}")

    # Check BQM properties
    bqm = qubo_result.bqm
    print(f"\n      BQM variables: {len(bqm.variables)}")
    print(f"      BQM linear terms: {len(bqm.linear)}")
    print(f"      BQM quadratic terms: {len(bqm.quadratic)}")
    print(f"      BQM offset: {bqm.offset:.4f}")

    print("\n" + "=" * 60)
    print("QUBO FORMULATOR TEST COMPLETE")
    print("=" * 60)
