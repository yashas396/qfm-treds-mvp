"""
QGAI Quantum Financial Modeling - TReDS MVP
Graph Builder Module

This module constructs the transaction graph for QUBO-based ring detection:
- Nodes: All entities (buyers + suppliers)
- Edges: Transaction relationships with weights
- Modularity matrix for community detection

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import get_config


@dataclass
class GraphBuildResult:
    """Result container for graph construction."""
    graph: nx.DiGraph
    undirected_graph: nx.Graph
    n_nodes: int
    n_edges: int
    node_list: List[str]
    adjacency_matrix: np.ndarray
    modularity_matrix: Optional[np.ndarray]
    total_edge_weight: float


class TransactionGraphBuilder:
    """
    Build transaction graph from invoice data for QUBO ring detection.

    This class constructs:
    - Directed weighted graph (buyer → supplier)
    - Undirected version for community detection
    - Adjacency and modularity matrices

    The modularity matrix B is used in QUBO formulation:
    B[i,j] = A[i,j] - (k_i * k_j) / (2m)

    Attributes:
        weight_by: Edge weight attribute ('amount', 'count', or 'both')
        min_edge_weight: Minimum edge weight to include

    Example:
        >>> builder = TransactionGraphBuilder()
        >>> result = builder.build(invoices_df)
        >>> print(f"Graph has {result.n_nodes} nodes, {result.n_edges} edges")
    """

    def __init__(
        self,
        weight_by: str = 'amount',
        min_edge_weight: float = 0,
        normalize_weights: bool = True
    ):
        """
        Initialize TransactionGraphBuilder.

        Args:
            weight_by: How to weight edges ('amount', 'count', 'both')
            min_edge_weight: Minimum edge weight to include
            normalize_weights: Whether to normalize edge weights
        """
        self.weight_by = weight_by
        self.min_edge_weight = min_edge_weight
        self.normalize_weights = normalize_weights

    def build(
        self,
        invoices_df: pd.DataFrame,
        compute_modularity: bool = True
    ) -> GraphBuildResult:
        """
        Build transaction graph from invoices.

        Args:
            invoices_df: DataFrame with invoice data
            compute_modularity: Whether to compute modularity matrix

        Returns:
            GraphBuildResult: Result with graph and matrices
        """
        # Build directed graph
        G = self._build_directed_graph(invoices_df)

        # Convert to undirected for community detection
        G_undirected = self._to_undirected(G)

        # Get node list
        node_list = list(G_undirected.nodes())

        # Build adjacency matrix
        adjacency_matrix = nx.to_numpy_array(G_undirected, nodelist=node_list)

        # Compute modularity matrix
        modularity_matrix = None
        if compute_modularity:
            modularity_matrix = self._compute_modularity_matrix(
                G_undirected, node_list, adjacency_matrix
            )

        # Calculate total edge weight
        total_weight = sum(data.get('weight', 1) for _, _, data in G.edges(data=True))

        return GraphBuildResult(
            graph=G,
            undirected_graph=G_undirected,
            n_nodes=G_undirected.number_of_nodes(),
            n_edges=G_undirected.number_of_edges(),
            node_list=node_list,
            adjacency_matrix=adjacency_matrix,
            modularity_matrix=modularity_matrix,
            total_edge_weight=total_weight
        )

    def _build_directed_graph(self, invoices_df: pd.DataFrame) -> nx.DiGraph:
        """Build directed weighted transaction graph."""
        G = nx.DiGraph()

        # Aggregate by buyer-supplier pair
        if self.weight_by == 'amount':
            edge_data = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
                'amount': 'sum',
                'invoice_id': 'count'
            }).reset_index()
            edge_data.columns = ['buyer_id', 'supplier_id', 'weight', 'count']

        elif self.weight_by == 'count':
            edge_data = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
                'invoice_id': 'count',
                'amount': 'sum'
            }).reset_index()
            edge_data.columns = ['buyer_id', 'supplier_id', 'weight', 'total_amount']

        else:  # 'both'
            edge_data = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
                'amount': 'sum',
                'invoice_id': 'count'
            }).reset_index()
            edge_data.columns = ['buyer_id', 'supplier_id', 'amount', 'count']
            # Combined weight
            edge_data['weight'] = np.sqrt(edge_data['amount']) * edge_data['count']

        # Filter by minimum weight
        edge_data = edge_data[edge_data['weight'] >= self.min_edge_weight]

        # Normalize weights if requested
        if self.normalize_weights and len(edge_data) > 0:
            max_weight = edge_data['weight'].max()
            if max_weight > 0:
                edge_data['weight'] = edge_data['weight'] / max_weight

        # Add edges to graph
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['buyer_id'],
                row['supplier_id'],
                weight=row['weight'],
                **{k: row[k] for k in row.index if k not in ['buyer_id', 'supplier_id', 'weight']}
            )

        return G

    def _to_undirected(self, G: nx.DiGraph) -> nx.Graph:
        """
        Convert directed graph to undirected.

        For edges in both directions, sum the weights.
        """
        G_undirected = nx.Graph()

        # Add all nodes
        G_undirected.add_nodes_from(G.nodes())

        # Add edges with combined weights
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1)
            if G_undirected.has_edge(u, v):
                # Add to existing weight
                G_undirected[u][v]['weight'] += weight
            else:
                G_undirected.add_edge(u, v, weight=weight)

        return G_undirected

    def _compute_modularity_matrix(
        self,
        G: nx.Graph,
        node_list: List[str],
        adjacency_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Compute modularity matrix B.

        B[i,j] = A[i,j] - (k_i * k_j) / (2m)

        Where:
        - A is the adjacency matrix
        - k_i is the degree of node i
        - m is the total number of edges (or sum of weights)
        """
        n = len(node_list)

        # Get weighted degrees
        degrees = np.array([
            G.degree(node, weight='weight') for node in node_list
        ])

        # Total edge weight (2m in the formula)
        m = G.size(weight='weight')

        if m == 0:
            return np.zeros((n, n))

        # Modularity matrix: B[i,j] = A[i,j] - (k_i * k_j) / (2m)
        expected = np.outer(degrees, degrees) / (2 * m)
        B = adjacency_matrix - expected

        return B

    def get_subgraph(
        self,
        G: nx.Graph,
        node_ids: List[str]
    ) -> nx.Graph:
        """
        Extract subgraph for specific nodes.

        Args:
            G: Original graph
            node_ids: List of node IDs to include

        Returns:
            Subgraph containing only specified nodes
        """
        return G.subgraph(node_ids).copy()

    def get_graph_statistics(self, result: GraphBuildResult) -> Dict:
        """
        Compute graph statistics.

        Args:
            result: GraphBuildResult from build()

        Returns:
            Dict with graph statistics
        """
        G = result.undirected_graph

        stats = {
            'n_nodes': result.n_nodes,
            'n_edges': result.n_edges,
            'density': nx.density(G),
            'total_weight': result.total_edge_weight,
        }

        if result.n_nodes > 0:
            # Degree statistics
            degrees = [d for _, d in G.degree()]
            stats['avg_degree'] = np.mean(degrees)
            stats['max_degree'] = max(degrees)
            stats['min_degree'] = min(degrees)

            # Weighted degree statistics
            w_degrees = [d for _, d in G.degree(weight='weight')]
            stats['avg_weighted_degree'] = np.mean(w_degrees)

            # Clustering coefficient (if graph has enough structure)
            if result.n_edges > 0:
                stats['avg_clustering'] = nx.average_clustering(G)

            # Connected components
            stats['n_components'] = nx.number_connected_components(G)

        return stats

    def find_dense_subgraphs(
        self,
        G: nx.Graph,
        min_density: float = 0.5,
        min_size: int = 3
    ) -> List[Set[str]]:
        """
        Find dense subgraphs (potential rings).

        Args:
            G: Graph to analyze
            min_density: Minimum density threshold
            min_size: Minimum subgraph size

        Returns:
            List of node sets forming dense subgraphs
        """
        dense_subgraphs = []

        # Use clique finding as starting point
        for clique in nx.find_cliques(G):
            if len(clique) >= min_size:
                subgraph = G.subgraph(clique)
                density = nx.density(subgraph)
                if density >= min_density:
                    dense_subgraphs.append(set(clique))

        return dense_subgraphs


def build_transaction_graph(
    invoices_df: pd.DataFrame,
    weight_by: str = 'amount'
) -> GraphBuildResult:
    """
    Convenience function to build transaction graph.

    Args:
        invoices_df: DataFrame with invoice data
        weight_by: Edge weight method

    Returns:
        GraphBuildResult: Graph and matrices
    """
    builder = TransactionGraphBuilder(weight_by=weight_by)
    return builder.build(invoices_df)


if __name__ == "__main__":
    # Test graph building
    print("=" * 60)
    print("TRANSACTION GRAPH BUILDER TEST")
    print("=" * 60)

    # Generate test data
    from src.data_generation import EntityGenerator, InvoiceGenerator

    entity_gen = EntityGenerator()
    entities_df = entity_gen.generate()

    invoice_gen = InvoiceGenerator()
    invoices_df = invoice_gen.generate(entities_df)

    # Build graph
    builder = TransactionGraphBuilder(weight_by='amount')
    result = builder.build(invoices_df)

    print(f"\nGraph Statistics:")
    print(f"  Nodes: {result.n_nodes}")
    print(f"  Edges: {result.n_edges}")
    print(f"  Total weight: {result.total_edge_weight:.2f}")

    stats = builder.get_graph_statistics(result)
    print(f"\n  Density: {stats['density']:.4f}")
    print(f"  Avg degree: {stats['avg_degree']:.2f}")
    print(f"  Components: {stats['n_components']}")
    print(f"  Avg clustering: {stats.get('avg_clustering', 0):.4f}")

    print(f"\nAdjacency matrix shape: {result.adjacency_matrix.shape}")
    print(f"Modularity matrix shape: {result.modularity_matrix.shape}")

    # Find dense subgraphs
    dense = builder.find_dense_subgraphs(result.undirected_graph, min_density=0.3, min_size=3)
    print(f"\nDense subgraphs (potential rings): {len(dense)}")
    for i, subgraph in enumerate(dense[:3]):
        print(f"  Subgraph {i+1}: {len(subgraph)} nodes")
