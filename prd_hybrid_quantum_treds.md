# Product Requirements Document (PRD)

## Hybrid Classical-Quantum TReDS Invoice Fraud Detection System

### Targeting & Prediction | Quantum Financial Modeling

**Document Classification:** Internal - Confidential  
**Version:** 1.0  
**Status:** Draft for Executive Review  
**Date:** January 6, 2026  
**Author:** QGAI Quantum Financial Modeling Team  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Understanding](#2-problem-understanding)
3. [Hybrid Methodology Overview](#3-hybrid-methodology-overview)
4. [MVP Architecture](#4-mvp-architecture)
5. [Classical Component: Default Prediction](#5-classical-component-default-prediction)
6. [Quantum Component: QUBO Ring Detection](#6-quantum-component-qubo-ring-detection)
7. [Feature Engineering Design](#7-feature-engineering-design)
8. [Synthetic Data Generation](#8-synthetic-data-generation)
9. [Implementation Specifications](#9-implementation-specifications)
10. [Explainability Framework](#10-explainability-framework)
11. [Sample Outputs & Targeting Strategy](#11-sample-outputs--targeting-strategy)
12. [Future Quantum Hardware Integration](#12-future-quantum-hardware-integration)
13. [Technical Appendices](#13-technical-appendices)

---

## 1. Executive Summary

### 1.1 Purpose

This document defines requirements for a **Hybrid Classical-Quantum System** for trade finance invoice risk assessment. The system employs a principled division of labor:

| Problem | Method | Rationale |
|---------|--------|-----------|
| **Invoice Default Prediction** | Classical ML (Random Forest) | Supervised classification with labeled training data |
| **Fake Invoice Ring Detection** | Quantum Optimization (QUBO) | Community detection is NP-hard; maps naturally to quantum annealing |

### 1.2 Why Hybrid? The Right Tool for Each Problem

**Default Prediction** is a supervised learning problem:
- We have labeled data (invoices that defaulted vs. settled)
- Features predict a binary outcome
- Classical ML excels here; quantum offers no structural advantage

**Ring Detection** is a combinatorial optimization problem:
- Find community structure in transaction network that reveals fraud rings
- This is modularity maximization—an NP-hard problem
- Maps directly to QUBO (Quadratic Unconstrained Binary Optimization)
- Quantum annealing explores solution space via quantum tunneling
- D-Wave and QAOA implementations are well-established

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    HYBRID ARCHITECTURE SUMMARY                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  CLASSICAL TRACK                    QUANTUM TRACK                       │
│  ───────────────                    ─────────────                       │
│                                                                         │
│  Invoice Features ──▶ Random     Transaction Graph ──▶ QUBO            │
│                       Forest                           Formulation      │
│        │                │               │                   │           │
│        ▼                ▼               ▼                   ▼           │
│  P(default)        Feature       Modularity          Simulated          │
│                   Importance      Matrix             Annealing /        │
│                                                      D-Wave Hybrid      │
│        │                                                    │           │
│        │                                                    ▼           │
│        │                                            Community           │
│        │                                            Assignments         │
│        │                                                    │           │
│        └──────────────────┬─────────────────────────────────┘           │
│                           │                                             │
│                           ▼                                             │
│                   COMPOSITE RISK SCORE                                  │
│                   + Ring Membership                                     │
│                   + Targeting List                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Strategic Value

| Dimension | Value Proposition |
|-----------|------------------|
| **India Exclusivity** | TReDS + GST e-Invoice integration—uniquely Indian infrastructure |
| **Quantum Problem Fit** | Community detection has proven QUBO formulation (Negre et al., 2020) |
| **Competitive Moat** | First quantum-native trade finance fraud system in India |
| **Regulatory Alignment** | RBI mandate driving TReDS adoption (₹250Cr+ companies must register) |
| **MVP to Production Path** | Simulated annealing → D-Wave Advantage → Gate-based QAOA |

### 1.4 Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Default Model AUC-ROC | ≥ 0.75 | Cross-validated on synthetic data |
| Ring Detection Modularity | ≥ 0.3 | QUBO solution quality (benchmark: Louvain) |
| Ring Recovery Rate | ≥ 70% | Known synthetic rings correctly identified |
| Explainability Coverage | 100% | All predictions have justification |
| Quantum Readiness | Full | Architecture supports D-Wave swap-in |

---

## 2. Problem Understanding

### 2.1 TReDS Ecosystem Context

**Trade Receivables Discounting System (TReDS)** is an RBI-regulated platform enabling MSMEs to convert unpaid invoices into immediate working capital.

```
┌─────────────┐     Invoice      ┌─────────────┐
│   MSME      │ ────────────────▶│  Corporate  │
│  Supplier   │                  │   Buyer     │
└──────┬──────┘                  └──────┬──────┘
       │                                │
       │ Upload                         │ Accept
       ▼                                ▼
┌─────────────────────────────────────────────────┐
│                TReDS Platform                    │
│  (RXIL, M1xchange, Invoicemart, KredX DTX)      │
└─────────────────────────────────────────────────┘
       │                                │
       │ Auction                        │ Auto-debit (Due Date)
       ▼                                ▼
┌─────────────┐                  ┌─────────────┐
│  Financier  │ ◀───── Funds ────│   Buyer's   │
│  (Bank/NBFC)│                  │    Bank     │
└─────────────┘                  └─────────────┘
```

**Operational Parameters:**
- Discount rates: 2-4% annualized
- Payment terms: 30-90 days
- Default rate: ~1-3% (industry estimate)
- Without recourse: Financier bears default risk

### 2.2 Problem 1: Invoice Default Risk

**Nature:** Binary classification (defaulted vs. settled)  
**Method:** Classical supervised learning  
**Why Classical:** Labeled training data available; no combinatorial structure

**Risk Drivers:**
| Category | Features |
|----------|----------|
| Buyer creditworthiness | Credit rating, payment history, financial health |
| Invoice characteristics | Amount, timing, acceptance delay |
| Relationship dynamics | Age, frequency, concentration |
| Platform signals | Late acceptances, disputes |

### 2.3 Problem 2: Fake Invoice Ring Detection

**Nature:** Community detection in transaction graph  
**Method:** QUBO-based quantum optimization  
**Why Quantum:** NP-hard combinatorial optimization; proven QUBO mapping

#### 2.3.1 Ring Fraud Mechanism

Coordinated entities create circular invoice patterns to extract cash:

```
        Entity A (Buyer)
            │
   Invoice  │  ▲  Invoice
   A→B      │  │  C→A
            ▼  │
        Entity B ──────▶ Entity C
        (Supplier)  Invoice B→C

Each invoice in the cycle is discounted → Cash extraction
Ring appears as tight-knit community in transaction graph
```

#### 2.3.2 Why Graph Community Detection?

| Detection Approach | Capability | Limitation |
|-------------------|------------|------------|
| Rule-based | Catches known patterns | Easily evaded |
| Invoice-level ML | Anomaly detection | Cannot see network structure |
| Entity-level ML | Behavioral patterns | Cannot detect coordination |
| **Graph Community Detection** | **Reveals ring structure** | **Computationally hard** |

**Key Insight:** Fraud rings form densely connected communities within the transaction graph. Finding these communities is the modularity maximization problem—which maps to QUBO.

### 2.4 Academic Foundation

| Source | Contribution |
|--------|--------------|
| Negre et al. (PLOS ONE, 2020) | QUBO formulation for modularity-based community detection |
| AWS Quantum Blog (2023) | QBSolv hybrid solver for community detection |
| D-Wave Documentation | dimod, neal libraries for QUBO solving |
| safe-graph/graph-fraud-detection-papers | GNN approaches for fraud ring detection |
| Feld et al. (2019) | Hybrid quantum-classical optimization patterns |

---

## 3. Hybrid Methodology Overview

### 3.1 Division of Labor

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     METHODOLOGY ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                        RAW DATA                                         │
│                          │                                              │
│          ┌───────────────┴───────────────┐                              │
│          │                               │                              │
│          ▼                               ▼                              │
│   ┌─────────────┐                 ┌─────────────┐                       │
│   │  INVOICE    │                 │ TRANSACTION │                       │
│   │  FEATURES   │                 │   GRAPH     │                       │
│   │             │                 │             │                       │
│   │ Amount, due │                 │ Nodes:      │                       │
│   │ date, delay │                 │  Entities   │                       │
│   │ buyer stats │                 │ Edges:      │                       │
│   │ etc.        │                 │  Invoices   │                       │
│   └──────┬──────┘                 └──────┬──────┘                       │
│          │                               │                              │
│          │ CLASSICAL                     │ QUANTUM                      │
│          │                               │                              │
│          ▼                               ▼                              │
│   ┌─────────────┐                 ┌─────────────┐                       │
│   │  RANDOM     │                 │   QUBO      │                       │
│   │  FOREST     │                 │ MODULARITY  │                       │
│   │             │                 │             │                       │
│   │ Supervised  │                 │ H = -x'Bx   │                       │
│   │ classifica- │                 │ + P*penalty │                       │
│   │ tion        │                 │             │                       │
│   └──────┬──────┘                 └──────┬──────┘                       │
│          │                               │                              │
│          ▼                               ▼                              │
│   ┌─────────────┐                 ┌─────────────┐                       │
│   │ P(default)  │                 │ ANNEALING   │                       │
│   │             │                 │ SOLVER      │                       │
│   │ Per-invoice │                 │             │                       │
│   │ probability │                 │ SimAnneal / │                       │
│   │             │                 │ D-Wave      │                       │
│   └──────┬──────┘                 └──────┬──────┘                       │
│          │                               │                              │
│          │                               ▼                              │
│          │                        ┌─────────────┐                       │
│          │                        │ COMMUNITY   │                       │
│          │                        │ ASSIGNMENTS │                       │
│          │                        │             │                       │
│          │                        │ Entity →    │                       │
│          │                        │ Community   │                       │
│          │                        │ mapping     │                       │
│          │                        └──────┬──────┘                       │
│          │                               │                              │
│          │                               ▼                              │
│          │                        ┌─────────────┐                       │
│          │                        │ RING        │                       │
│          │                        │ SCORING     │                       │
│          │                        │             │                       │
│          │                        │ Community   │                       │
│          │                        │ → Ring prob │                       │
│          │                        └──────┬──────┘                       │
│          │                               │                              │
│          └───────────────┬───────────────┘                              │
│                          │                                              │
│                          ▼                                              │
│                   ┌─────────────┐                                       │
│                   │ COMPOSITE   │                                       │
│                   │ RISK SCORE  │                                       │
│                   │             │                                       │
│                   │ Default +   │                                       │
│                   │ Ring Risk   │                                       │
│                   └─────────────┘                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why This Hybrid Approach Works

| Aspect | Classical ML | Quantum Optimization |
|--------|--------------|---------------------|
| **Problem Type** | Supervised learning | Combinatorial optimization |
| **Input** | Feature vectors | Graph structure |
| **Output** | Probability scores | Optimal partitions |
| **Training** | Learns from labeled data | Minimizes energy function |
| **Strength** | Pattern recognition | Solution space exploration |
| **Default Prediction** | ✓ Appropriate | ✗ No structural advantage |
| **Ring Detection** | ✗ Cannot see graph structure | ✓ Natural QUBO formulation |

### 3.3 Quantum Advantage Hypothesis

For ring detection via community detection:

1. **Classical Heuristics (Louvain, Girvan-Newman):** Greedy local optimization; can get stuck in local minima
2. **Quantum Annealing:** Explores solution space via quantum tunneling; can escape local minima
3. **Adversarial Structure:** Fraud rings are designed to evade detection; quantum may find non-obvious partitions

**Research Support:** Negre et al. (2020) demonstrated quantum annealing achieving comparable or better modularity than classical methods on benchmark networks.

---

## 4. MVP Architecture

### 4.1 System Components

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MVP SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐                                                    │
│  │  SYNTHETIC DATA │                                                    │
│  │   GENERATOR     │                                                    │
│  │                 │                                                    │
│  │ • Entities      │                                                    │
│  │ • Invoices      │                                                    │
│  │ • Known rings   │                                                    │
│  │ • Default labels│                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    DATA PROCESSING LAYER                         │   │
│  │                                                                   │   │
│  │  ┌───────────────┐              ┌───────────────────────────────┐│   │
│  │  │ Feature       │              │ Graph Construction             ││   │
│  │  │ Engineering   │              │                                ││   │
│  │  │               │              │ • Adjacency matrix            ││   │
│  │  │ Invoice, buyer│              │ • Edge weights (amounts)      ││   │
│  │  │ supplier,     │              │ • Modularity matrix B         ││   │
│  │  │ relationship  │              │                                ││   │
│  │  └───────────────┘              └───────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                                    │                        │
│           │                                    │                        │
│           ▼                                    ▼                        │
│  ┌─────────────────────┐          ┌─────────────────────────────────┐  │
│  │ CLASSICAL COMPONENT │          │ QUANTUM COMPONENT                │  │
│  │                     │          │                                  │  │
│  │ Random Forest       │          │ QUBO Formulation                 │  │
│  │ Classifier          │          │ ────────────────                 │  │
│  │                     │          │ H = -x'Bx + P*(constraints)      │  │
│  │ Features → P(def)   │          │                                  │  │
│  │                     │          │ Solver: SimulatedAnnealingSampler│  │
│  │ sklearn             │          │ (D-Wave dimod/neal)              │  │
│  └──────────┬──────────┘          │                                  │  │
│             │                     │ Output: Community assignments    │  │
│             │                     └───────────────┬─────────────────┘  │
│             │                                     │                     │
│             │                                     ▼                     │
│             │                     ┌─────────────────────────────────┐  │
│             │                     │ RING PROBABILITY SCORING        │  │
│             │                     │                                  │  │
│             │                     │ Community properties:            │  │
│             │                     │ • Size, density                  │  │
│             │                     │ • Internal edge weight           │  │
│             │                     │ • Entity age distribution        │  │
│             │                     │ → P(ring | community)            │  │
│             │                     └───────────────┬─────────────────┘  │
│             │                                     │                     │
│             └──────────────┬──────────────────────┘                     │
│                            │                                            │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      OUTPUT LAYER                                │   │
│  │                                                                   │   │
│  │  • Invoice-level: P(default), community_id, P(ring)              │   │
│  │  • Entity-level: Community membership, ring association          │   │
│  │  • Network-level: Detected communities, ring candidates          │   │
│  │  • Ranked targeting list for investigation                       │   │
│  │  • Explainability reports (SHAP + community structure)           │   │
│  │                                                                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Language | Python 3.9+ | Primary implementation |
| Classical ML | scikit-learn 1.3+ | Random Forest, evaluation |
| Quantum/QUBO | dimod, dwave-samplers | QUBO modeling, simulated annealing |
| Graph Operations | networkx | Graph construction, metrics |
| Data Handling | pandas, numpy | Data manipulation |
| Explainability | shap | Model interpretation |
| Visualization | matplotlib, seaborn | Reporting |

### 4.3 MVP Scope Boundaries

| In Scope | Out of Scope |
|----------|--------------|
| Synthetic TReDS-like data | Real bank/platform data |
| Classical default prediction | Deep learning models |
| QUBO community detection | Actual D-Wave hardware (simulated) |
| Simulated annealing solver | Gate-based QAOA |
| Console/file outputs | Production dashboards |
| SHAP explainability | Real-time API |

---

## 5. Classical Component: Default Prediction

### 5.1 Problem Formulation

**Input:** Invoice feature vector x ∈ ℝ^d  
**Output:** P(default | x) ∈ [0, 1]  
**Model:** Random Forest Classifier

### 5.2 Feature Set

```python
DEFAULT_PREDICTION_FEATURES = [
    # Invoice-level
    'amount_log',                    # log(1 + invoice_amount)
    'days_to_due',                   # Due date - Invoice date
    'acceptance_delay_days',         # Days for buyer to accept
    'amount_zscore_buyer',           # (amount - buyer_mean) / buyer_std
    'is_round_amount',               # amount % 10000 == 0
    'is_month_end',                  # Invoice day >= 25
    
    # Buyer-level
    'buyer_total_invoices_30d',      # Transaction velocity
    'buyer_avg_invoice_amount',      # Typical transaction size
    'buyer_acceptance_rate',         # Historical acceptance ratio
    'buyer_avg_acceptance_delay',    # Typical processing time
    'buyer_default_rate_historical', # Past default behavior
    'buyer_credit_rating_encoded',   # Categorical → numeric
    
    # Supplier-level
    'supplier_age_days',             # Platform tenure
    'supplier_unique_buyers_30d',    # Customer diversity
    
    # Relationship-level
    'relationship_age_days',         # Trading history length
    'relationship_invoice_count',    # Transaction frequency
    'is_new_relationship',           # < 30 days
]
```

### 5.3 Model Configuration

```python
from sklearn.ensemble import RandomForestClassifier

DEFAULT_MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 5,
    'class_weight': 'balanced',  # Handle class imbalance
    'random_state': 42,
    'n_jobs': -1
}

default_model = RandomForestClassifier(**DEFAULT_MODEL_CONFIG)
```

### 5.4 Evaluation Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| AUC-ROC | ≥ 0.75 | Overall discrimination ability |
| Average Precision | ≥ 0.30 | Performance on minority class |
| Recall @ 10% FPR | ≥ 0.50 | Catch rate at acceptable false positive level |
| F1-Score | ≥ 0.40 | Balance of precision and recall |

---

## 6. Quantum Component: QUBO Ring Detection

### 6.1 Problem Formulation: Community Detection as QUBO

**Goal:** Partition transaction graph into communities that maximize modularity  
**Modularity Q** measures quality of partition:

```
Q = (1/2m) Σᵢⱼ [Aᵢⱼ - (kᵢkⱼ/2m)] δ(cᵢ, cⱼ)
```

Where:
- A = adjacency matrix
- kᵢ = degree of node i
- m = total edges
- δ(cᵢ, cⱼ) = 1 if nodes i,j in same community

**Modularity Matrix B:**
```
Bᵢⱼ = Aᵢⱼ - (kᵢkⱼ/2m)
```

**QUBO Formulation (k communities):**

For each node i and community c, binary variable xᵢ,c = 1 if node i in community c.

**Objective (Maximize Modularity):**
```
H_modularity = -(1/m) Σc xc' B xc
```

**Constraint (Each node in exactly one community):**
```
Σc xᵢ,c = 1  for all nodes i
```

**Penalty Form:**
```
H_penalty = P * Σᵢ (Σc xᵢ,c - 1)²
```

**Combined QUBO Hamiltonian:**
```
H = H_modularity + H_penalty = x' Q x
```

Where Q is the QUBO matrix encoding both objective and constraints.

### 6.2 QUBO Construction Code

```python
import numpy as np
import networkx as nx
from dimod import BinaryQuadraticModel

def build_modularity_qubo(G, k_communities, penalty_weight=1.0):
    """
    Build QUBO for k-community modularity maximization.
    
    Args:
        G: networkx Graph (transaction network)
        k_communities: Number of communities to find
        penalty_weight: Weight for one-hot constraint penalty
        
    Returns:
        BinaryQuadraticModel ready for sampling
    """
    n_nodes = G.number_of_nodes()
    nodes = list(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Build adjacency and modularity matrices
    A = nx.to_numpy_array(G)
    degrees = np.array([G.degree(node) for node in nodes])
    m = G.number_of_edges()
    
    if m == 0:
        raise ValueError("Graph has no edges")
    
    # Modularity matrix: B[i,j] = A[i,j] - (k_i * k_j) / (2m)
    B = A - np.outer(degrees, degrees) / (2 * m)
    
    # Build QUBO dictionary
    Q = {}
    n_vars = n_nodes * k_communities
    
    # Variable naming: x_{node_idx}_{community_idx}
    def var_name(node_idx, comm_idx):
        return f"x_{node_idx}_{comm_idx}"
    
    # Modularity terms: -B[i,j] * x_{i,c} * x_{j,c} for same community
    for i in range(n_nodes):
        for j in range(i, n_nodes):
            for c in range(k_communities):
                vi = var_name(i, c)
                vj = var_name(j, c)
                
                coeff = -B[i, j] / m  # Negative for minimization
                
                if i == j:
                    # Linear term (diagonal)
                    Q[(vi, vi)] = Q.get((vi, vi), 0) + coeff
                else:
                    # Quadratic term
                    Q[(vi, vj)] = Q.get((vi, vj), 0) + 2 * coeff
    
    # One-hot constraint penalty: P * (Σc x_{i,c} - 1)²
    # Expands to: P * (Σc x_{i,c})² - 2P * (Σc x_{i,c}) + P
    # = P * Σc Σc' x_{i,c} * x_{i,c'} - 2P * Σc x_{i,c} + P
    
    P = penalty_weight
    for i in range(n_nodes):
        # Quadratic terms within same node across communities
        for c1 in range(k_communities):
            for c2 in range(c1, k_communities):
                vi = var_name(i, c1)
                vj = var_name(i, c2)
                
                if c1 == c2:
                    # x_{i,c}² = x_{i,c} (binary), contributes: P - 2P = -P
                    Q[(vi, vi)] = Q.get((vi, vi), 0) + P - 2*P
                else:
                    # Cross terms: 2P * x_{i,c1} * x_{i,c2}
                    Q[(vi, vj)] = Q.get((vi, vj), 0) + 2 * P
    
    # Create BQM from QUBO
    bqm = BinaryQuadraticModel.from_qubo(Q)
    
    return bqm, nodes, k_communities

def decode_community_solution(sample, nodes, k_communities):
    """
    Decode QUBO solution to community assignments.
    
    Args:
        sample: Dict of variable assignments from solver
        nodes: List of node identifiers
        k_communities: Number of communities
        
    Returns:
        Dict mapping node → community_id
    """
    assignments = {}
    
    for i, node in enumerate(nodes):
        for c in range(k_communities):
            var_name = f"x_{i}_{c}"
            if sample.get(var_name, 0) == 1:
                assignments[node] = c
                break
        else:
            # No community assigned (constraint violation)
            assignments[node] = -1
    
    return assignments
```

### 6.3 Solver Configuration

```python
from dwave.samplers import SimulatedAnnealingSampler

def solve_community_qubo(bqm, num_reads=1000, seed=42):
    """
    Solve community detection QUBO using simulated annealing.
    
    For MVP: Uses classical simulated annealing
    For Production: Swap in D-Wave hybrid sampler
    
    Args:
        bqm: BinaryQuadraticModel
        num_reads: Number of annealing runs
        seed: Random seed for reproducibility
        
    Returns:
        SampleSet with solutions
    """
    sampler = SimulatedAnnealingSampler()
    
    sampleset = sampler.sample(
        bqm,
        num_reads=num_reads,
        num_sweeps=1000,
        beta_range=[0.1, 3.0],
        seed=seed
    )
    
    return sampleset

def get_best_community_assignment(sampleset, nodes, k_communities):
    """
    Extract best solution from sample set.
    """
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    assignments = decode_community_solution(best_sample, nodes, k_communities)
    
    return assignments, best_energy
```

### 6.4 Ring Probability Scoring

After community detection, score each community for ring likelihood:

```python
def score_community_ring_likelihood(G, community_assignments):
    """
    Score each community for fraud ring characteristics.
    
    Ring indicators:
    - Small, tight-knit community (high density)
    - High internal transaction volume
    - Reciprocal edges (A→B and B→A)
    - Similar entity ages (created together)
    - Round transaction amounts
    
    Args:
        G: Transaction graph with node/edge attributes
        community_assignments: Dict node → community_id
        
    Returns:
        Dict community_id → ring_probability
    """
    communities = {}
    for node, comm_id in community_assignments.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    ring_scores = {}
    
    for comm_id, members in communities.items():
        if len(members) < 3:
            # Too small to be a ring
            ring_scores[comm_id] = 0.0
            continue
        
        # Extract subgraph
        subgraph = G.subgraph(members)
        
        # Feature 1: Density (rings are tightly connected)
        n = len(members)
        e = subgraph.number_of_edges()
        max_edges = n * (n - 1)  # Directed graph
        density = e / max_edges if max_edges > 0 else 0
        
        # Feature 2: Reciprocity (rings have bidirectional transactions)
        reciprocal_pairs = 0
        for u, v in subgraph.edges():
            if subgraph.has_edge(v, u):
                reciprocal_pairs += 1
        reciprocity = reciprocal_pairs / e if e > 0 else 0
        
        # Feature 3: Size penalty (rings are typically 3-15 entities)
        size_score = 1.0 if 3 <= n <= 15 else 0.5
        
        # Feature 4: Transaction regularity (similar amounts suggest coordination)
        amounts = [G[u][v].get('weight', 1) for u, v in subgraph.edges()]
        if len(amounts) > 1:
            cv = np.std(amounts) / np.mean(amounts) if np.mean(amounts) > 0 else 1
            regularity = 1 / (1 + cv)  # Lower CV = more regular = more suspicious
        else:
            regularity = 0.5
        
        # Combine scores
        ring_probability = (
            0.35 * density +
            0.35 * reciprocity +
            0.15 * size_score +
            0.15 * regularity
        )
        
        ring_scores[comm_id] = min(ring_probability, 1.0)
    
    return ring_scores
```

### 6.5 Complete Ring Detection Pipeline

```python
class QUBORingDetector:
    """
    Hybrid quantum-classical ring detection system.
    
    Uses QUBO formulation for community detection,
    then scores communities for ring characteristics.
    """
    
    def __init__(self, k_communities=5, penalty_weight=1.0, num_reads=1000):
        self.k_communities = k_communities
        self.penalty_weight = penalty_weight
        self.num_reads = num_reads
        self.bqm = None
        self.nodes = None
        self.community_assignments = None
        self.ring_scores = None
        
    def fit(self, invoices_df):
        """
        Build transaction graph and detect communities.
        
        Args:
            invoices_df: DataFrame with buyer_id, supplier_id, amount columns
        """
        # Build transaction graph
        G = self._build_transaction_graph(invoices_df)
        self.graph = G
        
        # Build QUBO
        self.bqm, self.nodes, _ = build_modularity_qubo(
            G, self.k_communities, self.penalty_weight
        )
        
        # Solve QUBO
        sampleset = solve_community_qubo(self.bqm, self.num_reads)
        
        # Decode solution
        self.community_assignments, self.energy = get_best_community_assignment(
            sampleset, self.nodes, self.k_communities
        )
        
        # Score communities for ring likelihood
        self.ring_scores = score_community_ring_likelihood(
            G, self.community_assignments
        )
        
        return self
    
    def _build_transaction_graph(self, invoices_df):
        """
        Construct weighted directed graph from invoices.
        """
        G = nx.DiGraph()
        
        # Aggregate invoices by buyer-supplier pair
        edge_data = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
            'amount': ['sum', 'count'],
            'invoice_id': 'count'
        }).reset_index()
        
        edge_data.columns = ['buyer_id', 'supplier_id', 'total_amount', 
                            'invoice_count', 'n_invoices']
        
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['buyer_id'],
                row['supplier_id'],
                weight=row['total_amount'],
                count=row['invoice_count']
            )
        
        return G
    
    def predict_entity_ring_probability(self):
        """
        Get ring probability for each entity based on community membership.
        """
        entity_probs = {}
        for node, comm_id in self.community_assignments.items():
            entity_probs[node] = self.ring_scores.get(comm_id, 0.0)
        return entity_probs
    
    def predict_invoice_ring_probability(self, invoices_df):
        """
        Get ring probability for each invoice (max of buyer/supplier).
        """
        entity_probs = self.predict_entity_ring_probability()
        
        def get_invoice_ring_prob(row):
            buyer_prob = entity_probs.get(row['buyer_id'], 0.0)
            supplier_prob = entity_probs.get(row['supplier_id'], 0.0)
            return max(buyer_prob, supplier_prob)
        
        return invoices_df.apply(get_invoice_ring_prob, axis=1)
    
    def get_detected_rings(self, threshold=0.5):
        """
        Return communities likely to be fraud rings.
        """
        rings = []
        for comm_id, score in self.ring_scores.items():
            if score >= threshold:
                members = [node for node, c in self.community_assignments.items() 
                          if c == comm_id]
                rings.append({
                    'community_id': comm_id,
                    'ring_probability': score,
                    'members': members,
                    'size': len(members)
                })
        return sorted(rings, key=lambda x: -x['ring_probability'])
```

---

## 7. Feature Engineering Design

### 7.1 Feature Taxonomy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FEATURE HIERARCHY                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  LEVEL 1: INVOICE FEATURES (Used by Classical Default Model)            │
│  ───────────────────────────────────────────────────────────            │
│  • Transaction-specific attributes                                      │
│  • Direct observables from single invoice                               │
│                                                                         │
│  LEVEL 2: ENTITY FEATURES (Used by Classical Default Model)             │
│  ───────────────────────────────────────────────────────────            │
│  • Aggregated buyer/supplier statistics                                 │
│  • Behavioral patterns                                                  │
│                                                                         │
│  LEVEL 3: GRAPH STRUCTURE (Used by Quantum Ring Detector)               │
│  ───────────────────────────────────────────────────────────            │
│  • Adjacency matrix                                                     │
│  • Edge weights (transaction volumes)                                   │
│  • Modularity matrix for QUBO                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Invoice-Level Features

| Feature | Formula | Type | Default Signal |
|---------|---------|------|----------------|
| `amount_log` | log(1 + amount) | Continuous | Medium |
| `days_to_due` | due_date - invoice_date | Integer | High |
| `acceptance_delay_days` | acceptance_date - invoice_date | Integer | High |
| `amount_zscore_buyer` | (amount - μ_buyer) / σ_buyer | Continuous | Medium |
| `is_round_amount` | amount % 10000 == 0 | Binary | Low |
| `is_month_end` | day >= 25 | Binary | Low |
| `time_since_last_invoice` | Days since previous in relationship | Integer | Medium |

### 7.3 Entity-Level Features

**Buyer Features:**

| Feature | Description | Default Signal |
|---------|-------------|----------------|
| `buyer_total_invoices_30d` | Transaction velocity | Medium |
| `buyer_unique_suppliers_30d` | Supplier diversity | Low |
| `buyer_avg_invoice_amount` | Typical transaction size | Medium |
| `buyer_acceptance_rate` | Historical acceptance ratio | High |
| `buyer_avg_acceptance_delay` | Processing time pattern | High |
| `buyer_default_rate_historical` | Past default behavior | Very High |
| `buyer_credit_rating_encoded` | AAA=5, AA=4, A=3, BBB=2, BB=1 | Very High |

**Supplier Features:**

| Feature | Description | Default Signal |
|---------|-------------|----------------|
| `supplier_age_days` | Platform tenure | Medium |
| `supplier_unique_buyers_30d` | Customer diversity | Low |
| `supplier_avg_invoice_amount` | Typical transaction size | Medium |

**Relationship Features:**

| Feature | Description | Default Signal |
|---------|-------------|----------------|
| `relationship_age_days` | Trading history length | High |
| `relationship_invoice_count` | Transaction frequency | Medium |
| `is_new_relationship` | relationship_age < 30 | High |

### 7.4 Graph Structure (For QUBO)

The QUBO ring detector uses the raw transaction graph structure:

- **Nodes:** All unique entities (buyers + suppliers)
- **Edges:** Directed edges from buyer → supplier
- **Edge Weights:** Total transaction amount between pair
- **Derived:** Modularity matrix B for community detection

---

## 8. Synthetic Data Generation

### 8.1 Design Requirements

| Requirement | Implementation |
|-------------|----------------|
| Realistic distributions | Invoice amounts follow log-normal |
| Known ground truth | Injected fraud rings with labels |
| TReDS characteristics | 30-90 day terms, 1-3% default rate |
| Network structure | Power-law degree distribution |
| Reproducibility | Seeded random generation |

### 8.2 Entity Generation

```python
def generate_entities(n_buyers=100, n_suppliers=200, n_rings=3, 
                      ring_sizes=[5, 7, 8], seed=42):
    """
    Generate synthetic entities including fraud ring participants.
    
    Args:
        n_buyers: Number of legitimate corporate buyers
        n_suppliers: Number of legitimate MSME suppliers
        n_rings: Number of fraud rings to inject
        ring_sizes: List of sizes for each ring
        seed: Random seed
        
    Returns:
        DataFrame with entity information and ring labels
    """
    np.random.seed(seed)
    entities = []
    
    # Legitimate buyers
    for i in range(n_buyers):
        entities.append({
            'entity_id': f'B{i:04d}',
            'entity_type': 'buyer',
            'registration_date': pd.Timestamp('2023-01-01') + 
                                pd.Timedelta(days=np.random.randint(0, 365)),
            'turnover_cr': np.random.lognormal(mean=6, sigma=1),
            'credit_rating': np.random.choice(
                ['AAA', 'AA', 'A', 'BBB', 'BB'], 
                p=[0.1, 0.25, 0.35, 0.2, 0.1]
            ),
            'industry_sector': np.random.choice(
                ['Manufacturing', 'IT', 'Pharma', 'Auto', 'FMCG']
            ),
            'is_ring_member': False,
            'ring_id': None
        })
    
    # Legitimate suppliers
    for i in range(n_suppliers):
        entities.append({
            'entity_id': f'S{i:04d}',
            'entity_type': 'supplier',
            'registration_date': pd.Timestamp('2023-01-01') + 
                                pd.Timedelta(days=np.random.randint(0, 365)),
            'turnover_cr': np.random.lognormal(mean=2, sigma=1.5),
            'credit_rating': np.random.choice(
                ['A', 'BBB', 'BB', 'B', 'NR'], 
                p=[0.1, 0.2, 0.3, 0.3, 0.1]
            ),
            'industry_sector': np.random.choice(
                ['Manufacturing', 'IT', 'Pharma', 'Auto', 'FMCG']
            ),
            'is_ring_member': False,
            'ring_id': None
        })
    
    # Fraud ring entities (dual-role: can be buyer or supplier)
    for ring_idx in range(n_rings):
        ring_id = f'RING_{ring_idx:02d}'
        ring_size = ring_sizes[ring_idx] if ring_idx < len(ring_sizes) else 5
        
        # Ring entities registered around same time (suspicious)
        ring_start_date = pd.Timestamp('2024-06-01') + \
                         pd.Timedelta(days=np.random.randint(0, 30))
        
        for j in range(ring_size):
            entities.append({
                'entity_id': f'R{ring_idx}{j:02d}',
                'entity_type': 'dual',  # Can act as buyer or supplier
                'registration_date': ring_start_date + 
                                    pd.Timedelta(days=np.random.randint(0, 14)),
                'turnover_cr': np.random.lognormal(mean=3, sigma=0.3),  # Suspiciously similar
                'credit_rating': np.random.choice(['BBB', 'BB', 'B'], p=[0.3, 0.4, 0.3]),
                'industry_sector': 'Manufacturing',  # Same sector
                'is_ring_member': True,
                'ring_id': ring_id
            })
    
    return pd.DataFrame(entities)
```

### 8.3 Invoice Generation

```python
def generate_invoices(entities_df, n_legitimate=5000, n_ring_invoices=500,
                      start_date='2024-01-01', end_date='2024-12-31', seed=42):
    """
    Generate synthetic invoices including ring transactions.
    
    Args:
        entities_df: Entity DataFrame
        n_legitimate: Number of legitimate invoices
        n_ring_invoices: Number of ring-related invoices
        start_date: Invoice period start
        end_date: Invoice period end
        seed: Random seed
        
    Returns:
        DataFrame with invoices and labels
    """
    np.random.seed(seed)
    invoices = []
    
    buyers = entities_df[entities_df['entity_type'].isin(['buyer', 'dual'])]
    suppliers = entities_df[entities_df['entity_type'].isin(['supplier', 'dual'])]
    legit_suppliers = suppliers[~suppliers['is_ring_member']]
    ring_entities = entities_df[entities_df['is_ring_member']]
    
    # Default probabilities by credit rating
    default_probs = {
        'AAA': 0.005, 'AA': 0.01, 'A': 0.02, 
        'BBB': 0.05, 'BB': 0.10, 'B': 0.15, 'NR': 0.20
    }
    
    # Generate legitimate invoices
    for i in range(n_legitimate):
        buyer = buyers[~buyers['is_ring_member']].sample(1).iloc[0]
        supplier = legit_suppliers.sample(1).iloc[0]
        
        invoice_date = pd.Timestamp(start_date) + pd.Timedelta(
            days=np.random.randint(0, 365)
        )
        
        # Amount correlated with buyer turnover
        base_amount = buyer['turnover_cr'] * 1e5 * np.random.lognormal(0, 0.5)
        amount = np.clip(base_amount, 10000, 1e8)
        
        # Default based on credit rating
        is_defaulted = np.random.random() < default_probs.get(
            buyer['credit_rating'], 0.1
        )
        
        invoices.append({
            'invoice_id': f'INV{i:06d}',
            'buyer_id': buyer['entity_id'],
            'supplier_id': supplier['entity_id'],
            'invoice_date': invoice_date,
            'due_date': invoice_date + pd.Timedelta(
                days=np.random.choice([30, 45, 60, 90])
            ),
            'acceptance_date': invoice_date + pd.Timedelta(
                days=np.random.randint(1, 6)
            ),
            'amount': amount,
            'is_defaulted': is_defaulted,
            'is_in_ring': False,
            'ring_id': None
        })
    
    # Generate ring invoices (circular patterns)
    ring_groups = ring_entities.groupby('ring_id')
    inv_counter = n_legitimate
    
    for ring_id, members in ring_groups:
        member_list = members['entity_id'].tolist()
        n_members = len(member_list)
        invoices_per_ring = n_ring_invoices // len(ring_groups)
        
        for j in range(invoices_per_ring):
            # Circular: A→B, B→C, C→A
            idx = j % n_members
            buyer_id = member_list[idx]
            supplier_id = member_list[(idx + 1) % n_members]
            
            invoice_date = pd.Timestamp('2024-06-01') + pd.Timedelta(
                days=np.random.randint(0, 180)
            )
            
            # Ring invoices: round amounts (suspicious)
            amount = np.random.choice([100000, 500000, 1000000, 2500000])
            
            # Higher default rate in rings
            is_defaulted = np.random.random() < 0.35
            
            invoices.append({
                'invoice_id': f'INV{inv_counter:06d}',
                'buyer_id': buyer_id,
                'supplier_id': supplier_id,
                'invoice_date': invoice_date,
                'due_date': invoice_date + pd.Timedelta(days=45),
                'acceptance_date': invoice_date,  # Instant acceptance (suspicious)
                'amount': amount,
                'is_defaulted': is_defaulted,
                'is_in_ring': True,
                'ring_id': ring_id
            })
            inv_counter += 1
    
    return pd.DataFrame(invoices)
```

---

## 9. Implementation Specifications

### 9.1 Complete Pipeline Code

```python
"""
TReDS Hybrid Quantum-Classical Invoice Fraud Detection
Main Pipeline Implementation
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import shap
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler


class TReDSFraudDetectionPipeline:
    """
    Hybrid classical-quantum pipeline for invoice fraud detection.
    
    Classical: Random Forest for default prediction
    Quantum: QUBO community detection for ring identification
    """
    
    def __init__(self, k_communities=5, penalty_weight=1.0, 
                 num_reads=1000, random_state=42):
        """
        Initialize pipeline.
        
        Args:
            k_communities: Number of communities for QUBO
            penalty_weight: QUBO constraint penalty
            num_reads: Simulated annealing iterations
            random_state: Random seed
        """
        self.k_communities = k_communities
        self.penalty_weight = penalty_weight
        self.num_reads = num_reads
        self.random_state = random_state
        
        # Models
        self.default_model = None
        self.ring_detector = None
        self.label_encoders = {}
        
        # Results
        self.community_assignments = None
        self.ring_scores = None
        self.feature_importances = None
        
    def fit(self, entities_df, invoices_df):
        """
        Train both classical and quantum components.
        
        Args:
            entities_df: Entity information
            invoices_df: Invoice records with labels
        """
        print("=" * 60)
        print("TRAINING HYBRID CLASSICAL-QUANTUM PIPELINE")
        print("=" * 60)
        
        # Step 1: Feature Engineering
        print("\n[1/4] Engineering features...")
        features_df = self._engineer_features(entities_df, invoices_df)
        
        # Step 2: Train Classical Default Model
        print("\n[2/4] Training classical default prediction model...")
        self._train_default_model(features_df)
        
        # Step 3: Build Transaction Graph
        print("\n[3/4] Building transaction graph...")
        self.graph = self._build_transaction_graph(invoices_df)
        print(f"      Graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        
        # Step 4: QUBO Ring Detection
        print("\n[4/4] Running QUBO community detection...")
        self._run_qubo_ring_detection()
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        
        return self
    
    def _engineer_features(self, entities_df, invoices_df):
        """
        Create features for default prediction.
        """
        # Merge entity info
        df = invoices_df.merge(
            entities_df[['entity_id', 'turnover_cr', 'credit_rating']].rename(
                columns={'entity_id': 'buyer_id', 
                        'turnover_cr': 'buyer_turnover',
                        'credit_rating': 'buyer_credit_rating'}
            ),
            on='buyer_id', how='left'
        )
        
        # Invoice-level features
        df['amount_log'] = np.log1p(df['amount'])
        df['days_to_due'] = (df['due_date'] - df['invoice_date']).dt.days
        df['acceptance_delay_days'] = (
            df['acceptance_date'] - df['invoice_date']
        ).dt.days
        df['is_round_amount'] = (df['amount'] % 10000 == 0).astype(int)
        df['is_month_end'] = (df['invoice_date'].dt.day >= 25).astype(int)
        
        # Buyer-level aggregates
        buyer_stats = invoices_df.groupby('buyer_id').agg({
            'amount': ['mean', 'std', 'count'],
            'is_defaulted': 'mean'
        }).reset_index()
        buyer_stats.columns = ['buyer_id', 'buyer_avg_amount', 'buyer_std_amount',
                              'buyer_invoice_count', 'buyer_default_rate']
        df = df.merge(buyer_stats, on='buyer_id', how='left')
        
        # Amount z-score
        df['amount_zscore_buyer'] = (
            (df['amount'] - df['buyer_avg_amount']) / 
            df['buyer_std_amount'].replace(0, 1)
        )
        
        # Encode credit rating
        le = LabelEncoder()
        df['buyer_credit_rating_encoded'] = le.fit_transform(
            df['buyer_credit_rating'].fillna('NR')
        )
        self.label_encoders['credit_rating'] = le
        
        # Relationship features
        rel_stats = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
            'invoice_date': ['min', 'count']
        }).reset_index()
        rel_stats.columns = ['buyer_id', 'supplier_id', 
                            'first_invoice_date', 'relationship_count']
        df = df.merge(rel_stats, on=['buyer_id', 'supplier_id'], how='left')
        
        df['relationship_age_days'] = (
            df['invoice_date'] - df['first_invoice_date']
        ).dt.days
        df['is_new_relationship'] = (df['relationship_age_days'] < 30).astype(int)
        
        self.features_df = df
        return df
    
    def _train_default_model(self, features_df):
        """
        Train Random Forest for default prediction.
        """
        feature_cols = [
            'amount_log', 'days_to_due', 'acceptance_delay_days',
            'amount_zscore_buyer', 'is_round_amount', 'is_month_end',
            'buyer_avg_amount', 'buyer_invoice_count', 'buyer_default_rate',
            'buyer_credit_rating_encoded', 'relationship_age_days',
            'relationship_count', 'is_new_relationship'
        ]
        
        X = features_df[feature_cols].fillna(0)
        y = features_df['is_defaulted'].astype(int)
        
        self.default_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, 
                           random_state=self.random_state)
        scores = cross_val_score(self.default_model, X, y, 
                                cv=cv, scoring='roc_auc')
        
        print(f"      Default Model CV AUC-ROC: {scores.mean():.4f} "
              f"(+/- {scores.std()*2:.4f})")
        
        # Fit on full data
        self.default_model.fit(X, y)
        
        # Feature importance
        self.feature_importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.default_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_cols = feature_cols
        
    def _build_transaction_graph(self, invoices_df):
        """
        Build directed weighted transaction graph.
        """
        G = nx.DiGraph()
        
        # Aggregate by buyer-supplier pair
        edges = invoices_df.groupby(['buyer_id', 'supplier_id']).agg({
            'amount': 'sum',
            'invoice_id': 'count'
        }).reset_index()
        edges.columns = ['buyer_id', 'supplier_id', 'total_amount', 'n_invoices']
        
        for _, row in edges.iterrows():
            G.add_edge(
                row['buyer_id'], 
                row['supplier_id'],
                weight=row['total_amount'],
                count=row['n_invoices']
            )
        
        return G
    
    def _run_qubo_ring_detection(self):
        """
        Execute QUBO-based community detection.
        """
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        # Build QUBO
        bqm, nodes, k = self._build_modularity_qubo(G_undirected)
        self.nodes = nodes
        
        print(f"      QUBO size: {len(bqm.variables)} variables")
        
        # Solve with simulated annealing
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(
            bqm,
            num_reads=self.num_reads,
            num_sweeps=1000,
            seed=self.random_state
        )
        
        # Decode solution
        best_sample = sampleset.first.sample
        self.qubo_energy = sampleset.first.energy
        
        self.community_assignments = self._decode_communities(
            best_sample, nodes, self.k_communities
        )
        
        # Score communities for ring likelihood
        self.ring_scores = self._score_ring_likelihood(G_undirected)
        
        # Report detected rings
        n_high_risk = sum(1 for s in self.ring_scores.values() if s >= 0.5)
        print(f"      Detected {n_high_risk} high-risk communities")
        print(f"      QUBO energy: {self.qubo_energy:.4f}")
        
    def _build_modularity_qubo(self, G):
        """
        Build QUBO for modularity maximization.
        """
        nodes = list(G.nodes())
        n = len(nodes)
        k = self.k_communities
        
        # Adjacency and modularity matrices
        A = nx.to_numpy_array(G, nodelist=nodes)
        degrees = np.array([G.degree(node) for node in nodes])
        m = G.number_of_edges()
        
        if m == 0:
            # Handle empty graph
            return BinaryQuadraticModel('BINARY'), nodes, k
        
        B = A - np.outer(degrees, degrees) / (2 * m)
        
        # Build QUBO
        Q = {}
        P = self.penalty_weight
        
        for i in range(n):
            for c in range(k):
                vi = f"x_{i}_{c}"
                
                # Modularity diagonal
                Q[(vi, vi)] = Q.get((vi, vi), 0) - B[i, i] / m
                
                # Penalty diagonal: -P (from expanding (sum - 1)^2)
                Q[(vi, vi)] = Q.get((vi, vi), 0) - P
                
                # Modularity off-diagonal (same community)
                for j in range(i+1, n):
                    vj = f"x_{j}_{c}"
                    Q[(vi, vj)] = Q.get((vi, vj), 0) - 2 * B[i, j] / m
                
                # Penalty cross-terms (same node, different communities)
                for c2 in range(c+1, k):
                    vj = f"x_{i}_{c2}"
                    Q[(vi, vj)] = Q.get((vi, vj), 0) + 2 * P
        
        bqm = BinaryQuadraticModel.from_qubo(Q)
        return bqm, nodes, k
    
    def _decode_communities(self, sample, nodes, k):
        """
        Decode QUBO solution to community assignments.
        """
        assignments = {}
        for i, node in enumerate(nodes):
            for c in range(k):
                if sample.get(f"x_{i}_{c}", 0) == 1:
                    assignments[node] = c
                    break
            else:
                assignments[node] = -1
        return assignments
    
    def _score_ring_likelihood(self, G):
        """
        Score each community for fraud ring characteristics.
        """
        # Group nodes by community
        communities = {}
        for node, comm in self.community_assignments.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        
        scores = {}
        for comm_id, members in communities.items():
            if len(members) < 3:
                scores[comm_id] = 0.0
                continue
            
            subgraph = G.subgraph(members)
            n = len(members)
            e = subgraph.number_of_edges()
            
            # Density
            max_e = n * (n - 1) / 2
            density = e / max_e if max_e > 0 else 0
            
            # Clustering coefficient
            clustering = nx.average_clustering(subgraph) if e > 0 else 0
            
            # Size factor (rings typically 3-15 entities)
            size_factor = 1.0 if 3 <= n <= 15 else 0.5
            
            # Combine
            score = 0.4 * density + 0.4 * clustering + 0.2 * size_factor
            scores[comm_id] = min(score, 1.0)
        
        return scores
    
    def predict(self, entities_df, invoices_df):
        """
        Generate predictions for new invoices.
        
        Returns:
            DataFrame with default_prob, ring_prob, composite_score
        """
        # Engineer features
        features_df = self._engineer_features(entities_df, invoices_df)
        
        # Default predictions
        X = features_df[self.feature_cols].fillna(0)
        default_probs = self.default_model.predict_proba(X)[:, 1]
        
        # Ring predictions (based on entity community membership)
        entity_ring_probs = {
            node: self.ring_scores.get(comm, 0)
            for node, comm in self.community_assignments.items()
        }
        
        ring_probs = features_df.apply(
            lambda row: max(
                entity_ring_probs.get(row['buyer_id'], 0),
                entity_ring_probs.get(row['supplier_id'], 0)
            ),
            axis=1
        )
        
        # Composite score
        composite = 0.5 * default_probs + 0.5 * ring_probs
        
        # Risk categories
        def categorize(score):
            if score >= 0.7: return 'Critical'
            if score >= 0.5: return 'High'
            if score >= 0.3: return 'Moderate'
            return 'Low'
        
        results = pd.DataFrame({
            'invoice_id': features_df['invoice_id'],
            'buyer_id': features_df['buyer_id'],
            'supplier_id': features_df['supplier_id'],
            'amount': features_df['amount'],
            'default_probability': default_probs,
            'ring_probability': ring_probs,
            'composite_risk_score': composite,
            'risk_category': [categorize(s) for s in composite]
        })
        
        return results.sort_values('composite_risk_score', ascending=False)
    
    def get_detected_rings(self, threshold=0.5):
        """
        Return communities identified as likely fraud rings.
        """
        rings = []
        for comm_id, score in self.ring_scores.items():
            if score >= threshold:
                members = [n for n, c in self.community_assignments.items() 
                          if c == comm_id]
                rings.append({
                    'community_id': comm_id,
                    'ring_probability': score,
                    'members': members,
                    'size': len(members)
                })
        return sorted(rings, key=lambda x: -x['ring_probability'])
    
    def explain_prediction(self, invoice_idx):
        """
        Generate SHAP-based explanation for single prediction.
        """
        X = self.features_df[self.feature_cols].fillna(0)
        explainer = shap.TreeExplainer(self.default_model)
        shap_values = explainer.shap_values(X.iloc[[invoice_idx]])
        
        explanation = pd.DataFrame({
            'feature': self.feature_cols,
            'value': X.iloc[invoice_idx].values,
            'shap_value': shap_values[1][0] if isinstance(shap_values, list) 
                         else shap_values[0]
        }).sort_values('shap_value', key=abs, ascending=False)
        
        return explanation


# Main execution
if __name__ == "__main__":
    # Generate synthetic data
    print("Generating synthetic data...")
    entities_df = generate_entities(
        n_buyers=80, n_suppliers=150, 
        n_rings=3, ring_sizes=[5, 6, 7]
    )
    invoices_df = generate_invoices(
        entities_df, 
        n_legitimate=3000, 
        n_ring_invoices=300
    )
    
    # Initialize and train pipeline
    pipeline = TReDSFraudDetectionPipeline(
        k_communities=5,
        penalty_weight=1.0,
        num_reads=500
    )
    pipeline.fit(entities_df, invoices_df)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = pipeline.predict(entities_df, invoices_df)
    
    # Display results
    print("\n" + "=" * 60)
    print("TOP 10 HIGH-RISK INVOICES")
    print("=" * 60)
    print(predictions.head(10).to_string(index=False))
    
    # Detected rings
    print("\n" + "=" * 60)
    print("DETECTED FRAUD RINGS")
    print("=" * 60)
    rings = pipeline.get_detected_rings(threshold=0.4)
    for ring in rings:
        print(f"Ring {ring['community_id']}: "
              f"Score={ring['ring_probability']:.3f}, "
              f"Size={ring['size']}, "
              f"Members={ring['members'][:5]}...")
```

---

## 10. Explainability Framework

### 10.1 Dual Explainability Approach

| Component | Method | Output |
|-----------|--------|--------|
| Default Prediction | SHAP TreeExplainer | Feature contribution to P(default) |
| Ring Detection | Community structure analysis | Why community is flagged as ring |

### 10.2 Default Prediction Explanation

```python
def explain_default_prediction(pipeline, invoice_id):
    """
    Generate executive-friendly explanation for default risk.
    """
    idx = pipeline.features_df[
        pipeline.features_df['invoice_id'] == invoice_id
    ].index[0]
    
    explanation = pipeline.explain_prediction(idx)
    
    print(f"DEFAULT RISK EXPLANATION: {invoice_id}")
    print("-" * 50)
    print("\nTop Risk Factors:")
    for _, row in explanation.head(3).iterrows():
        direction = "increases" if row['shap_value'] > 0 else "decreases"
        print(f"  • {row['feature']}: {row['value']:.2f} "
              f"({direction} risk by {abs(row['shap_value']):.3f})")
    
    print("\nTop Protective Factors:")
    for _, row in explanation.tail(3).iterrows():
        direction = "increases" if row['shap_value'] > 0 else "decreases"
        print(f"  • {row['feature']}: {row['value']:.2f} "
              f"({direction} risk by {abs(row['shap_value']):.3f})")
```

### 10.3 Ring Detection Explanation

```python
def explain_ring_detection(pipeline, community_id):
    """
    Explain why a community is flagged as potential ring.
    """
    members = [n for n, c in pipeline.community_assignments.items() 
               if c == community_id]
    subgraph = pipeline.graph.subgraph(members)
    
    n = len(members)
    e = subgraph.number_of_edges()
    density = e / (n * (n-1)) if n > 1 else 0
    
    # Check for circular patterns
    try:
        cycles = list(nx.simple_cycles(subgraph))
        n_cycles = len([c for c in cycles if len(c) >= 3])
    except:
        n_cycles = 0
    
    print(f"RING DETECTION EXPLANATION: Community {community_id}")
    print("-" * 50)
    print(f"\nCommunity Statistics:")
    print(f"  • Members: {n} entities")
    print(f"  • Transactions: {e} edges")
    print(f"  • Density: {density:.3f} (high density suggests coordination)")
    print(f"  • Circular patterns: {n_cycles} detected")
    print(f"\nMembers: {members}")
    
    if density > 0.5:
        print("\n⚠️  HIGH DENSITY: Unusually tight-knit transaction cluster")
    if n_cycles > 0:
        print("⚠️  CIRCULAR PATTERNS: Potential invoice cycling detected")
```

---

## 11. Sample Outputs & Targeting Strategy

### 11.1 Output Schema

```json
{
    "invoice_id": "INV003421",
    "buyer_id": "R0103",
    "supplier_id": "R0104",
    "amount": 2500000.00,
    "default_probability": 0.32,
    "ring_probability": 0.78,
    "composite_risk_score": 0.55,
    "risk_category": "High",
    "community_id": 2,
    "top_default_factors": [
        "buyer_default_rate: 0.35 (+0.12)",
        "is_new_relationship: 1 (+0.08)",
        "acceptance_delay_days: 0 (+0.05)"
    ],
    "ring_indicators": [
        "Community density: 0.72",
        "Circular patterns: 3 detected",
        "Members registered within 14 days"
    ],
    "recommended_action": "Escalate to fraud investigation team"
}
```

### 11.2 Targeting Priority Matrix

```
                    Ring Probability
                    Low (<0.3)  │  High (≥0.3)
                ─────────────────┼──────────────────
     Default    │               │
     Probability│   TIER 3      │   TIER 1
     High (≥0.5)│   (Credit     │   (IMMEDIATE
                │    Review)    │    INVESTIGATION)
                │               │
                ├───────────────┼──────────────────
                │               │
     Default    │   TIER 4      │   TIER 2
     Probability│   (Standard   │   (Ring
     Low (<0.5) │    Process)   │    Investigation)
                │               │
```

---

## 12. Future Quantum Hardware Integration

### 12.1 D-Wave Integration Path

```python
# MVP: Simulated Annealing
from dwave.samplers import SimulatedAnnealingSampler
sampler = SimulatedAnnealingSampler()

# Production: D-Wave Hybrid Solver
from dwave.system import LeapHybridSampler
sampler = LeapHybridSampler()

# Same interface - just swap sampler
sampleset = sampler.sample(bqm, ...)
```

### 12.2 QAOA Alternative (Gate-Based)

For gate-based quantum computers:

```python
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.algorithms import QAOA
from qiskit.primitives import Sampler

# Convert QUBO to Qiskit format
qp = QuadraticProgram()
# ... add variables and objective from QUBO

# Solve with QAOA
qaoa = QAOA(sampler=Sampler(), reps=3)
optimizer = MinimumEigenOptimizer(qaoa)
result = optimizer.solve(qp)
```

### 12.3 Scalability Considerations

| Problem Size | Recommended Solver |
|-------------|-------------------|
| < 50 nodes | Exact solver (BruteForce) |
| 50-500 nodes | Simulated Annealing |
| 500-5000 nodes | D-Wave Hybrid (QBSolv) |
| > 5000 nodes | Hierarchical decomposition |

---

## 13. Technical Appendices

### Appendix A: QUBO Mathematical Derivation

**Modularity Maximization:**

Given graph G = (V, E) with adjacency A, the modularity Q for partition into k communities is:

$$Q = \frac{1}{2m} \sum_{ij} \left[ A_{ij} - \frac{k_i k_j}{2m} \right] \delta(c_i, c_j)$$

**QUBO Variables:**
- $x_{ic} \in \{0,1\}$: Node i assigned to community c

**Constraint:**
$$\sum_{c=1}^{k} x_{ic} = 1 \quad \forall i$$

**QUBO Hamiltonian:**
$$H = -\frac{1}{m}\sum_c \mathbf{x}_c^T B \mathbf{x}_c + P \sum_i \left(\sum_c x_{ic} - 1\right)^2$$

### Appendix B: Benchmark Networks

| Network | Nodes | Edges | Communities | Best Modularity |
|---------|-------|-------|-------------|-----------------|
| Zachary Karate | 34 | 78 | 2-4 | 0.42 |
| Dolphins | 62 | 159 | 2-4 | 0.53 |
| Football | 115 | 613 | 12 | 0.60 |

### Appendix C: Dependencies

```
# requirements.txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
networkx>=2.6.0
dimod>=0.12.0
dwave-samplers>=1.0.0
shap>=0.41.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## Document Approval

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Author | QGAI QFM Team | | |
| Technical Review | | | |
| Product Owner | | | |
| Executive Sponsor | | | |

---

*This document is confidential and intended for internal use only.*
