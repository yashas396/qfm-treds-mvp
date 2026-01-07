"""
QGAI TReDS MVP - Data Loader

Loads sample data for the UI demonstration.
Generates realistic synthetic data matching the backend pipeline output.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path


def initialize_session_state():
    """Initialize all session state variables."""
    defaults = {
        'current_page': 'Dashboard',
        'view_invoice_id': None,
        'view_ring_id': None,
        'selected_invoices': [],
        'filters': {
            'risk_levels': ['Critical', 'High', 'Medium', 'Low'],
            'ring_only': False,
            'amount_min': 0,
            'amount_max': 100000000,
            'search_query': ''
        },
        'decision_log': [],
        'flagged_count': 47,
        'ring_count': 3
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_sample_data():
    """Load or generate sample data for the application."""
    # Try to load from output files first
    outputs_dir = Path(__file__).parent.parent.parent / "data" / "outputs"

    # Load sample predictions
    predictions_file = outputs_dir / "sample_predictions.json"
    if predictions_file.exists():
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)
        st.session_state.predictions = predictions_data
    else:
        st.session_state.predictions = generate_sample_predictions()

    # Load detected rings
    rings_file = outputs_dir / "detected_rings.json"
    if rings_file.exists():
        with open(rings_file, 'r') as f:
            rings_data = json.load(f)
        st.session_state.rings = rings_data
    else:
        st.session_state.rings = generate_sample_rings()

    # Load risk report
    report_file = outputs_dir / "risk_report.json"
    if report_file.exists():
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        st.session_state.risk_report = report_data
    else:
        st.session_state.risk_report = generate_sample_report()

    # Generate invoice DataFrame for queue
    st.session_state.invoices_df = generate_invoice_dataframe()

    # Generate entities DataFrame
    st.session_state.entities_df = generate_entities_dataframe()

    # Update counts
    df = st.session_state.invoices_df
    st.session_state.flagged_count = len(df[df['risk_level'].isin(['Critical', 'High'])])
    st.session_state.ring_count = 3


def generate_sample_predictions():
    """Generate sample predictions data."""
    return {
        "metadata": {
            "system": "QGAI Hybrid Classical-Quantum TReDS MVP",
            "version": "1.0.0",
            "generated_at": datetime.now().isoformat(),
            "model_auc_roc": 0.8389,
            "total_predictions": 50
        },
        "predictions": [],
        "summary_statistics": {
            "risk_distribution": {
                "Critical": 12,
                "High": 35,
                "Medium": 127,
                "Low": 5060
            }
        }
    }


def generate_sample_rings():
    """Generate sample rings data."""
    return {
        "detected_rings": [
            {
                "ring_id": "RING_001",
                "confidence_score": 0.92,
                "members": [
                    {"entity_id": "R0101", "role": "hub"},
                    {"entity_id": "R0102", "role": "intermediary"},
                    {"entity_id": "R0103", "role": "intermediary"},
                    {"entity_id": "R0104", "role": "intermediary"},
                    {"entity_id": "R0105", "role": "peripheral"}
                ],
                "statistics": {
                    "member_count": 5,
                    "density": 0.60,
                    "total_transaction_amount": 45000000
                }
            }
        ]
    }


def generate_sample_report():
    """Generate sample risk report."""
    return {
        "executive_summary": {
            "overall_risk_level": "ELEVATED",
            "key_findings": [
                "3 fraud rings detected with total exposure of INR 25.2 Cr",
                "Classical model AUC-ROC of 0.8389 exceeds target"
            ]
        },
        "classical_model_performance": {
            "auc_roc": 0.8389
        }
    }


def generate_invoice_dataframe() -> pd.DataFrame:
    """Generate invoice DataFrame for the queue view."""
    np.random.seed(42)

    # Ring-related invoices (high risk)
    ring_invoices = []
    ring_entities = ['R0101', 'R0102', 'R0103', 'R0104', 'R0105', 'R0106', 'R0107']

    for i in range(15):
        seller = np.random.choice(ring_entities)
        buyer = np.random.choice([e for e in ring_entities if e != seller])
        ring_invoices.append({
            'invoice_id': f'INV{3400 + i:06d}',
            'seller_id': seller,
            'buyer_id': buyer,
            'seller_name': f'Entity {seller}',
            'buyer_name': f'Entity {buyer}',
            'amount': np.random.choice([1000000, 2500000, 5000000, 10000000, 25000000]),
            'invoice_date': datetime.now() - timedelta(days=np.random.randint(1, 30)),
            'due_date': datetime.now() + timedelta(days=np.random.randint(7, 45)),
            'default_prob': np.random.uniform(0.25, 0.55),
            'ring_prob': np.random.uniform(0.65, 0.95),
            'is_ring_member': True,
            'ring_id': 'RING_001' if i < 8 else 'RING_002',
            'status': 'Pending Review'
        })

    # Regular invoices (mixed risk)
    regular_invoices = []
    for i in range(35):
        risk = np.random.choice(['low', 'medium', 'high'], p=[0.7, 0.2, 0.1])
        if risk == 'low':
            default_p = np.random.uniform(0.01, 0.15)
            ring_p = np.random.uniform(0.01, 0.10)
        elif risk == 'medium':
            default_p = np.random.uniform(0.15, 0.35)
            ring_p = np.random.uniform(0.05, 0.25)
        else:
            default_p = np.random.uniform(0.35, 0.55)
            ring_p = np.random.uniform(0.10, 0.30)

        seller_id = f'S{np.random.randint(1, 200):04d}'
        buyer_id = f'B{np.random.randint(1, 100):04d}'

        regular_invoices.append({
            'invoice_id': f'INV{3000 + i:06d}',
            'seller_id': seller_id,
            'buyer_id': buyer_id,
            'seller_name': f'Supplier {seller_id}',
            'buyer_name': f'Buyer {buyer_id}',
            'amount': np.random.randint(100000, 50000000),
            'invoice_date': datetime.now() - timedelta(days=np.random.randint(1, 60)),
            'due_date': datetime.now() + timedelta(days=np.random.randint(7, 90)),
            'default_prob': default_p,
            'ring_prob': ring_p,
            'is_ring_member': False,
            'ring_id': None,
            'status': 'Pending Review'
        })

    # Combine all invoices
    all_invoices = ring_invoices + regular_invoices
    df = pd.DataFrame(all_invoices)

    # Calculate composite risk and risk level
    df['composite_risk'] = 0.5 * df['default_prob'] + 0.5 * df['ring_prob']

    def get_risk_level(composite):
        if composite >= 0.5:
            return 'Critical'
        elif composite >= 0.35:
            return 'High'
        elif composite >= 0.2:
            return 'Medium'
        else:
            return 'Low'

    df['risk_level'] = df['composite_risk'].apply(get_risk_level)

    # Sort by risk level
    risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    df['risk_order'] = df['risk_level'].map(risk_order)
    df = df.sort_values(['risk_order', 'composite_risk'], ascending=[True, False])
    df = df.drop('risk_order', axis=1)

    return df.reset_index(drop=True)


def generate_entities_dataframe() -> pd.DataFrame:
    """Generate entities DataFrame."""
    entities = []

    # Ring members
    ring_entities = ['R0101', 'R0102', 'R0103', 'R0104', 'R0105', 'R0106', 'R0107']
    ring_names = [
        'Apex Trading Co.', 'Prime Industries Ltd.', 'Nova Exports Pvt Ltd',
        'Global Commerce Inc.', 'Star Enterprises', 'Metro Supplies', 'Delta Corp'
    ]

    for i, (eid, name) in enumerate(zip(ring_entities, ring_names)):
        entities.append({
            'entity_id': eid,
            'entity_name': name,
            'entity_type': 'dual',
            'registration_date': datetime.now() - timedelta(days=45 + i * 2),
            'turnover_cr': np.random.uniform(5, 50),
            'credit_rating': np.random.choice(['BBB', 'BB', 'B']),
            'is_ring_member': True,
            'ring_id': 'RING_001' if i < 5 else 'RING_002',
            'total_invoices': np.random.randint(10, 50),
            'default_rate': np.random.uniform(0.25, 0.45)
        })

    # Regular entities
    for i in range(20):
        etype = np.random.choice(['buyer', 'supplier'])
        prefix = 'B' if etype == 'buyer' else 'S'
        entities.append({
            'entity_id': f'{prefix}{i+1:04d}',
            'entity_name': f'Company {prefix}{i+1}',
            'entity_type': etype,
            'registration_date': datetime.now() - timedelta(days=np.random.randint(90, 730)),
            'turnover_cr': np.random.uniform(10, 500),
            'credit_rating': np.random.choice(['AAA', 'AA', 'A', 'BBB', 'BB'], p=[0.1, 0.2, 0.3, 0.25, 0.15]),
            'is_ring_member': False,
            'ring_id': None,
            'total_invoices': np.random.randint(20, 200),
            'default_rate': np.random.uniform(0.01, 0.10)
        })

    return pd.DataFrame(entities)


def get_invoice_by_id(invoice_id: str) -> dict:
    """Get invoice details by ID."""
    df = st.session_state.invoices_df
    invoice = df[df['invoice_id'] == invoice_id]

    if len(invoice) == 0:
        return None

    inv = invoice.iloc[0].to_dict()

    # Add explanation data
    inv['risk_explanation'] = generate_risk_explanation(inv)
    inv['ring_explanation'] = generate_ring_explanation(inv) if inv['is_ring_member'] else None

    return inv


def generate_risk_explanation(invoice: dict) -> dict:
    """Generate risk explanation for an invoice."""
    factors = []

    if invoice['default_prob'] > 0.3:
        factors.append({
            'factor': 'Buyer historical default rate',
            'value': f"{invoice['default_prob']*100:.0f}%",
            'impact': '+0.12',
            'direction': 'increases_risk'
        })

    if invoice['is_ring_member']:
        factors.append({
            'factor': 'Ring member involvement',
            'value': 'Yes',
            'impact': '+0.15',
            'direction': 'increases_risk'
        })

    if invoice['amount'] > 10000000:
        factors.append({
            'factor': 'Large transaction amount',
            'value': f"INR {invoice['amount']/10000000:.1f} Cr",
            'impact': '+0.05',
            'direction': 'increases_risk'
        })

    # Add some protective factors
    protective = []
    if invoice['default_prob'] < 0.2:
        protective.append({
            'factor': 'Good buyer credit history',
            'value': 'AA rating',
            'impact': '-0.08',
            'direction': 'decreases_risk'
        })

    return {
        'risk_factors': factors,
        'protective_factors': protective,
        'summary': f"Invoice flagged as {invoice['risk_level']} risk based on hybrid classical-quantum analysis."
    }


def generate_ring_explanation(invoice: dict) -> dict:
    """Generate ring explanation for a ring-related invoice."""
    return {
        'ring_id': invoice['ring_id'],
        'indicators': [
            {
                'indicator': 'Reciprocal transactions',
                'description': 'Buyer and seller have invoiced each other',
                'severity': 'high'
            },
            {
                'indicator': 'Circular patterns',
                'description': '5 triangular transaction patterns detected',
                'severity': 'high'
            },
            {
                'indicator': 'High clustering',
                'description': '82% clustering coefficient (normal: <30%)',
                'severity': 'medium'
            },
            {
                'indicator': 'Synchronized registration',
                'description': 'Entities registered within 14-day window',
                'severity': 'medium'
            }
        ],
        'community_info': {
            'members': 7,
            'total_exposure': 42000000,
            'density': 0.72
        }
    }


def add_decision(invoice_id: str, action: str, reason: str, analyst: str = "Priya Sharma"):
    """Add a decision to the log."""
    decision = {
        'timestamp': datetime.now(),
        'invoice_id': invoice_id,
        'action': action,
        'reason': reason,
        'analyst': analyst,
        'risk_at_decision': st.session_state.invoices_df[
            st.session_state.invoices_df['invoice_id'] == invoice_id
        ]['risk_level'].values[0] if invoice_id else 'N/A'
    }
    st.session_state.decision_log.insert(0, decision)

    # Update invoice status
    if invoice_id:
        idx = st.session_state.invoices_df['invoice_id'] == invoice_id
        st.session_state.invoices_df.loc[idx, 'status'] = action


def get_ring_by_id(ring_id: str) -> dict:
    """Get ring details by ID."""
    rings = st.session_state.rings.get('detected_rings', [])

    # If no rings loaded, generate sample
    if not rings:
        return generate_sample_ring_detail(ring_id)

    for ring in rings:
        if ring.get('ring_id') == ring_id:
            return ring

    return generate_sample_ring_detail(ring_id)


def generate_sample_ring_detail(ring_id: str) -> dict:
    """Generate detailed ring data."""
    members = [
        {'entity_id': 'R0101', 'entity_name': 'Apex Trading Co.', 'role': 'hub', 'in_degree': 4, 'out_degree': 3},
        {'entity_id': 'R0102', 'entity_name': 'Prime Industries', 'role': 'intermediary', 'in_degree': 3, 'out_degree': 3},
        {'entity_id': 'R0103', 'entity_name': 'Nova Exports', 'role': 'intermediary', 'in_degree': 2, 'out_degree': 3},
        {'entity_id': 'R0104', 'entity_name': 'Global Commerce', 'role': 'intermediary', 'in_degree': 3, 'out_degree': 2},
        {'entity_id': 'R0105', 'entity_name': 'Star Enterprises', 'role': 'peripheral', 'in_degree': 2, 'out_degree': 1},
        {'entity_id': 'R0106', 'entity_name': 'Metro Supplies', 'role': 'peripheral', 'in_degree': 1, 'out_degree': 2},
        {'entity_id': 'R0107', 'entity_name': 'Delta Corp', 'role': 'peripheral', 'in_degree': 1, 'out_degree': 1}
    ]

    edges = [
        ('R0101', 'R0102', 8500000),
        ('R0102', 'R0103', 5200000),
        ('R0103', 'R0104', 7800000),
        ('R0104', 'R0101', 6500000),  # Circular
        ('R0102', 'R0101', 4200000),  # Reciprocal
        ('R0103', 'R0105', 3100000),
        ('R0104', 'R0106', 2800000),
        ('R0105', 'R0102', 1500000),
        ('R0106', 'R0107', 1200000),
        ('R0107', 'R0104', 900000)
    ]

    return {
        'ring_id': ring_id,
        'confidence_score': 0.87,
        'members': members,
        'edges': edges,
        'statistics': {
            'member_count': len(members),
            'internal_edges': len(edges),
            'density': 0.72,
            'reciprocity': 0.67,
            'total_transaction_amount': sum(e[2] for e in edges),
            'registration_window_days': 14
        },
        'indicators': [
            {'name': 'High density', 'value': '0.72', 'threshold': '<0.3', 'severity': 'high'},
            {'name': 'Circular patterns', 'value': '5 triangles', 'threshold': '0', 'severity': 'high'},
            {'name': 'Reciprocal edges', 'value': '4 bidirectional', 'threshold': '0', 'severity': 'high'},
            {'name': 'Similar registration', 'value': '14 days', 'threshold': '>90 days', 'severity': 'medium'}
        ]
    }
