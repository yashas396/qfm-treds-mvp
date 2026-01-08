"""
QGAI TReDS MVP - Quantum Invoice Ring Detection Platform
Professional Risk Analyst Dashboard

A hybrid classical-quantum fraud detection system for TReDS invoice financing.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from fpdf import FPDF
import io
import base64

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="QGAI TReDS | Risk Platform",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Headers */
    h1 { color: #1e293b; font-weight: 700; }
    h2 { color: #334155; font-weight: 600; }
    h3 { color: #475569; font-weight: 600; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    [data-testid="metric-container"] label {
        color: #64748b;
        font-weight: 500;
    }

    /* Risk badges */
    .risk-critical { background: #fee2e2; color: #dc2626; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .risk-high { background: #ffedd5; color: #ea580c; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .risk-medium { background: #fef3c7; color: #d97706; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }
    .risk-low { background: #dcfce7; color: #16a34a; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; }

    /* Cards */
    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    /* Alert cards */
    .alert-warning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .alert-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 4px solid #dc2626;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .alert-quantum {
        background: linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%);
        border-left: 4px solid #7c3aed;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #000000 !important;
    }

    [data-testid="stSidebar"] button {
        color: #000000 !important;
    }

    /* Tables */
    .dataframe { font-size: 0.9rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA GENERATION
# =============================================================================
@st.cache_data
def generate_invoice_data() -> pd.DataFrame:
    """Generate sample invoice data for the MVP."""
    np.random.seed(42)

    # Ring member invoices (high risk)
    ring_data = []
    ring_entities = ['R0101', 'R0102', 'R0103', 'R0104', 'R0105', 'R0106', 'R0107']
    ring_names = ['Apex Trading', 'Prime Industries', 'Nova Exports', 'Global Commerce',
                  'Star Enterprises', 'Metro Supplies', 'Delta Corp']

    for i in range(12):
        seller_idx = i % len(ring_entities)
        buyer_idx = (i + 1) % len(ring_entities)
        ring_data.append({
            'invoice_id': f'INV-2025-{1000 + i:04d}',
            'seller_id': ring_entities[seller_idx],
            'seller_name': ring_names[seller_idx],
            'buyer_id': ring_entities[buyer_idx],
            'buyer_name': ring_names[buyer_idx],
            'amount': np.random.choice([2500000, 5000000, 7500000, 10000000]),
            'invoice_date': datetime.now() - timedelta(days=np.random.randint(5, 25)),
            'due_date': datetime.now() + timedelta(days=np.random.randint(15, 45)),
            'default_prob': round(np.random.uniform(0.25, 0.45), 3),
            'ring_prob': round(np.random.uniform(0.70, 0.95), 3),
            'is_ring_member': True,
            'ring_id': 'RING-001' if i < 7 else 'RING-002'
        })

    # Regular invoices
    regular_data = []
    for i in range(38):
        risk_tier = np.random.choice(['low', 'medium', 'high'], p=[0.6, 0.3, 0.1])
        if risk_tier == 'low':
            default_p, ring_p = np.random.uniform(0.02, 0.12), np.random.uniform(0.01, 0.08)
        elif risk_tier == 'medium':
            default_p, ring_p = np.random.uniform(0.12, 0.30), np.random.uniform(0.08, 0.20)
        else:
            default_p, ring_p = np.random.uniform(0.30, 0.50), np.random.uniform(0.15, 0.35)

        regular_data.append({
            'invoice_id': f'INV-2025-{2000 + i:04d}',
            'seller_id': f'S{np.random.randint(100, 999)}',
            'seller_name': f'Supplier {np.random.randint(100, 999)}',
            'buyer_id': f'B{np.random.randint(100, 999)}',
            'buyer_name': f'Buyer Corp {np.random.randint(100, 999)}',
            'amount': np.random.randint(500000, 25000000),
            'invoice_date': datetime.now() - timedelta(days=np.random.randint(1, 45)),
            'due_date': datetime.now() + timedelta(days=np.random.randint(7, 90)),
            'default_prob': round(default_p, 3),
            'ring_prob': round(ring_p, 3),
            'is_ring_member': False,
            'ring_id': None
        })

    df = pd.DataFrame(ring_data + regular_data)

    # Calculate composite risk
    df['composite_risk'] = (0.5 * df['default_prob'] + 0.5 * df['ring_prob']).round(3)

    # Assign risk levels
    def get_risk_level(score):
        if score >= 0.45: return 'Critical'
        if score >= 0.30: return 'High'
        if score >= 0.15: return 'Medium'
        return 'Low'

    df['risk_level'] = df['composite_risk'].apply(get_risk_level)

    # Sort by risk
    risk_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
    df['_sort'] = df['risk_level'].map(risk_order)
    df = df.sort_values(['_sort', 'composite_risk'], ascending=[True, False]).drop('_sort', axis=1)

    return df.reset_index(drop=True)


@st.cache_data
def generate_ring_data() -> List[Dict]:
    """Generate sample ring detection data."""
    return [
        {
            'ring_id': 'RING-001',
            'confidence': 0.92,
            'member_count': 5,
            'total_exposure': 45000000,
            'density': 0.72,
            'status': 'Under Investigation',
            'members': [
                {'id': 'R0101', 'name': 'Apex Trading', 'role': 'Hub', 'in_deg': 4, 'out_deg': 3},
                {'id': 'R0102', 'name': 'Prime Industries', 'role': 'Intermediary', 'in_deg': 3, 'out_deg': 3},
                {'id': 'R0103', 'name': 'Nova Exports', 'role': 'Intermediary', 'in_deg': 2, 'out_deg': 3},
                {'id': 'R0104', 'name': 'Global Commerce', 'role': 'Peripheral', 'in_deg': 2, 'out_deg': 2},
                {'id': 'R0105', 'name': 'Star Enterprises', 'role': 'Peripheral', 'in_deg': 1, 'out_deg': 2}
            ],
            'edges': [
                ('R0101', 'R0102', 8500000), ('R0102', 'R0103', 5200000),
                ('R0103', 'R0104', 7800000), ('R0104', 'R0101', 6500000),
                ('R0102', 'R0101', 4200000), ('R0103', 'R0105', 3100000),
                ('R0105', 'R0102', 1500000)
            ]
        },
        {
            'ring_id': 'RING-002',
            'confidence': 0.87,
            'member_count': 7,
            'total_exposure': 82000000,
            'density': 0.58,
            'status': 'Flagged',
            'members': [
                {'id': 'R0106', 'name': 'Metro Supplies', 'role': 'Hub', 'in_deg': 5, 'out_deg': 4},
                {'id': 'R0107', 'name': 'Delta Corp', 'role': 'Hub', 'in_deg': 4, 'out_deg': 5},
                {'id': 'R0108', 'name': 'Sigma Tech', 'role': 'Intermediary', 'in_deg': 3, 'out_deg': 3},
                {'id': 'R0109', 'name': 'Alpha Solutions', 'role': 'Intermediary', 'in_deg': 3, 'out_deg': 2},
                {'id': 'R0110', 'name': 'Beta Industries', 'role': 'Peripheral', 'in_deg': 2, 'out_deg': 2},
                {'id': 'R0111', 'name': 'Gamma Corp', 'role': 'Peripheral', 'in_deg': 2, 'out_deg': 1},
                {'id': 'R0112', 'name': 'Omega Trading', 'role': 'Peripheral', 'in_deg': 1, 'out_deg': 1}
            ],
            'edges': [
                ('R0106', 'R0107', 12000000), ('R0107', 'R0108', 8500000),
                ('R0108', 'R0109', 6200000), ('R0109', 'R0106', 9800000),
                ('R0107', 'R0110', 5500000), ('R0110', 'R0111', 3200000),
                ('R0111', 'R0106', 2800000), ('R0108', 'R0112', 1900000),
                ('R0112', 'R0107', 1500000)
            ]
        },
        {
            'ring_id': 'RING-003',
            'confidence': 0.78,
            'member_count': 4,
            'total_exposure': 28000000,
            'density': 0.65,
            'status': 'New',
            'members': [
                {'id': 'R0113', 'name': 'Zenith Corp', 'role': 'Hub', 'in_deg': 3, 'out_deg': 3},
                {'id': 'R0114', 'name': 'Apex Solutions', 'role': 'Intermediary', 'in_deg': 2, 'out_deg': 2},
                {'id': 'R0115', 'name': 'Peak Industries', 'role': 'Peripheral', 'in_deg': 2, 'out_deg': 1},
                {'id': 'R0116', 'name': 'Summit Trading', 'role': 'Peripheral', 'in_deg': 1, 'out_deg': 2}
            ],
            'edges': [
                ('R0113', 'R0114', 9500000), ('R0114', 'R0115', 6200000),
                ('R0115', 'R0113', 5800000), ('R0113', 'R0116', 3500000),
                ('R0116', 'R0114', 3000000)
            ]
        }
    ]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def format_inr(amount: float) -> str:
    """Format amount in Indian Rupee notation."""
    if amount >= 10000000:
        return f"₹{amount/10000000:.2f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.2f} L"
    else:
        return f"₹{amount:,.0f}"


def get_risk_badge(level: str) -> str:
    """Return HTML badge for risk level."""
    css_class = f"risk-{level.lower()}"
    return f'<span class="{css_class}">{level}</span>'


def generate_ring_report_pdf(ring: Dict) -> bytes:
    """Generate a PDF report for a detected fraud ring."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(124, 58, 237)  # Purple
    pdf.cell(0, 15, 'QGAI TReDS - Ring Investigation Report', ln=True, align='C')
    pdf.ln(5)

    # Report metadata
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}', ln=True, align='C')
    pdf.cell(0, 6, 'Classification: CONFIDENTIAL', ln=True, align='C')
    pdf.ln(10)

    # Ring Summary Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(10, pdf.get_y(), 190, 45, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, f'Ring ID: {ring["ring_id"]}', ln=True)

    pdf.set_x(15)
    pdf.set_font('Helvetica', '', 11)
    pdf.cell(60, 7, f'Confidence Score: {ring["confidence"]*100:.0f}%')
    pdf.cell(60, 7, f'Status: {ring["status"]}')
    pdf.cell(60, 7, f'Members: {ring["member_count"]}', ln=True)

    pdf.set_x(15)
    pdf.cell(60, 7, f'Graph Density: {ring["density"]:.2f}')

    # Format exposure
    exposure = ring["total_exposure"]
    if exposure >= 10000000:
        exposure_str = f'Rs. {exposure/10000000:.2f} Cr'
    else:
        exposure_str = f'Rs. {exposure/100000:.2f} L'
    pdf.cell(60, 7, f'Total Exposure: {exposure_str}', ln=True)

    pdf.ln(25)

    # Risk Assessment Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(220, 38, 38)  # Red
    pdf.cell(0, 10, 'Risk Assessment', ln=True)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 41, 59)

    risk_indicators = [
        ('High Graph Density', f'{ring["density"]:.2f}', '< 0.30', 'ALERT'),
        ('Circular Transaction Patterns', 'Detected', 'None', 'ALERT'),
        ('Reciprocal Edges', f'{len(ring["edges"]) - ring["member_count"]}', '0', 'ALERT'),
        ('Member Clustering', 'Abnormal', 'Normal', 'WARNING')
    ]

    pdf.set_fill_color(254, 242, 242)
    for indicator, value, normal, severity in risk_indicators:
        pdf.set_x(15)
        pdf.cell(70, 7, indicator)
        pdf.cell(40, 7, f'Value: {value}')
        pdf.cell(40, 7, f'Normal: {normal}')
        pdf.set_text_color(220, 38, 38) if severity == 'ALERT' else pdf.set_text_color(217, 119, 6)
        pdf.cell(30, 7, severity, ln=True)
        pdf.set_text_color(30, 41, 59)

    pdf.ln(10)

    # Ring Members Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 10, 'Ring Members', ln=True)

    # Table header
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(71, 85, 105)
    pdf.cell(30, 8, 'Entity ID', 1, 0, 'C', True)
    pdf.cell(60, 8, 'Name', 1, 0, 'C', True)
    pdf.cell(35, 8, 'Role', 1, 0, 'C', True)
    pdf.cell(30, 8, 'In-Degree', 1, 0, 'C', True)
    pdf.cell(30, 8, 'Out-Degree', 1, 1, 'C', True)

    # Table rows
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(30, 41, 59)

    for i, member in enumerate(ring['members']):
        fill = i % 2 == 0
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(30, 7, member['id'], 1, 0, 'C', fill)
        pdf.cell(60, 7, member['name'][:25], 1, 0, 'L', fill)
        pdf.cell(35, 7, member['role'], 1, 0, 'C', fill)
        pdf.cell(30, 7, str(member['in_deg']), 1, 0, 'C', fill)
        pdf.cell(30, 7, str(member['out_deg']), 1, 1, 'C', fill)

    pdf.ln(10)

    # Transaction Flows Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 10, 'Transaction Flows', ln=True)

    # Table header
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(71, 85, 105)
    pdf.cell(50, 8, 'Source', 1, 0, 'C', True)
    pdf.cell(50, 8, 'Target', 1, 0, 'C', True)
    pdf.cell(50, 8, 'Amount', 1, 1, 'C', True)

    # Table rows
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(30, 41, 59)

    for i, edge in enumerate(ring['edges']):
        fill = i % 2 == 0
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        amount = edge[2]
        if amount >= 10000000:
            amount_str = f'Rs. {amount/10000000:.2f} Cr'
        else:
            amount_str = f'Rs. {amount/100000:.2f} L'
        pdf.cell(50, 7, edge[0], 1, 0, 'C', fill)
        pdf.cell(50, 7, edge[1], 1, 0, 'C', fill)
        pdf.cell(50, 7, amount_str, 1, 1, 'C', fill)

    pdf.ln(10)

    # Quantum Analysis Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 10, 'Quantum Analysis', ln=True)

    pdf.set_fill_color(237, 233, 254)
    pdf.set_draw_color(167, 139, 250)
    pdf.rect(10, pdf.get_y(), 190, 30, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(91, 33, 182)
    pdf.multi_cell(180, 5,
        'This ring was detected using QUBO-based modularity maximization algorithm. '
        'The optimization explored 10,000 candidate community partitions using simulated annealing. '
        'This approach is quantum-ready and compatible with D-Wave quantum annealers for enhanced performance.')

    pdf.ln(15)

    # Recommendations
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(220, 38, 38)
    pdf.cell(0, 10, 'Recommended Actions', ln=True)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 41, 59)

    recommendations = [
        '1. Immediately flag all invoices involving ring members for manual review',
        '2. Request additional documentation from all participating entities',
        '3. Cross-verify entity registration dates and ownership structures',
        '4. Review historical transaction patterns for additional anomalies',
        '5. Consider blocking new invoice submissions from ring members pending investigation'
    ]

    for rec in recommendations:
        pdf.cell(0, 7, rec, ln=True)

    pdf.ln(10)

    # Footer
    pdf.set_y(-30)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, 'This report is generated by QGAI TReDS Quantum Ring Detection Platform', ln=True, align='C')
    pdf.cell(0, 5, 'For internal use only. Handle according to data classification policies.', ln=True, align='C')

    return bytes(pdf.output())


def generate_invoice_report_pdf(invoice: pd.Series) -> bytes:
    """Generate a PDF report for an invoice risk assessment."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title
    pdf.set_font('Helvetica', 'B', 20)
    pdf.set_text_color(124, 58, 237)  # Purple
    pdf.cell(0, 15, 'QGAI TReDS - Invoice Risk Report', ln=True, align='C')
    pdf.ln(5)

    # Report metadata
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}', ln=True, align='C')
    pdf.cell(0, 6, 'Classification: CONFIDENTIAL', ln=True, align='C')
    pdf.ln(10)

    # Risk Level Badge
    risk_colors = {
        'Critical': (220, 38, 38),
        'High': (234, 88, 12),
        'Medium': (217, 119, 6),
        'Low': (22, 163, 74)
    }
    risk_color = risk_colors.get(invoice['risk_level'], (107, 114, 128))

    pdf.set_fill_color(*risk_color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font('Helvetica', 'B', 12)
    risk_text = f"RISK LEVEL: {invoice['risk_level'].upper()}"
    pdf.cell(0, 10, risk_text, ln=True, align='C', fill=True)
    pdf.ln(10)

    # Invoice Summary Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(10, pdf.get_y(), 190, 50, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, f'Invoice: {invoice["invoice_id"]}', ln=True)

    # Format amount
    amount = invoice['amount']
    if amount >= 10000000:
        amount_str = f'Rs. {amount/10000000:.2f} Cr'
    elif amount >= 100000:
        amount_str = f'Rs. {amount/100000:.2f} L'
    else:
        amount_str = f'Rs. {amount:,.0f}'

    pdf.set_x(15)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 10, f'Amount: {amount_str}', ln=True)

    pdf.set_x(15)
    pdf.set_font('Helvetica', '', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(95, 7, f'Invoice Date: {invoice["invoice_date"].strftime("%d %B %Y")}')
    pdf.cell(95, 7, f'Due Date: {invoice["due_date"].strftime("%d %B %Y")}', ln=True)

    pdf.ln(25)

    # Parties Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, 'Transaction Parties', ln=True)

    # Seller Box
    pdf.set_fill_color(243, 232, 255)
    pdf.set_draw_color(167, 139, 250)
    pdf.rect(10, pdf.get_y(), 92, 30, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, 'SELLER', ln=True)

    pdf.set_x(15)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, invoice['seller_name'][:30], ln=True)

    pdf.set_x(15)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(124, 58, 237)
    pdf.cell(0, 5, invoice['seller_id'], ln=True)

    # Buyer Box
    pdf.set_xy(107, pdf.get_y() - 21)
    pdf.set_fill_color(219, 234, 254)
    pdf.set_draw_color(96, 165, 250)
    pdf.rect(107, pdf.get_y(), 92, 30, 'DF')

    pdf.set_xy(112, pdf.get_y() + 5)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 5, 'BUYER', ln=True)

    pdf.set_x(112)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 6, invoice['buyer_name'][:30], ln=True)

    pdf.set_x(112)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 5, invoice['buyer_id'], ln=True)

    pdf.ln(20)

    # Risk Assessment Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(220, 38, 38)
    pdf.cell(0, 10, 'Risk Assessment', ln=True)

    # Risk Scores Table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_fill_color(71, 85, 105)
    pdf.cell(63, 8, 'Metric', 1, 0, 'C', True)
    pdf.cell(63, 8, 'Score', 1, 0, 'C', True)
    pdf.cell(63, 8, 'Risk Level', 1, 1, 'C', True)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 41, 59)

    risk_metrics = [
        ('Composite Risk Score', f'{invoice["composite_risk"]*100:.1f}%', invoice['risk_level']),
        ('Default Probability', f'{invoice["default_prob"]*100:.1f}%', 'High' if invoice['default_prob'] > 0.3 else 'Medium' if invoice['default_prob'] > 0.15 else 'Low'),
        ('Ring Likelihood', f'{invoice["ring_prob"]*100:.1f}%', 'High' if invoice['ring_prob'] > 0.3 else 'Medium' if invoice['ring_prob'] > 0.15 else 'Low')
    ]

    for i, (metric, score, level) in enumerate(risk_metrics):
        fill = i % 2 == 0
        pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
        pdf.cell(63, 7, metric, 1, 0, 'L', fill)
        pdf.cell(63, 7, score, 1, 0, 'C', fill)

        level_color = risk_colors.get(level, (107, 114, 128))
        pdf.set_text_color(*level_color)
        pdf.cell(63, 7, level, 1, 1, 'C', fill)
        pdf.set_text_color(30, 41, 59)

    pdf.ln(10)

    # Ring Membership Alert
    if invoice['is_ring_member']:
        pdf.set_fill_color(254, 226, 226)
        pdf.set_draw_color(220, 38, 38)
        pdf.rect(10, pdf.get_y(), 190, 35, 'DF')

        pdf.set_xy(15, pdf.get_y() + 5)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 7, 'RING DETECTION ALERT', ln=True)

        pdf.set_x(15)
        pdf.set_font('Helvetica', '', 10)
        pdf.set_text_color(30, 41, 59)
        pdf.multi_cell(180, 5,
            f'This invoice involves entities that are members of detected fraud ring {invoice["ring_id"]}. '
            'Circular transaction patterns and abnormal clustering have been identified. '
            'Manual review is strongly recommended before approval.')

        pdf.ln(15)
    else:
        pdf.ln(5)

    # Risk Factors Section
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(217, 119, 6)
    pdf.cell(0, 10, 'Risk Factors Analysis', ln=True)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 41, 59)

    # Default risk factors
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'Default Risk Contributors:', ln=True)

    pdf.set_font('Helvetica', '', 10)
    default_factors = [
        ('Buyer credit history assessment', '+12%' if invoice['default_prob'] > 0.2 else '+5%'),
        ('Transaction amount relative to average', '+5%' if invoice['amount'] > 5000000 else '+2%'),
        ('Industry sector risk factor', '+3%'),
        ('Payment history patterns', '+2%' if invoice['default_prob'] > 0.15 else '+1%')
    ]

    for factor, contrib in default_factors:
        pdf.set_x(15)
        pdf.cell(140, 6, f'- {factor}')
        pdf.set_text_color(220, 38, 38)
        pdf.cell(40, 6, contrib, ln=True)
        pdf.set_text_color(30, 41, 59)

    pdf.ln(5)

    # Ring risk factors
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 7, 'Ring Detection Indicators:', ln=True)

    pdf.set_font('Helvetica', '', 10)
    ring_factors = [
        ('Transaction graph clustering coefficient', 'Abnormal' if invoice['ring_prob'] > 0.3 else 'Normal'),
        ('Entity relationship density', 'High' if invoice['ring_prob'] > 0.2 else 'Normal'),
        ('Bidirectional transaction patterns', 'Detected' if invoice['is_ring_member'] else 'None'),
        ('Registration date correlation', 'Flagged' if invoice['is_ring_member'] else 'Clear')
    ]

    for factor, status in ring_factors:
        pdf.set_x(15)
        pdf.cell(140, 6, f'- {factor}')
        if status in ['Abnormal', 'High', 'Detected', 'Flagged']:
            pdf.set_text_color(220, 38, 38)
        else:
            pdf.set_text_color(22, 163, 74)
        pdf.cell(40, 6, status, ln=True)
        pdf.set_text_color(30, 41, 59)

    pdf.ln(10)

    # Recommendations
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(0, 10, 'Recommended Actions', ln=True)

    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(30, 41, 59)

    if invoice['risk_level'] in ['Critical', 'High']:
        recommendations = [
            '1. Request additional supporting documentation from seller',
            '2. Verify buyer payment capacity and credit standing',
            '3. Cross-check entity registration details',
            '4. Review historical transaction patterns between parties',
            '5. Consider escalating to senior risk analyst for approval'
        ]
    else:
        recommendations = [
            '1. Standard documentation verification recommended',
            '2. Monitor for any pattern changes in future transactions',
            '3. Proceed with normal approval workflow'
        ]

    for rec in recommendations:
        pdf.cell(0, 6, rec, ln=True)

    pdf.ln(10)

    # Quantum Analysis Note
    pdf.set_fill_color(237, 233, 254)
    pdf.set_draw_color(167, 139, 250)
    pdf.rect(10, pdf.get_y(), 190, 25, 'DF')

    pdf.set_xy(15, pdf.get_y() + 5)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(91, 33, 182)
    pdf.cell(0, 5, 'Quantum-Enhanced Analysis', ln=True)

    pdf.set_x(15)
    pdf.set_font('Helvetica', '', 9)
    pdf.multi_cell(180, 4,
        'Risk scores incorporate QUBO-based community detection for ring identification. '
        'This hybrid classical-quantum approach provides enhanced fraud pattern recognition.')

    # Footer
    pdf.set_y(-30)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, 'This report is generated by QGAI TReDS Quantum Ring Detection Platform', ln=True, align='C')
    pdf.cell(0, 5, 'For internal use only. Handle according to data classification policies.', ln=True, align='C')

    return bytes(pdf.output())


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    if 'page' not in st.session_state:
        st.session_state.page = 'Dashboard'
    if 'selected_invoice' not in st.session_state:
        st.session_state.selected_invoice = None
    if 'selected_ring' not in st.session_state:
        st.session_state.selected_ring = None
    if 'decisions' not in st.session_state:
        st.session_state.decisions = []
    if 'invoices' not in st.session_state:
        st.session_state.invoices = generate_invoice_data()
    if 'rings' not in st.session_state:
        st.session_state.rings = generate_ring_data()


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================
def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("### 🔮 QGAI TReDS")
        st.caption("Quantum Ring Detection Platform")
        st.markdown("---")

        # Navigation
        pages = ['Dashboard', 'Invoice Queue', 'Ring Analysis', 'Decision Log']

        for page in pages:
            icon = {'Dashboard': '📊', 'Invoice Queue': '📋', 'Ring Analysis': '🔗', 'Decision Log': '📝'}[page]
            if st.button(f"{icon} {page}", key=f"nav_{page}", width='stretch'):
                st.session_state.page = page
                st.session_state.selected_invoice = None
                st.session_state.selected_ring = None
                st.rerun()

        st.markdown("---")

        # Quick Stats
        df = st.session_state.invoices
        flagged = len(df[df['risk_level'].isin(['Critical', 'High'])])
        rings = len(st.session_state.rings)

        st.metric("Flagged Invoices", flagged, delta="-3 from yesterday")
        st.metric("Active Rings", rings)

        st.markdown("---")
        st.caption("**Analyst:** Risk Team")
        st.caption(f"**Session:** {datetime.now().strftime('%H:%M')}")


# =============================================================================
# DASHBOARD PAGE
# =============================================================================
def render_dashboard():
    """Render the main dashboard."""
    st.title("📊 Risk Overview Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%d %B %Y, %H:%M')}")

    df = st.session_state.invoices
    rings = st.session_state.rings

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Invoices", len(df), delta="+12 today")
    with col2:
        flagged = len(df[df['risk_level'].isin(['Critical', 'High'])])
        st.metric("Flagged (High Risk)", flagged, delta="-3", delta_color="inverse")
    with col3:
        st.metric("Rings Detected", len(rings), delta="+1 this week")
    with col4:
        exposure = df[df['risk_level'].isin(['Critical', 'High'])]['amount'].sum()
        st.metric("Exposure at Risk", format_inr(exposure))

    st.markdown("---")

    # Two columns: Risk Distribution + Alerts
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Risk Distribution")

        risk_counts = df['risk_level'].value_counts()
        colors = {'Critical': '#dc2626', 'High': '#ea580c', 'Medium': '#d97706', 'Low': '#16a34a'}

        fig = go.Figure(data=[
            go.Bar(
                x=list(risk_counts.index),
                y=list(risk_counts.values),
                marker_color=[colors.get(r, '#6b7280') for r in risk_counts.index],
                text=list(risk_counts.values),
                textposition='auto'
            )
        ])
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis_title="Risk Level",
            yaxis_title="Count",
            showlegend=False
        )
        st.plotly_chart(fig, width='stretch')

    with col2:
        st.subheader("Recent Alerts")

        alerts = [
            ("🔴", "Ring RING-001 confidence increased to 92%", "2h ago"),
            ("🟠", "3 new high-risk invoices detected", "4h ago"),
            ("🟣", "QUBO optimization completed - 3 communities found", "6h ago"),
            ("🟢", "Model AUC-ROC stable at 0.8389", "1d ago")
        ]

        for icon, msg, time in alerts:
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 8px 0; border-bottom: 1px solid #e2e8f0;">
                <span style="font-size: 1.2rem; margin-right: 12px;">{icon}</span>
                <div style="flex: 1;">
                    <div style="font-size: 0.9rem; color: #1e293b;">{msg}</div>
                    <div style="font-size: 0.75rem; color: #94a3b8;">{time}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Requires Attention Table
    st.subheader("⚠️ Requires Immediate Attention")

    critical_df = df[df['risk_level'].isin(['Critical', 'High'])].head(5)

    for _, row in critical_df.iterrows():
        col1, col2, col3, col4, col5 = st.columns([2, 2, 1.5, 1.5, 1])

        with col1:
            st.markdown(f"**{row['invoice_id']}**")
            st.caption(f"{row['invoice_date'].strftime('%d %b %Y')}")
        with col2:
            st.markdown(f"{row['seller_id']} → {row['buyer_id']}")
        with col3:
            st.markdown(f"**{format_inr(row['amount'])}**")
        with col4:
            st.markdown(get_risk_badge(row['risk_level']), unsafe_allow_html=True)
        with col5:
            if st.button("Review", key=f"dash_review_{row['invoice_id']}"):
                st.session_state.selected_invoice = row['invoice_id']
                st.session_state.page = 'Invoice Queue'
                st.rerun()

        st.markdown("<hr style='margin: 8px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)

    # Model Performance
    st.markdown("---")
    st.subheader("🎯 System Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #16a34a;">0.8389</div>
            <div style="color: #64748b;">Classical Model AUC-ROC</div>
            <div style="font-size: 0.8rem; color: #16a34a;">✓ Target: 0.75</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #7c3aed;">0.4523</div>
            <div style="color: #64748b;">QUBO Modularity Score</div>
            <div style="font-size: 0.8rem; color: #16a34a;">✓ Target: 0.30</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card" style="text-align: center;">
            <div style="font-size: 2.5rem; font-weight: 700; color: #2563eb;">85%</div>
            <div style="color: #64748b;">Ring Recovery Rate</div>
            <div style="font-size: 0.8rem; color: #16a34a;">✓ Target: 70%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="alert-quantum">
        <strong>🔮 Quantum-Ready System</strong><br>
        <span style="color: #64748b;">Ring detection uses QUBO-based modularity maximization, compatible with D-Wave quantum annealers. Currently running on simulated annealing.</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# INVOICE QUEUE PAGE
# =============================================================================
def render_invoice_queue():
    """Render the invoice queue page."""

    # Check if viewing specific invoice
    if st.session_state.selected_invoice:
        render_invoice_detail()
        return

    st.title("📋 Invoice Risk Queue")

    df = st.session_state.invoices

    # Filters
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

    with col1:
        search = st.text_input("🔍 Search", placeholder="Invoice ID, Entity...", label_visibility="collapsed")
    with col2:
        risk_filter = st.multiselect("Risk Level", ['Critical', 'High', 'Medium', 'Low'],
                                      default=['Critical', 'High', 'Medium', 'Low'], label_visibility="collapsed")
    with col3:
        ring_filter = st.selectbox("Ring Filter", ['All', 'Ring Members', 'Non-Ring'], label_visibility="collapsed")
    with col4:
        if st.button("Clear Filters"):
            st.rerun()

    # Apply filters
    filtered_df = df[df['risk_level'].isin(risk_filter)]

    if ring_filter == 'Ring Members':
        filtered_df = filtered_df[filtered_df['is_ring_member'] == True]
    elif ring_filter == 'Non-Ring':
        filtered_df = filtered_df[filtered_df['is_ring_member'] == False]

    if search:
        search_lower = search.lower()
        filtered_df = filtered_df[
            filtered_df['invoice_id'].str.lower().str.contains(search_lower) |
            filtered_df['seller_id'].str.lower().str.contains(search_lower) |
            filtered_df['buyer_id'].str.lower().str.contains(search_lower)
        ]

    # Results header with export button
    col_count, col_export = st.columns([3, 1])
    with col_count:
        st.markdown(f"**Showing {len(filtered_df)} invoices**")
    with col_export:
        if len(filtered_df) > 0:
            # Prepare CSV export data
            export_df = filtered_df[['invoice_id', 'seller_id', 'seller_name', 'buyer_id', 'buyer_name',
                                      'amount', 'invoice_date', 'due_date', 'risk_level',
                                      'composite_risk', 'default_prob', 'ring_prob', 'is_ring_member', 'ring_id']].copy()
            export_df['invoice_date'] = export_df['invoice_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            export_df['due_date'] = export_df['due_date'].apply(lambda x: x.strftime('%Y-%m-%d'))
            export_df['composite_risk'] = (export_df['composite_risk'] * 100).round(1).astype(str) + '%'
            export_df['default_prob'] = (export_df['default_prob'] * 100).round(1).astype(str) + '%'
            export_df['ring_prob'] = (export_df['ring_prob'] * 100).round(1).astype(str) + '%'
            export_df = export_df.rename(columns={
                'invoice_id': 'Invoice ID',
                'seller_id': 'Seller ID',
                'seller_name': 'Seller Name',
                'buyer_id': 'Buyer ID',
                'buyer_name': 'Buyer Name',
                'amount': 'Amount (INR)',
                'invoice_date': 'Invoice Date',
                'due_date': 'Due Date',
                'risk_level': 'Risk Level',
                'composite_risk': 'Composite Risk',
                'default_prob': 'Default Probability',
                'ring_prob': 'Ring Probability',
                'is_ring_member': 'Ring Member',
                'ring_id': 'Ring ID'
            })
            csv_data = export_df.to_csv(index=False)

            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name=f"invoice_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary"
            )

    st.markdown("---")

    # Invoice Table
    if len(filtered_df) == 0:
        st.info("No invoices match your filters.")
    else:
        for _, row in filtered_df.head(15).iterrows():
            col1, col2, col3, col4, col5, col6 = st.columns([1.5, 2, 1.2, 1, 1.2, 0.8])

            with col1:
                st.markdown(f"**{row['invoice_id']}**")
                st.caption(row['invoice_date'].strftime('%d %b %Y'))
            with col2:
                st.markdown(f"**{row['seller_name'][:20]}**")
                st.caption(f"→ {row['buyer_name'][:20]}")
            with col3:
                st.markdown(f"**{format_inr(row['amount'])}**")
            with col4:
                st.markdown(get_risk_badge(row['risk_level']), unsafe_allow_html=True)
            with col5:
                if row['is_ring_member']:
                    st.markdown(f"🔗 **{row['ring_id']}**")
                else:
                    st.markdown("—")
            with col6:
                if st.button("Review", key=f"review_{row['invoice_id']}"):
                    st.session_state.selected_invoice = row['invoice_id']
                    st.rerun()

            st.markdown("<hr style='margin: 4px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


# =============================================================================
# INVOICE DETAIL PAGE
# =============================================================================
def render_invoice_detail():
    """Render invoice detail view."""
    invoice_id = st.session_state.selected_invoice
    df = st.session_state.invoices

    invoice = df[df['invoice_id'] == invoice_id].iloc[0]

    # Back button
    if st.button("← Back to Queue"):
        st.session_state.selected_invoice = None
        st.rerun()

    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"Invoice {invoice_id}")
    with col2:
        st.markdown(f"<div style='text-align: right;'>{get_risk_badge(invoice['risk_level'])}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Two columns: Details + Risk Assessment
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📄 Invoice Details")

        st.markdown(f"""
        <div class="info-card">
            <div style="margin-bottom: 16px;">
                <div style="color: #64748b; font-size: 0.8rem;">AMOUNT</div>
                <div style="font-size: 1.8rem; font-weight: 700; color: #1e293b;">{format_inr(invoice['amount'])}</div>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                <div>
                    <div style="color: #64748b; font-size: 0.8rem;">INVOICE DATE</div>
                    <div>{invoice['invoice_date'].strftime('%d %B %Y')}</div>
                </div>
                <div>
                    <div style="color: #64748b; font-size: 0.8rem;">DUE DATE</div>
                    <div>{invoice['due_date'].strftime('%d %B %Y')}</div>
                </div>
            </div>
            <hr style="margin: 16px 0;">
            <div style="margin-bottom: 12px;">
                <div style="color: #64748b; font-size: 0.8rem;">SELLER</div>
                <div style="font-weight: 600;">{invoice['seller_name']}</div>
                <div style="color: #7c3aed; font-family: monospace;">{invoice['seller_id']}</div>
            </div>
            <div>
                <div style="color: #64748b; font-size: 0.8rem;">BUYER</div>
                <div style="font-weight: 600;">{invoice['buyer_name']}</div>
                <div style="color: #2563eb; font-family: monospace;">{invoice['buyer_id']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader("🎯 Risk Assessment")

        # Risk gauge
        risk_color = {'Critical': '#dc2626', 'High': '#ea580c', 'Medium': '#d97706', 'Low': '#16a34a'}[invoice['risk_level']]

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=invoice['composite_risk'] * 100,
            title={'text': "Composite Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 15], 'color': '#dcfce7'},
                    {'range': [15, 30], 'color': '#fef3c7'},
                    {'range': [30, 45], 'color': '#ffedd5'},
                    {'range': [45, 100], 'color': '#fee2e2'}
                ]
            }
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, width='stretch')

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Default Risk", f"{invoice['default_prob']*100:.0f}%")
        with col_b:
            st.metric("Ring Likelihood", f"{invoice['ring_prob']*100:.0f}%")

    st.markdown("---")

    # Risk Explanation
    st.subheader("📊 Risk Explanation")

    col1, col2 = st.columns(2)

    with col1:
        if invoice['is_ring_member']:
            st.markdown(f"""
            <div class="alert-danger">
                <strong>⚠️ RING DETECTION ALERT</strong><br>
                This invoice involves entities in detected ring <strong>{invoice['ring_id']}</strong>:
                <ul style="margin: 8px 0;">
                    <li><strong>Circular patterns:</strong> Bidirectional transactions detected</li>
                    <li><strong>High clustering:</strong> 72% coefficient (normal: &lt;30%)</li>
                    <li><strong>Synchronized registration:</strong> Entities registered within 14 days</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="alert-warning">
            <strong>📈 DEFAULT RISK FACTORS</strong><br>
            Contributing to {invoice['default_prob']*100:.0f}% default probability:
            <ul style="margin: 8px 0;">
                <li>Buyer credit history: <span style="color: #dc2626;">+12%</span></li>
                <li>Transaction amount: <span style="color: #dc2626;">+5%</span></li>
                <li>Industry sector risk: <span style="color: #d97706;">+3%</span></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Actions
    st.subheader("⚡ Actions")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("✅ Approve", type="secondary", width='stretch'):
            st.session_state.decisions.append({
                'timestamp': datetime.now(),
                'invoice_id': invoice_id,
                'action': 'APPROVED',
                'risk_level': invoice['risk_level']
            })
            st.success(f"Invoice {invoice_id} approved!")

    with col2:
        if st.button("📝 Request Docs", type="secondary", width='stretch'):
            st.session_state.decisions.append({
                'timestamp': datetime.now(),
                'invoice_id': invoice_id,
                'action': 'DOCS_REQUESTED',
                'risk_level': invoice['risk_level']
            })
            st.info("Documentation requested from seller.")

    with col3:
        if invoice['is_ring_member']:
            if st.button("🔗 View Ring", type="secondary", width='stretch'):
                st.session_state.selected_ring = invoice['ring_id']
                st.session_state.page = 'Ring Analysis'
                st.rerun()

    with col4:
        if st.button("🚫 Block", type="primary", width='stretch'):
            st.session_state.decisions.append({
                'timestamp': datetime.now(),
                'invoice_id': invoice_id,
                'action': 'BLOCKED',
                'risk_level': invoice['risk_level']
            })
            st.error(f"Invoice {invoice_id} blocked!")

    with col5:
        pdf_bytes = generate_invoice_report_pdf(invoice)
        st.download_button(
            label="📥 Export PDF",
            data=pdf_bytes,
            file_name=f"{invoice_id}_risk_report.pdf",
            mime="application/pdf",
            type="secondary",
            width='stretch'
        )


# =============================================================================
# RING ANALYSIS PAGE
# =============================================================================
def render_ring_analysis():
    """Render ring analysis page."""

    rings = st.session_state.rings

    # Check if viewing specific ring
    if st.session_state.selected_ring:
        ring = next((r for r in rings if r['ring_id'] == st.session_state.selected_ring), None)
        if ring:
            render_ring_detail(ring)
            return

    st.title("🔗 Ring Analysis")
    st.caption("Fraud ring communities detected using QUBO-based modularity maximization")

    # Export buttons
    col_space, col_export1, col_export2 = st.columns([2, 1, 1])

    with col_export1:
        # Export rings summary CSV
        rings_summary = []
        for ring in rings:
            rings_summary.append({
                'Ring ID': ring['ring_id'],
                'Confidence': f"{ring['confidence']*100:.0f}%",
                'Member Count': ring['member_count'],
                'Total Exposure (INR)': ring['total_exposure'],
                'Graph Density': ring['density'],
                'Status': ring['status']
            })
        summary_df = pd.DataFrame(rings_summary)
        csv_summary = summary_df.to_csv(index=False)

        st.download_button(
            label="📥 Rings Summary",
            data=csv_summary,
            file_name=f"rings_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary"
        )

    with col_export2:
        # Export detailed members CSV
        members_data = []
        for ring in rings:
            for member in ring['members']:
                members_data.append({
                    'Ring ID': ring['ring_id'],
                    'Ring Status': ring['status'],
                    'Ring Confidence': f"{ring['confidence']*100:.0f}%",
                    'Entity ID': member['id'],
                    'Entity Name': member['name'],
                    'Role': member['role'],
                    'In-Degree': member['in_deg'],
                    'Out-Degree': member['out_deg']
                })
        members_df = pd.DataFrame(members_data)
        csv_members = members_df.to_csv(index=False)

        st.download_button(
            label="📥 All Members",
            data=csv_members,
            file_name=f"ring_members_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary"
        )

    st.markdown("---")

    # Ring cards
    for ring in rings:
        col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1.5, 1])

        status_colors = {'New': '#2563eb', 'Flagged': '#ea580c', 'Under Investigation': '#d97706', 'Blocked': '#dc2626'}

        with col1:
            st.markdown(f"**{ring['ring_id']}**")
            st.caption(f"{ring['member_count']} members")
        with col2:
            st.metric("Confidence", f"{ring['confidence']*100:.0f}%", label_visibility="collapsed")
        with col3:
            st.markdown(f"**{format_inr(ring['total_exposure'])}**")
            st.caption("Exposure")
        with col4:
            color = status_colors.get(ring['status'], '#6b7280')
            st.markdown(f"<span style='color: {color}; font-weight: 600;'>{ring['status']}</span>", unsafe_allow_html=True)
        with col5:
            if st.button("Investigate", key=f"inv_{ring['ring_id']}"):
                st.session_state.selected_ring = ring['ring_id']
                st.rerun()

        st.markdown("<hr style='margin: 12px 0;'>", unsafe_allow_html=True)


def render_ring_detail(ring: Dict):
    """Render ring detail view."""

    if st.button("← Back to Rings"):
        st.session_state.selected_ring = None
        st.rerun()

    col1, col2 = st.columns([3, 1])
    with col1:
        st.title(f"🔗 Ring Investigation: {ring['ring_id']}")
    with col2:
        st.markdown(f"""
        <div style="background: #7c3aed; color: white; padding: 8px 16px; border-radius: 8px; text-align: center; font-weight: 600;">
            {ring['confidence']*100:.0f}% Confidence
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Network Graph + Analysis
    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("Transaction Network")

        # Build network
        G = nx.DiGraph()
        for m in ring['members']:
            G.add_node(m['id'], **m)
        for e in ring['edges']:
            G.add_edge(e[0], e[1], weight=e[2])

        pos = nx.spring_layout(G, k=2, seed=42)

        # Edge trace
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Node trace
        node_x, node_y, node_colors, node_sizes, node_text = [], [], [], [], []
        role_colors = {'Hub': '#7c3aed', 'Intermediary': '#a78bfa', 'Peripheral': '#c4b5fd'}
        role_sizes = {'Hub': 40, 'Intermediary': 30, 'Peripheral': 22}

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            data = G.nodes[node]
            node_colors.append(role_colors.get(data['role'], '#c4b5fd'))
            node_sizes.append(role_sizes.get(data['role'], 22))
            node_text.append(f"{data['name']}<br>Role: {data['role']}")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=2, color='#94a3b8'),
            hoverinfo='none'
        ))

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_colors, line=dict(width=2, color='white')),
            text=[n for n in G.nodes()],
            textposition='bottom center',
            hovertext=node_text,
            hoverinfo='text'
        ))

        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=350,
            plot_bgcolor='#f8fafc'
        )

        st.plotly_chart(fig, width='stretch')

        st.caption("🟣 Hub  |  🟪 Intermediary  |  ⚪ Peripheral  |  — Transaction flow")

    with col2:
        st.subheader("Ring Analysis")

        st.metric("Members", ring['member_count'])
        st.metric("Graph Density", f"{ring['density']:.2f}")
        st.metric("Total Exposure", format_inr(ring['total_exposure']))

        st.markdown("---")

        st.markdown("**🚨 Risk Indicators**")

        indicators = [
            ("High density", f"{ring['density']:.2f}", "< 0.30"),
            ("Circular patterns", "Yes", "No"),
            ("Reciprocal edges", f"{len(ring['edges'])-ring['member_count']}", "0")
        ]

        for name, value, threshold in indicators:
            st.markdown(f"""
            <div style="padding: 8px; background: #fef3c7; border-radius: 6px; margin-bottom: 8px;">
                <div style="font-weight: 600;">⚠️ {name}: <span style="color: #dc2626;">{value}</span></div>
                <div style="font-size: 0.75rem; color: #64748b;">Normal: {threshold}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="alert-quantum">
            <strong>🔮 Quantum Insight</strong><br>
            <span style="font-size: 0.85rem;">Community detected using QUBO-based modularity maximization from 10,000 candidate solutions.</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Members Table
    st.subheader("Ring Members")

    members_df = pd.DataFrame(ring['members'])
    members_df.columns = ['Entity ID', 'Name', 'Role', 'In-Degree', 'Out-Degree']
    st.dataframe(members_df, width='stretch', hide_index=True)

    st.markdown("---")

    # Actions
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚫 Block Entire Ring", type="primary", width='stretch'):
            st.session_state.decisions.append({
                'timestamp': datetime.now(),
                'invoice_id': ring['ring_id'],
                'action': 'RING_BLOCKED',
                'risk_level': 'Critical'
            })
            st.error(f"Ring {ring['ring_id']} blocked! All {ring['member_count']} members flagged.")

    with col2:
        if st.button("🏴 Flag for Review", type="secondary", width='stretch'):
            st.info(f"Ring {ring['ring_id']} flagged for senior review.")

    with col3:
        pdf_bytes = generate_ring_report_pdf(ring)
        st.download_button(
            label="📥 Export Report",
            data=pdf_bytes,
            file_name=f"{ring['ring_id']}_investigation_report.pdf",
            mime="application/pdf",
            type="secondary",
            width='stretch'
        )


# =============================================================================
# DECISION LOG PAGE
# =============================================================================
def render_decision_log():
    """Render decision log page."""
    st.title("📝 Decision Log")
    st.caption("Audit trail of all analyst decisions")

    decisions = st.session_state.decisions

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    action_counts = {}
    for d in decisions:
        action_counts[d['action']] = action_counts.get(d['action'], 0) + 1

    with col1:
        st.metric("Total Decisions", len(decisions))
    with col2:
        st.metric("Approved", action_counts.get('APPROVED', 0))
    with col3:
        st.metric("Blocked", action_counts.get('BLOCKED', 0) + action_counts.get('RING_BLOCKED', 0))
    with col4:
        st.metric("Docs Requested", action_counts.get('DOCS_REQUESTED', 0))

    # Export CSV button
    if decisions:
        st.markdown("")
        col_export, col_space = st.columns([1, 3])
        with col_export:
            # Convert decisions to DataFrame for CSV export
            export_df = pd.DataFrame(decisions)
            export_df['timestamp'] = export_df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
            export_df = export_df.rename(columns={
                'timestamp': 'Timestamp',
                'invoice_id': 'Invoice/Ring ID',
                'action': 'Action',
                'risk_level': 'Risk Level'
            })
            csv_data = export_df.to_csv(index=False)

            st.download_button(
                label="📥 Export to CSV",
                data=csv_data,
                file_name=f"decision_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="secondary"
            )

    st.markdown("---")

    if not decisions:
        st.info("No decisions recorded yet. Review invoices to start logging decisions.")

        # Add sample data option
        if st.button("Load Sample Decisions"):
            st.session_state.decisions = [
                {'timestamp': datetime.now() - timedelta(hours=2), 'invoice_id': 'INV-2025-1001', 'action': 'BLOCKED', 'risk_level': 'Critical'},
                {'timestamp': datetime.now() - timedelta(hours=4), 'invoice_id': 'INV-2025-2015', 'action': 'APPROVED', 'risk_level': 'Low'},
                {'timestamp': datetime.now() - timedelta(hours=6), 'invoice_id': 'RING-001', 'action': 'RING_BLOCKED', 'risk_level': 'Critical'},
                {'timestamp': datetime.now() - timedelta(hours=8), 'invoice_id': 'INV-2025-2022', 'action': 'DOCS_REQUESTED', 'risk_level': 'Medium'},
            ]
            st.rerun()
    else:
        # Sort by timestamp
        sorted_decisions = sorted(decisions, key=lambda x: x['timestamp'], reverse=True)

        action_colors = {
            'APPROVED': '#16a34a',
            'BLOCKED': '#dc2626',
            'RING_BLOCKED': '#7c3aed',
            'DOCS_REQUESTED': '#2563eb'
        }

        for d in sorted_decisions:
            col1, col2, col3, col4 = st.columns([2, 2, 1.5, 2])

            with col1:
                st.markdown(f"**{d['timestamp'].strftime('%d %b %Y %H:%M')}**")
            with col2:
                st.markdown(f"`{d['invoice_id']}`")
            with col3:
                color = action_colors.get(d['action'], '#6b7280')
                st.markdown(f"<span style='background: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8rem;'>{d['action']}</span>", unsafe_allow_html=True)
            with col4:
                st.markdown(get_risk_badge(d['risk_level']), unsafe_allow_html=True)

            st.markdown("<hr style='margin: 8px 0; border-color: #e2e8f0;'>", unsafe_allow_html=True)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Route to correct page
    page = st.session_state.page

    if page == 'Dashboard':
        render_dashboard()
    elif page == 'Invoice Queue':
        render_invoice_queue()
    elif page == 'Ring Analysis':
        render_ring_analysis()
    elif page == 'Decision Log':
        render_decision_log()


if __name__ == "__main__":
    main()
