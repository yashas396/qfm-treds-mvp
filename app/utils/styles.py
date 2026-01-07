"""
QGAI TReDS MVP - Custom CSS Styles

Design System Implementation based on UI/UX Specification
Color Palette, Typography, and Component Styles
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS matching the design specification."""
    st.markdown("""
    <style>
    /* ============================================
       DESIGN SYSTEM - Color Palette
       ============================================ */
    :root {
        /* Semantic Colors - Risk Levels */
        --risk-critical: #DC2626;
        --risk-high: #EA580C;
        --risk-medium: #D97706;
        --risk-low: #16A34A;

        /* Neutral Colors */
        --bg-primary: #FFFFFF;
        --bg-secondary: #F9FAFB;
        --border-color: #E5E7EB;
        --text-primary: #111827;
        --text-secondary: #6B7280;

        /* Accent Colors */
        --accent-primary: #2563EB;
        --accent-hover: #1D4ED8;
        --accent-quantum: #7C3AED;

        /* Shadows */
        --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* ============================================
       TYPOGRAPHY
       ============================================ */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    .mono {
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
    }

    /* ============================================
       GLOBAL LAYOUT
       ============================================ */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1rem;
    }

    /* ============================================
       METRIC CARDS
       ============================================ */
    .metric-card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }

    .metric-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        line-height: 1.1;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.5rem;
    }

    .metric-trend {
        font-size: 0.75rem;
        margin-top: 0.25rem;
    }

    .metric-trend.positive {
        color: var(--risk-low);
    }

    .metric-trend.negative {
        color: var(--risk-critical);
    }

    /* ============================================
       RISK BADGES
       ============================================ */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
    }

    .risk-badge.critical {
        background-color: var(--risk-critical);
        color: white;
    }

    .risk-badge.high {
        background-color: var(--risk-high);
        color: white;
    }

    .risk-badge.medium {
        background-color: var(--risk-medium);
        color: white;
    }

    .risk-badge.low {
        background-color: var(--risk-low);
        color: white;
    }

    /* ============================================
       DATA TABLE
       ============================================ */
    .data-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.875rem;
    }

    .data-table th {
        background-color: var(--bg-secondary);
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        color: var(--text-primary);
        border-bottom: 2px solid var(--border-color);
    }

    .data-table td {
        padding: 1rem;
        border-bottom: 1px solid var(--border-color);
        vertical-align: middle;
    }

    .data-table tr:hover {
        background-color: #EFF6FF;
    }

    .data-table tr.selected {
        border-left: 3px solid var(--accent-primary);
        background-color: #EFF6FF;
    }

    /* ============================================
       CARDS & PANELS
       ============================================ */
    .card {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: var(--shadow-sm);
        overflow: hidden;
    }

    .card-header {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid var(--border-color);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .card-header h3 {
        margin: 0;
        font-size: 1.125rem;
        font-weight: 600;
    }

    .card-body {
        padding: 1.5rem;
    }

    /* Alert Cards */
    .alert-card {
        background: white;
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1rem 1.5rem;
        margin-bottom: 0.75rem;
    }

    .alert-card.warning {
        border-left-color: var(--risk-medium);
        background-color: #FFFBEB;
    }

    .alert-card.error {
        border-left-color: var(--risk-critical);
        background-color: #FEF2F2;
    }

    .alert-card.info {
        border-left-color: var(--accent-primary);
        background-color: #EFF6FF;
    }

    .alert-card.quantum {
        border-left-color: var(--accent-quantum);
        background-color: #F5F3FF;
    }

    /* ============================================
       PROGRESS BARS
       ============================================ */
    .progress-bar {
        height: 8px;
        background-color: var(--bg-secondary);
        border-radius: 4px;
        overflow: hidden;
    }

    .progress-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .progress-bar-fill.critical { background-color: var(--risk-critical); }
    .progress-bar-fill.high { background-color: var(--risk-high); }
    .progress-bar-fill.medium { background-color: var(--risk-medium); }
    .progress-bar-fill.low { background-color: var(--risk-low); }

    /* ============================================
       BUTTONS
       ============================================ */
    .btn {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-size: 0.875rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s ease;
        border: none;
    }

    .btn-primary {
        background-color: var(--accent-primary);
        color: white;
    }

    .btn-primary:hover {
        background-color: var(--accent-hover);
    }

    .btn-danger {
        background-color: var(--risk-critical);
        color: white;
    }

    .btn-success {
        background-color: white;
        border: 1px solid var(--risk-low);
        color: var(--risk-low);
    }

    .btn-quantum {
        background-color: var(--accent-quantum);
        color: white;
    }

    /* ============================================
       NETWORK GRAPH CONTAINER
       ============================================ */
    .graph-container {
        background: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        min-height: 400px;
    }

    /* ============================================
       EXPLANATION CARDS
       ============================================ */
    .explanation-card {
        background: white;
        border-radius: 8px;
        border-left: 4px solid;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .explanation-card.risk {
        border-left-color: var(--risk-high);
        background-color: #FFF7ED;
    }

    .explanation-card.ring {
        border-left-color: var(--accent-quantum);
        background-color: #F5F3FF;
    }

    .explanation-card h4 {
        margin: 0 0 0.75rem 0;
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }

    .explanation-card ul {
        margin: 0;
        padding-left: 1.25rem;
    }

    .explanation-card li {
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }

    /* ============================================
       LOG ENTRIES
       ============================================ */
    .log-entry {
        background: white;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }

    .log-entry-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.75rem;
    }

    .log-entry-action {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.75rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .log-entry-action.blocked {
        background-color: var(--risk-critical);
        color: white;
    }

    .log-entry-action.approved {
        background-color: var(--risk-low);
        color: white;
    }

    .log-entry-action.flagged {
        background-color: var(--risk-high);
        color: white;
    }

    .log-entry-time {
        font-size: 0.75rem;
        color: var(--text-secondary);
    }

    /* ============================================
       STREAMLIT COMPONENT OVERRIDES
       ============================================ */
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        border-bottom: 2px solid var(--accent-primary);
    }

    /* Buttons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-sm);
    }

    /* DataFrames */
    .stDataFrame {
        border: 1px solid var(--border-color);
        border-radius: 8px;
        overflow: hidden;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        border-radius: 6px;
    }

    /* Text Input */
    .stTextInput > div > div {
        border-radius: 6px;
    }

    /* ============================================
       PAGE HEADERS
       ============================================ */
    .page-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border-color);
    }

    .page-title {
        font-size: 1.875rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }

    .page-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }

    /* ============================================
       ENTITY LINKS
       ============================================ */
    .entity-link {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent-quantum);
        text-decoration: none;
        cursor: pointer;
    }

    .entity-link:hover {
        text-decoration: underline;
    }

    /* ============================================
       TOOLTIPS
       ============================================ */
    .tooltip {
        position: relative;
    }

    .tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.5rem 0.75rem;
        background-color: var(--text-primary);
        color: white;
        font-size: 0.75rem;
        border-radius: 4px;
        white-space: nowrap;
        z-index: 1000;
    }

    /* ============================================
       RESPONSIVE UTILITIES
       ============================================ */
    @media (max-width: 768px) {
        .metric-value {
            font-size: 1.75rem;
        }

        .page-title {
            font-size: 1.5rem;
        }
    }

    /* ============================================
       ANIMATIONS
       ============================================ */
    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .skeleton {
        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
        background-size: 200% 100%;
        animation: shimmer 1.5s infinite;
        border-radius: 4px;
    }

    </style>
    """, unsafe_allow_html=True)


def get_risk_color(risk_level: str) -> str:
    """Get color for risk level."""
    colors = {
        'critical': '#DC2626',
        'high': '#EA580C',
        'medium': '#D97706',
        'low': '#16A34A'
    }
    return colors.get(risk_level.lower(), '#6B7280')


def get_risk_badge_html(risk_level: str) -> str:
    """Generate HTML for risk badge."""
    return f'<span class="risk-badge {risk_level.lower()}">{risk_level}</span>'


def format_inr(amount: float) -> str:
    """Format amount in Indian Rupee format."""
    if amount >= 10000000:  # 1 Crore
        return f"INR {amount/10000000:.2f} Cr"
    elif amount >= 100000:  # 1 Lakh
        return f"INR {amount/100000:.2f} L"
    else:
        return f"INR {amount:,.0f}"
