"""
QGAI Quantum Financial Modeling - TReDS MVP
Main Streamlit Application

Quantum Invoice Ring Detection Platform
Enterprise Decision-Support Interface

Author: QGAI Quantum Financial Modeling Team
Version: 1.0.0
Date: January 2026
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="Quantum Ring Detection | TReDS MVP",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Quantum Invoice Ring Detection Platform\nHybrid Classical-Quantum TReDS MVP"
    }
)

# Import pages
from app.pages import dashboard, invoice_queue, invoice_detail, ring_investigation, decision_log
from app.utils.data_loader import load_sample_data, initialize_session_state
from app.utils.styles import apply_custom_css


def main():
    """Main application entry point."""
    # Apply custom CSS
    apply_custom_css()

    # Initialize session state
    initialize_session_state()

    # Load data if not already loaded
    if 'data_loaded' not in st.session_state:
        with st.spinner("Loading data..."):
            load_sample_data()
        st.session_state.data_loaded = True

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
                    padding: 12px 16px; border-radius: 8px; margin-bottom: 16px;">
            <h2 style="color: white; margin: 0; font-size: 1.25rem;">🔮 QGAI TReDS</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 4px 0 0 0; font-size: 0.75rem;">Quantum Ring Detection</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Navigation
        page = st.radio(
            "Navigation",
            options=["Dashboard", "Invoices", "Rings", "Decision Log"],
            index=["Dashboard", "Invoices", "Rings", "Decision Log"].index(
                st.session_state.get('current_page', 'Dashboard')
            ),
            label_visibility="collapsed"
        )
        st.session_state.current_page = page

        st.markdown("---")

        # Quick Stats
        st.markdown("### Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Flagged", st.session_state.get('flagged_count', 47))
        with col2:
            st.metric("Rings", st.session_state.get('ring_count', 3))

        st.markdown("---")

        # User Info
        st.markdown("**Analyst:** Priya Sharma")
        st.markdown("**Session:** Active")
        st.caption("Last refresh: 2 min ago")

    # Main Content Area
    if page == "Dashboard":
        dashboard.render()
    elif page == "Invoices":
        # Check if viewing specific invoice
        if st.session_state.get('view_invoice_id'):
            invoice_detail.render()
        else:
            invoice_queue.render()
    elif page == "Rings":
        # Check if viewing specific ring
        if st.session_state.get('view_ring_id'):
            ring_investigation.render()
        else:
            ring_investigation.render_list()
    elif page == "Decision Log":
        decision_log.render()


if __name__ == "__main__":
    main()
