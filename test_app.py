"""Simple test app to identify errors"""
import streamlit as st

st.set_page_config(page_title="Test App", page_icon="🔮", layout="wide")

st.title("Test App - QGAI TReDS")

try:
    st.write("Step 1: Basic Streamlit - OK")

    # Test imports
    import pandas as pd
    import numpy as np
    st.write("Step 2: Pandas/Numpy imports - OK")

    # Test plotly
    import plotly.graph_objects as go
    st.write("Step 3: Plotly import - OK")

    # Test networkx
    import networkx as nx
    st.write("Step 4: NetworkX import - OK")

    # Test data generation
    from app.utils.data_loader import generate_invoice_dataframe
    df = generate_invoice_dataframe()
    st.write(f"Step 5: Data generation - OK ({len(df)} invoices)")

    # Test styles
    from app.utils.styles import apply_custom_css
    apply_custom_css()
    st.write("Step 6: Custom CSS - OK")

    # Test session state
    from app.utils.data_loader import initialize_session_state, load_sample_data
    initialize_session_state()
    st.write("Step 7: Session state init - OK")

    load_sample_data()
    st.write("Step 8: Load sample data - OK")

    # Test dashboard render
    from app.pages import dashboard
    st.write("Step 9: Dashboard import - OK")

    st.markdown("---")
    st.success("All tests passed! Now rendering dashboard...")

    dashboard.render()

except Exception as e:
    st.error(f"Error: {type(e).__name__}: {e}")
    import traceback
    st.code(traceback.format_exc())
