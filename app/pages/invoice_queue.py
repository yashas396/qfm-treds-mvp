"""
QGAI TReDS MVP - Invoice Queue Page

Primary working screen for risk analysts.
Displays risk-sorted invoice queue with filtering capabilities.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from app.utils.styles import format_inr, get_risk_color


def render():
    """Render the invoice queue page."""
    df = st.session_state.invoices_df

    # Page Header
    col1, col2 = st.columns([3, 1])
    with col1:
        flagged_count = len(df[df['risk_level'].isin(['Critical', 'High'])])
        st.markdown(f"# Invoice Risk Queue")
        st.caption(f"{len(df)} total invoices | {flagged_count} flagged for review")
    with col2:
        if st.button("Export CSV", type="secondary"):
            export_to_csv(df)

    st.markdown("---")

    # Filters Row
    render_filters()

    # Apply filters
    filtered_df = apply_filters(df)

    st.markdown("")

    # Results count
    st.markdown(f"**Showing {len(filtered_df)} invoices**")

    # Invoice Table
    render_invoice_table(filtered_df)

    # Pagination (simplified for MVP)
    render_pagination(len(filtered_df))


def render_filters():
    """Render filter controls."""
    col1, col2, col3, col4 = st.columns([2, 1.5, 1.5, 1])

    with col1:
        search = st.text_input(
            "Search",
            placeholder="Search invoices, entities...",
            value=st.session_state.filters.get('search_query', ''),
            label_visibility="collapsed"
        )
        st.session_state.filters['search_query'] = search

    with col2:
        risk_levels = st.multiselect(
            "Risk Level",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=st.session_state.filters.get('risk_levels', ['Critical', 'High', 'Medium', 'Low']),
            label_visibility="collapsed"
        )
        st.session_state.filters['risk_levels'] = risk_levels

    with col3:
        ring_filter = st.selectbox(
            "Ring Filter",
            options=['All Invoices', 'Ring Members Only', 'Non-Ring Only'],
            label_visibility="collapsed"
        )
        st.session_state.filters['ring_only'] = ring_filter

    with col4:
        if st.button("Clear Filters"):
            st.session_state.filters = {
                'risk_levels': ['Critical', 'High', 'Medium', 'Low'],
                'ring_only': 'All Invoices',
                'amount_min': 0,
                'amount_max': 100000000,
                'search_query': ''
            }
            st.rerun()

    # Active filters display
    active_filters = []
    if risk_levels and len(risk_levels) < 4:
        active_filters.append(f"Risk: {', '.join(risk_levels)}")
    if ring_filter != 'All Invoices':
        active_filters.append(ring_filter)
    if search:
        active_filters.append(f"Search: '{search}'")

    if active_filters:
        st.markdown(f"**Active Filters:** {' | '.join(active_filters)}")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Apply current filters to the dataframe."""
    filtered = df.copy()

    # Risk level filter
    risk_levels = st.session_state.filters.get('risk_levels', ['Critical', 'High', 'Medium', 'Low'])
    if risk_levels:
        filtered = filtered[filtered['risk_level'].isin(risk_levels)]

    # Ring filter
    ring_filter = st.session_state.filters.get('ring_only', 'All Invoices')
    if ring_filter == 'Ring Members Only':
        filtered = filtered[filtered['is_ring_member'] == True]
    elif ring_filter == 'Non-Ring Only':
        filtered = filtered[filtered['is_ring_member'] == False]

    # Search filter
    search_query = st.session_state.filters.get('search_query', '').lower()
    if search_query:
        filtered = filtered[
            filtered['invoice_id'].str.lower().str.contains(search_query) |
            filtered['seller_id'].str.lower().str.contains(search_query) |
            filtered['buyer_id'].str.lower().str.contains(search_query)
        ]

    return filtered


def render_invoice_table(df: pd.DataFrame):
    """Render the invoice table with click interactions."""
    if len(df) == 0:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">&#x1F4CB;</div>
            <h3 style="color: #111827;">No invoices to review</h3>
            <p style="color: #6B7280;">All flagged invoices have been processed or no invoices match your filters.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Display invoices in a more interactive way
    for idx, row in df.head(10).iterrows():
        render_invoice_row(row)


def render_invoice_row(row):
    """Render a single invoice row."""
    risk_color = get_risk_color(row['risk_level'])

    col1, col2, col3, col4, col5, col6 = st.columns([1.2, 1.5, 1, 1, 1, 0.8])

    with col1:
        st.markdown(f"""
        <div style="font-family: 'JetBrains Mono', monospace; font-weight: 600;">
            {row['invoice_id']}
        </div>
        <div style="font-size: 0.75rem; color: #6B7280; margin-top: 2px;">
            {row['invoice_date'].strftime('%d %b %Y')}
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div>
            <span style="color: #7C3AED;">{row['seller_id']}</span>
            &rarr;
            <span style="color: #2563EB;">{row['buyer_id']}</span>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"**{format_inr(row['amount'])}**")

    with col4:
        # Risk badge
        st.markdown(f"""
        <span class="risk-badge {row['risk_level'].lower()}">{row['risk_level']}</span>
        """, unsafe_allow_html=True)

    with col5:
        if row['is_ring_member']:
            st.markdown(f"""
            <span style="color: #7C3AED; font-weight: 500;">
                &#x1F517; {row['ring_id']}
            </span>
            """, unsafe_allow_html=True)
        else:
            st.markdown("—")

    with col6:
        if st.button("Review", key=f"review_{row['invoice_id']}"):
            st.session_state.view_invoice_id = row['invoice_id']
            st.rerun()

    # Divider
    st.markdown("<hr style='margin: 8px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)


def render_pagination(total_items: int):
    """Render pagination controls."""
    st.markdown("")
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        items_per_page = 10
        total_pages = (total_items + items_per_page - 1) // items_per_page

        if total_pages > 1:
            st.markdown(f"""
            <div style="text-align: center; color: #6B7280;">
                Page 1 of {total_pages} | Showing 1-{min(10, total_items)} of {total_items}
            </div>
            """, unsafe_allow_html=True)


def export_to_csv(df: pd.DataFrame):
    """Export dataframe to CSV."""
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"invoice_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
