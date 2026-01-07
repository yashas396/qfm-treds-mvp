"""
QGAI TReDS MVP - Decision Log Page

Audit trail and compliance screen.
Shows all analyst decisions with timestamps and justifications.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from app.utils.styles import format_inr


def render():
    """Render the decision log page."""
    # Page Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# Decision Log")
        st.caption("Audit trail of all analyst decisions")
    with col2:
        if st.button("Export Audit Log", type="secondary"):
            export_audit_log()

    st.markdown("---")

    # Filters
    render_filters()

    st.markdown("")

    # Get decisions
    decisions = st.session_state.get('decisions', [])

    # Add sample decisions if empty
    if not decisions:
        decisions = get_sample_decisions()

    # Apply filters
    filtered_decisions = apply_filters(decisions)

    # Summary metrics
    render_summary_metrics(filtered_decisions)

    st.markdown("")

    # Decision table
    render_decision_table(filtered_decisions)


def render_filters():
    """Render filter controls."""
    col1, col2, col3, col4 = st.columns([1.5, 1.5, 1.5, 1])

    with col1:
        action_filter = st.multiselect(
            "Action Type",
            options=['APPROVED', 'BLOCKED', 'FLAGGED', 'DOCS_REQUESTED', 'RING_BLOCKED'],
            default=['APPROVED', 'BLOCKED', 'FLAGGED', 'DOCS_REQUESTED', 'RING_BLOCKED'],
            label_visibility="collapsed"
        )
        st.session_state.log_filters = st.session_state.get('log_filters', {})
        st.session_state.log_filters['actions'] = action_filter

    with col2:
        date_range = st.selectbox(
            "Date Range",
            options=['Last 24 hours', 'Last 7 days', 'Last 30 days', 'All time'],
            index=1,
            label_visibility="collapsed"
        )
        st.session_state.log_filters['date_range'] = date_range

    with col3:
        search = st.text_input(
            "Search",
            placeholder="Search invoice ID, reason...",
            label_visibility="collapsed"
        )
        st.session_state.log_filters['search'] = search

    with col4:
        if st.button("Clear Filters", key="clear_log_filters"):
            st.session_state.log_filters = {
                'actions': ['APPROVED', 'BLOCKED', 'FLAGGED', 'DOCS_REQUESTED', 'RING_BLOCKED'],
                'date_range': 'Last 7 days',
                'search': ''
            }
            st.rerun()


def apply_filters(decisions: list) -> list:
    """Apply filters to decisions list."""
    filters = st.session_state.get('log_filters', {})
    filtered = decisions.copy()

    # Action filter
    actions = filters.get('actions', [])
    if actions:
        filtered = [d for d in filtered if d['action'] in actions]

    # Date filter
    date_range = filters.get('date_range', 'Last 7 days')
    now = datetime.now()

    if date_range == 'Last 24 hours':
        cutoff = now - timedelta(days=1)
    elif date_range == 'Last 7 days':
        cutoff = now - timedelta(days=7)
    elif date_range == 'Last 30 days':
        cutoff = now - timedelta(days=30)
    else:
        cutoff = datetime(2020, 1, 1)

    filtered = [d for d in filtered if d['timestamp'] >= cutoff]

    # Search filter
    search = filters.get('search', '').lower()
    if search:
        filtered = [
            d for d in filtered
            if search in d.get('invoice_id', '').lower()
            or search in d.get('reason', '').lower()
            or search in d.get('analyst', '').lower()
        ]

    return filtered


def render_summary_metrics(decisions: list):
    """Render summary metrics for decisions."""
    col1, col2, col3, col4, col5 = st.columns(5)

    # Count by action type
    action_counts = {}
    for d in decisions:
        action = d['action']
        action_counts[action] = action_counts.get(action, 0) + 1

    with col1:
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-value" style="font-size: 1.5rem;">{len(decisions)}</div>
            <div class="metric-label">Total Decisions</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        approved = action_counts.get('APPROVED', 0)
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-value" style="font-size: 1.5rem; color: #16A34A;">{approved}</div>
            <div class="metric-label">Approved</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        blocked = action_counts.get('BLOCKED', 0) + action_counts.get('RING_BLOCKED', 0)
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-value" style="font-size: 1.5rem; color: #DC2626;">{blocked}</div>
            <div class="metric-label">Blocked</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        flagged = action_counts.get('FLAGGED', 0)
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-value" style="font-size: 1.5rem; color: #D97706;">{flagged}</div>
            <div class="metric-label">Flagged</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        docs_requested = action_counts.get('DOCS_REQUESTED', 0)
        st.markdown(f"""
        <div class="metric-card" style="padding: 12px;">
            <div class="metric-value" style="font-size: 1.5rem; color: #2563EB;">{docs_requested}</div>
            <div class="metric-label">Docs Requested</div>
        </div>
        """, unsafe_allow_html=True)


def render_decision_table(decisions: list):
    """Render the decision log table."""
    st.markdown(f"**Showing {len(decisions)} decisions**")

    if not decisions:
        st.markdown("""
        <div style="text-align: center; padding: 3rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">&#x1F4CB;</div>
            <h3 style="color: #111827;">No decisions found</h3>
            <p style="color: #6B7280;">No decisions match your current filters.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Sort by timestamp (newest first)
    decisions = sorted(decisions, key=lambda x: x['timestamp'], reverse=True)

    # Table HTML
    table_html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Timestamp</th>
                <th>Invoice/Ring</th>
                <th>Action</th>
                <th>Analyst</th>
                <th>Reason</th>
                <th>Risk Context</th>
            </tr>
        </thead>
        <tbody>
    """

    action_colors = {
        'APPROVED': '#16A34A',
        'BLOCKED': '#DC2626',
        'FLAGGED': '#D97706',
        'DOCS_REQUESTED': '#2563EB',
        'RING_BLOCKED': '#7C3AED'
    }

    for decision in decisions[:20]:  # Show max 20
        action = decision['action']
        color = action_colors.get(action, '#6B7280')

        # Format timestamp
        ts = decision['timestamp']
        time_str = ts.strftime('%d %b %Y %H:%M')

        # Format target (invoice or ring)
        target = decision.get('invoice_id') or decision.get('ring_id') or '—'

        # Risk context
        risk_level = decision.get('risk_level', '—')
        risk_color = {
            'Critical': '#DC2626',
            'High': '#EA580C',
            'Medium': '#D97706',
            'Low': '#16A34A'
        }.get(risk_level, '#6B7280')

        table_html += f"""
        <tr>
            <td>
                <div style="font-size: 0.875rem;">{time_str}</div>
            </td>
            <td>
                <span class="mono" style="color: #7C3AED;">{target}</span>
            </td>
            <td>
                <span style="background-color: {color}; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600;">
                    {action}
                </span>
            </td>
            <td>{decision.get('analyst', 'System')}</td>
            <td style="max-width: 300px; overflow: hidden; text-overflow: ellipsis;">
                {decision.get('reason', '—')}
            </td>
            <td>
                <span style="color: {risk_color}; font-weight: 500;">{risk_level}</span>
            </td>
        </tr>
        """

    table_html += """
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # Pagination info
    if len(decisions) > 20:
        st.markdown(f"""
        <div style="text-align: center; color: #6B7280; margin-top: 16px;">
            Showing 1-20 of {len(decisions)} decisions
        </div>
        """, unsafe_allow_html=True)


def get_sample_decisions() -> list:
    """Generate sample decision data."""
    now = datetime.now()

    return [
        {
            'timestamp': now - timedelta(hours=2),
            'invoice_id': 'INV_2025_00145',
            'action': 'BLOCKED',
            'analyst': 'Analyst_01',
            'reason': 'Part of suspected ring RING_001. High density connections detected.',
            'risk_level': 'Critical'
        },
        {
            'timestamp': now - timedelta(hours=4),
            'invoice_id': 'INV_2025_00132',
            'action': 'APPROVED',
            'analyst': 'Analyst_02',
            'reason': 'Low risk score. Verified entity history.',
            'risk_level': 'Low'
        },
        {
            'timestamp': now - timedelta(hours=6),
            'ring_id': 'RING_002',
            'action': 'RING_BLOCKED',
            'analyst': 'Analyst_01',
            'reason': 'Confirmed circular transaction pattern. All 7 members blocked.',
            'risk_level': 'Critical'
        },
        {
            'timestamp': now - timedelta(hours=8),
            'invoice_id': 'INV_2025_00128',
            'action': 'DOCS_REQUESTED',
            'analyst': 'Analyst_03',
            'reason': 'Requested GST returns and bank statements for verification.',
            'risk_level': 'Medium'
        },
        {
            'timestamp': now - timedelta(hours=12),
            'invoice_id': 'INV_2025_00115',
            'action': 'FLAGGED',
            'analyst': 'Analyst_02',
            'reason': 'Unusual transaction amount. Flagged for senior review.',
            'risk_level': 'High'
        },
        {
            'timestamp': now - timedelta(days=1, hours=2),
            'invoice_id': 'INV_2025_00098',
            'action': 'APPROVED',
            'analyst': 'Analyst_01',
            'reason': 'Risk factors mitigated. Long entity history confirmed.',
            'risk_level': 'Medium'
        },
        {
            'timestamp': now - timedelta(days=1, hours=5),
            'invoice_id': 'INV_2025_00089',
            'action': 'BLOCKED',
            'analyst': 'Analyst_03',
            'reason': 'Entity R0203 on watchlist. Suspected fraudulent activity.',
            'risk_level': 'Critical'
        },
        {
            'timestamp': now - timedelta(days=2),
            'invoice_id': 'INV_2025_00072',
            'action': 'APPROVED',
            'analyst': 'Analyst_02',
            'reason': 'Standard approval - low risk indicators.',
            'risk_level': 'Low'
        },
        {
            'timestamp': now - timedelta(days=2, hours=8),
            'ring_id': 'RING_001',
            'action': 'RING_BLOCKED',
            'analyst': 'Analyst_01',
            'reason': 'QUBO analysis confirmed community with 0.92 confidence. All 5 members blocked.',
            'risk_level': 'Critical'
        },
        {
            'timestamp': now - timedelta(days=3),
            'invoice_id': 'INV_2025_00065',
            'action': 'DOCS_REQUESTED',
            'analyst': 'Analyst_03',
            'reason': 'Missing supporting documentation for high-value invoice.',
            'risk_level': 'High'
        },
        {
            'timestamp': now - timedelta(days=3, hours=6),
            'invoice_id': 'INV_2025_00058',
            'action': 'APPROVED',
            'analyst': 'Analyst_01',
            'reason': 'Verified against external sources. Risk acceptable.',
            'risk_level': 'Medium'
        },
        {
            'timestamp': now - timedelta(days=4),
            'invoice_id': 'INV_2025_00042',
            'action': 'FLAGGED',
            'analyst': 'Analyst_02',
            'reason': 'New entity with limited history. Requires monitoring.',
            'risk_level': 'Medium'
        }
    ]


def export_audit_log():
    """Export audit log to CSV."""
    decisions = st.session_state.get('decisions', get_sample_decisions())

    # Convert to DataFrame
    df = pd.DataFrame(decisions)

    # Format for export
    if 'timestamp' in df.columns:
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download Audit Log CSV",
        data=csv,
        file_name=f"audit_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
