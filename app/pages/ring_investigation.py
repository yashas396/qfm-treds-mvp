"""
QGAI TReDS MVP - Ring Investigation Page

Network visualization and ring analysis screen.
Shows transaction graph and ring detection results.
"""

import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from app.utils.styles import format_inr
from app.utils.data_loader import get_ring_by_id, add_decision


def render():
    """Render the ring investigation page for a specific ring."""
    ring_id = st.session_state.get('view_ring_id')

    if not ring_id:
        render_list()
        return

    # Get ring data
    ring = get_ring_by_id(ring_id)

    if not ring:
        st.error(f"Ring {ring_id} not found.")
        return

    # Back button and header
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("&larr; Back"):
            st.session_state.view_ring_id = None
            st.rerun()
    with col2:
        confidence = ring.get('confidence_score', 0.87)
        st.markdown(f"""
        <span style="background-color: #7C3AED; color: white; padding: 8px 16px; border-radius: 4px; font-weight: 600;">
            {confidence*100:.0f}% Confidence
        </span>
        """, unsafe_allow_html=True)

    st.markdown(f"# Ring Investigation: {ring_id}")

    st.markdown("---")

    # Main content: Graph + Analysis
    col1, col2 = st.columns([1.5, 1])

    with col1:
        render_network_graph(ring)

    with col2:
        render_ring_analysis(ring)

    st.markdown("---")

    # Ring Members Table
    render_members_table(ring)

    st.markdown("---")

    # Actions
    render_ring_actions(ring)


def render_list():
    """Render list of all detected rings."""
    st.markdown("# Detected Rings")
    st.caption("Communities identified through QUBO-based modularity maximization")

    st.markdown("---")

    # Sample rings data
    rings = [
        {
            'ring_id': 'RING_001',
            'members': 5,
            'confidence': 0.92,
            'total_exposure': 45000000,
            'density': 0.60,
            'status': 'Under Investigation'
        },
        {
            'ring_id': 'RING_002',
            'members': 7,
            'confidence': 0.87,
            'total_exposure': 82000000,
            'density': 0.50,
            'status': 'Flagged'
        },
        {
            'ring_id': 'RING_003',
            'members': 8,
            'confidence': 0.78,
            'total_exposure': 125000000,
            'density': 0.50,
            'status': 'New'
        }
    ]

    for ring in rings:
        render_ring_card(ring)


def render_ring_card(ring: dict):
    """Render a card for a single ring in the list."""
    status_colors = {
        'New': '#2563EB',
        'Under Investigation': '#D97706',
        'Flagged': '#EA580C',
        'Blocked': '#DC2626'
    }

    status_color = status_colors.get(ring['status'], '#6B7280')

    col1, col2, col3, col4, col5 = st.columns([1.5, 1, 1, 1, 1])

    with col1:
        st.markdown(f"""
        <div>
            <div style="font-weight: 600; font-size: 1.125rem; color: #7C3AED;">{ring['ring_id']}</div>
            <div style="font-size: 0.875rem; color: #6B7280;">{ring['members']} members</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div>
            <div style="color: #6B7280; font-size: 0.75rem;">Confidence</div>
            <div style="font-weight: 600;">{ring['confidence']*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div>
            <div style="color: #6B7280; font-size: 0.75rem;">Exposure</div>
            <div style="font-weight: 600;">{format_inr(ring['total_exposure'])}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div>
            <div style="color: #6B7280; font-size: 0.75rem;">Status</div>
            <div style="color: {status_color}; font-weight: 500;">{ring['status']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        if st.button("Investigate", key=f"inv_{ring['ring_id']}"):
            st.session_state.view_ring_id = ring['ring_id']
            st.rerun()

    st.markdown("<hr style='margin: 16px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)


def render_network_graph(ring: dict):
    """Render the transaction network graph using Plotly."""
    st.markdown("### Transaction Network")

    # Create NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    for member in ring.get('members', []):
        G.add_node(member['entity_id'], **member)

    # Add edges
    for edge in ring.get('edges', []):
        G.add_edge(edge[0], edge[1], weight=edge[2])

    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create edge traces
    edge_x = []
    edge_y = []
    edge_colors = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

        # Check if reciprocal
        if G.has_edge(edge[1], edge[0]):
            edge_colors.extend(['#EA580C', '#EA580C', '#EA580C'])  # Orange for reciprocal
        else:
            edge_colors.extend(['#9CA3AF', '#9CA3AF', '#9CA3AF'])

    # Create node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        member = G.nodes[node]
        role = member.get('role', 'member')

        if role == 'hub':
            node_colors.append('#7C3AED')  # Violet for hub
            node_sizes.append(40)
        elif role == 'intermediary':
            node_colors.append('#A78BFA')  # Light violet
            node_sizes.append(32)
        else:
            node_colors.append('#C4B5FD')  # Lighter violet
            node_sizes.append(24)

        node_text.append(f"{node}<br>{member.get('entity_name', '')}<br>Role: {role}")

    # Create figure
    fig = go.Figure()

    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#9CA3AF'),
        hoverinfo='none',
        mode='lines'
    ))

    # Add nodes
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        hovertext=node_text,
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white')
        ),
        text=[n for n in G.nodes()],
        textposition='bottom center',
        textfont=dict(size=10, family='JetBrains Mono')
    ))

    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0, l=0, r=0, t=0),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        paper_bgcolor='#F9FAFB',
        plot_bgcolor='#F9FAFB',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown("""
    <div style="display: flex; gap: 24px; font-size: 0.75rem; color: #6B7280;">
        <div><span style="color: #7C3AED;">&#x25CF;</span> Hub node</div>
        <div><span style="color: #A78BFA;">&#x25CF;</span> Intermediary</div>
        <div><span style="color: #C4B5FD;">&#x25CF;</span> Peripheral</div>
        <div><span style="color: #EA580C;">&#x2015;</span> Reciprocal</div>
    </div>
    """, unsafe_allow_html=True)


def render_ring_analysis(ring: dict):
    """Render ring analysis panel."""
    st.markdown("### Ring Analysis")

    stats = ring.get('statistics', {})

    # Ring probability
    confidence = ring.get('confidence_score', 0.87)
    st.markdown(f"""
    <div class="card" style="margin-bottom: 16px;">
        <div class="card-body" style="text-align: center;">
            <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Ring Probability</div>
            <div style="font-size: 2rem; font-weight: 700; color: #7C3AED;">{confidence*100:.0f}%</div>
            <div class="progress-bar" style="margin-top: 8px;">
                <div class="progress-bar-fill" style="width: {confidence*100}%; background-color: #7C3AED;"></div>
            </div>
            <div style="font-weight: 500; color: #7C3AED; margin-top: 4px;">HIGH</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Statistics
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 8px;">Statistics</div>
    """, unsafe_allow_html=True)

    stat_items = [
        ('Members', f"{stats.get('member_count', 7)} entities"),
        ('Transactions', f"{stats.get('internal_edges', 10)}"),
        ('Total Value', format_inr(stats.get('total_transaction_amount', 42000000))),
        ('First Activity', f"{stats.get('registration_window_days', 45)} days ago")
    ]

    for label, value in stat_items:
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #E5E7EB;">
            <span style="color: #6B7280;">{label}</span>
            <span style="font-weight: 500;">{value}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Ring Indicators
    st.markdown("### Ring Indicators")

    indicators = ring.get('indicators', [
        {'name': 'High density', 'value': '0.72', 'threshold': '<0.3', 'severity': 'high'},
        {'name': 'Circular patterns', 'value': '5 triangles', 'threshold': '0', 'severity': 'high'},
        {'name': 'Reciprocal edges', 'value': '4 bidirectional', 'threshold': '0', 'severity': 'high'},
        {'name': 'Similar registration', 'value': '14 days', 'threshold': '>90 days', 'severity': 'medium'}
    ])

    for indicator in indicators:
        severity_color = '#DC2626' if indicator['severity'] == 'high' else '#D97706'
        st.markdown(f"""
        <div class="alert-card warning" style="padding: 12px; margin-bottom: 8px;">
            <div style="display: flex; justify-content: space-between;">
                <span style="font-weight: 500;">&#x26A0;&#xFE0F; {indicator['name']}</span>
                <span style="color: {severity_color}; font-weight: 600;">{indicator['value']}</span>
            </div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 4px;">
                Normal: {indicator['threshold']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quantum insight
    st.markdown("""
    <div class="alert-card quantum" style="margin-top: 16px;">
        <div style="font-weight: 600; color: #5B21B6; margin-bottom: 8px;">&#x1F52E; Quantum Insight</div>
        <div style="font-size: 0.875rem; color: #6B7280;">
            Community detected using QUBO-based modularity maximization.
            This partition was selected as optimal from 10,000 candidate solutions.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_members_table(ring: dict):
    """Render ring members table."""
    st.markdown("### Ring Members")

    members = ring.get('members', [])

    # Create table
    table_html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Entity ID</th>
                <th>Name</th>
                <th>Role</th>
                <th>In-Degree</th>
                <th>Out-Degree</th>
            </tr>
        </thead>
        <tbody>
    """

    for member in members:
        role_color = '#7C3AED' if member.get('role') == 'hub' else '#6B7280'
        table_html += f"""
        <tr>
            <td><span class="mono" style="color: #7C3AED;">{member['entity_id']}</span></td>
            <td>{member.get('entity_name', 'Unknown')}</td>
            <td style="color: {role_color}; font-weight: 500;">{member.get('role', 'member').title()}</td>
            <td>{member.get('in_degree', 0)}</td>
            <td>{member.get('out_degree', 0)}</td>
        </tr>
        """

    table_html += """
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)


def render_ring_actions(ring: dict):
    """Render action buttons for ring."""
    st.markdown("### Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Block Entire Ring", type="primary", key="block_ring"):
            st.session_state.show_ring_confirm = 'block'

    with col2:
        if st.button("Flag for Investigation", key="flag_ring"):
            st.session_state.show_ring_confirm = 'flag'

    with col3:
        if st.button("Export Report", key="export_ring"):
            st.info("Report export functionality coming soon.")

    # Confirmation
    if st.session_state.get('show_ring_confirm') == 'block':
        st.markdown("---")
        st.warning(f"**Block entire ring?** This will block all pending invoices from {len(ring.get('members', []))} entities.")

        reason = st.text_area("Reason (required):", key="ring_block_reason")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="cancel_ring_block"):
                st.session_state.show_ring_confirm = None
                st.rerun()
        with col2:
            if st.button("Confirm Block", type="primary", key="confirm_ring_block"):
                if not reason:
                    st.error("Reason is required.")
                else:
                    add_decision(
                        invoice_id=None,
                        action=f"RING_BLOCKED ({ring['ring_id']})",
                        reason=reason
                    )
                    st.success(f"Ring {ring['ring_id']} blocked successfully!")
                    st.session_state.show_ring_confirm = None
                    st.session_state.view_ring_id = None
                    st.rerun()
