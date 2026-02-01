"""
QGAI TReDS MVP - Dashboard Page

Main dashboard showing risk overview, key metrics, and alerts.
Entry point for the application.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from app.utils.styles import format_inr, get_risk_color


def render():
    """Render the dashboard page."""
    # Page Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("# Risk Overview Dashboard")
        st.caption(f"Last updated: {datetime.now().strftime('%H:%M')} | Today: {datetime.now().strftime('%d %B %Y')}")
    with col2:
        if st.button("Refresh Data", type="secondary"):
            st.rerun()

    st.markdown("---")

    # Key Metrics Row
    render_metrics_row()

    st.markdown("")  # Spacing

    # Two Column Layout: Risk Distribution + Alerts
    col1, col2 = st.columns([1.2, 1])

    with col1:
        render_risk_distribution()

    with col2:
        render_alerts_panel()

    st.markdown("")  # Spacing

    # Requires Immediate Attention
    render_attention_table()

    # Model Performance Summary
    render_model_performance()


def render_metrics_row():
    """Render the main metrics cards."""
    df = st.session_state.invoices_df

    # Calculate metrics
    total_invoices = len(df)
    flagged_count = len(df[df['risk_level'].isin(['Critical', 'High'])])
    ring_count = 3  # From detected rings
    total_exposure = df[df['risk_level'].isin(['Critical', 'High'])]['amount'].sum()

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">5,234</div>
            <div class="metric-label">Invoices Processed</div>
            <div class="metric-trend positive">+12% from last week</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #EA580C;">{flagged_count}</div>
            <div class="metric-label">Flagged (High Risk)</div>
            <div class="metric-trend negative">-8% from last week</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #7C3AED;">{ring_count}</div>
            <div class="metric-label">Rings Detected</div>
            <div class="metric-trend">No change</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        exposure_display = format_inr(total_exposure)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #DC2626;">{exposure_display}</div>
            <div class="metric-label">Exposure at Risk</div>
            <div class="metric-trend positive">-15% from last week</div>
        </div>
        """, unsafe_allow_html=True)


def render_risk_distribution():
    """Render risk distribution chart."""
    st.markdown("### Risk Distribution")

    df = st.session_state.invoices_df

    # Calculate distribution
    risk_counts = df['risk_level'].value_counts()
    risk_data = {
        'Critical': risk_counts.get('Critical', 0),
        'High': risk_counts.get('High', 0),
        'Medium': risk_counts.get('Medium', 0),
        'Low': risk_counts.get('Low', 0)
    }

    # Add synthetic data for visualization
    risk_data['Low'] = max(risk_data['Low'], 150)  # Ensure visible proportion

    # Create horizontal bar chart
    fig = go.Figure()

    colors = {
        'Critical': '#DC2626',
        'High': '#EA580C',
        'Medium': '#D97706',
        'Low': '#16A34A'
    }

    total = sum(risk_data.values())

    # Create stacked bar
    cumulative = 0
    for risk_level in ['Critical', 'High', 'Medium', 'Low']:
        count = risk_data[risk_level]
        percentage = (count / total) * 100 if total > 0 else 0

        fig.add_trace(go.Bar(
            y=['Risk'],
            x=[percentage],
            orientation='h',
            name=f'{risk_level} ({count})',
            marker_color=colors[risk_level],
            hovertemplate=f'{risk_level}: {count} invoices ({percentage:.1f}%)<extra></extra>'
        ))

    fig.update_layout(
        barmode='stack',
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=-0.3,
            xanchor='center',
            x=0.5
        ),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, width="stretch")

    # Detailed breakdown
    st.markdown("#### Breakdown by Category")
    for risk_level in ['Critical', 'High', 'Medium', 'Low']:
        count = risk_data[risk_level]
        color = colors[risk_level]
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 8px;">
            <span style="width: 12px; height: 12px; background-color: {color}; border-radius: 2px; margin-right: 8px;"></span>
            <span style="flex: 1;">{risk_level}</span>
            <span style="font-weight: 600;">{count}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("View All Invoices", key="view_all_invoices"):
        st.session_state.current_page = "Invoices"
        st.rerun()


def render_alerts_panel():
    """Render recent alerts panel."""
    st.markdown("### Recent Alerts")

    alerts = [
        {
            'type': 'ring',
            'title': 'New ring detected',
            'description': 'Community #3 identified with 8 members',
            'time': '2h ago',
            'severity': 'high'
        },
        {
            'type': 'surge',
            'title': 'High-risk invoice surge',
            'description': '+15% flagged invoices today',
            'time': '4h ago',
            'severity': 'medium'
        },
        {
            'type': 'entity',
            'title': 'Entity flagged',
            'description': 'R0203 added to watchlist',
            'time': '6h ago',
            'severity': 'high'
        },
        {
            'type': 'model',
            'title': 'Model performance stable',
            'description': 'AUC-ROC: 0.8389 (target: 0.75)',
            'time': '1d ago',
            'severity': 'low'
        }
    ]

    for alert in alerts:
        severity_colors = {
            'high': '#DC2626',
            'medium': '#D97706',
            'low': '#16A34A'
        }
        color = severity_colors.get(alert['severity'], '#6B7280')

        st.markdown(f"""
        <div class="alert-card" style="border-left-color: {color};">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div>
                    <div style="font-weight: 600; color: #111827;">{alert['title']}</div>
                    <div style="font-size: 0.875rem; color: #6B7280; margin-top: 2px;">{alert['description']}</div>
                </div>
                <div style="font-size: 0.75rem; color: #9CA3AF;">{alert['time']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("View All Alerts", key="view_all_alerts"):
        st.session_state.current_page = "Decision Log"
        st.rerun()


def render_attention_table():
    """Render the requires immediate attention table."""
    st.markdown("### Requires Immediate Attention")

    df = st.session_state.invoices_df
    critical_invoices = df[df['risk_level'].isin(['Critical', 'High'])].head(5)

    # Create table HTML
    table_html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Invoice ID</th>
                <th>Parties</th>
                <th>Amount</th>
                <th>Risk Level</th>
                <th>Ring</th>
                <th>Action</th>
            </tr>
        </thead>
        <tbody>
    """

    for _, row in critical_invoices.iterrows():
        risk_color = get_risk_color(row['risk_level'])
        ring_badge = f'<span style="color: #7C3AED;">Ring #{row["ring_id"][-1]}</span>' if row['is_ring_member'] else '—'

        table_html += f"""
        <tr style="cursor: pointer;" onclick="window.location.href='#'">
            <td><span class="mono">{row['invoice_id']}</span></td>
            <td>{row['seller_id']} &rarr; {row['buyer_id']}</td>
            <td>{format_inr(row['amount'])}</td>
            <td><span class="risk-badge {row['risk_level'].lower()}">{row['risk_level']}</span></td>
            <td>{ring_badge}</td>
            <td><button class="btn btn-primary" style="padding: 4px 12px; font-size: 12px;">Review</button></td>
        </tr>
        """

    table_html += """
        </tbody>
    </table>
    """

    st.markdown(table_html, unsafe_allow_html=True)

    # Make rows clickable using streamlit
    st.markdown("")
    cols = st.columns(len(critical_invoices))
    for i, (_, row) in enumerate(critical_invoices.iterrows()):
        with cols[i]:
            if st.button(f"Review {row['invoice_id']}", key=f"review_{row['invoice_id']}"):
                st.session_state.view_invoice_id = row['invoice_id']
                st.session_state.current_page = "Invoices"
                st.rerun()


def render_model_performance():
    """Render model performance summary."""
    st.markdown("---")
    st.markdown("### System Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-body" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #16A34A;">0.8389</div>
                <div style="color: #6B7280; margin-top: 4px;">Classical Model AUC-ROC</div>
                <div style="font-size: 0.75rem; color: #16A34A; margin-top: 4px;">Target: 0.75</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-body" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #7C3AED;">0.4523</div>
                <div style="color: #6B7280; margin-top: 4px;">QUBO Modularity</div>
                <div style="font-size: 0.75rem; color: #16A34A; margin-top: 4px;">Target: 0.30</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div class="card-body" style="text-align: center;">
                <div style="font-size: 2rem; font-weight: 700; color: #2563EB;">85%</div>
                <div style="color: #6B7280; margin-top: 4px;">Ring Recovery Rate</div>
                <div style="font-size: 0.75rem; color: #16A34A; margin-top: 4px;">Target: 70%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Quantum insight
    st.markdown("")
    st.markdown("""
    <div class="alert-card quantum">
        <div style="display: flex; align-items: flex-start;">
            <span style="font-size: 1.5rem; margin-right: 12px;">&#x1F52E;</span>
            <div>
                <div style="font-weight: 600; color: #5B21B6;">Quantum-Ready System</div>
                <div style="font-size: 0.875rem; color: #6B7280; margin-top: 4px;">
                    Ring detection uses QUBO-based modularity maximization, compatible with D-Wave quantum annealers.
                    Currently running on simulated annealing for MVP demonstration.
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
