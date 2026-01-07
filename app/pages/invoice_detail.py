"""
QGAI TReDS MVP - Invoice Detail Page

Single invoice investigation and decision screen.
Shows invoice details, risk assessment, and explanations.
"""

import streamlit as st
from datetime import datetime
from app.utils.styles import format_inr, get_risk_color
from app.utils.data_loader import get_invoice_by_id, add_decision


def render():
    """Render the invoice detail page."""
    invoice_id = st.session_state.get('view_invoice_id')

    if not invoice_id:
        st.warning("No invoice selected. Returning to queue.")
        st.session_state.current_page = "Invoices"
        st.rerun()
        return

    # Get invoice data
    invoice = get_invoice_by_id(invoice_id)

    if not invoice:
        st.error(f"Invoice {invoice_id} not found.")
        return

    # Back button and header
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("&larr; Back to Queue"):
            st.session_state.view_invoice_id = None
            st.rerun()
    with col2:
        risk_level = invoice['risk_level']
        st.markdown(f"""
        <span class="risk-badge {risk_level.lower()}" style="font-size: 1rem; padding: 8px 16px;">
            {risk_level} Risk
        </span>
        """, unsafe_allow_html=True)

    st.markdown(f"# Invoice {invoice_id}")

    st.markdown("---")

    # Main content: Two columns
    col1, col2 = st.columns([1, 1])

    with col1:
        render_invoice_details(invoice)

    with col2:
        render_risk_assessment(invoice)

    st.markdown("---")

    # Risk Explanation Section
    render_risk_explanation(invoice)

    st.markdown("---")

    # Actions Section
    render_action_buttons(invoice)


def render_invoice_details(invoice: dict):
    """Render invoice details panel."""
    st.markdown("### Invoice Details")

    st.markdown(f"""
    <div class="card">
        <div class="card-body">
            <div style="margin-bottom: 16px;">
                <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Invoice ID</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.125rem; font-weight: 600;">
                    {invoice['invoice_id']}
                </div>
            </div>

            <div style="margin-bottom: 16px;">
                <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Amount</div>
                <div style="font-size: 1.5rem; font-weight: 700; color: #111827;">
                    {format_inr(invoice['amount'])}
                </div>
            </div>

            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px;">
                <div>
                    <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Invoice Date</div>
                    <div>{invoice['invoice_date'].strftime('%d %B %Y')}</div>
                </div>
                <div>
                    <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Due Date</div>
                    <div>{invoice['due_date'].strftime('%d %B %Y')}</div>
                </div>
            </div>

            <div style="margin-bottom: 16px;">
                <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Status</div>
                <div>{invoice.get('status', 'Pending Review')}</div>
            </div>

            <hr style="border-color: #E5E7EB; margin: 16px 0;">

            <div style="margin-bottom: 16px;">
                <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Seller</div>
                <div style="font-weight: 600;">{invoice['seller_name']}</div>
                <div style="font-family: 'JetBrains Mono', monospace; color: #7C3AED; font-size: 0.875rem;">
                    {invoice['seller_id']}
                </div>
            </div>

            <div style="margin-bottom: 16px;">
                <div style="color: #6B7280; font-size: 0.75rem; text-transform: uppercase;">Buyer</div>
                <div style="font-weight: 600;">{invoice['buyer_name']}</div>
                <div style="font-family: 'JetBrains Mono', monospace; color: #2563EB; font-size: 0.875rem;">
                    {invoice['buyer_id']}
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_assessment(invoice: dict):
    """Render risk assessment panel."""
    st.markdown("### Risk Assessment")

    composite_risk = invoice['composite_risk']
    default_prob = invoice['default_prob']
    ring_prob = invoice['ring_prob']

    # Determine color based on risk
    if composite_risk >= 0.5:
        color = '#DC2626'
        category = 'CRITICAL'
    elif composite_risk >= 0.35:
        color = '#EA580C'
        category = 'HIGH'
    elif composite_risk >= 0.2:
        color = '#D97706'
        category = 'MEDIUM'
    else:
        color = '#16A34A'
        category = 'LOW'

    st.markdown(f"""
    <div class="card">
        <div class="card-body">
            <div style="text-align: center; padding: 20px 0;">
                <div style="color: #6B7280; font-size: 0.875rem; text-transform: uppercase; margin-bottom: 8px;">
                    Composite Risk
                </div>
                <div style="font-size: 3rem; font-weight: 700; color: {color};">
                    {composite_risk:.2f}
                </div>
                <div class="progress-bar" style="margin: 16px auto; max-width: 200px;">
                    <div class="progress-bar-fill" style="width: {composite_risk*100}%; background-color: {color};"></div>
                </div>
                <div style="font-weight: 600; color: {color};">{category}</div>
            </div>

            <hr style="border-color: #E5E7EB; margin: 16px 0;">

            <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 12px;">Risk Components</div>

            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Default Risk</span>
                    <span style="font-weight: 600;">{default_prob*100:.0f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-bar-fill {'high' if default_prob > 0.3 else 'medium' if default_prob > 0.15 else 'low'}"
                         style="width: {default_prob*100}%;"></div>
                </div>
            </div>

            <div style="margin-bottom: 16px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span>Ring Likelihood</span>
                    <span style="font-weight: 600;">{ring_prob*100:.0f}%</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-bar-fill"
                         style="width: {ring_prob*100}%; background-color: #7C3AED;"></div>
                </div>
            </div>
    """, unsafe_allow_html=True)

    # Ring information if applicable
    if invoice['is_ring_member']:
        st.markdown(f"""
            <hr style="border-color: #E5E7EB; margin: 16px 0;">

            <div style="color: #6B7280; font-size: 0.875rem; margin-bottom: 8px;">Community Assignment</div>
            <div style="font-weight: 600; color: #7C3AED;">
                {invoice['ring_id']}
            </div>
        """, unsafe_allow_html=True)

        if st.button("Investigate Ring", type="primary", key="investigate_ring"):
            st.session_state.view_ring_id = invoice['ring_id']
            st.session_state.current_page = "Rings"
            st.rerun()

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_risk_explanation(invoice: dict):
    """Render risk explanation section."""
    st.markdown("### Risk Explanation")
    st.markdown(f"**Why this invoice is flagged as {invoice['risk_level']} RISK:**")

    col1, col2 = st.columns(2)

    # Ring detection alert (if applicable)
    if invoice['is_ring_member'] and invoice.get('ring_explanation'):
        with col1:
            ring_exp = invoice['ring_explanation']
            st.markdown("""
            <div class="explanation-card ring">
                <h4 style="color: #5B21B6;">&#x26A0;&#xFE0F; RING DETECTION ALERT</h4>
                <p style="margin-bottom: 12px;">
                    This invoice involves entities in a detected transaction ring:
                </p>
                <ul>
            """, unsafe_allow_html=True)

            for indicator in ring_exp['indicators']:
                st.markdown(f"""
                    <li>
                        <strong>{indicator['indicator']}:</strong> {indicator['description']}
                    </li>
                """, unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)

    # Default risk factors
    risk_exp = invoice.get('risk_explanation', {})

    with col2 if invoice['is_ring_member'] else col1:
        st.markdown("""
        <div class="explanation-card risk">
            <h4 style="color: #C2410C;">&#x1F4CA; DEFAULT RISK FACTORS</h4>
            <p style="margin-bottom: 12px;">
                Contributing to default probability:
            </p>
            <ul>
        """, unsafe_allow_html=True)

        factors = risk_exp.get('risk_factors', [])
        if factors:
            for factor in factors:
                st.markdown(f"""
                    <li>
                        <strong>{factor['factor']}:</strong> {factor['value']}
                        <span style="color: #DC2626;">({factor['impact']} impact)</span>
                    </li>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <li>No significant risk factors identified</li>
            """, unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)

    # Protective factors
    protective = risk_exp.get('protective_factors', [])
    if protective:
        st.markdown("""
        <div class="alert-card info" style="margin-top: 16px;">
            <strong>Protective Factors:</strong>
            <ul style="margin: 8px 0 0 0;">
        """, unsafe_allow_html=True)

        for factor in protective:
            st.markdown(f"""
                <li>{factor['factor']}: {factor['value']} <span style="color: #16A34A;">({factor['impact']} impact)</span></li>
            """, unsafe_allow_html=True)

        st.markdown("</ul></div>", unsafe_allow_html=True)


def render_action_buttons(invoice: dict):
    """Render action buttons with confirmation."""
    st.markdown("### Actions")
    st.caption("Decision will be logged with timestamp and justification.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("&#x2713; Approve", key="approve_btn", type="secondary"):
            st.session_state.show_confirm = 'approve'

    with col2:
        if st.button("&#x1F4DD; Request Docs", key="request_docs_btn", type="secondary"):
            st.session_state.show_confirm = 'request_docs'

    with col3:
        if st.button("&#x1F50D; Investigate", key="investigate_btn"):
            if invoice['is_ring_member']:
                st.session_state.view_ring_id = invoice['ring_id']
                st.session_state.current_page = "Rings"
                st.rerun()
            else:
                st.info("Invoice is not part of a detected ring.")

    with col4:
        if st.button("&#x26D4; Block", key="block_btn", type="primary"):
            st.session_state.show_confirm = 'block'

    # Confirmation dialogs
    if st.session_state.get('show_confirm') == 'approve':
        render_confirmation_dialog(invoice, 'Approve', 'approved')
    elif st.session_state.get('show_confirm') == 'block':
        render_confirmation_dialog(invoice, 'Block', 'blocked')
    elif st.session_state.get('show_confirm') == 'request_docs':
        render_confirmation_dialog(invoice, 'Request Documents', 'docs_requested')


def render_confirmation_dialog(invoice: dict, action: str, status: str):
    """Render confirmation dialog."""
    st.markdown("---")
    st.markdown(f"### Confirm {action}")

    st.warning(f"You are about to **{action.upper()}** invoice {invoice['invoice_id']} ({format_inr(invoice['amount'])})")

    consequences = {
        'Approve': [
            "Invoice will proceed to normal processing",
            "Risk flag will be cleared",
            "Decision logged for audit"
        ],
        'Block': [
            "Invoice will be prevented from discounting",
            "Seller and buyer flagged for review",
            "Decision logged for audit"
        ],
        'Request Documents': [
            "Request sent to seller for additional documentation",
            "Invoice placed on hold",
            "Decision logged for audit"
        ]
    }

    st.markdown("**This action will:**")
    for consequence in consequences.get(action, []):
        st.markdown(f"- {consequence}")

    reason = st.text_area(
        "Reason (required for Block/Flag):",
        placeholder="Enter justification for this decision...",
        key="decision_reason"
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Cancel", key="cancel_confirm"):
            st.session_state.show_confirm = None
            st.rerun()

    with col2:
        btn_type = "primary" if action == 'Block' else "secondary"
        if st.button(f"Confirm {action}", key="confirm_action", type=btn_type):
            if action == 'Block' and not reason:
                st.error("Reason is required for blocking an invoice.")
            else:
                # Log the decision
                add_decision(
                    invoice_id=invoice['invoice_id'],
                    action=status.upper(),
                    reason=reason or f"{action} - No specific reason provided"
                )

                st.session_state.show_confirm = None
                st.success(f"Invoice {invoice['invoice_id']} {status} successfully!")

                # Return to queue after brief delay
                st.session_state.view_invoice_id = None
                st.rerun()
