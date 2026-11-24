"""
contact_strategy.py
The C4 contact strategy as code: one place that turns an account's situation into a channel, a time window
and a message. daily_call_list.py and the notebook both use it. The evidence for every rule is in
outputs/c4_contact_strategy_rules.csv (built by notebook Step 15 from the c4_* analysis tables).
"""
import numpy as np
import pandas as pd

RISK_BANDS = [(0.0, 0.25, 'Low'), (0.25, 0.50, 'Medium'), (0.50, 1.01, 'High')]   # same cut points as notebook 13.12

WINDOW_EVENING = '17:00-19:59'     # best right-party contact rate in c4_calls_by_hour / c4_calls_by_hour_band
WINDOW_AFTERNOON = '12:00-17:29'
WINDOW_MORNING = '09:00-11:59'

TECHNICAL_FAILURES = {'PROCESSOR_TIMEOUT', 'AUTH_ERROR'}

PLAIN = {   # plain-English labels for the SHAP reasons shown to agents
    'phone_valid': 'phone number on file', 'credit_score': 'credit score at application',
    'sms_failed_before_scoring': 'failed reminder texts', 'n_payment_attempts_before_scoring': 'payment attempts by the due date',
    'n_failed_attempts_before_scoring': 'failed payment attempts', 'last_failure_reason': 'reason the last payment failed',
    'acquisition_channel': 'how the customer joined', 'principal_gbp': 'loan size', 'total_repayable_gbp': 'amount to repay',
    'days_since_payday_at_due': 'days between payday and due date', 'tenure_days_at_disbursement': 'time as a customer',
    'enquiries_6m': 'recent credit applications', 'customer_loan_number': 'number of loans taken',
    'pre_due_sms_delivered': 'pre-due reminder delivered', 'due_day_sms_delivered': 'due-day reminder delivered',
    'repayment_method': 'repayment method', 'employment_status': 'employment status', 'income_band': 'income band',
    'missed_payments_12m': 'missed payments on credit file', 'unsecured_debt_gbp': 'other unsecured debt',
    'active_credit_accounts': 'open credit accounts', 'prior_late_count': 'earlier late loans',
    'prior_late_rate': 'share of earlier loans repaid late', 'prior_default30_count': 'earlier defaults',
    'prior_open_overdue_loans': 'other overdue loans', 'prior_calls': 'earlier collection calls',
    'prior_promises_to_pay': 'earlier promises to pay', 'prior_refusals': 'earlier refusals',
    'prior_wrong_number_calls': 'earlier wrong-number calls', 'share_paid_before_scoring': 'share already paid',
    'cost_of_credit_pct': 'cost of credit', 'term_days': 'loan term', 'device_os': 'phone type',
    'app_opens_14d': 'app opens in last 14 days', 'repayment_screen_views_14d': 'repayment screen views',
    'tap_pay_now_14d': 'pay-now taps', 'payment_errors_14d': 'payment errors in app', 'app_events_14d': 'app activity',
    'sms_link_clicks_14d': 'SMS link clicks', 'debit_failed_notif_opened_14d': 'failed-debit notices opened',
    'sms_invalid_number': 'text bounced as invalid number', 'bureau_available': 'credit file found',
    'bureau_thin_file': 'thin credit file', 'ccj_flag': 'county court judgment', 'days_since_bureau_pull': 'age of credit check',
    'sms_opt_in': 'text opt-in', 'payday_known': 'payday recorded', 'due_day_of_week': 'due day of week',
    'due_on_weekend': 'due at a weekend', 'due_day_of_month': 'due day of month',
}


def risk_band(p):
    for lo, hi, name in RISK_BANDS:
        if lo <= p < hi:
            return name
    return 'High'


def message_for(row):
    """Short message template and tone for an agent call or a digital contact."""
    amount = f"£{row['balance_gbp']:,.2f}"
    reason = row['last_failure_reason']
    if row['broken_promise']:
        return ('Follow-up, calm and direct: "You planned to pay on {d}. It has not reached us yet. '
                'Can we take {a} now, or agree a smaller amount today?"').format(d=row['last_promise_date'], a=amount)
    if row['last_call_outcome'] == 'Spoke - query raised':
        return 'Resolve first: "You raised a question last time. I have the answer for you." Then agree payment of ' + amount + '.'
    if row['days_past_due'] >= 15:
        return ('Clear and supportive: "Your account is {d} days overdue. If we cannot agree a plan by day 30 it will be '
                'treated as in default. Can we agree a payment or a plan today?" Signpost free debt advice.').format(d=row['days_past_due'])
    if reason in TECHNICAL_FAILURES:
        return f'Practical, low pressure: "Your payment of {amount} did not go through because of a technical problem, not anything you did. Can we retry it now?"'
    if reason == 'CARD_EXPIRED':
        return f'Practical: "The card we hold has expired. Can you update it in the app so we can take {amount}?"'
    payday_in_time = pd.notna(row['next_payday']) and row['days_to_next_payday'] <= 30 - row['days_past_due']
    if reason == 'INSUFFICIENT_FUNDS' and payday_in_time:
        return f'Supportive: "We know money can be tight before payday. Can we agree to take {amount} on {row["next_payday"]}?"'
    if reason == 'INSUFFICIENT_FUNDS':
        return f'Supportive: "Could you pay part of {amount} now and agree dates for the rest before day 30?"'
    return f'Friendly reminder: "Your repayment of {amount} was due on {row["due_date"]}. Can you pay today in the app or by card over the phone?"'


def digital_route(row):
    """Channel and send time for customers handled without a call (evidence: c4_nudges_by_phone, c4_sms_delivery)."""
    if row['phone_valid'] == 0:
        return 'In-app push, then email', '10:00', 'Ask the customer to update their phone number and pay in the app.'
    if row['sms_opt_in'] == 1:
        return 'WhatsApp, plus SMS on ROUTE_A only', '10:00', 'Short reminder with the amount and a pay link.'
    return 'WhatsApp, then in-app push', '10:00', 'Short reminder with the amount and a pay link (customer has not opted in to SMS).'


def assign_rule(row, run_date):
    """Return (action, rule_id, why) for one overdue account. First matching rule wins."""
    if row['phone_valid'] == 0:
        return 'Digital only', 'R1', 'Phone number marked invalid: every past call to these customers was a wrong number'
    if row['refused_on_loan']:
        return 'Team leader review', 'R2', 'Customer refused to pay on this loan: no payment followed any refusal'
    if row['active_promise']:
        return 'Promise reminder', 'R3', f"Promise to pay on {row['last_promise_date']} still open: remind on that date, call only if broken"
    if pd.notna(row['days_since_last_call']) and row['days_since_last_call'] < 2:
        return 'Rest day', 'R4', 'Called yesterday: past practice leaves at least 2 days between calls on a loan'
    return 'Call', 'R5', 'Call candidate ranked by expected balance at risk'
