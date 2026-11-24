"""
compass_core.py
Shared code for the Collections Compass model, used by train_and_save_models.py and daily_call_list.py.

It repeats the data preparation (notebook Step 2), outcome rebuild (Step 3) and feature building (Step 4)
from Collections_Compass_Model.ipynb, so the saved model and the daily call list use exactly the same
logic as the notebook. If you change a feature in the notebook, change it here too.
"""
import os
import numpy as np
import pandas as pd

FILES = {
    'loans': 'fact_loans.csv', 'loan_status': 'dim_loan_status.csv', 'payments': 'fact_payments.csv',
    'bureau': 'fact_bureau_pulls.csv', 'app_events': 'fact_app_events.csv', 'sms': 'fact_sms_log.csv',
    'nudges': 'fact_nudges.csv', 'calls': 'fact_collection_calls.csv', 'customer': 'dim_customer.csv',
    'agent': 'dim_agent.csv',
}

NUM = ['principal_gbp', 'total_repayable_gbp', 'cost_of_credit_pct', 'term_days', 'customer_loan_number',
       'due_day_of_week', 'due_on_weekend', 'due_day_of_month',
       'n_payment_attempts_before_scoring', 'n_failed_attempts_before_scoring', 'share_paid_before_scoring',
       'prior_loans_due', 'prior_late_count', 'prior_late_rate', 'prior_default30_count', 'prior_open_overdue_loans',
       'prior_calls', 'prior_promises_to_pay', 'prior_refusals', 'prior_wrong_number_calls',
       'bureau_available', 'bureau_thin_file', 'credit_score', 'ccj_flag', 'enquiries_6m', 'missed_payments_12m',
       'active_credit_accounts', 'unsecured_debt_gbp', 'days_since_bureau_pull',
       'sms_opt_in', 'phone_valid', 'tenure_days_at_disbursement', 'payday_known', 'days_since_payday_at_due',
       'pre_due_sms_delivered', 'due_day_sms_delivered', 'sms_failed_before_scoring', 'sms_invalid_number',
       'app_opens_14d', 'repayment_screen_views_14d', 'tap_pay_now_14d', 'payment_errors_14d',
       'sms_link_clicks_14d', 'debit_failed_notif_opened_14d', 'app_events_14d']
CAT = ['repayment_method', 'last_failure_reason', 'employment_status', 'income_band', 'acquisition_channel', 'device_os']
FEATURES = NUM + CAT

NAME_MAP = {'AppOpen': 'app_open', 'ViewRepayment': 'view_repayment_screen', 'SmsLinkClick': 'sms_link_click',
            'TapPayNow': 'tap_pay_now', 'PaymentConfirmed': 'payment_confirmed', 'NotificationOpened': 'notification_opened',
            'PaymentError': 'payment_error_shown', 'DebitFailedNotifOpened': 'debit_failed_notification_opened'}


def load_clean(data_dir='data', history_before=None):
    """Load the CSV files and apply the Step 2 fixes.
    history_before: if given (a date), every event on or after the start of that date is removed, so the
    result looks exactly like the data the team would have had at 00:00 on that day."""
    t = {k: pd.read_csv(os.path.join(data_dir, f), low_memory=False) for k, f in FILES.items()}
    cut = pd.Timestamp(history_before) if history_before is not None else None

    status_cols = ['LOAN_ID', 'CLOSED_DATE', 'TERM_DAYS', 'LOAN_STATUS', 'REPAYMENT_METHOD', 'CUSTOMER_LOAN_NUMBER']
    loans = t['loans'].merge(t['loan_status'][status_cols], on='LOAN_ID', how='left')        # 2.1
    for c in ['DISBURSEMENT_DATE', 'DUE_DATE', 'CLOSED_DATE']:
        loans[c] = pd.to_datetime(loans[c])
    if cut is not None:
        loans = loans[loans['DISBURSEMENT_DATE'] < cut].copy()           # loans not yet made are unknown

    pay = t['payments'].copy()                                                                   # 2.2, 2.3
    pay['ts'] = pd.to_datetime(pay['PAYMENT_DATETIME_UTC']).dt.tz_localize('UTC').dt.tz_convert('Europe/London').dt.tz_localize(None)
    pay['paid_gbp'] = np.where(pay['PAYMENT_STATUS'] == 'Success', pay['AMOUNT_PAID_GBP'], 0.0)

    ev = t['app_events'].copy()                                                                  # 2.4, 2.5
    ev['ts'] = pd.to_datetime(ev['Event_Date_Local']) + (pd.to_datetime(ev['Event_Time_Local']) - pd.Timestamp('1899-12-30'))
    ev['event'] = ev['EVENT_NAME'].replace(NAME_MAP)

    sms = t['sms'].copy(); sms['ts'] = pd.to_datetime(sms['SEND_DATETIME_LOCAL'])                # 2.6
    nud = t['nudges'].copy(); nud['ts'] = pd.to_datetime(nud['SENT_DATETIME_LOCAL'])
    calls = t['calls'].copy(); calls['ts'] = pd.to_datetime(calls['CALL_DATETIME_LOCAL'])
    calls['PROMISE_TO_PAY_DATE'] = pd.to_datetime(calls['PROMISE_TO_PAY_DATE'])
    calls.loc[calls['CALL_DURATION_SEC'] < 0, 'CALL_DURATION_SEC'] = np.nan                     # 2.7
    bureau = t['bureau'].copy(); bureau['PULL_DATE'] = pd.to_datetime(bureau['PULL_DATE'])
    bureau.loc[bureau['CREDIT_SCORE'] == 0, 'CREDIT_SCORE'] = np.nan                            # 2.8
    cust = t['customer'].copy(); cust['SIGNUP_DATE'] = pd.to_datetime(cust['SIGNUP_DATE'])

    known = loans['LOAN_ID']                                                                     # 2.9
    sms = sms[sms['LOAN_ID'].isin(known)].copy()
    nud = nud[nud['LOAN_ID'].isin(known)].copy()
    calls = calls[calls['LOAN_ID'].isin(known)].copy()
    ev = ev[ev['LOAN_ID'].isna() | ev['LOAN_ID'].isin(known)].copy()
    pay = pay[pay['LOAN_ID'].isin(known)].copy()

    if cut is not None:
        pay = pay[pay['ts'] < cut]; ev = ev[ev['ts'] < cut]; sms = sms[sms['ts'] < cut]
        nud = nud[nud['ts'] < cut]; calls = calls[calls['ts'] < cut]; bureau = bureau[bureau['PULL_DATE'] < cut]

    # Step 3: rebuild the moment each loan was fully repaid from successful payments
    p = pay.merge(loans[['LOAN_ID', 'TOTAL_REPAYABLE_GBP']], on='LOAN_ID').sort_values('ts')
    p['cum_paid'] = p.groupby('LOAN_ID')['paid_gbp'].cumsum()
    full_ts = p[p['cum_paid'] >= p['TOTAL_REPAYABLE_GBP'] - 0.01].groupby('LOAN_ID')['ts'].min().rename('FULLY_REPAID_TS')
    loans = loans.merge(full_ts, on='LOAN_ID', how='left')
    loans['days_to_full'] = (loans['FULLY_REPAID_TS'].dt.normalize() - loans['DUE_DATE']).dt.days
    loans['on_time'] = loans['days_to_full'] <= 0
    loans['default_30'] = ~(loans['days_to_full'] <= 30)
    loans['paid_to_date'] = loans['LOAN_ID'].map(pay.groupby('LOAN_ID')['paid_gbp'].sum()).fillna(0.0)
    loans = loans.merge(cust[['CUSTOMER_ID', 'LAST_NAME', 'FIRST_NAME']], on='CUSTOMER_ID', how='left')
    return {'loans': loans, 'pay': pay, 'ev': ev, 'sms': sms, 'nud': nud, 'calls': calls, 'bureau': bureau,
            'cust': cust, 'agent': t['agent']}


def days_since_payday(due, payday_dom):
    if pd.isna(payday_dom):
        return np.nan
    def payday_in(month_start):
        last_day = (month_start + pd.offsets.MonthEnd(0)).day
        return month_start + pd.Timedelta(days=min(int(payday_dom), last_day) - 1)
    candidate = payday_in(due.replace(day=1))
    if candidate > due:
        previous_month = (due.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        candidate = payday_in(previous_month)
    return (due - candidate).days


def next_payday(day, payday_dom):
    """First payday on or after `day` (payday 31 means the last day of the month)."""
    if pd.isna(payday_dom):
        return pd.NaT
    for k in range(0, 3):
        month_start = (day.replace(day=1) + pd.DateOffset(months=k))
        last_day = (month_start + pd.offsets.MonthEnd(0)).day
        cand = month_start + pd.Timedelta(days=min(int(payday_dom), last_day) - 1)
        if cand >= day:
            return cand
    return pd.NaT


def build_features(D, base, score_ts):
    """Notebook Step 4. `base` is a table of loans, `score_ts` a Series (same index) or one timestamp."""
    loans, pay, calls, bureau, cust, sms, ev = D['loans'], D['pay'], D['calls'], D['bureau'], D['cust'], D['sms'], D['ev']
    base = base.reset_index(drop=True)
    F = base[['LOAN_ID', 'CUSTOMER_ID', 'DISBURSEMENT_DATE', 'DUE_DATE']].copy()
    F['score_ts'] = score_ts.values if isinstance(score_ts, pd.Series) else score_ts
    # 4.2 loan terms
    F['principal_gbp'] = base['PRINCIPAL_AMOUNT_GBP']
    F['total_repayable_gbp'] = base['TOTAL_REPAYABLE_GBP']
    F['cost_of_credit_pct'] = (base['TOTAL_REPAYABLE_GBP'] / base['PRINCIPAL_AMOUNT_GBP'] - 1) * 100
    F['term_days'] = base['TERM_DAYS']
    F['repayment_method'] = base['REPAYMENT_METHOD']
    F['customer_loan_number'] = base['CUSTOMER_LOAN_NUMBER']
    F['due_day_of_week'] = F['DUE_DATE'].dt.dayofweek
    F['due_on_weekend'] = (F['due_day_of_week'] >= 5).astype(int)
    F['due_day_of_month'] = F['DUE_DATE'].dt.day
    # 4.3 payments before scoring
    pp = pay.merge(F[['LOAN_ID', 'score_ts']], on='LOAN_ID')
    pp = pp[pp['ts'] < pp['score_ts']].sort_values('ts')
    failed = pp[pp['PAYMENT_STATUS'] == 'Failed']
    paid = pp.groupby('LOAN_ID')['paid_gbp'].sum()
    F['n_payment_attempts_before_scoring'] = F['LOAN_ID'].map(pp.groupby('LOAN_ID').size()).fillna(0)
    F['n_failed_attempts_before_scoring'] = F['LOAN_ID'].map(failed.groupby('LOAN_ID').size()).fillna(0)
    F['last_failure_reason'] = F['LOAN_ID'].map(failed.groupby('LOAN_ID')['FAILURE_REASON'].last()).fillna('No failed attempt')
    paid_so_far = F['LOAN_ID'].map(paid).fillna(0)
    F['share_paid_before_scoring'] = paid_so_far / F['total_repayable_gbp']
    F['balance_at_scoring'] = (F['total_repayable_gbp'] - paid_so_far).round(2)
    # 4.4 earlier loans
    hist = loans[['LOAN_ID', 'CUSTOMER_ID', 'DUE_DATE', 'FULLY_REPAID_TS']]
    pr = F[['LOAN_ID', 'CUSTOMER_ID', 'score_ts']].merge(hist, on='CUSTOMER_ID', suffixes=('', '_prior'))
    pr = pr[(pr['LOAN_ID_prior'] != pr['LOAN_ID']) & (pr['DUE_DATE'] + pd.Timedelta(days=1) <= pr['score_ts'])].copy()
    pr['late'] = ~(pr['FULLY_REPAID_TS'] < pr['DUE_DATE'] + pd.Timedelta(days=1))
    outcome_known = pr['DUE_DATE'] + pd.Timedelta(days=31) <= pr['score_ts']
    pr['def30'] = outcome_known & ~(pr['FULLY_REPAID_TS'] < pr['DUE_DATE'] + pd.Timedelta(days=31))
    pr['open_overdue'] = ~(pr['FULLY_REPAID_TS'] < pr['score_ts'])
    g = pr.groupby('LOAN_ID')
    F['prior_loans_due'] = F['LOAN_ID'].map(g.size()).fillna(0)
    F['prior_late_count'] = F['LOAN_ID'].map(g['late'].sum()).fillna(0)
    F['prior_late_rate'] = F['prior_late_count'] / F['prior_loans_due'].replace(0, np.nan)
    F['prior_default30_count'] = F['LOAN_ID'].map(g['def30'].sum()).fillna(0)
    F['prior_open_overdue_loans'] = F['LOAN_ID'].map(g['open_overdue'].sum()).fillna(0)
    # 4.5 earlier calls
    pc = calls.merge(F[['LOAN_ID', 'CUSTOMER_ID', 'score_ts']], on='CUSTOMER_ID', suffixes=('_call', ''))
    pc = pc[pc['ts'] < pc['score_ts']]
    F['prior_calls'] = F['LOAN_ID'].map(pc.groupby('LOAN_ID').size()).fillna(0)
    for outcome, col in {'Spoke - promise to pay': 'prior_promises_to_pay', 'Spoke - refused to pay': 'prior_refusals',
                         'Wrong number': 'prior_wrong_number_calls'}.items():
        F[col] = F['LOAN_ID'].map(pc[pc['CALL_OUTCOME'] == outcome].groupby('LOAN_ID').size()).fillna(0)
    # 4.6 bureau
    left = F[['LOAN_ID', 'CUSTOMER_ID', 'DISBURSEMENT_DATE']].sort_values('DISBURSEMENT_DATE')
    x = pd.merge_asof(left, bureau.sort_values('PULL_DATE'), left_on='DISBURSEMENT_DATE', right_on='PULL_DATE',
                      by='CUSTOMER_ID', direction='backward').set_index('LOAN_ID')
    F['bureau_available'] = F['LOAN_ID'].map(x['PULL_DATE'].notna().astype(int))
    F['bureau_thin_file'] = F['LOAN_ID'].map((x['FILE_STATUS'] == 'Thin file').astype(float)).where(F['bureau_available'] == 1)
    for source, col in {'CREDIT_SCORE': 'credit_score', 'CCJ_FLAG': 'ccj_flag', 'ENQUIRIES_6M': 'enquiries_6m',
                        'MISSED_PAYMENTS_12M': 'missed_payments_12m', 'ACTIVE_CREDIT_ACCOUNTS': 'active_credit_accounts',
                        'TOTAL_UNSECURED_DEBT_GBP': 'unsecured_debt_gbp'}.items():
        F[col] = F['LOAN_ID'].map(x[source])
    F['days_since_bureau_pull'] = F['LOAN_ID'].map((x['DISBURSEMENT_DATE'] - x['PULL_DATE']).dt.days)
    # 4.7 customer
    c = cust.set_index('CUSTOMER_ID')
    F['employment_status'] = F['CUSTOMER_ID'].map(c['EMPLOYMENT_STATUS']).fillna('Unknown')
    F['income_band'] = F['CUSTOMER_ID'].map(c['MONTHLY_INCOME_BAND']).fillna('Unknown')
    F['acquisition_channel'] = F['CUSTOMER_ID'].map(c['ACQUISITION_CHANNEL'])
    F['device_os'] = F['CUSTOMER_ID'].map(c['DEVICE_OS'])
    F['sms_opt_in'] = (F['CUSTOMER_ID'].map(c['SMS_OPT_IN']) == 'Y').astype(int)
    F['phone_valid'] = (F['CUSTOMER_ID'].map(c['PHONE_VALID']) == 'Y').astype(int)
    F['tenure_days_at_disbursement'] = (F['DISBURSEMENT_DATE'] - F['CUSTOMER_ID'].map(c['SIGNUP_DATE'])).dt.days
    payday = F['CUSTOMER_ID'].map(c['PAYDAY_DAY_OF_MONTH'])
    F['payday_day_of_month'] = payday
    F['payday_known'] = payday.notna().astype(int)
    F['days_since_payday_at_due'] = [days_since_payday(d, pdom) for d, pdom in zip(F['DUE_DATE'], payday)]
    # 4.8 SMS before scoring
    s = sms.merge(F[['LOAN_ID', 'score_ts']], on='LOAN_ID')
    s = s[s['ts'] < s['score_ts']]
    def delivered(message_type):
        rows = s[s['MESSAGE_TYPE'] == message_type]
        return rows.groupby('LOAN_ID')['DELIVERY_STATUS'].apply(lambda v: int((v == 'DELIVERED').any()))
    F['pre_due_sms_delivered'] = F['LOAN_ID'].map(delivered('PRE_DUE_REMINDER')).fillna(0)
    F['due_day_sms_delivered'] = F['LOAN_ID'].map(delivered('DUE_DAY_REMINDER')).fillna(0)
    F['sms_failed_before_scoring'] = F['LOAN_ID'].map(s[s['DELIVERY_STATUS'] == 'FAILED'].groupby('LOAN_ID').size()).fillna(0)
    F['sms_invalid_number'] = F['LOAN_ID'].map(s[s['FAILURE_CODE'] == 'INVALID_NUMBER'].groupby('LOAN_ID').size()).fillna(0).clip(upper=1)
    # 4.9 app activity in the 14 days before scoring
    a = ev.merge(F[['LOAN_ID', 'CUSTOMER_ID', 'score_ts']], on='CUSTOMER_ID', suffixes=('_ev', ''))
    a = a[(a['ts'] < a['score_ts']) & (a['ts'] >= a['score_ts'] - pd.Timedelta(days=14))]
    for event_name, col in {'app_open': 'app_opens_14d', 'view_repayment_screen': 'repayment_screen_views_14d',
                            'tap_pay_now': 'tap_pay_now_14d', 'payment_error_shown': 'payment_errors_14d',
                            'sms_link_click': 'sms_link_clicks_14d', 'debit_failed_notification_opened': 'debit_failed_notif_opened_14d'}.items():
        F[col] = F['LOAN_ID'].map(a[a['event'] == event_name].groupby('LOAN_ID').size()).fillna(0)
    F['app_events_14d'] = F['LOAN_ID'].map(a.groupby('LOAN_ID').size()).fillna(0)
    return F


def to_lgb_frame(df, categories):
    X = df[FEATURES].copy()
    for col in CAT:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def prior_correct(p, weight):
    odds = p / (1 - p) / weight
    return odds / (1 + odds)


def score(bundle, F):
    """Return the prior-corrected probability of default_30 for each row of F."""
    X = to_lgb_frame(F, bundle['categories'])
    raw = bundle['model'].predict_proba(X)[:, 1]
    return prior_correct(raw, bundle['pos_weight']), X
