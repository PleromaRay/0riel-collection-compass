"""
daily_call_list.py
Builds the prioritised call list for Oriel Credit's 24 collections agents for one day.

Usage (from the project folder, with the virtual environment active):
    python daily_call_list.py                         # list for the day after the latest data (5 Nov 2026)
    python daily_call_list.py --run-date 2026-10-20   # list for any other day (uses only data from before that day)

What it does
 1. Loads the 14 data files and keeps only what was known before 00:00 on the run date.
 2. Finds every open loan 1 to 30 days past due, plus every loan due on the run date ("due today").
 3. Scores overdue loans with the saved Model B (models/model_B_day1.joblib) and due-today loans with Model A.
 4. Ranks overdue loans by expected balance at risk = probability of default x outstanding balance.
 5. Applies the contact strategy (contact_strategy.py): invalid phone numbers go digital only, refusals go to a team
    leader, open promises to pay get a reminder instead of a call, nobody is called two days running.
 6. Gives each agent up to their daily capacity (90th percentile of calls per agent per day over the last 120 days
    of the call log), putting the highest-value accounts in the 17:00-19:59 window covered by the partner team.
 7. Writes outputs/call_lists/call_list_YYYY-MM-DD.csv, digital_contacts_YYYY-MM-DD.csv and run_summary_YYYY-MM-DD.json.
"""
import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
import warnings
import shap
warnings.filterwarnings('ignore')
from compass_core import load_clean, build_features, score, FEATURES, next_payday
from contact_strategy import (assign_rule, message_for, digital_route, risk_band, PLAIN,
                              WINDOW_EVENING, WINDOW_AFTERNOON, WINDOW_MORNING)

MAX_DPD = 30                 # day 30 is the default point in the agreed target definition (notebook Step 3)
CAPACITY_LOOKBACK_DAYS = 120
CAPACITY_PERCENTILE = 0.90


def latest_data_day(data_dir):
    pay = pd.read_csv(os.path.join(data_dir, 'fact_payments.csv'), usecols=['PAYMENT_DATETIME_UTC'])
    return pd.to_datetime(pay['PAYMENT_DATETIME_UTC']).max().normalize()


def agent_capacity(calls, run_date):
    recent = calls[(calls['ts'] < run_date) & (calls['ts'] >= run_date - pd.Timedelta(days=CAPACITY_LOOKBACK_DAYS))]
    per_agent_day = recent.groupby([recent['ts'].dt.normalize(), 'AGENT_ID']).size()
    return int(np.ceil(per_agent_day.quantile(CAPACITY_PERCENTILE))), per_agent_day


def top_reasons(shap_row, X_row, n=3):
    s = pd.Series(shap_row, index=FEATURES)
    up = s[s > 0].sort_values(ascending=False).head(n)
    parts = []
    for feat, val in up.items():
        v = X_row[feat]
        if isinstance(v, (float, np.floating, int, np.integer)) and not pd.isna(v):
            v = int(round(v)) if float(v).is_integer() or abs(v) >= 10 else round(float(v), 2)
        parts.append(f"{PLAIN.get(feat, feat)} ({v})")
    return '; '.join(parts)


def contact_history(D, loan_ids, run_date):
    c = D['calls'][D['calls']['LOAN_ID'].isin(loan_ids)].sort_values('ts')
    g = c.groupby('LOAN_ID')
    h = pd.DataFrame(index=pd.Index(loan_ids, name='LOAN_ID'))
    h['calls_on_loan'] = g.size().reindex(h.index).fillna(0).astype(int)
    h['last_call_ts'] = g['ts'].max().reindex(h.index)
    h['last_call_outcome'] = g['CALL_OUTCOME'].last().reindex(h.index)
    h['refused_on_loan'] = c[c['CALL_OUTCOME'] == 'Spoke - refused to pay'].groupby('LOAN_ID').size().reindex(h.index).fillna(0) > 0
    ptp = c[c['PROMISE_TO_PAY_DATE'].notna()].groupby('LOAN_ID')['PROMISE_TO_PAY_DATE'].last().reindex(h.index)
    h['last_promise_date'] = ptp.dt.date
    # a promise counts as kept if paid by the promised date plus one day (notebook 13.6), so it is "open" until then
    h['active_promise'] = ptp.notna() & (ptp + pd.Timedelta(days=1) >= run_date)
    h['broken_promise'] = ptp.notna() & (ptp + pd.Timedelta(days=1) < run_date)
    h['days_since_last_call'] = (run_date - h['last_call_ts'].dt.normalize()).dt.days
    return h


def build_lists(run_date, data_dir='data', model_dir='models', calls_per_agent=None):
    run_date = pd.Timestamp(run_date).normalize()
    D = load_clean(data_dir, history_before=run_date)
    loans = D['loans']
    bundle_B = joblib.load(os.path.join(model_dir, 'model_B_day1.joblib'))
    bundle_A = joblib.load(os.path.join(model_dir, 'model_A_due_date.joblib'))

    open_loan = ~(loans['FULLY_REPAID_TS'] < run_date)
    dpd = (run_date - loans['DUE_DATE']).dt.days
    overdue = loans[open_loan & dpd.between(1, MAX_DPD)].copy()
    due_today = loans[open_loan & (dpd == 0)].copy()

    if overdue.empty:
        raise SystemExit(f'No open loans 1 to {MAX_DPD} days past due on {run_date.date()}: nothing to call.')
    # ---- score overdue loans with Model B, at the same point it was trained on (start of day 1)
    F = build_features(D, overdue, overdue['DUE_DATE'].reset_index(drop=True) + pd.Timedelta(days=1))
    F['risk_score'], X = score(bundle_B, F)
    explainer = shap.TreeExplainer(bundle_B['model'])
    sv = explainer.shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    F['top_reasons'] = [top_reasons(sv[i], F.iloc[i]) for i in range(len(F))]

    o = overdue.set_index('LOAN_ID')
    F['days_past_due'] = F['LOAN_ID'].map((run_date - o['DUE_DATE']).dt.days).astype(int)
    F['balance_gbp'] = F['LOAN_ID'].map((o['TOTAL_REPAYABLE_GBP'] - o['paid_to_date']).round(2))
    F = F[F['balance_gbp'] > 0.01].copy()
    F['expected_balance_at_risk_gbp'] = (F['risk_score'] * F['balance_gbp']).round(2)
    F['risk_band'] = F['risk_score'].map(risk_band)
    F['customer_name'] = F['LOAN_ID'].map(o['FIRST_NAME'] + ' ' + o['LAST_NAME'])
    F['due_date'] = F['DUE_DATE'].dt.date
    F['next_payday'] = [next_payday(run_date, p) for p in F['payday_day_of_month']]
    F['next_payday'] = pd.to_datetime(F['next_payday']).dt.date
    F['days_to_next_payday'] = (pd.to_datetime(F['next_payday']) - run_date).dt.days
    h = contact_history(D, F['LOAN_ID'].tolist(), run_date)
    F = F.merge(h, left_on='LOAN_ID', right_index=True, how='left')

    rules = [assign_rule(r, run_date) for _, r in F.iterrows()]
    F['action'] = [r[0] for r in rules]
    F['rule_id'] = [r[1] for r in rules]
    F['rule_reason'] = [r[2] for r in rules]
    F['message'] = [message_for(r) for _, r in F.iterrows()]
    F['payday_note'] = np.where(F['payday_known'] == 1,
                                'Next payday ' + F['next_payday'].astype(str) + ' (' + F['days_to_next_payday'].astype('Int64').astype(str) + ' days)',
                                'No fixed payday recorded')

    # ---- capacity and agents
    cap_auto, per_agent_day = agent_capacity(D['calls'], run_date)
    cap = int(calls_per_agent) if calls_per_agent else cap_auto
    agents = D['agent'].copy()
    agents['team_short'] = np.where(agents['TEAM'] == 'In-house', 'In-house', 'Partner')
    agents['shift'] = (pd.to_datetime(agents['SHIFT_START']).dt.strftime('%H:%M') + '-' + pd.to_datetime(agents['SHIFT_END']).dt.strftime('%H:%M'))
    partner = agents[agents['team_short'] == 'Partner'].reset_index(drop=True)
    inhouse = agents[agents['team_short'] == 'In-house'].reset_index(drop=True)

    cands = F[F['action'] == 'Call'].sort_values('expected_balance_at_risk_gbp', ascending=False).reset_index(drop=True)
    # queries are best handled by the in-house team, so they skip the partner evening slots
    slots = []   # (agent row, window) in priority order, snake-drafted inside each block
    def block(team_df, window, per_agent):
        order = []
        for r in range(per_agent):
            idx = list(range(len(team_df))) if r % 2 == 0 else list(range(len(team_df)))[::-1]
            order += [(team_df.loc[i], window) for i in idx]
        return order
    morning_share = 1 / 3      # in-house: about a third of calls before noon, matching 3,040 of 10,338 in-house calls in the log
    inhouse_morning = int(round(cap * morning_share))
    evening_slots = block(partner, WINDOW_EVENING, cap)
    afternoon_slots = block(inhouse, WINDOW_AFTERNOON, cap - inhouse_morning)
    morning_slots = block(inhouse, WINDOW_MORNING, inhouse_morning)

    assigned = []
    queues = {'evening': evening_slots, 'afternoon': afternoon_slots, 'morning': morning_slots}
    for _, r in cands.iterrows():
        pref = ['afternoon', 'morning'] if r['last_call_outcome'] == 'Spoke - query raised' else ['evening', 'afternoon', 'morning']
        slot = None
        for q in pref:
            if queues[q]:
                slot = queues[q].pop(0)
                break
        if slot is None:
            assigned.append(None)
            continue
        assigned.append(slot)
    cands['on_list'] = [a is not None for a in assigned]
    cands['agent_id'] = [a[0]['AGENT_ID'] if a else None for a in assigned]
    cands['agent_name'] = [a[0]['AGENT_NAME'] if a else None for a in assigned]
    cands['agent_team'] = [a[0]['team_short'] if a else None for a in assigned]
    cands['time_window'] = [a[1] if a else None for a in assigned]

    call_list = cands[cands['on_list']].copy()
    call_list['priority_rank'] = np.arange(1, len(call_list) + 1)
    call_list['recommended_channel'] = 'Phone call (agent)'
    call_list['run_date'] = run_date.date()
    call_list['risk_score'] = call_list['risk_score'].round(3)
    cols = ['run_date', 'priority_rank', 'agent_id', 'agent_name', 'agent_team', 'time_window', 'customer_id', 'customer_name',
            'loan_id', 'due_date', 'days_past_due', 'balance_gbp', 'risk_score', 'risk_band', 'expected_balance_at_risk_gbp',
            'top_reasons', 'recommended_channel', 'message', 'payday_note', 'last_failure_reason', 'calls_on_loan',
            'last_call_outcome', 'employment_status']
    call_list = call_list.rename(columns={'CUSTOMER_ID': 'customer_id', 'LOAN_ID': 'loan_id'})[cols]
    call_list = call_list.sort_values(['agent_id', 'time_window', 'priority_rank'])

    # ---- digital list: everyone not called today
    over_cap = cands[~cands['on_list']].copy(); over_cap['action'] = 'Digital (over call capacity)'
    not_called = pd.concat([F[F['action'] != 'Call'], over_cap], ignore_index=True)
    dig = []
    for _, r in not_called.iterrows():
        ch, when, note = digital_route(r)
        if r['action'] == 'Promise reminder':
            ch, when = ('SMS on ROUTE_A' if r['sms_opt_in'] == 1 and r['phone_valid'] == 1 else 'In-app push'), f"on {r['last_promise_date']} at 09:00"
            note = f"Reminder of the promise to pay {r['balance_gbp']:,.2f} GBP on {r['last_promise_date']}."
        if r['action'] == 'Team leader review':
            ch, when, note = 'Team leader review (no routine call)', 'today', 'Check affordability and hardship options before any further contact.'
        if r['action'] == 'Rest day':
            ch, when, note = 'None today', 'tomorrow', 'Called yesterday; back on the call list tomorrow if still unpaid.'
        dig.append({'run_date': run_date.date(), 'customer_id': r['CUSTOMER_ID'], 'customer_name': r['customer_name'],
                    'loan_id': r['LOAN_ID'], 'days_past_due': r['days_past_due'], 'balance_gbp': r['balance_gbp'],
                    'risk_score': round(r['risk_score'], 3), 'risk_band': r['risk_band'],
                    'expected_balance_at_risk_gbp': r['expected_balance_at_risk_gbp'], 'action': r['action'],
                    'rule_id': r['rule_id'], 'why': r['rule_reason'], 'recommended_channel': ch, 'send_time': when,
                    'message': note, 'top_reasons': r['top_reasons']})
    # ---- due today: Model A early warning, digital only (reminder SMS goes at 08:00 already)
    FA = build_features(D, due_today, due_today['DUE_DATE'].reset_index(drop=True)) if len(due_today) else pd.DataFrame()
    if len(FA):
        FA['risk_score'], _ = score(bundle_A, FA)
        thr = bundle_A.get('top20_threshold')
        dueA = due_today.set_index('LOAN_ID')
        for _, r in FA.sort_values('risk_score', ascending=False).iterrows():
            r = r.copy(); r['customer_name'] = dueA.loc[r['LOAN_ID'], 'FIRST_NAME'] + ' ' + dueA.loc[r['LOAN_ID'], 'LAST_NAME']
            high = thr is not None and r['risk_score'] >= thr
            ch, when, note = digital_route(r)
            if not high:
                ch, when, note = 'Standard due-day SMS on ROUTE_A', '08:00', 'Standard due-day reminder.'
            dig.append({'run_date': run_date.date(), 'customer_id': r['CUSTOMER_ID'], 'customer_name': r['customer_name'],
                        'loan_id': r['LOAN_ID'], 'days_past_due': 0, 'balance_gbp': round(float(dueA.loc[r['LOAN_ID'], 'TOTAL_REPAYABLE_GBP'] - dueA.loc[r['LOAN_ID'], 'paid_to_date']), 2),
                        'risk_score': round(float(r['risk_score']), 3), 'risk_band': 'Due today: top 20% risk' if high else 'Due today',
                        'expected_balance_at_risk_gbp': None, 'action': 'Due today', 'rule_id': 'R6' if high else 'R7',
                        'why': 'Model A early warning (top 20% of due-date risk)' if high else 'Due today, lower early-warning risk',
                        'recommended_channel': ch, 'send_time': when, 'message': note, 'top_reasons': ''})
    digital = pd.DataFrame(dig)

    summary = {
        'run_date': str(run_date.date()), 'data_used_up_to': str((run_date - pd.Timedelta(days=1)).date()),
        'open_loans_1_to_30_days_past_due': int(len(F)), 'due_today': int(len(due_today)),
        'actions': F['action'].value_counts().to_dict(),
        'call_candidates': int(len(cands)), 'calls_planned': int(len(call_list)),
        'calls_per_agent_capacity': cap, 'capacity_rule': f'{int(CAPACITY_PERCENTILE*100)}th percentile of calls per agent per day, last {CAPACITY_LOOKBACK_DAYS} days ({len(per_agent_day)} agent-days)',
        'agents': int(len(agents)), 'balance_on_call_list_gbp': round(float(call_list['balance_gbp'].sum()), 2),
        'expected_balance_at_risk_on_list_gbp': round(float(call_list['expected_balance_at_risk_gbp'].sum()), 2),
        'expected_balance_at_risk_all_open_gbp': round(float(F['expected_balance_at_risk_gbp'].sum()), 2),
        'calls_by_window': call_list['time_window'].value_counts().to_dict(),
        'risk_band_on_list': call_list['risk_band'].value_counts().to_dict(),
    }
    return call_list, digital, summary, F


def main():
    ap = argparse.ArgumentParser(description='Build the daily prioritised call list.')
    ap.add_argument('--run-date', help='YYYY-MM-DD. Default: the day after the latest payment in the data.')
    ap.add_argument('--data', default='data')
    ap.add_argument('--models', default='models')
    ap.add_argument('--out', default=os.path.join('outputs', 'call_lists'))
    ap.add_argument('--calls-per-agent', type=int, help='Override the capacity worked out from the call log.')
    a = ap.parse_args()
    run_date = pd.Timestamp(a.run_date) if a.run_date else latest_data_day(a.data) + pd.Timedelta(days=1)
    call_list, digital, summary, _ = build_lists(run_date, a.data, a.models, a.calls_per_agent)
    os.makedirs(a.out, exist_ok=True)
    d = run_date.date()
    call_list.to_csv(os.path.join(a.out, f'call_list_{d}.csv'), index=False)
    digital.to_csv(os.path.join(a.out, f'digital_contacts_{d}.csv'), index=False)
    with open(os.path.join(a.out, f'run_summary_{d}.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))
    print(f'Saved {a.out}/call_list_{d}.csv ({len(call_list)} calls) and digital_contacts_{d}.csv ({len(digital)} contacts)')


if __name__ == '__main__':
    main()
