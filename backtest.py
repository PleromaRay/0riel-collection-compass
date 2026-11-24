"""
backtest.py
Replays every working day of the test period (July to early November 2026) and compares the current
alphabetical call order with the new risk-based call list. Used by notebook Step 16.

It uses the model trained only on loans due before 1 July 2026 (models/model_B_day1_evaluated.joblib), so no
test-period outcome is used to rank accounts. It measures REACH (whether an account got onto the list),
not the effect of the call itself; turning reach into recovered money needs the team's own assumptions.
"""
import numpy as np
import pandas as pd


def replay(test, capacity, include_invalid_phones, order, min_gap_days=2, first_day=None, last_day=None):
    """test: Model B test table with risk_score, balance_at_scoring, default_30, phone_valid, names, DUE_DATE, FULLY_REPAID_TS.
    order: 'alphabetical', 'risk', 'risk_x_balance' or 'random'."""
    df = test.copy()
    df['ev'] = df['risk_score'] * df['balance_at_scoring']
    df['rand'] = np.random.default_rng(0).random(len(df))
    df['full_day'] = df['FULLY_REPAID_TS'].dt.normalize()
    sort = {'alphabetical': (['LAST_NAME', 'FIRST_NAME', 'LOAN_ID'], True), 'risk': (['risk_score'], False),
            'risk_x_balance': (['ev'], False), 'random': (['rand'], True)}[order]
    last_call = pd.Series(pd.NaT, index=df.index)
    first_call_dpd = pd.Series(np.nan, index=df.index)
    n_calls = pd.Series(0, index=df.index)
    first_day = first_day or df['DUE_DATE'].min() + pd.Timedelta(days=1)
    last_day = last_day or df['DUE_DATE'].max() + pd.Timedelta(days=30)
    daily_rows = []
    for day in pd.date_range(first_day, last_day):
        if day.dayofweek == 6:              # no Sunday calls in the log
            continue
        dpd = (day - df['DUE_DATE']).dt.days
        pool = dpd.between(1, 30) & ~(df['full_day'] < day)
        rested = (day - last_call).dt.days >= min_gap_days
        pool &= rested | last_call.isna()
        if not include_invalid_phones:
            pool &= df['phone_valid'] == 1
        if not pool.any():
            continue
        todays = df[pool].sort_values(*sort[:1], ascending=sort[1]).head(capacity).index
        last_call[todays] = day
        n_calls[todays] += 1
        valid_today = todays[df.loc[todays, 'phone_valid'] == 1]      # a call to an invalid number reaches nobody
        new = valid_today[first_call_dpd[valid_today].isna()]
        first_call_dpd[new] = dpd[new]
        daily_rows.append({'day': day, 'pool': int(pool.sum()), 'called': len(todays),
                           'defaulting_balance_in_pool': float((df.loc[pool, 'balance_at_scoring'] * df.loc[pool, 'default_30']).sum()),
                           'defaulting_balance_called': float((df.loc[valid_today, 'balance_at_scoring'] * df.loc[valid_today, 'default_30']).sum())})
    d = df['default_30'].astype(bool)   # "reached" below means a call to a valid number
    reached = first_call_dpd.notna()
    daily = pd.DataFrame(daily_rows)
    return {
        'defaults_reached_share': reached[d].mean(),
        'defaulting_balance_reached_share': df.loc[d & reached, 'balance_at_scoring'].sum() / df.loc[d, 'balance_at_scoring'].sum(),
        'defaults_reached_by_day_7_share': (first_call_dpd[d] <= 7).mean(),
        'defaulting_balance_reached_by_day_7_share': df.loc[d & (first_call_dpd <= 7), 'balance_at_scoring'].sum() / df.loc[d, 'balance_at_scoring'].sum(),
        'median_first_call_day_for_defaults': float(first_call_dpd[d & reached].median()),
        'calls_made': int(n_calls.sum()),
        'calls_to_invalid_numbers_share': n_calls[df['phone_valid'] == 0].sum() / max(n_calls.sum(), 1),
        'calls_to_accounts_that_repaid_by_day_30_share': n_calls[~d].sum() / max(n_calls.sum(), 1),
        'daily_share_of_defaulting_balance_on_list': daily['defaulting_balance_called'].sum() / daily['defaulting_balance_in_pool'].sum(),
    }
