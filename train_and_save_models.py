"""
train_and_save_models.py
Rebuilds the Model B (day 1) and Model A (due date) tables, checks the evaluated LightGBM model reproduces the
notebook's test AUC, then refits each model on every loan with a known 30-day outcome and saves it with joblib.

Run from the project folder:   python train_and_save_models.py
Creates: models/model_B_day1.joblib, models/model_A_due_date.joblib, models/model_card.json
"""
import json
import os
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from compass_core import load_clean, build_features, to_lgb_frame, FEATURES, CAT, NUM

SPLIT = pd.Timestamp('2026-07-01')       # same as notebook Step 5
# Settings chosen on the validation slice in notebook Step 6.4 (outputs/summary_numbers.json)
SETTINGS = {'B': {'num_leaves': 7, 'min_child_samples': 100, 'n_estimators': 200, 'learning_rate': 0.03},
            'A': {'num_leaves': 4, 'min_child_samples': 200, 'n_estimators': 300, 'learning_rate': 0.02}}


def make_lgbm(settings, pos_weight):
    return lgb.LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                              scale_pos_weight=pos_weight, random_state=42, verbose=-1, **settings)


def fit_bundle(df, settings, label):
    y = df['default_30'].astype(int).values
    pos_weight = (y == 0).sum() / (y == 1).sum()
    categories = {c: pd.Categorical(df[c]).categories for c in CAT}
    model = make_lgbm(settings, pos_weight).fit(to_lgb_frame(df, categories), y)
    return {'model': model, 'categories': categories, 'pos_weight': float(pos_weight), 'features': FEATURES,
            'settings': settings, 'label': label, 'trained_on_loans': int(len(df)),
            'trained_due_from': str(df['DUE_DATE'].min().date()), 'trained_due_to': str(df['DUE_DATE'].max().date()),
            'train_default_rate': float(y.mean())}


def main():
    D = load_clean('data')
    loans = D['loans']
    cutoff = D['pay']['ts'].max().normalize()                    # 4 November 2026
    loans['outcome_30_known'] = loans['DUE_DATE'] + pd.Timedelta(days=30) <= cutoff
    pop_B = loans[(~loans['on_time']) & loans['outcome_30_known']]
    pop_A = loans[loans['outcome_30_known'] & ~(loans['FULLY_REPAID_TS'] < loans['DUE_DATE'])]
    tables = {'B': build_features(D, pop_B, pop_B['DUE_DATE'].reset_index(drop=True) + pd.Timedelta(days=1)),
              'A': build_features(D, pop_A, pop_A['DUE_DATE'].reset_index(drop=True))}
    for name in tables:
        tables[name] = tables[name].merge(loans[['LOAN_ID', 'default_30']], on='LOAN_ID')

    os.makedirs('models', exist_ok=True)
    card = {'data_cutoff': str(cutoff.date())}
    for name, F in tables.items():
        train, test = F[F['DUE_DATE'] < SPLIT], F[F['DUE_DATE'] >= SPLIT]
        evaluated = fit_bundle(train, SETTINGS[name], f'Model {name} evaluated (due before {SPLIT.date()})')
        auc = roc_auc_score(test['default_30'].astype(int), evaluated['model'].predict_proba(to_lgb_frame(test, evaluated['categories']))[:, 1])
        production = fit_bundle(F, SETTINGS[name], f'Model {name} production refit (all loans with a known outcome)')
        production['evaluated_test_auc'] = float(auc)
        # threshold for 'top 20% risk' (Model A early-warning flag, as in notebook 10.4), from the most recent period
        from compass_core import prior_correct
        recent = to_lgb_frame(test, production['categories'])
        p_recent = prior_correct(production['model'].predict_proba(recent)[:, 1], production['pos_weight'])
        production['top20_threshold'] = float(np.quantile(p_recent, 0.8))
        file_name = {'B': 'model_B_day1.joblib', 'A': 'model_A_due_date.joblib'}[name]
        joblib.dump(production, os.path.join('models', file_name))
        joblib.dump(evaluated, os.path.join('models', file_name.replace('.joblib', '_evaluated.joblib')))
        card[f'model_{name}'] = {k: v for k, v in production.items() if k not in ('model', 'categories', 'features')}
        print(f'Model {name}: test AUC of the evaluated model {auc:.4f} (notebook value should match); '
              f'production refit on {len(F):,} loans saved to models/{file_name}')
    with open(os.path.join('models', 'model_card.json'), 'w') as f:
        json.dump(card, f, indent=1)


if __name__ == '__main__':
    main()
