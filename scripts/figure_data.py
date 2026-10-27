"""Small CSV inputs for the figures (results/fig_*.csv).

Reads results/ (parquet and CSV) and the fitted models. Sums over the three held out groups (DARPA TC pooled,
OpTC host 051, ATLASv2) unless stated.

    python scripts/figure_data.py
"""
import math

import pandas as pd

from common import CATEGORIES, OUT, folds, load_model

ACTIONS3 = ('file', 'network', 'process')
GROUPS = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')
FOLD_OF = {'velox_holdout_DARPA_TC': 'DARPA TC', 'velox_holdout_OpTC': 'OpTC', 'velox_holdout_ATLASv2': 'ATLASv2'}
RS = (5, 10, 20)


def odds(p):
    return math.inf if p >= 1 else p / (1 - p)


def prob(o):
    return 1.0 if math.isinf(o) else o / (1 + o)


def strengths(model, mask=ACTIONS3):
    """Likelihood ratio of every single answer of every question, per category, under the fitted model."""
    out = {}
    for c in CATEGORIES:
        q0 = model.posterior(c, mask, {})
        for a in mask:
            for s, _, q in model.predict(c, mask, {}, a):
                if s in ('empty', 'known', 'novel'):
                    out.setdefault(c, []).append(odds(q) / odds(q0))
    return out


def zone(lrs, r):
    """Starting probabilities [p_low, p_high) from which some answer moves the decision across 1/(1+r)."""
    t = 1 / r
    best, worst = max(lrs), min(lrs)
    lo = t / best if best > 1 else t
    hi = (t / worst if worst > 0 else math.inf) if worst < 1 else t
    return prob(lo), prob(hi)


def f2():
    rows = []
    for f in folds():
        model = load_model(f['id'])
        st = strengths(model)
        st['any'] = [x for c in CATEGORIES for x in st[c]]
        for cat in CATEGORIES + ('any',):
            for r in RS:
                p_low, p_high = zone(st[cat], r)
                prior = model.initial(model.context_map.get(cat, cat)) if cat != 'any' else float('nan')
                rows.append(dict(fold=FOLD_OF[f['id']], category=cat, r=r, p_low=p_low, p_high=p_high, prior=prior))
    pd.DataFrame(rows).to_csv(OUT / 'results' / 'fig_f2_zones.csv', index=False, float_format='%.10g')
    starts = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str})
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    alarms = count.loc[(count.B == 'inf') & (count.r == 10) & (count.c == 0) & count.dataset.isin(GROUPS)]
    s = starts.merge(alarms[['fold', 'queue', 'alarms']], on=['fold', 'queue'])
    s['fold'] = s.fold.map(FOLD_OF)
    s[['fold', 'queue', 'p_N', 'alarms']].to_csv(OUT / 'results' / 'fig_f2_starts.csv', index=False, float_format='%.10g')


def f3():
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    c = count.loc[(count.B == 'inf') & count.dataset.isin(GROUPS)].groupby(['queue', 'r', 'c'], as_index=False)[
        ['alarms', 'alarms_admissible']].sum()
    w = pd.read_csv(OUT / 'results' / 'count_windows.csv').groupby(['r', 'c'], as_index=False)[
        ['windows', 'windows_admissible']].sum().rename(columns={'windows': 'alarms', 'windows_admissible': 'alarms_admissible'})
    w['queue'] = 'windows'
    out = pd.concat([c, w], ignore_index=True)
    out['share'] = out.alarms_admissible / out.alarms
    out.to_csv(OUT / 'results' / 'fig_f3.csv', index=False, float_format='%.10g')


def f4():
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    x = res.loc[res.dataset.isin(GROUPS) & res.r.isin(RS)]
    g = x.groupby(['queue', 'r', 'c', 'B', 'criterion'], as_index=False)[
        ['labelled', 'attack_windows', 'caught', 'false', 'questions_known', 'groups_present', 'groups_reached', 'D']].sum()
    g.to_csv(OUT / 'results' / 'fig_f4.csv', index=False, float_format='%.10g')


def f5():
    v = pd.read_csv(OUT / 'results' / 'value.csv', dtype={'queue': str, 'B': str})
    a = v.loc[(v.B == 'inf') & v.r.isin(RS) & v.criterion.isin(['edv', 'ig', 'random'])].copy()
    w = pd.read_csv(OUT / 'results' / 'value_windows.csv')
    w = w.loc[(w.dataset == 'all_tested') & (w['mode'] == 'filtered') & w.r.isin(RS) & w.criterion.isin(['edv', 'ig', 'random'])].copy()
    w['queue'], w['B'] = 'windows', 'inf'
    cols = ['queue', 'r', 'c', 'criterion', 'questions_known', 'value_per_question']
    pd.concat([a[cols], w[cols]], ignore_index=True).to_csv(OUT / 'results' / 'fig_f5.csv', index=False, float_format='%.10g')


def f6():
    cal = pd.read_csv(OUT / 'results' / 'calibration.csv', dtype={'queue': str, 'B': str})
    x = cal.loc[(cal.B == 'inf') & cal.r.isin(RS) & cal.c.isin([0.001, 0.01, 0.03, 0.1, 0.3]) &
                cal.criterion.isin(['edv', 'ig', 'random']) & cal.queue.isin(['1', '3', '10', '30', 'alert', 'windows'])]
    # the model's 95% predictive interval of the realised value (results/robustness_predictive.csv)
    pred = pd.read_csv(OUT / 'results' / 'robustness_predictive.csv', dtype={'queue': str})
    x = x.merge(pred[['queue', 'r', 'c', 'criterion', 'pred_lo', 'pred_hi', 'outside']], on=['queue', 'r', 'c', 'criterion'],
                how='left')
    x[['unit', 'queue', 'r', 'c', 'criterion', 'questions_known', 'expected_value_per_question', 'value_per_question',
       'pred_lo', 'pred_hi', 'outside']].to_csv(OUT / 'results' / 'fig_f6.csv', index=False, float_format='%.10g')


def main():
    f2(), f3(), f4(), f5(), f6()
    for name in ('fig_f2_zones', 'fig_f2_starts', 'fig_f3', 'fig_f4', 'fig_f5', 'fig_f6'):
        print(name, len(pd.read_csv(OUT / 'results' / f'{name}.csv')))


if __name__ == '__main__':
    main()
