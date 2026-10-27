"""The value of a question.

Alarms: value per question v = (D_none - D) / Q and break even price (= v), from results/replay.parquet; the
expected value per question (mean EDV of the questions asked, under the model) against the realised value is the
calibration of the model on alarms.

Ordinary windows: every test window of the tested collections at its category prior, one question per window,
no shared budget, vectorised by category: (i) without filter (IG, random, fixed order ask on every window; EDV
where its EDV exceeds 1e-12); (ii) with the filter at every price, the rule used for alarms. Raw counts per
collection.

    python scripts/value.py

Writes results/value.parquet (per collection and pooled), results/value.csv (sums over the three held out
groups), results/value_windows.csv, results/calibration.csv and checks/value_check.json.
"""
import itertools
import json

import numpy as np
import pandas as pd

import policies as P
from common import ACTIONS, OUT, ShiftedBelief, folds, load_model, load_partition, now, read_labels, rel, write_json

SCRIPT = rel(__file__)
TOL = P.TOL
COST_RATIOS = P.COST_RATIOS
SEEDS = P.SEEDS
PRICES = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
ORDERS = tuple(''.join(a[0] for a in perm) for perm in itertools.permutations(ACTIONS))
ORDER_ACTIONS = {o: tuple(next(a for a in ACTIONS if a[0] == ch) for ch in o) for o in ORDERS}
SUMMARY = ('edv', 'ig', 'random') + tuple(f'order_{o}' for o in ORDERS)
GROUPS = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')


# ---------------------------------------------------------------- alarms

def alarm_values():
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    res = res.loc[~res.criterion.isin(['none'])].copy()
    res['break_even_price'] = res.value_per_question
    keep = ['fold', 'dataset', 'queue', 'r', 'c', 'B', 'criterion', 'labelled', 'attack_windows', 'caught', 'false',
            'questions_known', 'D_none', 'D', 'L', 'value_per_question', 'net_value_per_question', 'break_even_price',
            'expected_value', 'expected_value_per_question']
    res = res[keep]
    res.to_parquet(OUT / 'results' / 'value.parquet', index=False)
    grp = res.loc[res.dataset.isin(GROUPS)].groupby(['queue', 'r', 'c', 'B', 'criterion'])[
        ['labelled', 'attack_windows', 'caught', 'false', 'questions_known', 'D_none', 'D', 'expected_value']].sum().reset_index()
    q = grp.questions_known.where(grp.questions_known > 0)
    grp['L'] = grp.D + grp.c * grp.questions_known
    grp['value_per_question'] = (grp.D_none - grp.D) / q
    grp['net_value_per_question'] = grp.value_per_question - grp.c
    grp['break_even_price'] = grp.value_per_question
    grp['expected_value_per_question'] = grp.expected_value / q
    grp.to_csv(OUT / 'results' / 'value.csv', index=False, float_format='%.10g')
    return res, grp


# ---------------------------------------------------------------- windows

def window_tables(models):
    """Per tested collection: one row per window (category, answers, label, groups, random order ranks)."""
    out = {}
    for f in folds():
        model = models[f['id']]
        for t in f['target_evaluation']:
            ds = t['dataset']
            labels = read_labels(ds, 'test')
            rows = []
            for case in load_partition(ds, 'test'):
                if case.warm_up:
                    continue
                y, g = labels[tuple(case.key)]
                row = dict(fold=f['id'], dataset=ds, category=case.category,
                           context=model.context_map.get(case.category, case.category), mask=','.join(case.available),
                           label=y, groups=json.dumps(sorted(int(x) for x in g)),
                           **{f'ans_{a}': st for a, st in zip(ACTIONS, case.responses)})
                for sd in SEEDS:
                    order = P.random_order(case.key, sd, list(case.available))
                    for k, a in enumerate(ACTIONS):
                        row[f'rank_{sd}_{a}'] = order.index(a) if a in order else 99
                rows.append(row)
            out[ds] = pd.DataFrame(rows)
    return out


def windows_replay(models, tables):
    """Counts per collection, r, c, filter mode and criterion (random: mean over seeds), vectorised by category."""
    rows = []
    for ds, w in tables.items():
        model = models[w.fold.iloc[0]]
        n = len(w)
        att, ben = (w.label == 1).to_numpy(), (w.label == 0).to_numpy()
        known = att | ben
        groups = [set(json.loads(x)) if a else set() for x, a in zip(w.groups, att)]
        present = set().union(*groups) if groups else set()
        att_rows = np.flatnonzero(att)
        ans = {a: w[f'ans_{a}'].to_numpy() for a in ACTIONS}
        ranks = {sd: np.stack([w[f'rank_{sd}_{a}'].to_numpy() for a in ACTIONS], axis=1) for sd in SEEDS}
        cells = {key: idx.to_numpy() for key, idx in w.groupby(['context', 'mask']).groups.items()}
        nq_all = np.array([len(m.split(',')) for m in w['mask']], dtype=float)
        for r in COST_RATIOS:
            d_none = np.zeros(n, dtype=int)
            d_all = np.zeros(n, dtype=int)
            after = np.zeros((n, 3), dtype=int)        # decision after answering each question
            edv_arr = np.full((n, 3), -1.0)
            info = {}
            for key, idx in cells.items():
                context, avail = key[0], tuple(key[1].split(','))
                view = ShiftedBelief(model, context, avail, model.initial(context))
                q = view.posterior({})
                edv = {a: P.expected_decision_value(view, {}, q, a, r) for a in avail}
                ig = {a: P.information_gain(view, {}, q, a) for a in avail}
                info[key] = (edv, ig, avail)
                d_none[idx] = P.decide(q, r)
                for k, a in enumerate(ACTIONS):
                    if a in avail:
                        states = ans[a][idx]
                        lookup = {st: P.decide(view.posterior({a: st}), r) for st in set(states)}
                        after[idx, k] = [lookup[st] for st in states]
                        edv_arr[idx, k] = edv[a]
                combos = pd.Series(list(zip(*[ans[a][idx] for a in avail])))
                lookup = {cmb: P.decide(view.posterior(dict(zip(avail, cmb))), r) for cmb in set(combos)}
                d_all[idx] = [lookup[cmb] for cmb in combos]
            for mode, prices in (('unfiltered', (0.0,)), ('filtered', PRICES)):
                for c in prices:
                    for crit in SUMMARY + ('none', 'all'):
                        names = [f'random_{sd}' for sd in SEEDS] if crit == 'random' else [crit]
                        acc = []
                        for name in names:
                            if crit == 'none':
                                d, nq, ev = d_none, np.zeros(n), np.zeros(n)
                            elif crit == 'all':
                                d, nq, ev = d_all, nq_all, np.zeros(n)
                            else:
                                chosen = np.full(n, -1)
                                for key, idx in cells.items():
                                    edv, ig, avail = info[key]
                                    if mode == 'filtered':
                                        adm = [a for a in ACTIONS if a in edv and edv[a] > max(c, TOL)]
                                    elif crit == 'edv':
                                        adm = [a for a in ACTIONS if a in edv and edv[a] > TOL]
                                    else:
                                        adm = [a for a in ACTIONS if a in avail]
                                    if not adm:
                                        continue
                                    if crit == 'edv':
                                        chosen[idx] = ACTIONS.index(P.choose({a: edv[a] for a in adm}))
                                    elif crit == 'ig':
                                        chosen[idx] = ACTIONS.index(P.choose({a: ig[a] for a in adm}))
                                    elif crit.startswith('order_'):
                                        chosen[idx] = ACTIONS.index(next(a for a in ORDER_ACTIONS[crit[6:]] if a in adm))
                                    else:
                                        rk = ranks[int(name[7:])][idx].copy()
                                        for k, a in enumerate(ACTIONS):
                                            if a not in adm:
                                                rk[:, k] = 999
                                        chosen[idx] = rk.argmin(axis=1)
                                asked = chosen >= 0
                                rows_i = np.flatnonzero(asked)
                                d = d_none.copy()
                                d[rows_i] = after[rows_i, chosen[rows_i]]
                                nq = asked.astype(float)
                                ev = np.zeros(n)
                                ev[rows_i] = edv_arr[rows_i, chosen[rows_i]]
                            reached = set()
                            for i in att_rows:
                                if d[i] == 1:
                                    reached |= groups[i]
                            acc.append(dict(caught=int((att & (d == 1)).sum()), missed=int((att & (d == 0)).sum()),
                                            false=int((ben & (d == 1)).sum()), questions_known=float((nq * known).sum()),
                                            expected_value=float((ev * known).sum()), groups_reached=len(reached)))
                        m = {k: float(np.mean([x[k] for x in acc])) for k in acc[0]}
                        rows.append(dict(dataset=ds, fold=w.fold.iloc[0], r=r, c=c, mode=mode, criterion=crit,
                                         windows=n, labelled=int(known.sum()), attack_windows=int(att.sum()),
                                         groups_present=len(present), **m))
        print(f'windows {ds}: done', flush=True)
    return pd.DataFrame(rows)


def window_values(wr):
    wr = wr.copy()
    wr['D'] = wr.r * wr.missed + wr['false']
    none = wr.loc[wr.criterion == 'none', ['dataset', 'r', 'c', 'mode', 'D']].rename(columns={'D': 'D_none'})
    wr = wr.merge(none, on=['dataset', 'r', 'c', 'mode'])
    total = wr.groupby(['r', 'c', 'mode', 'criterion'], as_index=False)[
        ['windows', 'labelled', 'attack_windows', 'caught', 'missed', 'false', 'questions_known', 'expected_value',
         'D', 'D_none']].sum()
    total['dataset'] = 'all_tested'
    total['fold'] = 'all'
    wr = pd.concat([wr, total], ignore_index=True)
    q = wr.questions_known.where(wr.questions_known > 0)
    wr['L'] = wr.D + wr.c * wr.questions_known
    wr['value_per_question'] = (wr.D_none - wr.D) / q
    wr['net_value_per_question'] = wr.value_per_question - wr.c
    wr['break_even_price'] = wr.value_per_question
    wr['expected_value_per_question'] = wr.expected_value / q
    return wr


def calibration(grp, wr):
    """Expected value per question (model) against realised value per question (labels)."""
    a = grp.loc[~grp.criterion.isin(['all'])].copy()
    a['unit'] = 'alarms'
    w = wr.loc[(wr.dataset == 'all_tested') & (wr['mode'] == 'filtered') & ~wr.criterion.isin(['none', 'all'])].copy()
    w['unit'] = 'windows'
    w['queue'], w['B'] = 'windows', 'inf'
    cols = ['unit', 'queue', 'r', 'c', 'B', 'criterion', 'questions_known', 'D_none', 'D', 'expected_value',
            'expected_value_per_question', 'value_per_question']
    out = pd.concat([a[cols], w[cols]], ignore_index=True)
    out['realised_minus_expected'] = out.value_per_question - out.expected_value_per_question
    out['realised_over_expected'] = out.value_per_question / out.expected_value_per_question.where(out.expected_value_per_question > 0)
    out.to_csv(OUT / 'results' / 'calibration.csv', index=False, float_format='%.10g')
    return out


def main():
    res, grp = alarm_values()
    # automatic check: every value rebuilt from two losses and a count of questions
    recomputed = (res.D_none - res.D) / res.questions_known.where(res.questions_known > 0)
    alarm_diff = float((recomputed - res.value_per_question).abs().max())
    loss_diff = float((res.D - (res.r * (res.attack_windows - res.caught) + res['false'])).abs().max())
    models = {f['id']: load_model(f['id']) for f in folds()}
    wr = windows_replay(models, window_tables(models))
    wv = window_values(wr)
    wv.to_csv(OUT / 'results' / 'value_windows.csv', index=False, float_format='%.10g')
    cal = calibration(grp, wv)
    out = dict(generated=now(), script=SCRIPT,
               alarms=dict(rows=len(res), value_rebuilt_max_abs_difference=alarm_diff,
                           decision_loss_rebuilt_max_abs_difference=loss_diff,
                           passed=alarm_diff <= 1e-12 and loss_diff <= 1e-9),
               calibration_rows=len(cal))
    out['passed'] = out['alarms']['passed']
    write_json(OUT / 'checks' / 'value_check.json', out)
    print(json.dumps(out, default=str)[:1500])


if __name__ == '__main__':
    main()
