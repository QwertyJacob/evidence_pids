"""Post processing of the replay: realistic summary per criterion and an independent rebuild of every cell of
results/replay.parquet from results/by_alarm.parquet.

    python scripts/replay_summary.py

Writes results/replay_realistic.csv and adds the rebuild to checks/replay_check.json.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, now, rel, write_json

SCRIPT = rel(__file__)
SEEDS = range(10)
ORDERS = ('fnp', 'fpn', 'nfp', 'npf', 'pfn', 'pnf')
SUMMARY = ('edv', 'ig', 'random') + tuple(f'order_{o}' for o in ORDERS)
BUDGETS = (1, 2, 3, 5, 10, None)
REALISTIC = dict(queue=('1', '3', '10', '30', 'alert'), r=(5, 10, 20), c=(0.001, 0.01, 0.03, 0.1, 0.3))
GROUP_DATASETS = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')


def realistic_summary(res):
    sel = res.loc[res.dataset.isin(GROUP_DATASETS) & res.queue.isin(REALISTIC['queue']) & res.r.isin(REALISTIC['r'])
                  & res.c.isin(REALISTIC['c'])]
    cols = ['alarms', 'labelled', 'attack_windows', 'caught', 'false', 'questions_known', 'changed', 'differs_edv',
            'D', 'groups_present', 'groups_reached']
    sums = sel.groupby(['queue', 'r', 'c', 'B', 'criterion'])[cols].sum().reset_index()
    rows = []
    for (queue, crit), g in sums.groupby(['queue', 'criterion']):
        row = dict(queue=queue, criterion=crit, cells=len(g), alarms=int(g.alarms.iloc[0]),
                   labelled=int(g.labelled.iloc[0]), attack_windows=int(g.attack_windows.iloc[0]),
                   attack_groups=int(g.groups_present.iloc[0]))
        for k in ('caught', 'false', 'questions_known', 'changed', 'differs_edv', 'groups_reached'):
            row[f'{k}_min'], row[f'{k}_max'] = float(g[k].min()), float(g[k].max())
        rows.append(row)
    return pd.DataFrame(rows)


def rebuild(by, res):
    """Independent per cell loop: caught, missed, false, questions on labelled alarms, groups reached."""
    worst, n, failures = 0.0, 0, []
    ref = res.set_index(['fold', 'dataset', 'queue', 'r', 'c', 'B', 'criterion']).sort_index()
    for (fold, ds, queue, r, c), g in by.groupby(['fold', 'dataset', 'queue', 'r', 'c'], sort=False):
        targets = [ds] + (['DARPA_TC'] if fold == 'velox_holdout_DARPA_TC' else [])
        recs = g.to_dict('records')
        for crit in SUMMARY + ('none', 'all'):
            for b in (BUDGETS if crit not in ('none', 'all') else ('any',)):
                names = [f'random_{s}' for s in SEEDS] if crit == 'random' else [crit]
                tot = np.zeros(5)
                for name in names:
                    caught = missed = false = qk = 0
                    reached = set()
                    for x in recs:
                        if crit == 'none':
                            d, q = x['decision_none'], 0
                        elif crit == 'all':
                            d, q = x['decision_all'], x['questions_all']
                        else:
                            asked = x['has_admissible'] and (b is None or x['adm_rank'] < b)
                            d, q = (x[f'd_{name}'], 1) if asked else (x['decision_none'], 0)
                        if x['label'] == 1:
                            caught += d == 1
                            missed += d == 0
                            if d == 1:
                                reached |= {(x['dataset'], gg) for gg in json.loads(x['attack_groups'])}
                        elif x['label'] == 0:
                            false += d == 1
                        if x['label'] in (0, 1):
                            qk += q
                    tot += np.array([caught, missed, false, qk, len(reached)], dtype=float)
                tot /= len(names)
                for t in targets:
                    if t == 'DARPA_TC':
                        continue      # pooled rows are checked as sums below
                    row = ref.loc[(fold, t, queue, r, c, 'any' if b == 'any' else ('inf' if b is None else str(b)), crit)]
                    mine = dict(caught=tot[0], missed=tot[1], false=tot[2], questions_known=tot[3], groups_reached=tot[4])
                    for k, v in mine.items():
                        d_ = abs(v - float(row[k]))
                        worst, n = max(worst, d_), n + 1
                        if d_ > 1e-9:
                            failures.append(dict(fold=fold, dataset=t, queue=queue, r=r, c=c, B=str(b), crit=crit, field=k,
                                                 mine=v, stored=float(row[k])))
    # pooled DARPA TC rows equal the sums of their three collections (groups qualified, so additive)
    d = res.loc[res.fold == 'velox_holdout_DARPA_TC']
    parts = d.loc[d.dataset != 'DARPA_TC'].groupby(['queue', 'r', 'c', 'B', 'criterion'])[
        ['caught', 'missed', 'false', 'questions_known', 'groups_reached', 'alarms', 'labelled']].sum()
    pooled = d.loc[d.dataset == 'DARPA_TC'].set_index(['queue', 'r', 'c', 'B', 'criterion'])[parts.columns]
    pool_diff = float((parts.sort_index() - pooled.sort_index()).abs().max().max())
    return dict(cells_fields_compared=n, max_abs_difference=worst, failures=len(failures), examples=failures[:10],
                pooled_darpa_max_abs_difference=pool_diff,
                passed=(not failures and pool_diff <= 1e-9))


def main():
    by = pd.read_parquet(OUT / 'results' / 'by_alarm.parquet')
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    realistic_summary(res).to_csv(OUT / 'results' / 'replay_realistic.csv', index=False, float_format='%.10g')
    out = rebuild(by, res)
    out.update(generated=now(), script=SCRIPT)
    check = json.loads((OUT / 'checks' / 'replay_check.json').read_text())
    check['rebuild_from_by_alarm'] = out
    check['passed'] = bool(check['passed'] and out['passed'])
    write_json(OUT / 'checks' / 'replay_check.json', check)
    print(json.dumps({k: v for k, v in out.items() if k != 'examples'}))


if __name__ == '__main__':
    main()
