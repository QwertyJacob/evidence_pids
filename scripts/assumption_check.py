"""Does an answer keep on alarms the strength it has over all windows of its Velox category?

Checked where labels may be read: on the source collections of each fold (validation and test splits), never on
the fold's tested collections. For every queue built on the sources (the same rule as the tested queues) and for
all `high` windows of the sources (outside the warm up, the set the model's `high` likelihoods were fitted on),
the frequency of each answer of each question among attack windows and among benign windows, with Wilson 95%
intervals, and the resulting likelihood ratio with a conservative interval (ratio of the interval ends). The
fitted model's smoothed single answer likelihood ratio for `high` (three question mask) is given for reference.

The OpTC and ATLASv2 alerts keep their own prior but use the likelihood tables pooled over all categories, so all
source windows outside the warm up (`all_windows`) are added as a second comparator, with the model's ratio for
the category the alerts use (`model_lr_alert`). The alert queue is compared against all windows separately.

    python scripts/assumption_check.py

Writes results/assumption_check.csv and checks/assumption_check.json.
"""
import math

import pandas as pd

from common import ACTIONS, OUT, QUEUES, ShiftedBelief, folds, load_model, load_partition, now, read_labels, rel, write_json

SCRIPT = rel(__file__)
Z = 1.959963984540054
STATES = ('empty', 'known', 'novel')


def wilson(k, n):
    if n == 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    den = n + Z * Z
    centre = (k + Z * Z / 2) / den
    half = Z * math.sqrt(k * (n - k) / n + Z * Z / 4) / den
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def ratio(a, b):
    if b == 0:
        return math.inf if a > 0 else math.nan
    return a / b


def model_lr(model, question, state, category='high'):
    context = model.context_map.get(category, category)
    view = ShiftedBelief(model, context, ACTIONS, 0.5)
    table = view.table
    if state not in table.index[question]:
        return math.nan
    sets = {question: [table.index[question][state]]}
    l1, l0 = model._event_likelihoods(table, view.stratum, sets)[:2]
    return l1 / l0


def main():
    fold_list = folds()
    membership = pd.read_parquet(OUT / 'results' / 'queue_membership.parquet')
    membership['queue'] = membership['queue'].astype(str)
    needed = sorted({(e['dataset'], s) for f in fold_list for e in f['fitting'] for s in e['splits']})
    labels, cases = {}, {}
    for ds, split in needed:
        labels[(ds, split)] = read_labels(ds, split)
        cases[(ds, split)] = {(c.subject_id, c.window_start_ns): c for c in load_partition(ds, split)}
    rows = []
    for f in fold_list:
        model = load_model(f['id'])
        sources = [(e['dataset'], s) for e in f['fitting'] for s in e['splits']]
        sets = {}
        for queue in QUEUES:
            items = []
            for ds, split in sources:
                sel = membership.loc[(membership.dataset == ds) & (membership.split == split) & (membership.queue == queue)]
                items += [(ds, split, cases[(ds, split)][(int(s), int(w))]) for s, w in zip(sel.subject_id, sel.window_start_ns)]
            sets[queue] = items
        sets['all_high'] = [(ds, split, c) for ds, split in sources for c in cases[(ds, split)].values()
                            if c.category == 'high' and not c.warm_up]
        sets['all_windows'] = [(ds, split, c) for ds, split in sources for c in cases[(ds, split)].values()
                               if not c.warm_up]
        for name, items in sets.items():
            for qi, question in enumerate(ACTIONS):
                tally = {(y, st): 0 for y in (0, 1) for st in STATES}
                n = {0: 0, 1: 0}
                for ds, split, case in items:
                    y, _ = labels[(ds, split)][(ds, split, int(case.subject_id), int(case.window_start_ns))]
                    state = case.responses[qi]
                    if y not in (0, 1) or state == 'unavailable':
                        continue
                    n[y] += 1
                    tally[(y, state)] += 1
                for st in STATES:
                    f1, lo1, hi1 = wilson(tally[(1, st)], n[1])
                    f0, lo0, hi0 = wilson(tally[(0, st)], n[0])
                    rows.append(dict(fold=f['id'], set=name, question=question, answer=st,
                                     attack_windows=n[1], attack_with_answer=tally[(1, st)], freq_attack=f1,
                                     freq_attack_lo=lo1, freq_attack_hi=hi1,
                                     benign_windows=n[0], benign_with_answer=tally[(0, st)], freq_benign=f0,
                                     freq_benign_lo=lo0, freq_benign_hi=hi0,
                                     lr=ratio(f1, f0), lr_lo=ratio(lo1, hi0), lr_hi=ratio(hi1, lo0),
                                     model_lr_high=model_lr(model, question, st),
                                     model_lr_alert=model_lr(model, question, st, 'alert')))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'results' / 'assumption_check.csv', index=False, float_format='%.6g')
    # Comparison of each queue with all high windows: do the Wilson intervals of the answer frequencies overlap?
    comps = compare(res.loc[~res.set.isin(['all_high', 'all_windows'])], res.loc[res.set == 'all_high'])
    comps_all = compare(res.loc[res.set == 'alert'], res.loc[res.set == 'all_windows'])
    summary = comps.groupby(['set', 'cls', 'status']).size().unstack(fill_value=0).reset_index()
    summary_all = comps_all.groupby(['fold', 'cls', 'status']).size().unstack(fill_value=0).reset_index()
    attack_n = res.loc[~res.set.isin(['all_high', 'all_windows']) & (res.answer == 'empty')].groupby(['fold', 'set']).attack_windows.max()
    out = dict(generated=now(), script=SCRIPT, rows=len(res),
               comparisons=summary.to_dict('records'),
               disjoint=comps.loc[comps.status == 'intervals disjoint'].to_dict('records'),
               alert_against_all_windows=summary_all.to_dict('records'),
               attack_windows_in_source_alarms=attack_n.reset_index().to_dict('records'),
               note='Exploratory; Wilson 95% intervals; few attack windows in source alarms make intervals wide.')
    write_json(OUT / 'checks' / 'assumption_check.json', out)
    print(summary.to_string())
    print(summary_all.to_string())
    print(attack_n.to_string())


def compare(res, reference):
    """Do the Wilson intervals of each set's answer frequencies overlap those of the reference set?"""
    hi = reference.set_index(['fold', 'question', 'answer'])
    comps = []
    for row in res.itertuples(index=False):
        ref = hi.loc[(row.fold, row.question, row.answer)]
        for cls, lo, up, rlo, rup, n in (('attack', row.freq_attack_lo, row.freq_attack_hi, ref.freq_attack_lo,
                                          ref.freq_attack_hi, row.attack_windows),
                                         ('benign', row.freq_benign_lo, row.freq_benign_hi, ref.freq_benign_lo,
                                          ref.freq_benign_hi, row.benign_windows)):
            if n == 0:
                status = 'no windows'
            elif lo > rup or up < rlo:
                status = 'intervals disjoint'
            else:
                status = 'intervals overlap'
            comps.append(dict(fold=row.fold, set=row.set, question=row.question, answer=row.answer, cls=cls,
                              windows=n, status=status))
    return pd.DataFrame(comps)


if __name__ == '__main__':
    main()
