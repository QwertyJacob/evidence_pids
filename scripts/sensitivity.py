"""Sensitivity analyses of the budgeted replay, one default varied at a time.

Variants: assignment by EDV (alarms of a day in descending order of their largest admissible EDV at the start);
successive questions on the same alarm (up to all available, while the budget lasts and some unasked question is
admissible at the current probability); smoothing strength tau = 1e5 and 1e6 through with_tau (default design).
Every budget is run explicitly. Labels join the traces after the replay, read through read_labels.

    python scripts/sensitivity.py

Writes results/sensitivity.parquet (whole sweep), results/sensitivity.csv (extract at r = 10 and c = 0.01) and
checks/sensitivity_check.json.
"""
import pandas as pd

import budget_replay as B
from common import OUT, now, rel, write_json

SCRIPT = rel(__file__)
VARIANTS = (
    dict(variant='assignment_edv', assignment='edv', max_questions=1, tau=None),
    dict(variant='successive', assignment='queue', max_questions=3, tau=None),
    dict(variant='tau_1e5', assignment='queue', max_questions=1, tau=1e5),
    dict(variant='tau_1e6', assignment='queue', max_questions=1, tau=1e6),
)


def rows_for(engine, fold, dss, darpa, queue, start, days, labels, r, c, b, crit, var):
    seeds = B.SEEDS if crit == 'random' else (None,)
    per = {}
    for ds in dss:
        per[ds] = [B.run_unit(engine, fold, ds, queue, start, days[(fold, ds, queue)], labels, r, c, b, crit, s,
                              assignment=var['assignment'], max_questions=var['max_questions'], tau=var['tau'])[0]
                   for s in seeds]
    units = dict(per)
    if darpa:
        units['DARPA_TC'] = [B.Metrics().combine(per[dss[0]][k]).combine(per[dss[1]][k]).combine(per[dss[2]][k])
                             for k in range(len(seeds))]
    out = []
    for ds, us in units.items():
        m = B.summarise(us, r)
        out.append(dict(variant=var['variant'], fold=fold, dataset=ds, queue=queue, r=r, c=c,
                        B=B.blabel(b) if crit not in ('none', 'all') else 'any', criterion=crit, **m))
    return out


def main():
    fold_list, models, starts, days, labels = B.load_inputs()
    for f in fold_list:
        for tau in (1e5, 1e6):
            models[(f['id'], tau)] = models[(f['id'], None)].with_tau(tau)
    engine = B.Engine(models)
    rows = []
    for var in VARIANTS:
        for f in fold_list:
            dss = [t['dataset'] for t in f['target_evaluation']]
            darpa = f['target_group'] == 'DARPA_TC'
            for queue in B.QUEUES:
                start = starts[(f['id'], queue)]
                for r in B.COST_RATIOS:
                    refs = {crit: rows_for(engine, f['id'], dss, darpa, queue, start, days, labels, r, 0.0, None, crit, var)
                            for crit in ('none', 'all')}
                    for c in B.PRICES:
                        for crit in ('none', 'all'):
                            rows.extend(dict(row, c=c) for row in refs[crit])
                        for b in B.BUDGETS:
                            for crit in B.SUMMARY:
                                base = 'random' if crit == 'random' else crit
                                rows.extend(rows_for(engine, f['id'], dss, darpa, queue, start, days, labels, r, c, b, base, var))
            print(f'{var["variant"]} {f["id"]}: {len(rows)} rows', flush=True)
    res = pd.DataFrame(rows)
    res = res.rename(columns={'labelled_alarms': 'labelled', 'attack_windows_escalated': 'caught',
                              'attack_windows_missed': 'missed', 'false_escalations': 'false'})
    res['D'] = res.r * res.missed + res['false']
    res['L'] = res.D + res.c * res.questions_asked_known
    none = res.loc[res.criterion == 'none', ['variant', 'fold', 'dataset', 'queue', 'r', 'c', 'D']].rename(columns={'D': 'D_none'})
    res = res.merge(none, on=['variant', 'fold', 'dataset', 'queue', 'r', 'c'], how='left')
    qk = res.questions_asked_known.where(res.questions_asked_known > 0)
    res['value_per_question'] = (res.D_none - res.D) / qk
    res['expected_value_per_question'] = res.expected_value_known / qk
    res.to_parquet(OUT / 'results' / 'sensitivity.parquet', index=False)
    res.loc[(res.r == 10) & (res.c == 0.01)].to_csv(OUT / 'results' / 'sensitivity.csv', index=False, float_format='%.10g')
    # Checks: assignment by EDV and successive questions coincide with the default at B = infinity with one question
    # per alarm (assignment) and at B = 1 (successive questions, first question identical); tau changes nothing for
    # the reference "none".
    default = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    key = ['fold', 'dataset', 'queue', 'r', 'c', 'B', 'criterion']
    cols = ['caught', 'false', 'questions_known']
    d = default.set_index(key)[cols]
    s = res.rename(columns={'questions_asked_known': 'questions_known'})

    def compare(variant, budgets):
        part = s.loc[(s.variant == variant) & s.B.isin(budgets) & ~s.criterion.isin(['none', 'all'])].set_index(key)[cols]
        joined = part.join(d, rsuffix='_default', how='inner')
        diff = max(float((joined[c] - joined[f'{c}_default']).abs().max()) for c in cols)
        return dict(rows=len(joined), max_abs_difference=diff, passed=len(joined) == len(part) and diff <= 1e-9)

    none_tau = s.loc[(s.criterion == 'none') & s.variant.str.startswith('tau')].set_index(['variant'] + key)[cols]
    none_def = d.loc[d.index.get_level_values('criterion') == 'none']
    tau_none_diff = max(float((none_tau.xs(v, level='variant')[c] - none_def[c]).abs().max()) for v in ('tau_1e5', 'tau_1e6') for c in cols)
    out = dict(generated=now(), script=SCRIPT, rows=len(res),
               assignment_edv_equals_default_at_B_inf=compare('assignment_edv', ['inf']),
               successive_equals_default_at_B_1=compare('successive', ['1']),
               tau_none_equals_default=dict(max_abs_difference=tau_none_diff, passed=tau_none_diff <= 1e-9))
    out['passed'] = all(v['passed'] for k, v in out.items() if isinstance(v, dict))
    write_json(OUT / 'checks' / 'sensitivity_check.json', out)
    print(out)


if __name__ == '__main__':
    main()
