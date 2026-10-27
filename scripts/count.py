"""The label free count.

For every tested queue, cost ratio r and price c: which questions are admissible (EDV > c from the alarm's
start), how many alarms ask for a question per day against the budgets B, which question every criterion
chooses, and the decision it then takes on the recorded answers. Answers are observed data, not labels;
no label is read. Also the share of all test windows with an admissible question, each window starting at the
prior of its category (the contrast with ordinary windows).

    python scripts/count.py

Writes results/count_by_alarm.parquet, results/count.csv, results/count_pairs.parquet,
results/count_windows.csv and checks/count_check.json.
"""
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd

import policies as P
from common import ACTIONS, OUT, QUEUES, ShiftedBelief, folds, load_model, load_partition, now, rel, write_json

SCRIPT = rel(__file__)
TOL = P.TOL
COST_RATIOS = P.COST_RATIOS
PRICES = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BUDGETS = (1, 2, 3, 5, 10, None)            # None is no limit
SEEDS = P.SEEDS
ORDERS = tuple(''.join(a[0] for a in perm) for perm in itertools.permutations(ACTIONS))   # fnp, fpn, ...
ORDER_ACTIONS = {name: tuple(next(a for a in ACTIONS if a[0] == ch) for ch in name) for name in ORDERS}
CRITERIA = ('edv', 'ig') + tuple(f'order_{o}' for o in ORDERS) + tuple(f'random_{s}' for s in SEEDS)
SUMMARY = ('edv', 'ig', 'random') + tuple(f'order_{o}' for o in ORDERS)   # random is the mean over seeds


def budget_label(b):
    return 'inf' if b is None else str(b)


class Cell:
    """Model quantities shared by every alarm of one fold, start, category and mask (label free)."""

    def __init__(self, model, context, available, start):
        self.view = ShiftedBelief(model, context, available, start)
        self.start, self.available = float(start), tuple(available)
        self._scores, self._post, self._pdiff = {}, {}, {}

    def scores(self, r):
        if r not in self._scores:
            q = self.view.posterior({})
            self._scores[r] = ({a: P.expected_decision_value(self.view, {}, q, a, r) for a in self.available},
                               {a: P.information_gain(self.view, {}, q, a) for a in self.available})
        return self._scores[r]

    def posterior(self, history):
        key = tuple(sorted(history.items()))
        if key not in self._post:
            self._post[key] = self.view.posterior(history)
        return self._post[key]

    def p_differ(self, a1, a2, r):
        """Model probability that asking a1 or asking a2 leads to different decisions (joint answers)."""
        if a1 == a2:
            return 0.0
        key = (a1, a2, r) if a1 < a2 else (a2, a1, r)
        if key not in self._pdiff:
            total = 0.0
            for e1, p1, q1 in self.view.predict({}, a1):
                if p1 <= 0:
                    continue
                d1 = P.decide(q1, r)
                for e2, p2, _ in self.view.predict({a1: e1}, a2):
                    total += p1 * p2 * (d1 != P.decide(self.posterior({a2: e2}), r))
            self._pdiff[key] = total
        return self._pdiff[key]


def admissible(edv, c):
    return tuple(a for a in ACTIONS if a in edv and edv[a] > max(c, TOL))


def alarm_rows(fold, model, dataset, queue, alarms, cases, start, cells):
    """One row per alarm, r and c: admissible questions, choices and recorded-answer decisions of every criterion."""
    rows = []
    for alarm in alarms.itertuples(index=False):
        case = cases[(int(alarm.subject_id), int(alarm.window_start_ns))]
        if case.category != alarm.category:
            raise AssertionError('Queue category differs from the window table')
        context = model.context_map.get(case.category, case.category)
        key = (context, case.available, start)
        if key not in cells:
            cells[key] = Cell(model, context, case.available, start)
        cell = cells[key]
        answers = dict(zip(ACTIONS, case.responses))
        rand = {s: P.random_order(case.key, s, list(case.available)) for s in SEEDS}
        all_decision = {r: P.decide(cell.posterior({a: answers[a] for a in case.available}), r) for r in COST_RATIOS}
        for r in COST_RATIOS:
            edv, ig = cell.scores(r)
            none_decision = P.decide(cell.start, r)
            for c in PRICES:
                adm = admissible(edv, c)
                row = dict(fold=fold, dataset=dataset, queue=queue, r=r, c=c, day=alarm.day, rank=int(alarm.rank),
                           subject_id=int(alarm.subject_id), window_start_ns=int(alarm.window_start_ns),
                           category=case.category, model_category=context, mask=','.join(case.available),
                           start=start, n_admissible=len(adm), admissible=','.join(adm),
                           decision_none=none_decision, decision_all=all_decision[r],
                           **{f'edv_{a}': edv.get(a) for a in ACTIONS}, **{f'ig_{a}': ig.get(a) for a in ACTIONS})
                if adm:
                    choice = {'edv': P.choose({a: edv[a] for a in adm}), 'ig': P.choose({a: ig[a] for a in adm})}
                    for name in ORDERS:
                        choice[f'order_{name}'] = next(a for a in ORDER_ACTIONS[name] if a in adm)
                    for s in SEEDS:
                        choice[f'random_{s}'] = next(a for a in rand[s] if a in adm)
                    for crit, a in choice.items():
                        row[f'q_{crit}'] = a
                        row[f'd_{crit}'] = P.decide(cell.posterior({a: answers[a]}), r)
                        row[f'edv_asked_{crit}'] = edv[a]
                    # model probability of different decisions for the summary pairs (random: mean over seeds)
                    row['pdiff_edv_ig'] = cell.p_differ(choice['edv'], choice['ig'], r)
                    row['pdiff_edv_random'] = np.mean([cell.p_differ(choice['edv'], choice[f'random_{s}'], r) for s in SEEDS])
                    row['pdiff_ig_random'] = np.mean([cell.p_differ(choice['ig'], choice[f'random_{s}'], r) for s in SEEDS])
                else:
                    for crit in CRITERIA:
                        row[f'q_{crit}'] = ''
                        row[f'd_{crit}'] = none_decision
                        row[f'edv_asked_{crit}'] = np.nan
                    row['pdiff_edv_ig'] = row['pdiff_edv_random'] = row['pdiff_ig_random'] = 0.0
                rows.append(row)
    return rows


def expand(name):
    return [f'random_{s}' for s in SEEDS] if name == 'random' else [name]


def per_row_columns(frame):
    """Per alarm row: pair differences (random averaged over seeds), any difference, decision changed by asking."""
    cols = {}
    q = {c_: frame[f'q_{c_}'].to_numpy() for c_ in CRITERIA}
    d = {c_: frame[f'd_{c_}'].to_numpy() for c_ in CRITERIA}
    for x, y in itertools.combinations(SUMMARY, 2):
        cols[f'dq__{x}__{y}'] = np.mean([q[a] != q[b] for a in expand(x) for b in expand(y)], axis=0)
        cols[f'dd__{x}__{y}'] = np.mean([d[a] != d[b] for a in expand(x) for b in expand(y)], axis=0)
    qs = np.stack([q[c_] for c_ in CRITERIA])
    ds = np.stack([d[c_] for c_ in CRITERIA])
    cols['any_q'] = (qs != qs[0]).any(axis=0).astype(float)
    cols['any_d'] = (ds != ds[0]).any(axis=0).astype(float)
    none = frame.decision_none.to_numpy()
    for crit in SUMMARY:
        cols[f'changed__{crit}'] = np.mean([d[n] != none for n in expand(crit)], axis=0)
    return pd.DataFrame(cols, index=frame.index)


def aggregate(frame):
    """count.csv (per cell and budget) and count_pairs.parquet (every pair of summary criteria)."""
    cell = ['fold', 'dataset', 'queue', 'r', 'c']
    f = frame.copy()
    f['has'] = (f.n_admissible > 0).astype(int)
    # rank among alarms with an admissible question, per collection and day (the unit of the budget)
    f['adm_rank'] = f.groupby(cell + ['day']).has.cumsum() - 1
    extra = per_row_columns(f)
    f = pd.concat([f, extra], axis=1)
    darpa = f.loc[f.fold == 'velox_holdout_DARPA_TC'].copy()
    darpa['dataset'] = 'DARPA_TC'
    darpa['day'] = darpa['day'] + '@' + f.loc[f.fold == 'velox_holdout_DARPA_TC', 'dataset']
    f = pd.concat([f, darpa], ignore_index=True)
    per_day = f.groupby(cell + ['day'], sort=False).agg(adm=('has', 'sum'), adm_q=('n_admissible', 'sum')).reset_index()
    base = f.groupby(cell, sort=False).agg(
        alarms=('has', 'size'), alarms_admissible=('has', 'sum'),
        alarms_adm_1=('n_admissible', lambda s: int((s == 1).sum())),
        alarms_adm_2=('n_admissible', lambda s: int((s == 2).sum())),
        alarms_adm_3=('n_admissible', lambda s: int((s == 3).sum())),
        admissible_questions=('n_admissible', 'sum'))
    days = per_day.groupby(cell, sort=False).agg(days=('adm', 'size'), days_with_admissible=('adm', lambda s: int((s > 0).sum())),
                                                 max_admissible_alarms_day=('adm', 'max'),
                                                 max_admissible_questions_day=('adm_q', 'max'))
    base = base.join(days)
    value_cols = list(extra.columns) + ['pdiff_edv_ig', 'pdiff_edv_random', 'pdiff_ig_random']
    rows, pairs = [], []
    for b in BUDGETS:
        asked = (f.has == 1) if b is None else ((f.has == 1) & (f.adm_rank < b))
        sums = f[value_cols].multiply(asked.astype(float), axis=0).groupby([f[k] for k in cell], sort=False).sum()
        n_asked = asked.groupby([f[k] for k in cell], sort=False).sum().rename('alarms_questioned')
        binds = (per_day.adm > (b if b is not None else np.inf)).groupby([per_day[k] for k in cell], sort=False).sum()
        out = base.join(n_asked).join(binds.rename('days_budget_binds')).join(sums)
        out['B'] = budget_label(b)
        out['alarms_unserved'] = out.alarms_admissible - out.alarms_questioned
        rows.append(out.reset_index())
    full = pd.concat(rows, ignore_index=True)
    count = full[cell + ['B', 'alarms', 'days', 'alarms_admissible', 'alarms_adm_1', 'alarms_adm_2', 'alarms_adm_3',
                         'admissible_questions', 'days_with_admissible', 'max_admissible_alarms_day',
                         'max_admissible_questions_day', 'days_budget_binds', 'alarms_questioned', 'alarms_unserved']].copy()
    for x, y in (('edv', 'ig'), ('edv', 'random'), ('ig', 'random')):
        count[f'diffq_{x}_{y}'] = full[f'dq__{x}__{y}']
        count[f'diffd_{x}_{y}'] = full[f'dd__{x}__{y}']
        count[f'pdiff_{x}_{y}'] = full[f'pdiff_{x}_{y}']
    count['diffq_any'] = full.any_q
    count['diffd_any'] = full.any_d
    for crit in SUMMARY:
        count[f'changed_{crit}'] = full[f'changed__{crit}']
    for x, y in itertools.combinations(SUMMARY, 2):
        part = full[cell + ['B', 'alarms_questioned']].copy()
        part['x'], part['y'] = x, y
        part['diff_question'] = full[f'dq__{x}__{y}']
        part['diff_decision'] = full[f'dd__{x}__{y}']
        pairs.append(part)
    return count, pd.concat(pairs, ignore_index=True)


def windows_count(models):
    """Share of all test windows with an admissible question, each at its category prior (label free)."""
    rows = []
    for fold in folds():
        model = models[fold['id']]
        for target in fold['target_evaluation']:
            dataset = target['dataset']
            cases = load_partition(dataset, 'test')
            counts = defaultdict(int)
            for case in cases:
                if case.warm_up:
                    continue
                counts[(case.category, case.available)] += 1
            n = sum(counts.values())
            for r in COST_RATIOS:
                for c in PRICES:
                    adm = 0
                    for (category, available), k in counts.items():
                        context = model.context_map.get(category, category)
                        prior = model.initial(context)
                        view = ShiftedBelief(model, context, available, prior)
                        q = view.posterior({})
                        edv = {a: P.expected_decision_value(view, {}, q, a, r) for a in available}
                        adm += k * bool(admissible(edv, c))
                    rows.append(dict(fold=fold['id'], dataset=dataset, r=r, c=c, windows=n, windows_admissible=adm,
                                     share=adm / n))
    return pd.DataFrame(rows)


def checks(frame, count, windows):
    out = dict(generated=now(), script=SCRIPT, checks=[])

    def add(name, ok, **detail):
        out['checks'].append(dict(check=name, passed=bool(ok), **detail))

    # Under the default assignment every criterion questions the same alarms: questioned implies n_admissible > 0.
    asked_any = frame[[f'q_{c}' for c in CRITERIA]].ne('').any(axis=1)
    asked_all = frame[[f'q_{c}' for c in CRITERIA]].ne('').all(axis=1)
    add('every criterion asks exactly on the alarms with an admissible question', (asked_any == asked_all).all() and
        (asked_all == (frame.n_admissible > 0)).all())
    chosen_ok = all(all(q in adm.split(',') for q, adm in zip(frame[f'q_{c}'], frame.admissible) if q)
                    for c in CRITERIA)
    add('every chosen question is admissible', chosen_ok)
    sub = frame.loc[frame.q_edv != '']
    edv_best = all(v >= max(getattr(row, f'edv_{a}') for a in row.admissible.split(',')) - TOL
                   for v, row in zip(sub.edv_asked_edv, sub.itertuples(index=False)))
    add('EDV criterion picks a largest EDV among admissible questions', edv_best)
    mono = count.sort_values('c').groupby(['fold', 'dataset', 'queue', 'r', 'B']).alarms_admissible.apply(
        lambda s: (s.diff().dropna() <= 0).all()).all()
    add('admissible alarms never increase with the price', mono)
    wmono = windows.sort_values('c').groupby(['fold', 'dataset', 'r']).windows_admissible.apply(
        lambda s: (s.diff().dropna() <= 0).all()).all()
    add('admissible windows never increase with the price', wmono)
    out['all_passed'] = all(c['passed'] for c in out['checks'])
    write_json(OUT / 'checks' / 'count_check.json', out)
    return out


def main():
    fold_list = folds()
    models = {f['id']: load_model(f['id']) for f in fold_list}
    membership = pd.read_parquet(OUT / 'results' / 'queue_membership.parquet')
    membership['queue'] = membership['queue'].astype(str)
    starts = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str}, float_precision='round_trip')
    starts = {(row.fold, row.queue): float(row.p_N) for row in starts.itertuples(index=False)}
    rows = []
    for fold in fold_list:
        model, cells = models[fold['id']], {}
        for target in fold['target_evaluation']:
            dataset = target['dataset']
            cases = {(c.subject_id, c.window_start_ns): c for c in load_partition(dataset, 'test')}
            for queue in QUEUES:
                alarms = membership.loc[(membership.dataset == dataset) & (membership.split == 'test') &
                                        (membership.queue == queue)]
                alarms = alarms.sort_values(['day', 'velox_score', 'subject_id', 'window_start_ns'],
                                            ascending=[True, False, True, True], kind='mergesort')
                alarms = alarms.assign(rank=alarms.groupby('day').cumcount() + 1)
                rows.extend(alarm_rows(fold['id'], model, dataset, queue, alarms, cases, starts[(fold['id'], queue)], cells))
            print(f'{fold["id"]}/{dataset}: alarm rows so far {len(rows)}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(OUT / 'results' / 'count_by_alarm.parquet', index=False)
    count, pairs = aggregate(frame)
    count.to_csv(OUT / 'results' / 'count.csv', index=False, float_format='%.10g')
    pairs.to_parquet(OUT / 'results' / 'count_pairs.parquet', index=False)
    windows = windows_count(models)
    windows.to_csv(OUT / 'results' / 'count_windows.csv', index=False, float_format='%.10g')
    out = checks(frame, count, windows)
    for c in out['checks']:
        print(('ok    ' if c['passed'] else 'FAILED'), c['check'], {k: v for k, v in c.items() if k not in ('check', 'passed')})
    print('all passed:', out['all_passed'])


if __name__ == '__main__':
    main()
