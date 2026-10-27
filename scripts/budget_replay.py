"""The replay engine with shared budget and filter, and the default sweep.

The engine visits the alarms of one collection day in queue order (or by EDV for a sensitivity), asks admissible
questions (EDV > c from the current probability) chosen by a criterion, spends a shared daily budget, and decides
at the threshold of r. Labels are read only through read_labels and join the traces after the replay.

Selection, answer retrieval and escalation are automatic. The budget counts evidence actions, not analyst
minutes. There is no per-input human review charge, staffing constraint or action-latency simulation; benign
output escalations contribute the false-escalation cost, and questions contribute the separate c*Q term.

The default sweep runs every criterion without a budget limit and records, for each alarm, its rank among the
day's alarms with an admissible question; a finite budget B then questions the first B of them. This prefix rule
is verified against explicit runs at every finite budget.

    python scripts/budget_replay.py

Writes results/by_alarm.parquet, results/replay.parquet, results/replay.csv (extract at r = 10) and
checks/replay_check.json.
"""
import itertools
import json
import math
from collections import defaultdict

import numpy as np
import pandas as pd

import policies as P
from common import (ACTIONS, OUT, QUEUES, ShiftedBelief, folds, load_model, load_partition, now, read_labels, rel,
                    write_json)

SCRIPT = rel(__file__)
TOL = P.TOL
COST_RATIOS = P.COST_RATIOS
SEEDS = P.SEEDS
PRICES = (0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
BUDGETS = (1, 2, 3, 5, 10, None)
ORDERS = tuple(''.join(a[0] for a in perm) for perm in itertools.permutations(ACTIONS))
ORDER_ACTIONS = {o: tuple(next(a for a in ACTIONS if a[0] == ch) for ch in o) for o in ORDERS}
CRITERIA = ('edv', 'ig') + tuple(f'order_{o}' for o in ORDERS) + tuple(f'random_{s}' for s in SEEDS)
SUMMARY = ('edv', 'ig', 'random') + tuple(f'order_{o}' for o in ORDERS)


def blabel(b):
    return 'inf' if b is None else str(b)


class Cell:
    """Model quantities for one fold, smoothing, category, mask and start (label free, cached)."""

    def __init__(self, model, context, available, start):
        self.view = ShiftedBelief(model, context, available, start)
        self.available = tuple(available)
        self._scores = {}

    def posterior(self, history):
        return self.view.posterior(history)

    def scores(self, history, r):
        key = (tuple(sorted(history.items())), r)
        if key not in self._scores:
            q = self.view.posterior(history)
            feasible = [a for a in self.available if a not in history]
            self._scores[key] = (q, {a: P.expected_decision_value(self.view, history, q, a, r) for a in feasible},
                                 {a: P.information_gain(self.view, history, q, a) for a in feasible})
        return self._scores[key]


class Engine:
    def __init__(self, models):
        self.models, self.cells = models, {}

    def cell(self, fold, case, start, tau=None):
        model = self.models[(fold, tau)]
        context = model.context_map.get(case.category, case.category)
        key = (fold, tau, context, case.available, float(start))
        if key not in self.cells:
            self.cells[key] = Cell(model, context, case.available, start)
        return self.cells[key]

    def run_day(self, fold, cases, start, r, c, budget, criterion, seed=None, filtered=True, assignment='queue',
                max_questions=1, tau=None):
        """Automatically triage one day with an action-count budget; return traces in input queue order."""
        cells = [self.cell(fold, case, start, tau) for case in cases]
        order = list(range(len(cases)))
        if assignment == 'edv':
            best = []
            for cell in cells:
                _, edv, _ = cell.scores({}, r)
                adm = [v for v in edv.values() if v > max(c, TOL)]
                best.append(max(adm) if adm else -1.0)
            order.sort(key=lambda i: (-best[i], i))
        remaining = math.inf if budget is None else budget
        traces = [None] * len(cases)
        for i in order:
            case, cell = cases[i], cells[i]
            answers = dict(zip(ACTIONS, case.responses))
            history, asked, asked_edv = {}, [], []
            if criterion == 'all':
                for a in case.available:
                    history[a] = answers[a]
                    asked.append(a)
            elif criterion != 'none':
                while remaining >= 1 and len(asked) < max_questions:
                    q, edv, ig = cell.scores(history, r)
                    feasible = [a for a in ACTIONS if a in edv]
                    if not feasible:
                        break
                    if filtered or criterion == 'edv':
                        adm = [a for a in feasible if edv[a] > max(c if filtered else 0.0, TOL)]
                    else:
                        adm = feasible
                    if not adm:
                        break
                    if criterion == 'edv':
                        a = P.choose({x: edv[x] for x in adm})
                    elif criterion == 'ig':
                        a = P.choose({x: ig[x] for x in adm})
                    elif criterion.startswith('order_'):
                        a = next(x for x in ORDER_ACTIONS[criterion[6:]] if x in adm)
                    elif criterion == 'random':
                        a = next(x for x in P.random_order(case.key, seed, feasible) if x in adm)
                    else:
                        raise ValueError(criterion)
                    asked_edv.append(edv[a])
                    history[a] = answers[a]
                    asked.append(a)
                    remaining -= 1
            final = cell.posterior(history)
            traces[i] = dict(asked=tuple(asked), answers=tuple(history[a] for a in asked), final=final,
                             decision=P.decide(final, r), asked_edv=tuple(asked_edv))
        return traces


class Metrics:
    """Additive outcome counts of one queue and run, with collection qualified attack groups."""
    FIELDS = ('alarms', 'labelled_alarms', 'attack_windows', 'comparison_windows', 'unknown_alarms',
              'attack_windows_escalated', 'false_escalations', 'questions_asked', 'questions_asked_known',
              'escalations', 'unknown_escalations', 'expected_value_known')

    def __init__(self):
        for f in self.FIELDS:
            setattr(self, f, 0)
        self.present, self.reached = set(), set()

    def add(self, dataset, trace, label, groups):
        known = label in (0, 1)
        self.alarms += 1
        self.labelled_alarms += known
        self.attack_windows += label == 1
        self.comparison_windows += label == 0
        self.unknown_alarms += label == -1
        self.escalations += trace['decision']
        self.unknown_escalations += (label == -1) * trace['decision']
        self.attack_windows_escalated += (label == 1) * trace['decision']
        self.false_escalations += (label == 0) * trace['decision']
        self.questions_asked += len(trace['asked'])
        if known:
            self.questions_asked_known += len(trace['asked'])
            self.expected_value_known += sum(trace['asked_edv'])
        if label == 1:
            g = {(dataset, int(x)) for x in groups}
            self.present |= g
            if trace['decision']:
                self.reached |= g

    def combine(self, other):
        for f in self.FIELDS:
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.present |= other.present
        self.reached |= other.reached
        return self


def summarise(units, r):
    """Mean over seeds (one unit for deterministic criteria)."""
    out = {f: sum(getattr(u, f) for u in units) / len(units) for f in Metrics.FIELDS}
    out['attack_windows_missed'] = out['attack_windows'] - out['attack_windows_escalated']
    out['groups_present'] = sum(len(u.present) for u in units) / len(units)
    out['groups_reached'] = sum(len(u.reached) for u in units) / len(units)
    out['loss_total'] = r * out['attack_windows_missed'] + out['false_escalations']
    out['loss_per_100'] = None if not out['labelled_alarms'] else 100 * out['loss_total'] / out['labelled_alarms']
    return out


def load_inputs():
    fold_list = folds()
    models = {(f['id'], None): load_model(f['id']) for f in fold_list}
    membership = pd.read_parquet(OUT / 'results' / 'queue_membership.parquet')
    membership['queue'] = membership['queue'].astype(str)
    starts = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str}, float_precision='round_trip')
    starts = {(row.fold, row.queue): float(row.p_N) for row in starts.itertuples(index=False)}
    days, labels = {}, {}
    for f in fold_list:
        for t in f['target_evaluation']:
            ds = t['dataset']
            cases = {(c.subject_id, c.window_start_ns): c for c in load_partition(ds, 'test')}
            labels[ds] = read_labels(ds, 'test')
            for queue in QUEUES:
                sel = membership.loc[(membership.dataset == ds) & (membership.split == 'test') & (membership.queue == queue)]
                sel = sel.sort_values(['day', 'velox_score', 'subject_id', 'window_start_ns'],
                                      ascending=[True, False, True, True], kind='mergesort')
                days[(f['id'], ds, queue)] = [(day, [cases[(int(s), int(w))] for s, w in zip(g.subject_id, g.window_start_ns)])
                                              for day, g in sel.groupby('day', sort=True)]
    return fold_list, models, starts, days, labels


def run_unit(engine, fold, ds, queue, start, days, labels, r, c, budget, criterion, seed, **kw):
    m = Metrics()
    traces = []
    for day, cases in days:
        for case, trace in zip(cases, engine.run_day(fold, cases, start, r, c, budget, criterion, seed, **kw)):
            label, groups = labels[ds][tuple(case.key)]
            m.add(ds, trace, label, groups)
            traces.append((day, case, trace, label, groups))
    return m, traces


def default_sweep(engine, fold_list, starts, days, labels):
    """Default design (filter, queue order, one question per alarm) at B = infinity for every criterion, with the
    rank of each alarm among the day's alarms with an admissible question, so that every finite budget is the
    prefix rule; the prefix rule is verified against explicit runs at every finite budget."""
    rows, prefix_checked, prefix_failures = [], 0, []
    for f in fold_list:
        for t in f['target_evaluation']:
            ds = t['dataset']
            for queue in QUEUES:
                start, dlist = starts[(f['id'], queue)], days[(f['id'], ds, queue)]
                for r in COST_RATIOS:
                    none = [engine.run_day(f['id'], cases, start, r, 0.0, None, 'none') for _, cases in dlist]
                    allq = [engine.run_day(f['id'], cases, start, r, 0.0, None, 'all') for _, cases in dlist]
                    for c in PRICES:
                        runs = {}
                        for crit in CRITERIA:
                            base, seed = (crit, None) if not crit.startswith('random_') else ('random', int(crit[7:]))
                            runs[crit] = [engine.run_day(f['id'], cases, start, r, c, None, base, seed) for _, cases in dlist]
                            for b in BUDGETS[:-1]:       # prefix rule against explicit finite budgets
                                explicit = [engine.run_day(f['id'], cases, start, r, c, b, base, seed) for _, cases in dlist]
                                for k, (_, cases) in enumerate(dlist):
                                    rank = -1
                                    for j, case in enumerate(cases):
                                        has = bool(runs[crit][k][j]['asked'])
                                        rank += has
                                        expect = runs[crit][k][j] if (has and rank < b) else none[k][j]
                                        prefix_checked += 1
                                        if explicit[k][j]['decision'] != expect['decision'] or explicit[k][j]['asked'] != expect['asked']:
                                            prefix_failures.append(dict(fold=f['id'], dataset=ds, queue=queue, r=r, c=c, B=b, crit=crit))
                        for k, (day, cases) in enumerate(dlist):
                            rank = -1
                            for j, case in enumerate(cases):
                                label, groups = labels[ds][tuple(case.key)]
                                has = bool(runs['edv'][k][j]['asked'])
                                rank += has
                                row = dict(fold=f['id'], dataset=ds, queue=queue, r=r, c=c, day=day, position=j + 1,
                                           subject_id=int(case.subject_id), window_start_ns=int(case.window_start_ns),
                                           category=case.category, mask=','.join(case.available), start=start,
                                           has_admissible=has, adm_rank=rank if has else -1, label=label,
                                           attack_groups=json.dumps(sorted(int(g) for g in groups)),
                                           decision_none=none[k][j]['decision'], final_none=none[k][j]['final'],
                                           decision_all=allq[k][j]['decision'], final_all=allq[k][j]['final'],
                                           questions_all=len(allq[k][j]['asked']))
                                for crit in CRITERIA:
                                    tr = runs[crit][k][j]
                                    if bool(tr['asked']) != has:
                                        raise AssertionError('criteria question different alarms under the default design')
                                    row[f'q_{crit}'] = ','.join(tr['asked'])
                                    row[f'a_{crit}'] = ','.join(tr['answers'])
                                    row[f'p_{crit}'] = tr['final']
                                    row[f'd_{crit}'] = tr['decision']
                                    row[f'e_{crit}'] = tr['asked_edv'][0] if tr['asked_edv'] else np.nan
                                rows.append(row)
            print(f'default sweep {f["id"]}/{ds}: {len(rows)} rows', flush=True)
    return pd.DataFrame(rows), prefix_checked, prefix_failures


def expand(name):
    return [f'random_{s}' for s in SEEDS] if name == 'random' else [name]


def cell_metrics(frame):
    """replay.parquet: one row per fold, collection (and DARPA TC pooled), queue, r, c, B and criterion."""
    f = frame.copy()
    darpa = f.loc[f.fold == 'velox_holdout_DARPA_TC'].copy()
    darpa['group_dataset'] = 'DARPA_TC'
    f['group_dataset'] = f.dataset
    f = pd.concat([f, darpa], ignore_index=True)
    keys = ['fold', 'group_dataset', 'queue', 'r', 'c']
    known = f.label.isin([0, 1]).to_numpy()
    att = (f.label == 1).to_numpy()
    ben = (f.label == 0).to_numpy()
    groups = [set((ds, g) for g in json.loads(s)) if a else set() for ds, s, a in zip(f.dataset, f.attack_groups, att)]
    gkey = list(zip(*[f[k] for k in keys]))
    index = pd.MultiIndex.from_tuples(gkey, names=keys)
    att_rows = np.flatnonzero(att)
    present = defaultdict(set)
    for i in att_rows:
        present[gkey[i]] |= groups[i]
    base = pd.DataFrame(dict(alarms=1, labelled=known.astype(int), attack_windows=att.astype(int),
                             unknown=(~known).astype(int)), index=index).groupby(level=keys).sum()
    out = []

    def decisions(name, b):
        if name in ('none', 'all'):
            return f[f'decision_{name}'].to_numpy(), (np.zeros(len(f)) if name == 'none' else f.questions_all.to_numpy()), np.zeros(len(f))
        asked = f.has_admissible.to_numpy() & ((f.adm_rank < b).to_numpy() if b is not None else True)
        d = np.where(asked, f[f'd_{name}'].to_numpy(), f.decision_none.to_numpy())
        e = np.where(asked, f[f'e_{name}'].fillna(0).to_numpy(), 0.0)
        return d, asked.astype(float), e

    for b in BUDGETS:
        dnone = f.decision_none.to_numpy()
        for crit in SUMMARY + ('none', 'all'):
            per_seed = []
            for name in expand(crit):
                d, q, e = decisions(name, b)
                part = pd.DataFrame(dict(caught=(att & (d == 1)).astype(int), missed=(att & (d == 0)).astype(int),
                                         false=(ben & (d == 1)).astype(int), escalations=d.astype(int),
                                         questions=q, questions_known=q * known, expected_value=e * known,
                                         changed=(d != dnone).astype(int),
                                         differs_edv=(d != np.where(f.has_admissible.to_numpy() & ((f.adm_rank < b).to_numpy() if b is not None else True), f.d_edv.to_numpy(), dnone)).astype(int)),
                                    index=index).groupby(level=keys).sum()
                reached = defaultdict(set)
                for i in att_rows:
                    if d[i] == 1:
                        reached[gkey[i]] |= groups[i]
                part['groups_reached'] = [len(reached[k]) for k in part.index]
                per_seed.append(part)
            m = sum(per_seed) / len(per_seed)
            m = m.join(base)
            m['groups_present'] = [len(present[k]) for k in m.index]
            m['criterion'] = crit
            m['B'] = blabel(b) if crit not in ('none', 'all') else 'any'
            out.append(m.reset_index())
            if crit in ('none', 'all') and b is not None:
                out.pop()        # references do not depend on the budget: keep them once
    res = pd.concat(out, ignore_index=True).rename(columns={'group_dataset': 'dataset'})
    res['D'] = res.r * res.missed + res['false']
    res['L'] = res.D + res.c * res.questions_known
    res['L_per_labelled'] = res.L / res.labelled.where(res.labelled > 0)
    none = res.loc[res.criterion == 'none', ['fold', 'dataset', 'queue', 'r', 'c', 'D']].rename(columns={'D': 'D_none'})
    res = res.merge(none, on=['fold', 'dataset', 'queue', 'r', 'c'], how='left')
    res['value_per_question'] = (res.D_none - res.D) / res.questions_known.where(res.questions_known > 0)
    res['net_value_per_question'] = res.value_per_question - res.c
    res['expected_value_per_question'] = res.expected_value / res.questions_known.where(res.questions_known > 0)
    cols = ['fold', 'dataset', 'queue', 'r', 'c', 'B', 'criterion', 'alarms', 'labelled', 'unknown', 'attack_windows',
            'caught', 'missed', 'false', 'escalations', 'groups_present', 'groups_reached', 'questions',
            'questions_known', 'changed', 'differs_edv', 'D', 'D_none', 'L', 'L_per_labelled', 'value_per_question',
            'net_value_per_question', 'expected_value', 'expected_value_per_question']
    return res[cols]


def main():
    fold_list, models, starts, days, labels = load_inputs()
    engine = Engine(models)
    frame, prefix_checked, prefix_failures = default_sweep(engine, fold_list, starts, days, labels)
    frame.to_parquet(OUT / 'results' / 'by_alarm.parquet', index=False)
    res = cell_metrics(frame)
    res.to_parquet(OUT / 'results' / 'replay.parquet', index=False)
    res.loc[res.r == 10].to_csv(OUT / 'results' / 'replay.csv', index=False, float_format='%.10g')
    # label free consistency with scripts/count.py: same questions and decisions per criterion
    cba = pd.read_parquet(OUT / 'results' / 'count_by_alarm.parquet')
    key = ['fold', 'dataset', 'queue', 'r', 'c', 'subject_id', 'window_start_ns']
    m = frame.merge(cba, on=key, suffixes=('', '_count'))
    mism = sum(int((m[f'q_{c}'] != m[f'q_{c}_count']).sum() + (m[f'd_{c}'] != m[f'd_{c}_count']).sum()) for c in CRITERIA)
    mism += int((m.decision_none != m.decision_none_count).sum() + (m.decision_all != m.decision_all_count).sum())
    out = dict(generated=now(), script=SCRIPT, rows=len(frame), merged_rows=len(m),
               count_by_alarm_mismatches=mism, prefix_rule_checked=prefix_checked,
               prefix_rule_failures=len(prefix_failures), prefix_examples=prefix_failures[:10],
               passed=(mism == 0 and not prefix_failures and len(m) == len(frame) == len(cba)))
    write_json(OUT / 'checks' / 'replay_check.json', out)
    print(json.dumps({k: v for k, v in out.items() if k != 'prefix_examples'}))


if __name__ == '__main__':
    main()
