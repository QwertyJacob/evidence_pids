"""Uncertainty and robustness analyses on the existing data and models.

  1. bootstrap over collection days of the value per question;
  2. sensitivity of the starts: pseudo count 0.5 instead of 1, and source test splits only;
  3. predictive intervals of the realised value per question under the model;
  4. smoothing strength tau = 1e2 and 1e3;
  5. fitted single answer likelihood ratios and the tables each category uses;
  6. results per tested group and collection, attacks reached, spread over random seeds, six fixed orders.

Default design (filter, queue order, one question per alarm), no budget limit, sums over the three tested groups.

    python scripts/robustness.py

Writes results/robustness_bootstrap.csv, robustness_sensitivity.csv, robustness_starts.csv,
robustness_predictive.csv, robustness_fitted_lr.csv, robustness_per_group.csv and robustness_random_seeds.csv.
"""
import json

import numpy as np
import pandas as pd

import budget_replay as B
from common import OUT, ShiftedBelief, folds, now, read_labels, rel, write_json

SCRIPT = rel(__file__)
G = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')
QUEUES = ('1', '3', '10', '30', 'alert')
RS = (5, 10, 20)
CS = (0.001, 0.01, 0.03, 0.1, 0.3)
CRITS = ('edv', 'ig', 'random')
RNG = np.random.default_rng(20261024)
P = B.P


def value(dn, d, q):
    return (dn - d) / q if q > 0 else np.nan


# ---------------------------------------------------------------- 1. bootstrap over collection days

def bootstrap(by, n_boot=2000):
    rows = []
    for (queue, r, c), g in by.loc[by.queue.isin(QUEUES) & by.r.isin(RS) & by.c.isin(CS)].groupby(['queue', 'r', 'c']):
        g = g.loc[g.label.isin([0, 1])]
        att, ben = (g.label == 1).to_numpy(), (g.label == 0).to_numpy()
        asked = g.has_admissible.to_numpy()
        dn = g.decision_none.to_numpy()
        clusters = (g.dataset + '|' + g.day).to_numpy()
        ids, inv = np.unique(clusters, return_inverse=True)
        k = len(ids)
        loss_none = np.bincount(inv, weights=r * (att & (dn == 0)) + (ben & (dn == 1)), minlength=k)
        q = np.bincount(inv, weights=asked.astype(float), minlength=k)
        for crit in CRITS:
            names = [f'random_{s}' for s in range(10)] if crit == 'random' else [crit]
            d = np.mean([np.where(asked, g[f'd_{n}'].to_numpy(), dn) for n in names], axis=0)
            loss = np.bincount(inv, weights=r * (att * (1 - d)) + ben * d, minlength=k)
            point = value(loss_none.sum(), loss.sum(), q.sum())
            draws = RNG.integers(0, k, size=(n_boot, k))
            ls, ln, qq = loss[draws].sum(1), loss_none[draws].sum(1), q[draws].sum(1)
            with np.errstate(invalid='ignore', divide='ignore'):
                vb = np.where(qq > 0, (ln - ls) / qq, np.nan)
            lo, hi = np.nanpercentile(vb, [2.5, 97.5]) if np.isfinite(vb).any() else (np.nan, np.nan)
            rows.append(dict(queue=queue, r=r, c=c, criterion=crit, clusters=k, questions=q.sum(), value=point,
                             boot_lo=lo, boot_hi=hi, attack_windows=int(att.sum()),
                             attack_groups=len({(ds, x) for ds, s, a in zip(g.dataset, g.attack_groups, att) if a
                                                for x in json.loads(s)})))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. sensitivity of the starts

def alternative_starts(membership):
    """Starts with pseudo count 0.5 (from the stored counts) and with source test splits only (labels read)."""
    st = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str})
    jeff = {(row.fold, row.queue): (row.source_attack_windows + 0.5) / (row.source_labelled + 1) for row in st.itertuples()}
    test_only = {}
    labels = {}
    for f in folds():
        for e in f['fitting']:
            ds = e['dataset']
            if ds not in labels:
                labels[ds] = read_labels(ds, 'test')
    for f in folds():
        for queue in B.QUEUES:
            a = n = 0
            for e in f['fitting']:
                ds = e['dataset']
                sel = membership.loc[(membership.dataset == ds) & (membership.split == 'test') & (membership.queue == queue)]
                for s, w in zip(sel.subject_id, sel.window_start_ns):
                    y, _ = labels[ds][(ds, 'test', int(s), int(w))]
                    if y in (0, 1):
                        n += 1
                        a += y == 1
            test_only[(f['id'], queue)] = (a + 1) / (n + 2)
    return {'pseudo_count_0.5': jeff, 'source_test_only': test_only}


def replay_default(engine, fold_list, starts, days, labels, rs=RS, cs=CS, tau=None, queues=QUEUES):
    rows = []
    for f in fold_list:
        dss = [t['dataset'] for t in f['target_evaluation']]
        for queue in queues:
            start = starts[(f['id'], queue)]
            for r in rs:
                for c in cs:
                    for crit in ('none',) + CRITS:
                        seeds = range(10) if crit == 'random' else (None,)
                        for ds in dss:
                            units = [B.run_unit(engine, f['id'], ds, queue, start, days[(f['id'], ds, queue)], labels, r, c,
                                                None, crit, s, tau=tau)[0] for s in seeds]
                            m = B.summarise(units, r)
                            rows.append(dict(fold=f['id'], dataset=ds, queue=queue, r=r, c=c, criterion=crit, start=start,
                                             D=m['loss_total'], Q=m['questions_asked_known'], caught=m['attack_windows_escalated'],
                                             false=m['false_escalations'], expected=m['expected_value_known']))
    res = pd.DataFrame(rows)
    s = res.groupby(['queue', 'r', 'c', 'criterion'], as_index=False)[['D', 'Q', 'caught', 'false', 'expected']].sum()
    none = s.loc[s.criterion == 'none', ['queue', 'r', 'c', 'D']].rename(columns={'D': 'D_none'})
    s = s.merge(none, on=['queue', 'r', 'c'])
    s['value'] = (s.D_none - s.D) / s.Q.where(s.Q > 0)
    return s


# ---------------------------------------------------------------- 3. predictive intervals under the model

def joint_tables(model, ctx, avail):
    """P(full answer combination | Y) for y = 1 and 0, from the joint likelihoods (sequential predictive)."""
    out = {}
    for y in (1, 0):
        view = ShiftedBelief(model, ctx, avail, float(y))
        combos = [({}, 1.0)]
        for a in avail:
            nxt = []
            for hist, pr in combos:
                for sym, p_, _ in view.predict(hist, a):
                    if p_ > 0:
                        nxt.append(({**hist, a: sym}, pr * p_))
            combos = nxt
        tot = sum(p_ for _, p_ in combos)
        out[y] = ([h for h, _ in combos], np.array([p_ / tot for _, p_ in combos]))
    return out


def predictive(engine, fold_list, starts, days, by, n_draw=2000):
    """Labels and all answers drawn from the model (start and joint likelihoods of the alarm's category); the
    question each criterion asks is fixed by the design; realised value per question compared with the draws.
    The same draws (common random numbers) serve every cost ratio and price."""
    cases = {}
    for f in fold_list:
        for t in f['target_evaluation']:
            for queue in QUEUES:
                for day, cs_ in days[(f['id'], t['dataset'], queue)]:
                    for case in cs_:
                        cases[(t['dataset'], queue, case.subject_id, case.window_start_ns)] = (f['id'], case)
    sub = by.loc[by.queue.isin(QUEUES) & by.r.isin(RS) & by.c.isin(CS) & by.label.isin([0, 1])]
    tables, draws = {}, {}
    for key in sub[['dataset', 'queue', 'subject_id', 'window_start_ns']].drop_duplicates().itertuples(index=False):
        fold, case = cases[tuple(key)]
        model = engine.models[(fold, None)]
        ctx = model.context_map.get(case.category, case.category)
        tk = (fold, ctx, case.available)
        if tk not in tables:
            tables[tk] = joint_tables(model, ctx, case.available)
        start = starts[(fold, key.queue)]
        y = RNG.random(n_draw) < start
        ans = {}
        idx1 = RNG.choice(len(tables[tk][1][1]), size=n_draw, p=tables[tk][1][1])
        idx0 = RNG.choice(len(tables[tk][0][1]), size=n_draw, p=tables[tk][0][1])
        for a in case.available:
            s1 = np.array([h[a] for h in tables[tk][1][0]], dtype=object)[idx1]
            s0 = np.array([h[a] for h in tables[tk][0][0]], dtype=object)[idx0]
            ans[a] = np.where(y, s1, s0)
        draws[tuple(key)] = (y, ans, fold, case, start)
    rows = []
    for (queue, r, c), g in sub.groupby(['queue', 'r', 'c']):
        loss = {k: np.zeros(n_draw) for k in ('none',) + CRITS}
        real = {k: 0.0 for k in ('none',) + CRITS}
        q_real = 0.0
        for row in g.itertuples(index=False):
            y, ans, fold, case, start = draws[(row.dataset, queue, row.subject_id, row.window_start_ns)]
            cell = engine.cell(fold, case, start)
            dn = P.decide(start, r)
            lf = lambda d: np.where(y, r * (1 - d), d).astype(float)
            loss['none'] += lf(np.full(n_draw, float(dn)))
            real['none'] += (r * (1 - dn)) if row.label == 1 else dn
            asked = bool(row.has_admissible)
            q_real += asked
            for crit in CRITS:
                names = [f'random_{s}' for s in range(10)] if crit == 'random' else [crit]
                dsum = np.zeros(n_draw)
                for name in names:
                    qa = getattr(row, f'q_{name}')
                    if asked and qa:
                        lut = {sym: P.decide(cell.posterior({qa: sym}), r) for sym in set(ans[qa])}
                        dsum += np.array([lut[sym] for sym in ans[qa]], dtype=float)
                    else:
                        dsum += dn
                loss[crit] += lf(dsum / len(names))
                dr = np.mean([getattr(row, f'd_{name}') if asked else dn for name in names])
                real[crit] += r * (1 - dr) if row.label == 1 else dr
        for crit in CRITS:
            vr = value(real['none'], real[crit], q_real)
            if q_real > 0:
                vd = (loss['none'] - loss[crit]) / q_real
                lo, hi = np.percentile(vd, [2.5, 97.5])
                rows.append(dict(queue=queue, r=r, c=c, criterion=crit, questions=q_real, realised=vr,
                                 pred_mean=float(vd.mean()), pred_lo=lo, pred_hi=hi, outside=bool(vr < lo or vr > hi),
                                 p_lower=float((vd <= vr + 1e-12).mean())))
            else:
                rows.append(dict(queue=queue, r=r, c=c, criterion=crit, questions=0, realised=np.nan, pred_mean=np.nan,
                                 pred_lo=np.nan, pred_hi=np.nan, outside=False, p_lower=np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. fitted likelihood ratios

def fitted_lrs(models):
    rows = []
    for (fold, tau), model in models.items():
        if tau is not None:
            continue
        for mask in ('file,network,process', 'file,network'):
            avail = tuple(mask.split(','))
            for cat in ('unscored', 'low', 'high', 'alert'):
                ctx = model.context_map.get(cat, cat)
                view = ShiftedBelief(model, ctx, avail, 0.5)
                table, stratum = view.table, view.stratum
                for a in avail:
                    for s in ('empty', 'known', 'novel'):
                        if s not in table.index[a]:
                            continue
                        l1, l0 = model._event_likelihoods(table, stratum, {a: [table.index[a][s]]})[:2]
                        rows.append(dict(fold=fold, mask=mask, category=cat, model_category=ctx, prior=model.initial(ctx),
                                         likelihood_table=stratum, question=a, answer=s, lr=l1 / l0))
    return pd.DataFrame(rows)


def main():
    by = pd.read_parquet(OUT / 'results' / 'by_alarm.parquet')
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    bootstrap(by).to_csv(OUT / 'results' / 'robustness_bootstrap.csv', index=False, float_format='%.6g')
    print('bootstrap done', flush=True)
    fold_list, models, starts, days, labels = B.load_inputs()
    for f in fold_list:
        for tau in (1e2, 1e3):
            models[(f['id'], tau)] = models[(f['id'], None)].with_tau(tau)
    engine = B.Engine(models)
    fitted_lrs(models).to_csv(OUT / 'results' / 'robustness_fitted_lr.csv', index=False, float_format='%.6g')
    membership = pd.read_parquet(OUT / 'results' / 'queue_membership.parquet')
    membership['queue'] = membership['queue'].astype(str)
    alt = alternative_starts(membership)
    out = []
    for name, st in alt.items():
        s = replay_default(engine, fold_list, st, days, labels)
        s['variant'] = name
        out.append(s)
        print('starts', name, 'done', flush=True)
    base = replay_default(engine, fold_list, starts, days, labels)
    base['variant'] = 'default'
    out.append(base)
    for tau in (1e2, 1e3):
        s = replay_default(engine, fold_list, starts, days, labels, tau=tau)
        s['variant'] = f'tau_{tau:.0e}'
        out.append(s)
        print('tau', tau, 'done', flush=True)
    pd.concat(out, ignore_index=True).to_csv(OUT / 'results' / 'robustness_sensitivity.csv', index=False, float_format='%.6g')
    pd.DataFrame([dict(variant=k, fold=f, queue=q, start=v) for k, st in list(alt.items()) + [('default', starts)]
                  for (f, q), v in st.items()]).to_csv(OUT / 'results' / 'robustness_starts.csv', index=False, float_format='%.6g')
    predictive(engine, fold_list, starts, days, by).to_csv(OUT / 'results' / 'robustness_predictive.csv', index=False, float_format='%.6g')
    # 6, 7: tables from stored results
    x = res.loc[(res.r == 10) & (res.c == 0.01) & res.B.isin(['inf', 'any'])]
    per = x.loc[x.dataset.isin(G) | (x.fold == 'velox_holdout_DARPA_TC')]
    per[['fold', 'dataset', 'queue', 'criterion', 'labelled', 'attack_windows', 'caught', 'false', 'questions_known', 'D',
         'value_per_question', 'groups_present', 'groups_reached']].to_csv(OUT / 'results' / 'robustness_per_group.csv', index=False, float_format='%.6g')
    seeds = []
    b10 = by.loc[(by.r == 10) & (by.c == 0.01) & by.label.isin([0, 1])]
    for queue, g in b10.groupby('queue'):
        att, ben = g.label == 1, g.label == 0
        for s in range(10):
            d = np.where(g.has_admissible, g[f'd_random_{s}'], g.decision_none)
            seeds.append(dict(queue=queue, seed=s, D=float((10 * (att & (d == 0))).sum() + (ben & (d == 1)).sum())))
    pd.DataFrame(seeds).to_csv(OUT / 'results' / 'robustness_random_seeds.csv', index=False)
    write_json(OUT / 'checks' / 'robustness.json', dict(generated=now(), script=SCRIPT,
               outputs=['results/robustness_bootstrap.csv', 'results/robustness_sensitivity.csv', 'results/robustness_starts.csv',
                        'results/robustness_predictive.csv', 'results/robustness_fitted_lr.csv', 'results/robustness_per_group.csv',
                        'results/robustness_random_seeds.csv']))
    print('all done')


if __name__ == '__main__':
    main()
