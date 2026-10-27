"""Tables of the paper (results/tables.md).

Every table is computed from its source file and, where possible, cross-checked by a second route; a value whose
check fails is reported as MISMATCH at the end of the run. Labels: test labels of the tested collections through
read_labels (Table 1); stored results that embed labels.

Table identifiers follow the paper: T1 collections, T2 queues and starts, T4 admissible alarms, T5 differences
between criteria, T6 criteria over the realistic cells, T7 value per question with intervals, T8 sensitivity of
the value, T9 answer strength, T10 attacks needed; appendix A1 replay, A2 per collection, A3 fitted likelihood
ratios, A4 six fixed orders, A5 starts under other estimates, A6 blocks. Table 3 of the paper (prices of costly
actions) is taken from the literature and is not computed here.

    python scripts/tables.py
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DATASETS, NAMES, OUT, folds, read_json, read_labels, rel

SCRIPT = rel(__file__)
GROUPS = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')
FOLD_NAME = {'velox_holdout_DARPA_TC': 'DARPA TC', 'velox_holdout_OpTC': 'OpTC host 051',
             'velox_holdout_ATLASv2': 'ATLASv2'}
QUEUE_NAME = {'1': '1 a day', '3': '3 a day', '10': '10 a day', '30': '30 a day', '100': '100 a day',
              'alert': 'alert queue'}
REAL_QUEUES = ['1', '3', '10', '30', 'alert']
RS, CS = [5, 10, 20], [0.001, 0.01, 0.03, 0.1, 0.3]
ORDERS = {'order_fnp': 'file, network, process', 'order_fpn': 'file, process, network',
          'order_nfp': 'network, file, process', 'order_npf': 'network, process, file',
          'order_pfn': 'process, file, network', 'order_pnf': 'process, network, file'}
CRIT_NAME = {'edv': 'EDV', 'ig': 'IG', 'random': 'random', 'order_fnp': 'fixed order (file, network, process)',
             'none': 'no question', 'all': 'all questions'}
# Operating system and hosts of each collection (official E3 report for the DARPA TC hosts)
PLATFORM = {'CADETS_E3': ('FreeBSD', 1), 'CLEARSCOPE_E3': ('Android', 1), 'THEIA_E3': ('Linux', 1),
            'optc_h051': ('Windows', 1), 'ATLASV2_EDR': ('Windows', 2)}
PLATFORM_SRC = {'CADETS_E3': 'DARPA TC E3 ground truth report (CADETS FreeBSD)',
                'CLEARSCOPE_E3': 'DARPA TC E3 ground truth report (ClearScope Android 6.0.1)',
                'THEIA_E3': 'DARPA TC E3 ground truth report (THEIA Ubuntu 12.04)',
                'optc_h051': 'van Opstal and Arbaugh 2019 (Windows hosts); host 051 only',
                'ATLASV2_EDR': 'Riddle, Westfall and Bates 2023 (two Windows hosts)'}
VALUES = []


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 'n/a'
    if isinstance(x, (int, np.integer)) or (isinstance(x, float) and float(x).is_integer() and abs(x) < 1e9):
        return f'{int(round(x)):,}'
    if isinstance(x, float) and np.isinf(x):
        return 'none'
    r = float(f'{x:.{digits}g}')
    if abs(r) >= 1000:
        return f'{int(round(r)):,}'
    return f'{r:g}'


def fmt1(x):
    """Counts and losses: integers as integers, otherwise one decimal (means over ten seeds)."""
    x = float(x)
    return fmt(x) if abs(x - round(x)) < 1e-9 else f'{x:,.1f}'


def pct(x, decimals=1):
    return f'{100 * x:.{decimals}f}%'


def interval(lo, hi, digits=2):
    return 'n/a' if pd.isna(lo) else f'[{fmt(float(lo), digits)}, {fmt(float(hi), digits)}]'


def note(value, where, source, calc, check):
    VALUES.append(dict(number=value, where=where, source=source, calc=calc, check=check))
    return value


def md_table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
    out += ['| ' + ' | '.join(str(x) for x in row) + ' |' for row in rows]
    return '\n'.join(out)


def realistic(df):
    return df.loc[df.queue.isin(REAL_QUEUES) & df.r.isin(RS) & df.c.isin(CS)]


def t1():
    """Collections: platform, hosts, test days with scores, windows, categories, labelled and attack windows, attacks."""
    rows = []
    for ds in DATASETS:
        t = pq.read_table(DATA / 'windows' / ds / 'test.parquet',
                          columns=['subject_id', 'window_start_ns', 'category', 'warm_up', 'velox_score']).to_pandas()
        t = t.loc[~t.warm_up]
        days = pd.to_datetime(t.loc[t.velox_score.notna(), 'window_start_ns'], unit='ns', utc=True).dt.strftime('%Y-%m-%d').nunique()
        labels = read_labels(ds, 'test')
        ys = [labels[(ds, 'test', int(s), int(w))] for s, w in zip(t.subject_id, t.window_start_ns)]
        known = sum(y in (0, 1) for y, _ in ys)
        attacks = sum(y == 1 for y, _ in ys)
        groups = len({g for y, gs in ys if y == 1 for g in gs})
        cat = t.category.value_counts()
        ok = known + sum(y == -1 for y, _ in ys) == len(t) and sum(int(cat.get(k, 0)) for k in ('unscored', 'low', 'high', 'alert')) == len(t)
        check = 'counted from the window table and the labels' if ok else 'MISMATCH'
        src = f'data/windows/{ds}/test.parquet; data/labels/{ds}/test.parquet'
        os_, hosts = PLATFORM[ds]
        row = [NAMES[ds], os_, note(fmt(hosts), f'T1, {NAMES[ds]}, hosts', PLATFORM_SRC[ds], 'monitored hosts',
                                    'stated in the source')]
        vals = [days, len(t), int(cat.get('unscored', 0)), int(cat.get('low', 0)), int(cat.get('high', 0)),
                int(cat.get('alert', 0)), known, len(t) - known, attacks, groups]
        cols = ['UTC days with scores', 'windows', 'unscored', 'low', 'high', 'alert', 'labelled windows',
                'windows of unknown label', 'attack windows', 'distinct attacks']
        for v, col in zip(vals, cols):
            row.append(note(fmt(v), f'T1, {NAMES[ds]}, {col}', src, col, check))
        rows.append(row)
    tot = [sum(int(str(r[i]).replace(',', '')) for r in rows) for i in range(2, 13)]
    names = ['hosts', 'UTC days with scores', 'windows', 'unscored', 'low', 'high', 'alert', 'labelled windows',
             'windows of unknown label', 'attack windows', 'distinct attacks']
    total = ['Total', '', note(fmt(tot[0]), 'T1, Total, hosts', 'T1 rows above', 'sum of hosts', 'sum of the rows'), '']
    total += [note(fmt(v), f'T1, Total, {c}', 'T1 rows above', f'sum of {c}', 'sum of the rows')
              for v, c in zip(tot[2:], names[2:])]
    rows.append(total)
    return md_table(['Collection', 'OS', 'Hosts', 'Days', 'Windows', 'Unscored', 'Low', 'High', 'Alert',
                     'Labelled windows', 'Unknown label', 'Attack windows', 'Distinct attacks'], rows)


def t2(by):
    """Queues and starts per held out group, with the realised share of attack windows in the tested queue."""
    starts = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str}, float_precision='round_trip')
    b = by.loc[(by.r == 10) & (by.c == 0)]
    rows = []
    for f in folds():
        for q in QUEUE_NAME:
            s = starts.loc[(starts.fold == f['id']) & (starts.queue == q)].iloc[0]
            g = b.loc[(b.fold == f['id']) & (b.queue == q)]
            att = g.label == 1
            labelled = int(g.label.isin([0, 1]).sum())
            groups = len({(ds, x) for ds, gs in zip(g.dataset[att], g.attack_groups[att]) for x in json.loads(gs)})
            where = f'T2, {FOLD_NAME[f["id"]]}, {QUEUE_NAME[q]}'
            exact = (int(s.source_attack_windows) + 1) / (int(s.source_labelled) + 2)
            rows.append([FOLD_NAME[f['id']], QUEUE_NAME[q],
                         note(fmt(int(s.source_attack_windows)), where + ', source attack windows',
                              'results/starts.csv', 'source_attack_windows', 'count of labelled source alarms'),
                         note(fmt(int(s.source_labelled)), where + ', source alarms (labelled)',
                              'results/starts.csv', 'source_labelled', 'count of labelled source alarms'),
                         note(f'{100 * s.p_N:.2f}%', where + ', start', 'results/starts.csv',
                              'p_N = (attacks + 1)/(labelled + 2)',
                              'recomputed from the two counts (exact)' if abs(exact - s.p_N) < 1e-12 else 'MISMATCH'),
                         note(fmt(len(g)), where + ', tested alarms', 'results/by_alarm.parquet', 'rows at r = 10, c = 0',
                              'count of tested alarms'),
                         note(fmt(labelled), where + ', labelled', 'results/by_alarm.parquet',
                              'label in {0, 1}', 'count of tested alarms'),
                         note(fmt(int(att.sum())), where + ', attack windows', 'results/by_alarm.parquet', 'label = 1',
                              'count of tested alarms'),
                         note(pct(att.sum() / labelled) if labelled else 'n/a', where + ', realised share',
                              'results/by_alarm.parquet', 'attack windows / labelled', 'ratio of the two columns'),
                         note(fmt(groups), where + ', distinct attacks', 'results/by_alarm.parquet',
                              'distinct (collection, attack group) among label = 1', 'count of tested alarms')])
    return md_table(['Tested group', 'Queue', 'Source attack windows', 'Source alarms (labelled)', 'Start',
                     'Tested alarms', 'Labelled', 'Attack windows', 'Realised share', 'Distinct attacks'], rows)


def t4():
    """Alarms with an admissible question (label free): counts at r = 10 by price, shares over the realistic region."""
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    c = count.loc[(count.B == 'inf') & count.dataset.isin(GROUPS)].groupby(['queue', 'r', 'c'], as_index=False)[
        ['alarms', 'alarms_admissible']].sum()
    w = pd.read_csv(OUT / 'results' / 'count_windows.csv').groupby(['r', 'c'], as_index=False)[
        ['windows', 'windows_admissible']].sum().rename(columns={'windows': 'alarms', 'windows_admissible': 'alarms_admissible'})
    w['queue'] = 'windows'
    d = pd.concat([c, w], ignore_index=True)
    d['share'] = d.alarms_admissible / d.alarms
    rows = []
    for q in REAL_QUEUES + ['windows']:
        x = d.loc[d.queue == q]
        name = 'all test windows' if q == 'windows' else QUEUE_NAME[q]
        where = f'T4, {name}'
        src = 'results/count_windows.csv' if q == 'windows' else 'results/count.csv'
        n = int(x.alarms.iloc[0])
        row = [name, note(fmt(n), where + ', alarms', src, 'alarms (sum over the three tested groups)', 'equals T2 sums' if q != 'windows' else 'equals T1 total')]
        for price in [0.0] + CS[1:]:
            k = int(x.loc[(x.r == 10) & (x.c == price), 'alarms_admissible'].iloc[0])
            row.append(note(fmt(k), f'{where}, admissible at r = 10, c = {price:g}', src, 'alarms_admissible', 'sum over groups'))
        for band, prices in (('c in {0.001, 0.01}', [0.001, 0.01]), ('c in {0.1, 0.3}', [0.1, 0.3])):
            y = x.loc[x.r.isin(RS) & x.c.isin(prices), 'share']
            lo, hi = float(y.min()), float(y.max())
            dig = 2 if q == 'windows' else 0
            one = (lambda v: f'{fmt(100 * v, 2)}%') if q == 'windows' else (lambda v: f'{100 * v:.{dig}f}%')
            txt = f'{one(lo)} to {one(hi)}' if one(lo) != one(hi) else one(lo)
            row.append(note(txt, f'{where}, share over r in {{5, 10, 20}}, {band}', src, 'min and max of alarms_admissible / alarms',
                            'recomputed from the counts'))
        rows.append(row)
    return md_table(['Queue', 'Alarms', 'Admissible, r = 10, c = 0', 'c = 0.01', 'c = 0.03', 'c = 0.1', 'c = 0.3',
                     'Share, realistic r, c = 0.001 or 0.01', 'Share, realistic r, c = 0.1 or 0.3'], rows)


def t5():
    """Differences between criteria (label free): at r = 10, c = 0.01 without budget, and maxima over the realistic cells."""
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    s = count.loc[count.dataset.isin(GROUPS)].groupby(['queue', 'r', 'c', 'B'], as_index=False)[
        ['alarms', 'alarms_questioned', 'diffq_edv_ig', 'diffd_edv_ig', 'diffq_edv_random', 'diffd_edv_random']].sum()
    claims = pd.read_csv(OUT / 'results' / 'claims.csv', dtype={'scope': str})
    rows = []
    for q in REAL_QUEUES:
        x = s.loc[(s.queue == q) & (s.r == 10) & (s.c == 0.01) & (s.B == 'inf')].iloc[0]
        rr = realistic(s.loc[s.queue == q])
        where = f'T5, {QUEUE_NAME[q]}'
        row = [QUEUE_NAME[q], note(fmt(int(x.alarms)), where + ', alarms', 'results/count.csv', 'alarms', 'equals T4')]
        for col, label in (('alarms_questioned', 'questioned'), ('diffq_edv_ig', 'IG vs EDV, different questions'),
                           ('diffd_edv_ig', 'IG vs EDV, different decisions'),
                           ('diffq_edv_random', 'random vs EDV, different questions (mean of seeds)'),
                           ('diffd_edv_random', 'random vs EDV, different decisions (mean of seeds)')):
            row.append(note(fmt1(x[col]), f'{where}, r = 10, c = 0.01, {label}', 'results/count.csv', col, 'sum over groups'))
        for col, label, cl in (('diffd_edv_ig', 'max IG vs EDV different decisions', 'alarms where EDV and IG decide differently'),
                               ('diffd_edv_random', 'max random vs EDV different decisions',
                                'alarms where EDV and random decide differently (mean of seeds)')):
            mx = float(rr[col].max())
            ref = claims.loc[(claims.quantity == cl) & (claims.scope == q), 'max']
            ok = len(ref) == 1 and abs(float(ref.iloc[0]) - mx) < 1e-6
            row.append(note(fmt1(mx), f'{where}, realistic cells, {label}', 'results/count.csv', f'max of {col} over r, c, B',
                            'equals claims.csv C2' if ok else 'MISMATCH'))
        rows.append(row)
    return md_table(['Queue', 'Alarms', 'Questioned', 'IG vs EDV: different questions', 'IG vs EDV: different decisions',
                     'Random vs EDV: different questions', 'Random vs EDV: different decisions',
                     'Realistic cells, max: IG vs EDV decisions', 'Random vs EDV decisions'], rows)


def t6(res):
    """Per queue and criterion, range over the realistic cells of sums over the three groups (was Table 3)."""
    real = pd.read_csv(OUT / 'results' / 'replay_realistic.csv', dtype={'queue': str})
    sel = res.loc[res.dataset.isin(GROUPS) & res.queue.isin(REAL_QUEUES) & res.r.isin(RS) & res.c.isin(CS)]
    sums = sel.groupby(['queue', 'r', 'c', 'B', 'criterion'])[['caught', 'false', 'questions_known', 'differs_edv']].sum().reset_index()
    rows = []
    for q in REAL_QUEUES:
        for crit in ['edv', 'ig', 'random', 'order_fnp', 'none', 'all']:
            r = real.loc[(real.queue == q) & (real.criterion == crit)].iloc[0]
            g = sums.loc[(sums.queue == q) & (sums.criterion == crit)]
            where = f'T6, {QUEUE_NAME[q]}, {CRIT_NAME[crit]}'
            row = [QUEUE_NAME[q] if crit == 'edv' else '', CRIT_NAME[crit]]
            for k, label in (('caught', 'attack windows caught'), ('false', 'false escalations'),
                             ('questions_known', 'questions'), ('differs_edv', 'decisions different from EDV')):
                lo, hi = float(r[f'{k}_min']), float(r[f'{k}_max'])
                ok = abs(lo - g[k].min()) < 1e-9 and abs(hi - g[k].max()) < 1e-9
                txt = f'{fmt1(lo)} to {fmt1(hi)}' if lo != hi else fmt1(lo)
                row.append(note(txt, f'{where}, {label}', 'results/replay_realistic.csv', f'{k}_min, {k}_max',
                                'recomputed from replay.parquet' if ok else 'MISMATCH'))
            rows.append(row)
    head = ('Alarms (attack windows): '
            + ', '.join(f"{QUEUE_NAME[q]} {fmt(int(real.loc[real.queue == q, 'labelled'].iloc[0]))} "
                        f"({fmt(int(real.loc[real.queue == q, 'attack_windows'].iloc[0]))})" for q in REAL_QUEUES)
            + '.')
    for q in REAL_QUEUES:
        note(fmt(int(real.loc[real.queue == q, 'labelled'].iloc[0])), f'T6 caption, {QUEUE_NAME[q]}, labelled alarms',
             'results/replay_realistic.csv', 'labelled', 'equals T2 sums')
        note(fmt(int(real.loc[real.queue == q, 'attack_windows'].iloc[0])), f'T6 caption, {QUEUE_NAME[q]}, attack windows',
             'results/replay_realistic.csv', 'attack_windows', 'equals T2 sums')
    return head + '\n\n' + md_table(['Queue', 'Criterion', 'Attack windows caught', 'False escalations',
                                     'Questions asked', 'Decisions different from EDV'], rows)


def t7():
    """Value per question at r = 10, c = 0.01 and 0.03, without budget: bootstrap and model predictive intervals."""
    boot = pd.read_csv(OUT / 'results' / 'robustness_bootstrap.csv', dtype={'queue': str})
    pred = pd.read_csv(OUT / 'results' / 'robustness_predictive.csv', dtype={'queue': str})
    val = pd.read_csv(OUT / 'results' / 'value.csv', dtype={'queue': str, 'B': str})
    rows = []
    for q in REAL_QUEUES:
        for crit in ['edv', 'ig', 'random']:
            row = [QUEUE_NAME[q] if crit == 'edv' else '', CRIT_NAME[crit]]
            for c in (0.01, 0.03):
                b = boot.loc[(boot.queue == q) & (boot.r == 10) & (boot.c == c) & (boot.criterion == crit)].iloc[0]
                p = pred.loc[(pred.queue == q) & (pred.r == 10) & (pred.c == c) & (pred.criterion == crit)].iloc[0]
                v = val.loc[(val.queue == q) & (val.r == 10) & (val.c == c) & (val.B == 'inf') & (val.criterion == crit)].iloc[0]
                ok = (int(b.questions) == int(v.questions_known) and abs(b.value - v.value_per_question) < 1e-5
                      and abs(p.realised - v.value_per_question) < 1e-5)
                where = f'T7, {QUEUE_NAME[q]}, {CRIT_NAME[crit]}, c = {c:g}'
                chk = 'equals value.csv (B infinite) and the bootstrap point' if ok else 'MISMATCH'
                row.append(note(fmt(int(b.questions)), where + ', questions', 'results/robustness_bootstrap.csv', 'questions', chk))
                row.append(note(f'{fmt(float(b.value), 2)} {interval(b.boot_lo, b.boot_hi)}', where + ', value [bootstrap 95%]',
                                'results/robustness_bootstrap.csv', 'value, boot_lo, boot_hi (2,000 resamples of collection-days)', chk))
                row.append(note(interval(p.pred_lo, p.pred_hi) + (' *' if p.outside else ''), where + ', model 95% interval',
                                'results/robustness_predictive.csv', 'pred_lo, pred_hi (2,000 draws under the model); * outside',
                                'outside flag recomputed' if bool(p.outside) == bool(p.realised < p.pred_lo or p.realised > p.pred_hi) else 'MISMATCH'))
            rows.append(row)
    b = realistic(boot.loc[boot.questions > 0])
    p = realistic(pred.loc[pred.questions > 0])
    tail = (f"Realistic cells with questions: {note(fmt(len(b)), 'T7 note, realistic cells with questions', 'results/robustness_bootstrap.csv', 'rows with questions', 'equals robustness_predictive.csv rows (' + str(len(p)) + ')')}; "
            f"bootstrap intervals containing 0: {note(fmt(int(((b.boot_lo <= 0) & (b.boot_hi >= 0)).sum())), 'T7 note, bootstrap intervals containing 0', 'results/robustness_bootstrap.csv', 'boot_lo <= 0 <= boot_hi', 'count')}, "
            f"above 0: {note(fmt(int((b.boot_lo > 0).sum())), 'T7 note, bootstrap intervals above 0', 'results/robustness_bootstrap.csv', 'boot_lo > 0', 'count')}, "
            f"below 0: {note(fmt(int((b.boot_hi < 0).sum())), 'T7 note, bootstrap intervals below 0', 'results/robustness_bootstrap.csv', 'boot_hi < 0', 'count')}; "
            f"realised values outside the model's 95% interval: {note(fmt(int(p.outside.sum())), 'T7 note, realised outside the model interval', 'results/robustness_predictive.csv', 'outside', 'count')}.")
    return md_table(['Queue', 'Criterion', 'Questions, c = 0.01', 'Value [bootstrap 95%]', 'Model 95% interval',
                     'Questions, c = 0.03', 'Value [bootstrap 95%]', 'Model 95% interval'], rows) + '\n\n' + tail


def t8():
    """Sensitivity of the value per question to the starts and the smoothing, r = 10, c = 0.01, no budget."""
    s = pd.read_csv(OUT / 'results' / 'robustness_sensitivity.csv', dtype={'queue': str})
    val = pd.read_csv(OUT / 'results' / 'value.csv', dtype={'queue': str, 'B': str})
    variants = [('default', 'Default'), ('pseudo_count_0.5', 'Half a window per class'),
                ('source_test_only', 'Source test splits only'), ('tau_1e+02', '$\\tau = 10^2$'), ('tau_1e+03', '$\\tau = 10^3$')]
    rows = []
    for q in REAL_QUEUES:
        for crit in ['edv', 'ig', 'random']:
            row = [QUEUE_NAME[q] if crit == 'edv' else '', CRIT_NAME[crit]]
            for var, label in variants:
                x = s.loc[(s.variant == var) & (s.queue == q) & (s.r == 10) & (s.c == 0.01) & (s.criterion == crit)].iloc[0]
                chk = 'robustness_sensitivity.csv'
                if var == 'default':
                    v = val.loc[(val.queue == q) & (val.r == 10) & (val.c == 0.01) & (val.B == 'inf') & (val.criterion == crit)].iloc[0]
                    chk = 'equals value.csv (B infinite)' if abs(v.value_per_question - x.value) < 1e-5 else 'MISMATCH'
                txt = f'{fmt(float(x.value), 2)} ({fmt(int(x.Q))})' if x.Q > 0 else 'no question'
                row.append(note(txt, f'T8, {QUEUE_NAME[q]}, {CRIT_NAME[crit]}, {label}', 'results/robustness_sensitivity.csv',
                                'value (questions)', chk))
            rows.append(row)
    d = s.loc[s.variant == 'default'].set_index(['queue', 'r', 'c', 'criterion'])
    sign, only = ['Sign differs from default', ''], ['Only one of the two asks', '']
    for var, label in variants:
        if var == 'default':
            n = int(realistic(s.loc[(s.variant == 'default') & (s.Q > 0)]).shape[0])
            sign.append(note(f'{fmt(n)} cells with questions', 'T8, realistic cells with questions (default)',
                             'results/robustness_sensitivity.csv', 'Q > 0 over the realistic cells', 'equals T7 note'))
            only.append('')
            continue
        o = s.loc[s.variant == var].set_index(['queue', 'r', 'c', 'criterion'])
        j = realistic(d[['value', 'Q']].join(o[['value', 'Q']], rsuffix='_v').reset_index())
        both = j.loc[(j.Q > 0) & (j.Q_v > 0)]
        k = int((np.sign(both.value) != np.sign(both.value_v)).sum())
        one = int(((j.Q > 0) != (j.Q_v > 0)).sum())
        sign.append(note(f'{fmt(k)} of {fmt(len(both))}', f'T8, {label}, cells where the sign differs', 'results/robustness_sensitivity.csv',
                         'realistic cells where both ask; sign of value differs', 'count'))
        only.append(note(fmt(one), f'T8, {label}, cells where only one variant asks', 'results/robustness_sensitivity.csv',
                         'realistic cells; questions in exactly one variant', 'count'))
    rows += [sign, only]
    return md_table(['Queue', 'Criterion'] + [label for _, label in variants], rows)


def t9():
    """Answer strength inside alarms against all high windows (and, for the alerts with pooled tables, all windows)."""
    a = pd.read_csv(OUT / 'results' / 'assumption_check.csv', dtype={'set': str})
    set_name = {'10': '10 a day', '30': '30 a day', 'alert': 'alert queue', 'all_high': 'all high windows',
                'all_windows': 'all windows'}
    rows = []
    for f in folds():
        sets = ['10', '30', 'alert', 'all_high'] + (['all_windows'] if f['id'] != 'velox_holdout_DARPA_TC' else [])
        for st in sets:
            x = a.loc[(a.fold == f['id']) & (a.set == st)].set_index(['question', 'answer'])
            where = f'T9, {FOLD_NAME[f["id"]]}, {set_name[st]}'
            row = [FOLD_NAME[f['id']], set_name[st],
                   note(fmt(int(x.attack_windows.iloc[0])), where + ', attack windows', 'results/assumption_check.csv',
                        'attack_windows (file question)', 'count from load_labels on source collections')]
            for q, ans in (('file', 'novel'), ('network', 'novel'), ('network', 'known')):
                r = x.loc[(q, ans)]
                fa = 'n/a' if pd.isna(r.freq_attack) else f'{100 * r.freq_attack:.0f}%'
                both = note(f'{fa} / {100 * r.freq_benign:.0f}%', f'{where}, attack / benign {q} {ans}', 'results/assumption_check.csv',
                            'freq_attack / freq_benign', 'with_answer / windows in each class')
                lr = r.lr
                ok = np.isnan(lr) or np.isinf(lr) or abs(lr - r.freq_attack / r.freq_benign) < 1e-3 * max(1, lr)
                lrs = note(fmt(float(lr), 2) if np.isfinite(lr) else ('inf' if np.isinf(lr) else 'n/a'),
                           f'{where}, raw LR {q} {ans}', 'results/assumption_check.csv', 'lr = freq_attack / freq_benign (raw)',
                           'recomputed' if ok else 'MISMATCH')
                row += [both, lrs]
            rows.append(row)
    js = read_json(OUT / 'checks' / 'assumption_check.json')
    s = pd.DataFrame(js['comparisons'])
    lines = []
    for st in ['1', '3', '10', '30', '100', 'alert']:
        for cls in ('attack', 'benign'):
            r = s.loc[(s.set == st) & (s.cls == cls)].iloc[0]
            total = int(r.get('intervals disjoint', 0) + r.get('intervals overlap', 0))
            d = note(fmt(int(r.get('intervals disjoint', 0))), f'T9 note, queue {st}, {cls}, disjoint',
                     'checks/assumption_check.json', 'comparisons: intervals disjoint', 'count of Wilson comparisons')
            n = note(fmt(total), f'T9 note, queue {st}, {cls}, comparisons', 'checks/assumption_check.json',
                     'disjoint + overlap', 'count of Wilson comparisons')
            lines.append(f'{QUEUE_NAME[st]}, {cls} windows: {d} of {n} comparisons with disjoint intervals')
    s2 = pd.DataFrame(js['alert_against_all_windows'])
    lines2 = []
    for f in ('velox_holdout_OpTC', 'velox_holdout_ATLASv2'):
        for cls in ('attack', 'benign'):
            r = s2.loc[(s2.fold == f) & (s2.cls == cls)].iloc[0]
            total = int(r.get('intervals disjoint', 0) + r.get('intervals overlap', 0))
            d = note(fmt(int(r.get('intervals disjoint', 0))), f'T9 note 2, {FOLD_NAME[f]}, {cls}, disjoint against all windows',
                     'checks/assumption_check.json', 'alert_against_all_windows: intervals disjoint', 'count of Wilson comparisons')
            n = note(fmt(total), f'T9 note 2, {FOLD_NAME[f]}, {cls}, comparisons against all windows',
                     'checks/assumption_check.json', 'disjoint + overlap', 'count of Wilson comparisons')
            lines2.append(f'{FOLD_NAME[f]}, {cls} windows: {d} of {n}')
    table = md_table(['Tested group (sources)', 'Set', 'Attack windows', 'File novel: attack / benign', 'LR',
                      'Network novel: attack / benign', 'LR', 'Network known: attack / benign', 'LR'], rows)
    return (table + '\n\nWilson 95% intervals of answer frequencies, queue against all high windows, over the three groups, '
            'questions and answers: ' + '; '.join(lines) + '. Alert queue against all windows: ' + '; '.join(lines2) + '.')


def t10():
    """Distinct attacks needed to tell two criteria apart (model based), median over the cells with a difference."""
    p = pd.read_csv(OUT / 'results' / 'power_realistic.csv', dtype={'queue': str})
    full = pd.read_csv(OUT / 'results' / 'power.csv', dtype={'queue': str, 'B': str})
    full = full.loc[full.queue.isin(REAL_QUEUES) & full.r.isin(RS) & full.c.isin(CS) & (full.p_source == 'final_all')]
    rows = []
    for q in REAL_QUEUES:
        row = [QUEUE_NAME[q]]
        obs = int(p.loc[p.queue == q, 'observed_attacks'].iloc[0])
        row.append(note(fmt(obs), f'T10, {QUEUE_NAME[q]}, observed distinct attacks', 'results/power_realistic.csv',
                        'observed_attacks', 'equals T2 sums'))
        for pair in ('edv-random', 'ig-random', 'edv-ig', 'edv-none'):
            r = p.loc[(p.queue == q) & (p.pair == pair) & (p.p_source == 'final_all')].iloc[0]
            g = full.loc[(full.queue == q) & (full.pair == pair)]
            fin = g.loc[np.isfinite(g.attacks_needed), 'attacks_needed']
            med = float(fin.median()) if len(fin) else np.inf
            ok = (np.isinf(med) and np.isinf(r.attacks_needed_median)) or abs(med - r.attacks_needed_median) < 1e-5 * max(1, med)
            ndiff = int(r.cells) - int(r.cells_no_difference)
            ok = ok and ndiff == len(fin)
            txt = f'{fmt(med, 2)} ({ndiff})' if np.isfinite(med) else f'never ({ndiff})'
            row.append(note(txt, f'T10, {QUEUE_NAME[q]}, {pair}, median attacks needed (cells with a difference)',
                            'results/power_realistic.csv', 'attacks_needed_median; cells - cells_no_difference',
                            'recomputed from power.csv' if ok else 'MISMATCH'))
        rows.append(row)
    note('90', 'T10 caption, realistic cells per queue and pair', 'results/power_realistic.csv', 'cells', '3 r x 5 c x 6 B')
    return md_table(['Queue', 'Distinct attacks in the queue', 'EDV vs random', 'IG vs random', 'EDV vs IG',
                     'EDV vs no question'], rows)


def a1(res):
    """Replay at r = 10, c = 0.01; the queue of 1 a day never binds, so its budget rows are collapsed when identical."""
    x = res.loc[res.dataset.isin(GROUPS) & (res.r == 10) & (res.c == 0.01)]
    g = x.groupby(['queue', 'B', 'criterion'])[['caught', 'false', 'questions_known', 'D']].sum().reset_index()
    rows = []
    for q in QUEUE_NAME:
        budgets = ['1', '2', '3', '5', '10', 'inf', 'any']
        collapse = False
        if q == '1':
            y = g.loc[(g.queue == q) & g.B.isin(budgets[:-1])]
            collapse = bool((y.groupby('criterion')[['caught', 'false', 'questions_known', 'D']].nunique() == 1).all().all())
        for b in (['inf', 'any'] if collapse else budgets):
            for crit in ['edv', 'ig', 'random', 'order_fnp', 'none', 'all']:
                y = g.loc[(g.queue == q) & (g.B == b) & (g.criterion == crit)]
                if not len(y):
                    continue
                y = y.iloc[0]
                blabel = '∞' if b == 'inf' else ('any' if b == 'any' else b)
                if collapse and b == 'inf':
                    blabel = '1 to ∞'
                where = f'A1, {QUEUE_NAME[q]}, B = {b}, {CRIT_NAME[crit]}'
                rows.append([QUEUE_NAME[q], blabel, CRIT_NAME[crit]] +
                            [note(fmt1(float(y[k])), f'{where}, {k}', 'results/replay.parquet', f'sum of {k} over the three groups',
                                  'rebuilt from by_alarm.parquet (checks/replay_check.json)')
                             for k in ('caught', 'false', 'questions_known', 'D')])
    return md_table(['Queue', 'B', 'Criterion', 'Attack windows caught', 'False escalations', 'Questions', 'Loss D'], rows)


def a2():
    """Per tested collection at r = 10, c = 0.01, no budget: no question against EDV, with attacks reached."""
    p = pd.read_csv(OUT / 'results' / 'robustness_per_group.csv', dtype={'queue': str})
    rows = []
    for ds in DATASETS:
        for q in REAL_QUEUES:
            n = p.loc[(p.dataset == ds) & (p.queue == q) & (p.criterion == 'none')].iloc[0]
            e = p.loc[(p.dataset == ds) & (p.queue == q) & (p.criterion == 'edv')].iloc[0]
            where = f'A2, {NAMES[ds]}, {QUEUE_NAME[q]}'
            src = 'results/robustness_per_group.csv'
            v = 'n/a' if e.questions_known == 0 else fmt(float((n.D - e.D) / e.questions_known), 2)
            ok = e.questions_known == 0 or abs((n.D - e.D) / e.questions_known - e.value_per_question) < 1e-6
            rows.append([NAMES[ds], QUEUE_NAME[q],
                         note(fmt(int(e.labelled)), where + ', labelled', src, 'labelled', 'replay.parquet'),
                         note(fmt(int(e.attack_windows)), where + ', attack windows', src, 'attack_windows', 'replay.parquet'),
                         note(fmt(int(e.groups_present)), where + ', attacks present', src, 'groups_present', 'replay.parquet'),
                         note(f'{fmt1(n.caught)} / {fmt1(n.false)}', where + ', no question caught / false', src, 'caught, false', 'replay.parquet'),
                         note(fmt(int(e.questions_known)), where + ', EDV questions', src, 'questions_known', 'replay.parquet'),
                         note(f'{fmt1(e.caught)} / {fmt1(e.false)}', where + ', EDV caught / false', src, 'caught, false', 'replay.parquet'),
                         note(v, where + ', EDV value per question', src, '(D_none - D) / questions',
                              'equals value_per_question' if ok else 'MISMATCH'),
                         note(f'{fmt1(n.groups_reached)} / {fmt1(e.groups_reached)}', where + ', attacks reached (no question / EDV)',
                              src, 'groups_reached', 'replay.parquet')])
    # cross-check: DARPA TC pooled equals the sum of its three collections
    darpa = p.loc[p.dataset.isin(['CADETS_E3', 'CLEARSCOPE_E3', 'THEIA_E3'])].groupby(['queue', 'criterion'])[['caught', 'false', 'D']].sum()
    pooled = p.loc[p.dataset == 'DARPA_TC'].set_index(['queue', 'criterion'])[['caught', 'false', 'D']]
    assert float((darpa.sort_index() - pooled.loc[darpa.index].sort_index()).abs().max().max()) < 1e-9
    return md_table(['Collection', 'Queue', 'Labelled', 'Attack windows', 'Attacks', 'No question: caught / false',
                     'EDV: questions', 'EDV: caught / false', 'EDV: value per question', 'Attacks reached: none / EDV'], rows)


def a3():
    """Fitted single answer likelihood ratios per model and category (three question mask), with prior and tables used."""
    f = pd.read_csv(OUT / 'results' / 'robustness_fitted_lr.csv')
    f = f.loc[f['mask'] == 'file,network,process']
    a = pd.read_csv(OUT / 'results' / 'assumption_check.csv', dtype={'set': str})
    a = a.loc[a.set == 'all_high'].set_index(['fold', 'question', 'answer'])
    rows = []
    for fo in folds():
        for cat in ('unscored', 'low', 'high', 'alert'):
            x = f.loc[(f.fold == fo['id']) & (f.category == cat)]
            mc, tab, prior = x.model_category.iloc[0], x.likelihood_table.iloc[0], float(x.prior.iloc[0])
            uses = ('own prior, own tables' if tab == f'context:{cat}' else
                    ('prior and tables of high' if mc == 'high' and cat != 'high' else 'own prior, pooled tables'))
            for k, q in enumerate(('file', 'network', 'process')):
                y = x.loc[x.question == q].set_index('answer')
                where = f'A3, {FOLD_NAME[fo["id"]]}, {cat}, {q}'
                row = [FOLD_NAME[fo['id']] if (cat == 'unscored' and k == 0) else '', cat if k == 0 else '',
                       note(pct(prior, 2), f'A3, {FOLD_NAME[fo["id"]]}, {cat}, prior', 'results/robustness_fitted_lr.csv', 'prior (model.initial)',
                            'fitted model') if k == 0 else '', uses if k == 0 else '', q]
                for ans in ('empty', 'known', 'novel'):
                    lr = float(y.lr.get(ans, np.nan))
                    ok = True
                    if mc == 'high' and not np.isnan(lr):
                        ok = abs(lr - float(a.loc[(fo['id'], q, ans), 'model_lr_high'])) < 1e-4 * max(1, lr)
                    row.append(note(fmt(lr, 2), f'{where}, {ans}', 'results/robustness_fitted_lr.csv', 'lr = l1 / l0 (smoothed)',
                                    ('equals assumption_check.csv model_lr_high' if mc == 'high' else 'fitted model') if ok else 'MISMATCH'))
                rows.append(row)
    return md_table(['Model (tested group)', 'Category', 'Prior', 'Prior and tables used', 'Question', 'Empty', 'Known',
                     'Novel'], rows)


def a4(res):
    """Decision loss of every criterion, including the six fixed orders, at r = 10, c = 0.01, no budget."""
    p = pd.read_csv(OUT / 'results' / 'robustness_per_group.csv', dtype={'queue': str})
    s = p.loc[p.dataset.isin(GROUPS)].groupby(['queue', 'criterion'])[['D', 'questions_known']].sum()
    x = res.loc[res.dataset.isin(GROUPS) & (res.r == 10) & (res.c == 0.01) & res.B.isin(['inf', 'any'])]
    ref = x.groupby(['queue', 'criterion'])['D'].sum()
    crits = ['none', 'edv', 'ig', 'random'] + list(ORDERS) + ['all']
    rows = []
    for q in REAL_QUEUES:
        row = [QUEUE_NAME[q], note(fmt(int(s.loc[(q, 'edv'), 'questions_known'])), f'A4, {QUEUE_NAME[q]}, questions',
                                   'results/robustness_per_group.csv', 'questions_known', 'equals A1')]
        for crit in crits:
            d = float(s.loc[(q, crit), 'D'])
            ok = abs(d - float(ref.loc[(q, crit)])) < 1e-9
            row.append(note(fmt1(d), f'A4, {QUEUE_NAME[q]}, {crit}, D', 'results/robustness_per_group.csv', 'sum of D over the three groups',
                            'equals replay.parquet' if ok else 'MISMATCH'))
        rows.append(row)
    head = ['Queue', 'Questions', 'No question', 'EDV', 'IG', 'Random'] + [v for v in ORDERS.values()] + ['All questions']
    return md_table(head, rows)


def a5():
    """Starts under the alternative estimates."""
    st = pd.read_csv(OUT / 'results' / 'robustness_starts.csv', dtype={'queue': str})
    base = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str}, float_precision='round_trip')
    rows = []
    for f in folds():
        for q in QUEUE_NAME:
            row = [FOLD_NAME[f['id']], QUEUE_NAME[q]]
            for var in ('default', 'pseudo_count_0.5', 'source_test_only'):
                v = float(st.loc[(st.variant == var) & (st.fold == f['id']) & (st.queue == q), 'start'].iloc[0])
                chk = 'robustness_starts.csv'
                if var == 'default':
                    b = float(base.loc[(base.fold == f['id']) & (base.queue == q), 'p_N'].iloc[0])
                    chk = 'equals starts.csv' if abs(b - v) < 1e-6 else 'MISMATCH'
                if var == 'pseudo_count_0.5':
                    b = base.loc[(base.fold == f['id']) & (base.queue == q)].iloc[0]
                    exp = (b.source_attack_windows + 0.5) / (b.source_labelled + 1)
                    chk = '(a + 0.5)/(n + 1) recomputed' if abs(exp - v) < 1e-6 else 'MISMATCH'
                row.append(note(pct(v, 2), f'A5, {FOLD_NAME[f["id"]]}, {QUEUE_NAME[q]}, {var}', 'results/robustness_starts.csv', 'start', chk))
            rows.append(row)
    return md_table(['Tested group', 'Queue', 'Start (default)', 'Half a window per class', 'Source test splits only'], rows)


def a6(by):
    """Blocks of alarms that share start, prior, tables and available questions, at r = 10, c = 0.01, no budget."""
    f = pd.read_csv(OUT / 'results' / 'robustness_fitted_lr.csv')
    used = {(r.fold, r.category, r['mask']): (r.model_category, r.likelihood_table) for _, r in f.iterrows()}
    x = by.loc[(by.r == 10) & (by.c == 0.01) & by.has_admissible & by.queue.isin(REAL_QUEUES)].copy()
    x['uses'] = [used[(fo, c, m)] for fo, c, m in zip(x.fold, x.category, x['mask'])]
    x['prior_cat'] = [u[0] for u in x.uses]
    x['tables'] = [('pooled' if u[1] == 'pooled' else u[1].split(':')[1]) for u in x.uses]
    x['dd'] = x.d_edv != x.d_ig
    key = ['fold', 'queue', 'prior_cat', 'tables', 'mask']
    assert (x.groupby(key).q_edv.nunique() <= 1).all() and (x.groupby(key).q_ig.nunique() <= 1).all()
    g = x.groupby(key, sort=False).agg(alarms=('q_edv', 'size'), edv=('q_edv', 'first'), ig=('q_ig', 'first'),
                                       dd=('dd', 'sum'), categories=('category', lambda s: ', '.join(sorted(set(s))))).reset_index()
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    ref = count.loc[count.dataset.isin(GROUPS) & (count.r == 10) & (count.c == 0.01) & (count.B == 'inf')].groupby('queue')['diffd_edv_ig'].sum()
    for q in REAL_QUEUES:
        assert int(g.loc[g.queue == q, 'dd'].sum()) == int(ref.loc[q])
    order = {q: i for i, q in enumerate(REAL_QUEUES)}
    g['o1'] = g.fold.map({f['id']: i for i, f in enumerate(folds())})
    g['o2'] = g.queue.map(order)
    rows = []
    for r in g.sort_values(['o1', 'o2']).itertuples(index=False):
        where = f'A6, {FOLD_NAME[r.fold]}, {QUEUE_NAME[r.queue]}, {r.prior_cat}/{r.tables}, {r.mask}'
        rows.append([FOLD_NAME[r.fold], QUEUE_NAME[r.queue], r.categories, r.tables, r.mask.replace(',', ', '),
                     note(fmt(int(r.alarms)), where + ', alarms', 'results/by_alarm.parquet', 'alarms with an admissible question', 'count'),
                     r.edv, r.ig,
                     note(fmt(int(r.dd)), where + ', decisions differing', 'results/by_alarm.parquet', 'd_edv != d_ig',
                          'sums per queue equal count.csv diffd_edv_ig')])
    note(fmt(len(g)), 'A6, number of blocks', 'results/by_alarm.parquet', 'distinct blocks over the realistic queues', 'count')
    note(fmt(int((g.edv != g.ig).sum())), 'A6, blocks where EDV and IG ask different questions', 'results/by_alarm.parquet',
         'q_edv != q_ig', 'count')
    return md_table(['Tested group', 'Queue', 'Categories', 'Tables', 'Questions available', 'Alarms', 'EDV asks',
                     'IG asks', 'Decisions differing'], rows)


def main():
    by = pd.read_parquet(OUT / 'results' / 'by_alarm.parquet')
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    parts = [
        ('T1. Collections, test split', t1()),
        ('T2. Queues and starts', t2(by)),
        ('T4. Alarms with an admissible question (no labels; sums over the three tested groups)', t4()),
        ('T5. Differences between criteria (no labels; sums over the three tested groups)', t5()),
        ('T6. Criteria over the realistic cells (queues 1, 3, 10, 30 a day and the alerts; r = 5, 10, 20; c = 0.001 to 0.3; every B)', t6(res)),
        ('T7. Value per question at r = 10, no budget, with bootstrap and model intervals', t7()),
        ('T8. Sensitivity of the value per question at r = 10, c = 0.01, no budget: value (questions)', t8()),
        ('T9. Answer strength inside alarms against all high windows and all windows (source collections; raw ratios)', t9()),
        ('T10. Distinct attacks needed to tell two criteria apart (model based, realistic cells): median (cells with a difference)', t10()),
        ('A1. Replay at r = 10, c = 0.01, sums over the three tested groups', a1(res)),
        ('A2. Per tested collection at r = 10, c = 0.01, no budget', a2()),
        ('A3. Fitted single answer likelihood ratios (three question mask)', a3()),
        ('A4. Decision loss D of every criterion at r = 10, c = 0.01, no budget, sums over the three tested groups', a4(res)),
        ('A5. Starts under other estimates', a5()),
        ('A6. Blocks of alarms at r = 10, c = 0.01, no budget', a6(by)),
    ]
    text = ['# Tables', '', f'Generated by `{SCRIPT}`.', '']
    for title, body in parts:
        text += [f'## {title}', '', body, '']
    (OUT / 'results' / 'tables.md').write_text('\n'.join(text))
    bad = [n for n in VALUES if 'MISMATCH' in n['check']]
    print(len(VALUES), 'table values;', len(bad), 'mismatches')
    for n in bad[:10]:
        print(n)


if __name__ == '__main__':
    main()
