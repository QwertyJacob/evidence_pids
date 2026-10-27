"""Ranges over the realistic region of the main quantities quoted in the text of the paper (results/claims.csv).

Each row: claim group, quantity, queue (or scope), minimum and maximum over the realistic cells (queues 1, 3, 10,
30 a day and the alerts; r in {5, 10, 20}; c in {0.001, 0.01, 0.03, 0.1, 0.3}; budgets as stated), number of
cells, source file. Sums over the three held out groups. Groups: C1 how many decisions one answer can change, C2
differences between the criteria, C3 the value of a question, C4 attacks needed.

    python scripts/claims.py
"""
import numpy as np
import pandas as pd

from common import OUT

GROUPS = ('DARPA_TC', 'optc_h051', 'ATLASV2_EDR')
QUEUES = ('1', '3', '10', '30', 'alert')
RS = (5, 10, 20)
CS = (0.001, 0.01, 0.03, 0.1, 0.3)
ROWS = []


def add(claim, quantity, scope, values, source, note=''):
    v = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    ROWS.append(dict(claim=claim, quantity=quantity, scope=scope, min=float(v.min()) if len(v) else np.nan,
                     max=float(v.max()) if len(v) else np.nan, cells=len(v), source=source, note=note))


def main():
    count = pd.read_csv(OUT / 'results' / 'count.csv', dtype={'queue': str, 'B': str})
    cg = count.loc[count.dataset.isin(GROUPS)].groupby(['queue', 'r', 'c', 'B'], as_index=False).sum(numeric_only=True)
    real = cg.loc[cg.queue.isin(QUEUES) & cg.r.isin(RS) & cg.c.isin(CS)]
    # C1: how many decisions one answer can change (label free)
    for q in QUEUES:
        x = real.loc[(real.queue == q) & (real.B == 'inf')]
        add('C1', 'share of alarms with an admissible question', q, x.alarms_admissible / x.alarms, 'results/count.csv')
        add('C1', 'alarms with an admissible question', q, x.alarms_admissible, 'results/count.csv',
            f'out of {int(x.alarms.iloc[0])} alarms')
    for q in QUEUES:
        for band, cs in (('c <= 0.01', (0.001, 0.01)), ('c = 0.03', (0.03,)), ('c >= 0.1', (0.1, 0.3))):
            x = real.loc[(real.queue == q) & (real.B == 'inf') & real.c.isin(cs)]
            add('C1', f'share of alarms with an admissible question, {band}', q, x.alarms_admissible / x.alarms,
                'results/count.csv', 'r in {5, 10, 20}')
    w = pd.read_csv(OUT / 'results' / 'count_windows.csv').groupby(['r', 'c'], as_index=False)[['windows', 'windows_admissible']].sum()
    wr = w.loc[w.r.isin(RS) & w.c.isin(CS)]
    add('C1', 'windows with an admissible question', 'all test windows', wr.windows_admissible, 'results/count_windows.csv',
        f'out of {int(wr.windows.iloc[0])} windows')
    w0 = w.loc[w.r.isin(RS) & (w.c == 0)]
    add('C1', 'windows with an admissible question at c = 0', 'all test windows', w0.windows_admissible, 'results/count_windows.csv')
    # starts and required factor at r = 10
    starts = pd.read_csv(OUT / 'results' / 'starts.csv', dtype={'queue': str})
    for q in QUEUES + ('100',):
        p = starts.loc[starts.queue == q, 'p_N']
        add('C1', 'start of the queue (probability)', q, p, 'results/starts.csv')
        o = p / (1 - p)
        add('C1', 'factor needed at r = 10, 1/(10 O)', q, 1 / (10 * o), 'results/starts.csv')
    # C2: divergence of the criteria (label free), every budget
    for q in QUEUES:
        x = real.loc[real.queue == q]
        for col, name in (('diffq_edv_ig', 'alarms where EDV and IG ask different questions'),
                          ('diffd_edv_ig', 'alarms where EDV and IG decide differently'),
                          ('diffd_edv_random', 'alarms where EDV and random decide differently (mean of seeds)'),
                          ('diffd_ig_random', 'alarms where IG and random decide differently (mean of seeds)'),
                          ('diffd_any', 'alarms where some two criteria decide differently'),
                          ('alarms_questioned', 'alarms questioned')):
            add('C2', name, q, x[col], 'results/count.csv', 'all budgets')
        xi = x.loc[x.B == 'inf']
        add('C2', 'alarms where EDV and IG decide differently, B infinite', q, xi.diffd_edv_ig, 'results/count.csv')
    # C3: value of a question (labels), no shared budget
    v = pd.read_csv(OUT / 'results' / 'value.csv', dtype={'queue': str, 'B': str})
    vr = v.loc[v.queue.isin(QUEUES) & v.r.isin(RS) & v.c.isin(CS) & (v.B == 'inf')]
    for q in QUEUES:
        for crit in ('edv', 'ig', 'random'):
            x = vr.loc[(vr.queue == q) & (vr.criterion == crit)]
            add('C3', f'value per question ({crit})', q, x.value_per_question, 'results/value.csv', 'B infinite')
            for band, cs in (('c <= 0.01', (0.001, 0.01)), ('c >= 0.03', (0.03, 0.1, 0.3))):
                xb = x.loc[x.c.isin(cs)]
                add('C3', f'value per question ({crit}), {band}', q, xb.value_per_question, 'results/value.csv', 'B infinite')
            add('C3', f'cells with positive net value ({crit})', q,
                [int((x.net_value_per_question > 0).sum())], 'results/value.csv',
                f'of {int(x.value_per_question.notna().sum())} cells with questions')
    x = vr.loc[vr.criterion.isin(['edv', 'ig', 'random'])]
    add('C3', 'value per question, all realistic queues and criteria', 'all', x.value_per_question, 'results/value.csv')
    piv = x.pivot_table(index=['queue', 'r', 'c'], columns='criterion', values='net_value_per_question').dropna()
    add('C3', 'cells where the sign of the net value differs between EDV, IG and random', 'all',
        [int((np.sign(piv).nunique(axis=1) > 1).sum())], 'results/value.csv', f'of {len(piv)} cells with questions')
    add('C3', 'value per question in minutes, m = 10 to 40', 'all',
        list(x.value_per_question * 10) + list(x.value_per_question * 40), 'results/minutes.csv')
    vw = pd.read_csv(OUT / 'results' / 'value_windows.csv')
    vw = vw.loc[(vw.dataset == 'all_tested') & vw.r.isin(RS) & vw.criterion.isin(['edv', 'ig', 'random'])]
    add('C3', 'value per question on windows, unfiltered IG, random, fixed order', 'all test windows',
        vw.loc[(vw['mode'] == 'unfiltered') & vw.criterion.isin(['ig', 'random']), 'value_per_question'], 'results/value_windows.csv')
    add('C3', 'value per question on windows, unfiltered EDV', 'all test windows',
        vw.loc[(vw['mode'] == 'unfiltered') & (vw.criterion == 'edv'), 'value_per_question'], 'results/value_windows.csv')
    add('C3', 'questions on windows, unfiltered IG and random', 'all test windows',
        vw.loc[(vw['mode'] == 'unfiltered') & vw.criterion.isin(['ig', 'random']), 'questions_known'], 'results/value_windows.csv')
    add('C3', 'value per question on windows, filtered', 'all test windows',
        vw.loc[(vw['mode'] == 'filtered') & vw.c.isin(CS), 'value_per_question'], 'results/value_windows.csv')
    # calibration
    cal = pd.read_csv(OUT / 'results' / 'calibration.csv', dtype={'queue': str, 'B': str})
    cr = cal.loc[(cal.unit == 'alarms') & cal.queue.isin(QUEUES) & cal.r.isin(RS) & cal.c.isin(CS) & (cal.B == 'inf') &
                 cal.criterion.isin(['edv', 'ig', 'random'])]
    add('C3', 'expected value per question (model)', 'realistic alarms', cr.expected_value_per_question, 'results/calibration.csv')
    add('C3', 'realised value per question (labels)', 'realistic alarms', cr.value_per_question, 'results/calibration.csv')
    ok = cr.dropna(subset=['value_per_question'])
    add('C3', 'cells where the realised value is negative (expected always positive)', 'realistic alarms',
        [int((ok.value_per_question < 0).sum())], 'results/calibration.csv', f'of {len(ok)} cells; expected > 0 in {int((ok.expected_value_per_question > 0).sum())}')
    add('C3', 'cells where realised is below expected', 'realistic alarms',
        [int((ok.value_per_question < ok.expected_value_per_question).sum())], 'results/calibration.csv',
        f'of {len(ok)} cells')
    # budget: decision loss with B = 1 against no question and B infinite, r = 10, c = 0.01
    res = pd.read_parquet(OUT / 'results' / 'replay.parquet')
    rr = res.loc[res.dataset.isin(GROUPS) & (res.r == 10) & (res.c == 0.01)].groupby(['queue', 'B', 'criterion'], as_index=False)[['D']].sum()
    for q in QUEUES:
        x = rr.loc[rr.queue == q]
        none = float(x.loc[x.criterion == 'none', 'D'].iloc[0])
        for b in ('1', 'inf'):
            add('C3', f'decision loss D, B = {b}, EDV, IG, random (r = 10, c = 0.01)', q,
                x.loc[(x.B == b) & x.criterion.isin(['edv', 'ig', 'random']), 'D'], 'results/replay.parquet',
                f'no question: {none:g}')
    # C4: attacks needed (model based)
    pw = pd.read_csv(OUT / 'results' / 'power_realistic.csv', dtype={'queue': str})
    pw = pw.loc[pw.p_source == 'final_all']
    for pair in ('edv-random', 'ig-random', 'edv-ig', 'edv-none'):
        x = pw.loc[pw.pair == pair]
        add('C4', f'median distinct attacks needed, {pair}', 'by queue', x.attacks_needed_median, 'results/power_realistic.csv')
        add('C4', f'cells needing more attacks than observed, {pair}', 'by queue', x.cells_needing_more_than_observed,
            'results/power_realistic.csv', 'of 90 cells per queue')
    add('C4', 'observed distinct attacks', 'by queue', pw.loc[pw.pair == 'edv-random', 'observed_attacks'], 'results/power_realistic.csv')
    out = pd.DataFrame(ROWS)
    out.to_csv(OUT / 'results' / 'claims.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(out.drop(columns=['source']).to_string())


if __name__ == '__main__':
    main()
