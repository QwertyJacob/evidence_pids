"""Summary of results/power.csv over the realistic cells (queues 1, 3, 10, 30 a day and the alerts; r = 5, 10, 20;
c = 0.001 to 0.3); no label read.

    python scripts/power_summary.py
"""
import numpy as np
import pandas as pd

from common import OUT

REAL = dict(queue=('1', '3', '10', '30', 'alert'), r=(5, 10, 20), c=(0.001, 0.01, 0.03, 0.1, 0.3))


def main():
    p = pd.read_csv(OUT / 'results' / 'power.csv', dtype={'queue': str, 'B': str})
    s = p.loc[p.queue.isin(REAL['queue']) & p.r.isin(REAL['r']) & p.c.isin(REAL['c'])]
    rows = []
    for (pair, source, queue), g in s.groupby(['pair', 'p_source', 'queue']):
        fin = g.loc[np.isfinite(g.attacks_needed)]
        rows.append(dict(pair=pair, p_source=source, queue=queue, cells=len(g),
                         cells_no_difference=int((~np.isfinite(g.attacks_needed)).sum()),
                         observed_attacks=int(g.observed_attacks.iloc[0]),
                         attacks_needed_min=float(fin.attacks_needed.min()) if len(fin) else np.inf,
                         attacks_needed_median=float(fin.attacks_needed.median()) if len(fin) else np.inf,
                         attacks_needed_max=float(fin.attacks_needed.max()) if len(fin) else np.inf,
                         cells_needing_more_than_observed=int((g.attacks_needed > g.observed_attacks).sum())))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / 'results' / 'power_realistic.csv', index=False, float_format='%.6g')
    pd.set_option('display.width', 250)
    print(out.loc[out.p_source == 'final_all'].round(1).to_string())
    print(out.loc[out.p_source == 'start'].round(1).to_string())


if __name__ == '__main__':
    main()
