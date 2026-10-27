"""Express the currency of the study (false escalations) as equivalent analyst-time costs.

Questions are automated evidence actions. Multiplication by m changes cost units; it does not simulate human
participation in questions or elapsed execution time. Value and net value include weighted missed attacks,
so their minute equivalents are not measured human time savings. B*c*m values the full daily action allowance,
not actual spending when fewer actions are selected.

m minutes per false escalation: 10 (expert estimate of Kersten et al. 2023; lower end of Li et al. 2026), 20
(assumed by Hassan et al. 2019 and Sharif et al. 2024; geometric centre of 10 to 40), 40 (upper end quoted by
Hassan et al. 2019, Sharif et al. 2024 and Kim et al. 2025). For every cell: price c*m, value per question v*m,
net value (v - c)*m, break even price v*m, daily budget B*c*m (finite B). Alarms: sums over the three held out
groups (results/value.csv); ordinary windows: all tested windows (results/value_windows.csv, dataset all_tested).

    python scripts/minutes.py

Writes results/minutes.csv and checks/minutes_check.json.
"""
import numpy as np
import pandas as pd

from common import OUT, now, rel, write_json

SCRIPT = rel(__file__)
M = {10: 'expert estimate of tier 1 assessment time (Kersten et al. 2023); lower end of Li et al. 2026',
     20: 'assumed by Hassan et al. 2019 and Sharif et al. 2024; geometric centre of 10 to 40',
     40: 'upper end quoted by Hassan et al. 2019, Sharif et al. 2024 and Kim et al. 2025'}


def main():
    a = pd.read_csv(OUT / 'results' / 'value.csv', dtype={'queue': str, 'B': str})
    a['unit'] = 'alarms'
    w = pd.read_csv(OUT / 'results' / 'value_windows.csv')
    w = w.loc[(w.dataset == 'all_tested') & ~w.criterion.isin(['none'])].copy()
    w['unit'] = 'windows_' + w['mode']
    w['queue'], w['B'] = 'windows', 'inf'
    cols = ['unit', 'queue', 'r', 'c', 'B', 'criterion', 'questions_known', 'value_per_question',
            'net_value_per_question', 'break_even_price']
    out = pd.concat([a[cols], w[cols]], ignore_index=True)
    # Cost equivalent of using the whole finite allowance; not an analyst-time limit in the replay.
    budget = pd.to_numeric(out.B.replace({'inf': np.nan, 'any': np.nan}), errors='coerce')
    for m in M:
        out[f'price_min_m{m}'] = out.c * m
        out[f'value_min_m{m}'] = out.value_per_question * m
        out[f'net_value_min_m{m}'] = out.net_value_per_question * m
        out[f'break_even_min_m{m}'] = out.break_even_price * m
        out[f'budget_min_day_m{m}'] = budget * out.c * m
    out.to_csv(OUT / 'results' / 'minutes.csv', index=False, float_format='%.8g')
    check = max(float((out[f'net_value_min_m{m}'] - (out[f'value_min_m{m}'] - out[f'price_min_m{m}'])).abs().max())
                for m in M)
    write_json(OUT / 'checks' / 'minutes_check.json',
               dict(generated=now(), script=SCRIPT, m_minutes=M, rows=len(out),
                    net_equals_value_minus_price_max_abs_difference=check,
                    tolerance=1e-6, tolerance_reason='inputs are CSV values with 10 significant digits, times m up to 40',
                    passed=check <= 1e-6))
    print(len(out), check)


if __name__ == '__main__':
    main()
