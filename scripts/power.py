"""How many alarms and distinct attacks would be needed to rank two criteria? (model based)

The decisions of each criterion on each alarm are fixed by the recorded answers (results/by_alarm.parquet). Under
the model, the label of alarm i is drawn as Y_i ~ Bernoulli(p_i), with p_i the model's probability after all the
alarm's recorded answers (final_all; sensitivity: the start). The loss of a criterion with decision d on the alarm
is r (1 - d) Y + d (1 - Y), so the difference between two criteria X and Y is zero where they agree and
+-(1 - (1 + r) Y) where they disagree; its mean and variance follow in closed form (random: mean over seeds, a
fractional decision). Treating alarms as draws from the cell's queue, the number of alarms for which a two sided
test at 5% has power 0.8 is n = ((1.96 + 0.8416) sigma / |mu|)^2, with mu and sigma^2 the mean and variance of the
per alarm difference over the cell. Expected attack windows: n times the mean of p. Distinct attacks: attack
windows divided by the attack windows per annotated attack observed in the same queues (a labelled count,
declared as a conversion). A Monte Carlo draw checks the closed form.

    python scripts/power.py

Writes results/power.csv and checks/power_check.json.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, now, rel, write_json

SCRIPT = rel(__file__)
SEEDS = range(10)
Z = 1.959963984540054 + 0.8416212335729143
BUDGETS = (1, 2, 3, 5, 10, None)
PAIRS = (('edv', 'ig'), ('edv', 'random'), ('ig', 'random'), ('edv', 'order_fnp'), ('edv', 'none'))
QUEUES = ('1', '3', '10', '30', '100', 'alert')
RNG = np.random.default_rng(20261024)


def decisions(g, crit, b):
    if crit == 'none':
        return g.decision_none.to_numpy(dtype=float)
    asked = g.has_admissible.to_numpy() & (True if b is None else (g.adm_rank < b).to_numpy())
    names = [f'random_{s}' for s in SEEDS] if crit == 'random' else [crit]
    return np.mean([np.where(asked, g[f'd_{n}'].to_numpy(), g.decision_none.to_numpy()) for n in names], axis=0)


def moments(dx, dy, p, r):
    """Per alarm mean and variance of the loss difference under Y ~ Bernoulli(p)."""
    # loss(d) = r (1 - d) Y + d (1 - Y) = d + Y (r - (1 + r) d): linear in Y with slope r - (1 + r) d
    slope = (1 + r) * (dy - dx)            # slope of loss_X - loss_Y in Y
    const = dx - dy
    return const + slope * p, slope ** 2 * p * (1 - p)


def main():
    # labelled counts are used only for the observed attack windows and annotated attacks per queue
    by = pd.read_parquet(OUT / 'results' / 'by_alarm.parquet')
    by = by.loc[by.label.isin([0, 1])]          # alarms of known label, as in the loss
    rows, mc = [], []
    for (queue, r, c), g in by.groupby(['queue', 'r', 'c'], sort=False):
        att = g.label == 1
        windows_obs = int(att.sum())
        groups_obs = len({(ds, x) for ds, s in zip(g.dataset[att], g.attack_groups[att]) for x in json.loads(s)})
        per_attack = windows_obs / groups_obs if groups_obs else np.nan
        for source in ('final_all', 'start'):
            p = g[source].to_numpy()
            for b in BUDGETS:
                for x, y in PAIRS:
                    dx, dy = decisions(g, x, b), decisions(g, y, b)
                    m, v = moments(dx, dy, p, r)
                    mu = float(m.mean())
                    sigma2 = float(v.mean() + m.var())
                    n_needed = np.inf if abs(mu) < 1e-15 else Z ** 2 * sigma2 / mu ** 2
                    share = float(p.mean())
                    windows_needed = n_needed * share
                    rows.append(dict(queue=queue, r=int(r), c=float(c), B='inf' if b is None else str(b), pair=f'{x}-{y}',
                                     p_source=source, alarms=len(g), disagreeing_alarms=float(np.abs(dx - dy).sum()),
                                     mean_diff_per_alarm=mu, sd_diff_per_alarm=float(np.sqrt(sigma2)),
                                     model_attack_share=share, alarms_needed=n_needed,
                                     attack_windows_needed=windows_needed,
                                     observed_attack_windows=windows_obs, observed_attacks=groups_obs,
                                     attack_windows_per_attack=per_attack,
                                     attacks_needed=windows_needed / per_attack if per_attack else np.nan,
                                     needed_over_observed_attacks=(windows_needed / per_attack / groups_obs)
                                     if groups_obs else np.nan))
                    # Monte Carlo check of the closed form on a few cells
                    if source == 'final_all' and b is None and r == 10 and c == 0.01 and queue in ('3', '10', 'alert'):
                        draws = RNG.random((20000, len(p))) < p
                        lx = r * (1 - dx) * draws + dx * (1 - draws)
                        ly = r * (1 - dy) * draws + dy * (1 - draws)
                        diff = (lx - ly).sum(axis=1)
                        mc.append(dict(queue=queue, pair=f'{x}-{y}', closed_mean=float(m.sum()), mc_mean=float(diff.mean()),
                                       closed_var=float(v.sum()), mc_var=float(diff.var())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'results' / 'power.csv', index=False, float_format='%.8g')
    ok = all(abs(t['mc_mean'] - t['closed_mean']) <= 4 * np.sqrt(max(t['closed_var'], 1e-12) / 20000) + 1e-9 and
             (t['closed_var'] == 0 or abs(t['mc_var'] / t['closed_var'] - 1) < 0.05) for t in mc)
    write_json(OUT / 'checks' / 'power_check.json', dict(generated=now(), script=SCRIPT, rows=len(res),
                                                         monte_carlo=mc, passed=ok,
                                                         note='Model based: labels drawn from the model probabilities.'))
    print('rows', len(res), 'monte carlo ok', ok)


if __name__ == '__main__':
    main()
