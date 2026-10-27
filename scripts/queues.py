"""Daily alarm queues and the starts of the alarms.

Step 1 (no label). Each window's category follows from its Velox score and the thresholds of its collection
(data/velox_thresholds.csv: q99 and vmax of the scores on the validation days): unscored, low below q99, high
from q99 to vmax, alert above vmax; the stored categories are checked against this rule. On each UTC day of each
collection and split, the N scored windows with the highest Velox score form the daily queue of N alarms a day
(ties by subject, then window start), for N = 1, 3, 10, 30, 100; the alert queue holds every alert window.

Step 2 (labels of the source collections only). The start of a tested queue is the Laplace estimate
p_N = (a_N + 1) / (n_N + 2) over the same queue built on the validation and test splits of the source
collections of the fold, with n_N labelled source alarms and a_N attack windows among them.

    python scripts/queues.py

Writes results/queues.csv, results/day_counts.csv, results/queue_membership.parquet and results/starts.csv.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import DATA, DATASETS, OUT, QUEUES, SPLITS, folds, read_labels

CATEGORIES = ('low', 'high', 'alert')
KEY = ['subject_id', 'window_start_ns']
MEMBERSHIP_COLUMNS = ['dataset', 'split', 'queue', 'day'] + KEY + ['velox_score', 'category']


def check_categories():
    """Every stored category equals the threshold rule applied to the stored score."""
    th = pd.read_csv(DATA / 'velox_thresholds.csv').set_index('dataset')
    for dataset in DATASETS:
        q99, vmax = float(th.loc[dataset, 'q99']), float(th.loc[dataset, 'vmax'])
        for split in SPLITS:
            t = pq.read_table(DATA / 'windows' / dataset / f'{split}.parquet', columns=['velox_score', 'category']).to_pydict()
            rule = ['unscored' if x is None else 'low' if x < q99 else 'high' if x <= vmax else 'alert'
                    for x in t['velox_score']]
            if rule != t['category']:
                raise AssertionError(f'Category rule differs in {dataset}/{split}')


def build_queues():
    """Step 1. Read only score, category and identity columns, without labels."""
    summaries, memberships, days = [], [], []
    for path in sorted((DATA / 'windows').glob('*/*.parquet')):
        dataset, split = path.parent.name, path.stem
        if split not in SPLITS:
            continue
        table = pq.read_table(path, columns=KEY + ['velox_score', 'category']).to_pandas()
        scored = table.loc[table.velox_score.notna()].copy()
        if not np.isfinite(scored.velox_score).all():
            raise AssertionError(f'Nonfinite score in {dataset}/{split}')
        if scored.duplicated(KEY).any():
            raise AssertionError(f'Duplicate score key in {dataset}/{split}')
        if not set(scored.category).issubset(CATEGORIES):
            raise AssertionError(f'Scored category invalid in {dataset}/{split}')
        scored['day'] = pd.to_datetime(scored.window_start_ns, unit='ns', utc=True).dt.strftime('%Y-%m-%d')
        for day, daily in scored.groupby('day', sort=True):
            count = daily.category.value_counts()
            days.append(dict(dataset=dataset, split=split, day=day, scored_windows=len(daily),
                             min_window_start_ns=int(daily.window_start_ns.min()),
                             max_window_start_ns=int(daily.window_start_ns.max()),
                             **{category: int(count.get(category, 0)) for category in CATEGORIES}))
        scored = scored.sort_values(['day', 'velox_score', 'subject_id', 'window_start_ns'],
                                    ascending=[True, False, True, True], kind='mergesort')
        scored['rank_in_day'] = scored.groupby('day', sort=False).cumcount() + 1
        for queue in QUEUES:
            take = scored.category.eq('alert') if queue == 'alert' else scored.rank_in_day.le(int(queue))
            selected = scored.loc[take].copy()
            counts = selected.category.value_counts()
            summaries.append(dict(dataset=dataset, split=split, queue=queue, N=queue,
                                  scored_windows=len(scored), alarms=len(selected), days=scored.day.nunique(),
                                  queue_days=selected.day.nunique(),
                                  **{category: int(counts.get(category, 0)) for category in CATEGORIES}))
            selected = selected.assign(dataset=dataset, split=split, queue=queue)
            memberships.append(selected[MEMBERSHIP_COLUMNS])
    if not summaries:
        raise RuntimeError('No score partitions found')
    summary = pd.DataFrame(summaries)
    membership = pd.concat(memberships, ignore_index=True)
    if membership.duplicated(['dataset', 'split', 'queue'] + KEY).any():
        raise AssertionError('Duplicate queue membership')
    (OUT / 'results').mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / 'results/queues.csv', index=False)
    pd.DataFrame(days).to_csv(OUT / 'results/day_counts.csv', index=False)
    membership.to_parquet(OUT / 'results/queue_membership.parquet', index=False)
    return summary, membership


def source_starts(membership):
    """Step 2. Labels of the source collections of each fold only."""
    rows = []
    for fold in folds():
        allowed = {(entry['dataset'], split) for entry in fold['fitting'] for split in entry['splits']}
        target = {entry['dataset'] for entry in fold['target_evaluation']}
        if target.intersection(ds for ds, _ in allowed):
            raise AssertionError('Source and target overlap within a fold')
        accum = {queue: Counter() for queue in QUEUES}
        for dataset, split in sorted(allowed):
            labels = read_labels(dataset, split)
            selected = membership.loc[membership.dataset.eq(dataset) & membership.split.eq(split)]
            for queue in QUEUES:
                alarms = selected.loc[selected.queue.eq(queue)]
                counts = accum[queue]
                counts['alarms'] += len(alarms)
                for alarm in alarms.itertuples(index=False):
                    label, _ = labels[(dataset, split, int(alarm.subject_id), int(alarm.window_start_ns))]
                    if label not in (-1, 0, 1):
                        raise AssertionError('Unexpected label value')
                    counts['unknown' if label == -1 else 'labelled'] += 1
                    counts['attacks'] += label == 1
        for queue in QUEUES:
            count = accum[queue]
            rows.append(dict(fold=fold['id'], queue=queue, N=queue, source_alarms=count['alarms'],
                             source_labelled=count['labelled'], source_attack_windows=count['attacks'],
                             source_unknown=count['unknown'], p_N=(count['attacks'] + 1) / (count['labelled'] + 2),
                             source_attack_share=count['attacks'] / count['labelled'] if count['labelled'] else None))
    starts = pd.DataFrame(rows)
    starts.to_csv(OUT / 'results/starts.csv', index=False, float_format='%.17g')
    return starts


def main():
    check_categories()
    queues, membership = build_queues()
    print(f'Step 1: {len(queues)} queue summaries, {len(membership)} membership rows; no label read')
    starts = source_starts(membership)
    print(f'Step 2: {len(starts)} starts from the source collections of each fold')


if __name__ == '__main__':
    main()
