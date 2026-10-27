"""Fit the answer model of each fold on its source collections.

A fold tests one group (DARPA TC, OpTC host 051 or ATLASv2) and fits on the validation and test splits of the
other two. Fitting uses the labelled windows outside the first 24 hours of each collection (warm up). For every
category the prior is (a + 1) / (a + b + 2) with a attack and b benign source windows; a category keeps its own
prior only with at least 5 attack and 100 benign windows, and an alert without its own prior is counted as high.
For each set of answerable questions (mask) the joint answer counts are tabulated per class, pooled over all
categories and, where a category keeps its own prior and has at least 20 attack and 100 benign windows, per
category. The smoothing strength is tau = 10,000.

    python scripts/fit_models.py

Writes models/<fold>.json.
"""
import json

import numpy as np

from answer_model import CONTEXTS, mask_key, symbol_key
from common import ACTIONS, MODELS, folds, load_partition, read_labels

TAU = 10000
CTX_MIN_POS, CTX_MIN_COMP = 5, 100
LIK_MIN_POS, LIK_MIN_COMP = 20, 100
SYM_MIN = 20
AVAILABILITY = {'CADETS_E3': ('file', 'network'), 'CLEARSCOPE_E3': ACTIONS, 'THEIA_E3': ACTIONS,
                'optc_h051': ACTIONS, 'ATLASV2_EDR': ACTIONS}


class FittingData:
    """Labelled windows outside the warm up of the source collections of one fold."""

    def __init__(self, fold):
        label, context, datasets = [], [], []
        state = {a: [] for a in ACTIONS}
        for entry in fold['fitting']:
            ds = entry['dataset']
            for split in entry['splits']:
                labels = read_labels(ds, split)
                for w in load_partition(ds, split):
                    y, _ = labels[tuple(w.key)]
                    if y not in (0, 1) or w.warm_up:
                        continue
                    if w.available != AVAILABILITY[ds]:
                        raise ValueError('Availability mismatch')
                    label.append(y)
                    context.append(w.category)
                    datasets.append(ds)
                    for a, s in zip(ACTIONS, w.responses):
                        state[a].append(s)
        self.label = np.array(label, dtype=np.int8)
        self.context = np.array(context, dtype=object)
        self.datasets = np.array(datasets, dtype=object)
        self.state = {a: np.array(v, dtype=object) for a, v in state.items()}

    def __len__(self):
        return len(self.label)


def structural_masks():
    return sorted({tuple(a for a in ACTIONS if a in v) for v in AVAILABILITY.values()}, key=lambda m: (-len(m), m))


def build_mask(R, mask, ctx, strata_contexts):
    """Joint counts over every known fitting window whose availability covers the mask, after the merge rule."""
    avail = np.ones(len(R), bool)
    for a in mask:
        avail &= R.state[a] != 'unavailable'
    pop = np.flatnonzero(avail)
    y = R.label[pop]
    raw_counts, merged, syms = {}, {}, {}
    for a in mask:
        s = R.state[a][pop]
        raw_counts[a] = {k: dict(positive=int(((s == k) & (y == 1)).sum()), comparison=int(((s == k) & (y == 0)).sum()), total=int((s == k).sum()))
                         for k in ('empty', 'known', 'novel')}
        merged[a] = bool(raw_counts[a]['known']['total'] < SYM_MIN or raw_counts[a]['novel']['total'] < SYM_MIN)
        syms[a] = np.where(np.isin(s, ['known', 'novel']), 'nonempty', s) if merged[a] else s
    alph = {a: sorted(set(syms[a].tolist()), key=symbol_key) for a in mask}
    index = {a: {s: j for j, s in enumerate(v)} for a, v in alph.items()}
    shape = tuple(len(alph[a]) for a in mask)
    coords, marg = [], {}
    for a in mask:
        c = np.array([index[a][s] for s in syms[a]], dtype=np.int64)
        coords.append(c)
        marg[a] = (np.bincount(c, minlength=len(alph[a])) / max(len(pop), 1)).tolist()
    size = int(np.prod(shape)) if len(pop) and all(shape) else 0
    flat = np.ravel_multi_index(coords, shape) if size else np.zeros(0, np.int64)

    def table(sel):
        return {yy: np.bincount(flat[sel & (y == yy)], minlength=size).reshape(shape).tolist() if size else []
                for yy in (0, 1)}

    strata = {'pooled': table(np.ones(len(pop), bool))}
    support = dict(pooled=dict(positive=int((y == 1).sum()), comparison=int((y == 0).sum())), raw_symbol_counts=raw_counts, merged=merged)
    cpop = ctx[pop]
    for c in CONTEXTS:
        sel = cpop == c
        p, n = int(((y == 1) & sel).sum()), int(((y == 0) & sel).sum())
        own = c in strata_contexts and p >= LIK_MIN_POS and n >= LIK_MIN_COMP
        support[f'context:{c}'] = dict(positive=p, comparison=n, prior_supported=c in strata_contexts, own_stratum=own)
        if own:
            strata[f'context:{c}'] = table(sel)
    tp = int(np.asarray(strata['pooled'][1]).sum()) if size else 0
    tn = int(np.asarray(strata['pooled'][0]).sum()) if size else 0
    cons = dict(covering_known_cases=int(avail.sum()), table_cases=tp + tn, covering_positive=int((R.label[avail] == 1).sum()),
                table_positive=tp, covering_comparison=int((R.label[avail] == 0).sum()), table_comparison=tn)
    cons['conserved'] = cons['covering_known_cases'] == cons['table_cases'] and cons['covering_positive'] == tp and cons['covering_comparison'] == tn
    if not cons['conserved']:
        raise RuntimeError('Mask table does not conserve the covering fitting population')
    disabled = support['pooled']['positive'] == 0 or support['pooled']['comparison'] == 0 or size == 0
    return dict(mask=list(mask), alphabets=alph, merged=merged, marginals=marg, strata=strata, disabled=bool(disabled),
                population=cons, support=support)


def fit_fold(fold):
    F = FittingData(fold)
    N1, N0 = int((F.label == 1).sum()), int((F.label == 0).sum())
    if set(F.context) - set(CONTEXTS):
        raise ValueError('Unknown category in fitting data')
    global_prior = (N1 + 1) / (N1 + N0 + 2)
    context_map = {}
    sel = F.context == 'alert'
    if int((F.label[sel] == 1).sum()) < CTX_MIN_POS or int((F.label[sel] == 0).sum()) < CTX_MIN_COMP:
        context_map = {'alert': 'high'}         # an alert without its own prior is counted as high
        F.context = np.where(sel, 'high', F.context).astype(object)
    priors = {}
    for c in CONTEXTS:
        sel = F.context == c
        p, n = int((F.label[sel] == 1).sum()), int((F.label[sel] == 0).sum())
        sup = p >= CTX_MIN_POS and n >= CTX_MIN_COMP
        priors[c] = dict(positive=p, comparison=n, supported=sup, prior=(p + 1) / (p + n + 2) if sup else global_prior,
                         laplace=(p + 1) / (p + n + 2))
    strata_contexts = [c for c in CONTEXTS if priors[c]['supported']]
    masks = {mask_key(m): build_mask(F, m, F.context, strata_contexts) for m in structural_masks()}
    return dict(branch='history_v1', fold=fold['id'], context_symbols=list(CONTEXTS), context_priors=priors,
                global_prior=global_prior, masks=masks, tau=TAU, context_map=context_map)


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    for fold in folds():
        params = fit_fold(fold)
        (MODELS / f'{fold["id"]}.json').write_text(json.dumps(params, indent=1, allow_nan=False) + '\n')
        print(fold['id'], 'priors', {c: round(v['prior'], 5) for c, v in params['context_priors'].items()},
              'map', params['context_map'], flush=True)


if __name__ == '__main__':
    main()
