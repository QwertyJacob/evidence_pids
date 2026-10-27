"""The answer model: how often each answer occurs among attack and benign windows of each detector category.

Categories are the four Velox categories (unscored, low, high, alert); answers are empty, known or novel. For one
set of available questions (a mask) the model keeps joint counts of the answers, per class, in a stratum (one
category, or all categories pooled). With fitting counts n_y and totals N_y, N = N_1 + N_0, the likelihood of an
event E (a set of answers) is

    L_y(E) = (1 - lam) * n_y(E) / N_y + lam * B(E),   lam = tau / (N + tau),

where B is the product of the pooled marginal frequencies of the answers in E over every source window of the
mask. If n_1(E) = 0 then L_1(E) <= L_0(E): the likelihood ratio is at most one.

In each mask table a question whose known or novel answer has fewer than twenty covering source windows is
merged, so known and novel both become the single cell 'nonempty'. A realised answer absent from the merged
alphabet conditions on 'nonempty' when that cell exists, otherwise the question is ignored. Every conditioning
event of a question is one cell of a partition, so predictive probabilities sum to one and the predictive mean of
the updated probability equals the current probability.

ShiftedBelief binds the model to one start (the prior of an alarm's queue) instead of the category prior, keeping
the same likelihoods; it is the belief view used by every criterion of the replay.
"""
import json
import math
from pathlib import Path

import numpy as np

ACTIONS = ('file', 'network', 'process')
CONTEXTS = ('unscored', 'low', 'high', 'alert')
RESPONSES = ('empty', 'known', 'novel')
ORDER = {'empty': 0, 'known': 1, 'novel': 2, 'nonempty': 3}


def symbol_key(s):
    return ORDER[s]


def mask_key(actions):
    return ','.join(a for a in ACTIONS if a in actions)


class MaskTable:
    """Joint counts for one mask over the observed (possibly merged) cells."""

    def __init__(self, mask, alphabets, merged, marginals, strata, disabled, population=None, support=None):
        self.mask = tuple(mask)
        self.alphabets = {a: list(v) for a, v in alphabets.items()}
        self.merged = {a: bool(merged[a]) for a in self.mask}
        self.index = {a: {s: i for i, s in enumerate(v)} for a, v in self.alphabets.items()}
        self.marginals = {a: np.asarray(v, dtype=np.float64) for a, v in marginals.items()}
        self.strata = {k: {y: np.asarray(t, dtype=np.int64) for y, t in v.items()} for k, v in strata.items()}
        self.disabled = bool(disabled)
        self.population = population
        self.support = support
        self.shape = tuple(len(self.alphabets[a]) for a in self.mask)

    @classmethod
    def from_json(cls, d):
        return cls(d['mask'], d['alphabets'], d['merged'], d['marginals'],
                   {k: {int(y): t for y, t in v.items()} for k, v in d['strata'].items()}, d['disabled'],
                   d.get('population'), d.get('support'))

    def cell(self, action, symbol):
        """Map a realised or predicted symbol to (cell index or None, component)."""
        if symbol not in ORDER:
            raise ValueError(f'Unknown response symbol {symbol!r}')
        idx = self.index[action]
        s = symbol
        if self.merged[action] and s in ('known', 'novel'):
            s = 'nonempty'
        if s in idx:
            return idx[s], ('direct' if s == symbol else 'merged')
        if s in ('known', 'novel') and 'nonempty' in idx:
            return idx['nonempty'], 'merged'
        return None, 'ignored'

    def event(self, history):
        sets, comp = {}, {}
        for a, s in history.items():
            if a not in self.index:
                raise ValueError(f'Action {a} outside mask')
            j, c = self.cell(a, s)
            comp[a] = c
            if j is not None:
                sets[a] = [j]
        return sets, comp

    def event_count(self, table, sets):
        sub = table[np.ix_(*[np.asarray(sets[a]) if a in sets else np.arange(n) for a, n in zip(self.mask, self.shape)])]
        return float(sub.sum())


class AnswerModel:
    def __init__(self, params):
        self.params = params
        self.masks = {k: MaskTable.from_json(v) for k, v in params['masks'].items()}
        self.tau = float(params['tau'])
        self.priors = params['context_priors']
        self.global_prior = params['global_prior']
        self.context_map = params.get('context_map', {})       # an alert without source support maps to high

    def with_tau(self, tau):
        p = dict(self.params)
        p['tau'] = tau
        return AnswerModel(p)

    def _mask(self, available_actions):
        k = mask_key(available_actions)
        if set(k.split(',')) != set(available_actions):
            raise ValueError('Unknown action in mask')
        if k not in self.masks:
            raise ValueError(f'No fitted table for mask {k}')
        return self.masks[k]

    def _prior_stratum(self, c):
        c = self.context_map.get(c, c)
        return 'context' if self.priors.get(str(c), {}).get('supported') else 'global'

    def _prior(self, c):
        c = self.context_map.get(c, c)
        r = self.priors.get(str(c))
        if r is None:
            raise ValueError(f'Unknown context symbol {c!r}')
        return r['prior'] if r['supported'] else self.global_prior

    def _stratum(self, c, mt):
        c = self.context_map.get(c, c)
        key = f'context:{c}'
        if self._prior_stratum(c) == 'context' and key in mt.strata:
            return key
        return 'pooled'

    def _event_likelihoods(self, mt, stratum, sets):
        t = mt.strata[stratum]
        n1, n0 = mt.event_count(t[1], sets), mt.event_count(t[0], sets)
        N1, N0 = float(t[1].sum()), float(t[0].sum())
        lam = self.tau / (N1 + N0 + self.tau)
        B = 1.0
        for a, idx in sets.items():
            B *= float(mt.marginals[a][idx].sum())
        return (1 - lam) * n1 / N1 + lam * B, (1 - lam) * n0 / N0 + lam * B, n1, n0

    def initial(self, context_symbol):
        return self._prior(context_symbol)

    def _conditioned(self, context_symbol, available_actions, history):
        for a in history:
            if a not in available_actions:
                raise ValueError('History action not available')
        mt = self._mask(available_actions)
        prior = self._prior(context_symbol)
        info = dict(prior=prior, stratum='disabled', components={}, l1=None, l0=None, n1=None, n0=None)
        if mt.disabled:
            info['components'] = {a: 'disabled' for a in history}
            return prior, info
        sets, comp = mt.event(history)
        info['components'] = comp
        stratum = self._stratum(context_symbol, mt)
        info['stratum'] = stratum
        if not sets:
            return prior, info
        l1, l0, n1, n0 = self._event_likelihoods(mt, stratum, sets)
        info.update(l1=l1, l0=l0, n1=n1, n0=n0)
        if l1 == l0:
            return prior, info
        num = prior * l1
        den = num + (1 - prior) * l0
        if not den > 0:
            return prior, info
        q = num / den
        if n1 == 0:              # exact shared shrinkage direction despite rounding
            q = min(prior, q)
        return q, info

    def posterior(self, context_symbol, available_actions, history):
        return self._conditioned(context_symbol, available_actions, dict(history))[0]

    def predict(self, context_symbol, available_actions, history, action):
        """[(cell symbol, predictive probability, posterior after)] over the mask alphabet of `action`."""
        history = dict(history)
        if action not in available_actions or action in history:
            raise ValueError('Infeasible predicted action')
        mt = self._mask(available_actions)
        if mt.disabled or not mt.alphabets[action]:
            return []
        q, info = self._conditioned(context_symbol, available_actions, history)
        sets, _ = mt.event(history)
        stratum = info['stratum']
        base1, base0 = (self._event_likelihoods(mt, stratum, sets)[:2]) if sets else (1.0, 1.0)
        out = []
        for s in mt.alphabets[action]:
            s2 = dict(sets)
            s2[action] = [mt.index[action][s]]
            l1, l0, _, _ = self._event_likelihoods(mt, stratum, s2)
            p = q * l1 / base1 + (1 - q) * l0 / base0
            out.append((s, p, self.posterior(context_symbol, available_actions, {**history, action: s})))
        return out


class ShiftedBelief:
    """One start and category, preserving the joint likelihoods of the fitted model.

    A history H has probability p L1(H) + (1-p) L0(H), where p is the chosen start and Ly(H) is a stored joint
    event after the shrinkage. Prediction conditions those same joint events on H. The cell mapper keeps merged
    or ignored realised answers, and the zero attack count clamp of the model is preserved.
    """

    def __init__(self, model, context, available, start):
        if not math.isfinite(start) or not 0 <= start <= 1:
            raise ValueError('Start must be a finite probability')
        self.model = model
        self.context, self.available, self.start = context, tuple(available), float(start)
        self.table = model._mask(self.available)
        self.stratum = model._stratum(context, self.table)
        self._posterior, self._predict = {}, {}

    def posterior(self, history):
        key = tuple(dict(history).items())
        if key in self._posterior:
            return self._posterior[key]
        if any(action not in self.available for action in history):
            raise ValueError('History action unavailable')
        p = self.start
        if self.table.disabled:
            q = p
        else:
            sets, _ = self.table.event(dict(history))
            if not sets:
                q = p
            else:
                l1, l0, n1, _ = self.model._event_likelihoods(self.table, self.stratum, sets)
                numerator = p * l1
                denominator = numerator + (1 - p) * l0
                q = p if l1 == l0 or denominator <= 0 else numerator / denominator
                if n1 == 0:
                    q = min(p, q)
        self._posterior[key] = float(q)
        return float(q)

    def predict(self, history, action):
        history = dict(history)
        key = (tuple(history.items()), action)
        if key in self._predict:
            return self._predict[key]
        if action not in self.available or action in history:
            raise ValueError('Infeasible predicted action')
        table = self.table
        if table.disabled or not table.alphabets[action]:
            return []
        q = self.posterior(history)
        sets, _ = table.event(history)
        base1, base0 = self.model._event_likelihoods(table, self.stratum, sets)[:2] if sets else (1., 1.)
        out = []
        for symbol in table.alphabets[action]:
            after = dict(sets)
            after[action] = [table.index[action][symbol]]
            l1, l0, _, _ = self.model._event_likelihoods(table, self.stratum, after)
            predictive = q * l1 / base1 + (1 - q) * l0 / base0
            out.append((symbol, float(predictive), self.posterior({**history, action: symbol})))
        self._predict[key] = out
        return out


def load(path, fold=None):
    params = json.loads(Path(path).read_text())
    if fold is not None and params['fold'] != fold:
        raise ValueError(f'Model file belongs to {params["fold"]}, not {fold}')
    return AnswerModel(params)
