"""Shared paths, constants and loaders.

Inputs are read from data/: the window tables (data/windows/<collection>/<split>.parquet), the labels
(data/labels/<collection>/<split>.parquet) and the folds (data/folds.json). Outputs are written below the
directory named by the environment variable EVIDENCE_OUTPUT, by default the repository root: fitted answer
models in models/, results in results/, checks in checks/ and figures in figures/.

Labels are read only through read_labels, and only to fit the starts and the answer model on the source
collections of a fold or to score the outcomes of a replay; the selection of questions never sees them.
"""
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.dont_write_bytecode = True
import answer_model  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = Path(os.environ.get('EVIDENCE_OUTPUT', BASE)).resolve()
MODELS = OUT / 'models'
ACTIONS = ('file', 'network', 'process')
DATASETS = ('CADETS_E3', 'CLEARSCOPE_E3', 'THEIA_E3', 'optc_h051', 'ATLASV2_EDR')
SPLITS = ('val', 'test')
CATEGORIES = ('unscored', 'low', 'high', 'alert')
QUEUES = ('1', '3', '10', '30', '100', 'alert')
NAMES = {'CADETS_E3': 'CADETS E3', 'CLEARSCOPE_E3': 'ClearScope E3', 'THEIA_E3': 'THEIA E3',
         'optc_h051': 'OpTC h051', 'ATLASV2_EDR': 'ATLASv2', 'DARPA_TC': 'DARPA TC pooled'}
FOLD_NAMES = {'velox_holdout_DARPA_TC': 'DARPA TC', 'velox_holdout_OpTC': 'OpTC', 'velox_holdout_ATLASv2': 'ATLASv2'}


def now():
    return datetime.now(timezone.utc).isoformat()


def rel(path):
    path = Path(path).resolve()
    for root in (BASE, OUT):
        if path.is_relative_to(root):
            return str(path.relative_to(root))
    return str(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=False, default=str) + '\n')


def folds():
    return read_json(DATA / 'folds.json')['folds']


def mask_of(dataset):
    return 'file,network' if dataset == 'CADETS_E3' else 'file,network,process'


@dataclass(frozen=True)
class Window:
    """One process on one host in 15 minutes: Velox category and the recorded answer to each question."""
    dataset: str
    split: str
    subject_id: int
    window_start_ns: int
    category: str
    responses: tuple
    available: tuple
    warm_up: bool

    @property
    def key(self):
        return [self.dataset, self.split, int(self.subject_id), int(self.window_start_ns)]


def load_partition(dataset, split='test'):
    """Windows of one collection and split, in stored order; an unavailable question is left out of `available`."""
    t = pq.read_table(DATA / 'windows' / dataset / f'{split}.parquet',
                      columns=['subject_id', 'window_start_ns', 'category', 'warm_up', *ACTIONS]).to_pydict()
    out = []
    for i in range(len(t['subject_id'])):
        states = tuple(t[a][i] for a in ACTIONS)
        if any(state not in ('empty', 'known', 'novel', 'unavailable') for state in states):
            raise RuntimeError('Unknown recorded answer')
        available = tuple(a for a, state in zip(ACTIONS, states) if state != 'unavailable')
        out.append(Window(dataset, split, int(t['subject_id'][i]), int(t['window_start_ns'][i]), t['category'][i],
                          states, available, bool(t['warm_up'][i])))
    if len({tuple(w.key) for w in out}) != len(out):
        raise RuntimeError('Duplicate window')
    return out


def read_labels(dataset, split):
    """{(dataset, split, subject_id, window_start_ns): (label, attack groups)}; label 1 attack, 0 benign, -1 unknown."""
    t = pq.read_table(DATA / 'labels' / dataset / f'{split}.parquet')
    return {(dataset, split, int(s), int(w)): (int(y), [int(g) for g in gs])
            for s, w, y, gs in zip(t['subject_id'].to_pylist(), t['window_start_ns'].to_pylist(),
                                   t['label'].to_pylist(), t['attack_groups'].to_pylist())}


def load_model(fold):
    """Fitted answer model of one fold (scripts/fit_models.py)."""
    return answer_model.load(MODELS / f'{fold}.json', fold)


ShiftedBelief = answer_model.ShiftedBelief
