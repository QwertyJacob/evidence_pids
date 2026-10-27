"""Compare a fresh run of the pipeline with the results shipped in this repository.

    python scripts/compare_results.py /path/to/fresh/run [reference root, default: this repository]

For every file of models/ and results/ of the reference: Markdown and JSON files must be identical (JSON after
parsing); CSV and parquet files must hold the same rows and columns, with numbers equal to a relative tolerance
of 1e-9. The check files (checks/) carry timestamps and are compared without their 'generated' field. Figures are
not compared, since the rendering of PNG and SVG files depends on the matplotlib version.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RTOL = 1e-9


def frames_equal(a, b):
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False, 'different columns or row count'
    worst = 0.0
    for col in a.columns:
        x, y = a[col], b[col]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            x, y = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            same_nan = np.isnan(x) == np.isnan(y)
            if not same_nan.all():
                return False, f'missing values differ in {col}'
            ok = ~np.isnan(x)
            if ok.any():
                diff = np.abs(x[ok] - y[ok])
                scale = np.maximum(np.abs(x[ok]), np.abs(y[ok]))
                bad = diff > RTOL * np.maximum(scale, 1e-300)
                bad &= ~(np.isinf(x[ok]) & (x[ok] == y[ok]))
                if bad.any():
                    return False, f'{col}: max abs difference {np.nanmax(diff):.3g}'
                finite = np.isfinite(diff)
                worst = max(worst, float(diff[finite].max()) if finite.any() else 0.0)
        elif not x.astype(str).equals(y.astype(str)):
            return False, f'values differ in {col}'
    return True, f'max abs difference {worst:.3g}'


def without_generated(value):
    if isinstance(value, dict):
        return {k: without_generated(v) for k, v in value.items() if k not in ('generated', 'utc', 'elapsed_seconds')}
    if isinstance(value, list):
        return [without_generated(v) for v in value]
    return value


def compare(path_a, path_b):
    suffix = path_a.suffix
    if suffix == '.csv':
        if path_a.read_bytes() == path_b.read_bytes():
            return True, 'identical'
        return frames_equal(pd.read_csv(path_a), pd.read_csv(path_b))
    if suffix == '.parquet':
        return frames_equal(pd.read_parquet(path_a), pd.read_parquet(path_b))
    if suffix == '.json':
        a, b = without_generated(json.loads(path_a.read_text())), without_generated(json.loads(path_b.read_text()))
        return a == b, 'identical' if a == b else 'different content'
    same = path_a.read_bytes() == path_b.read_bytes()
    return same, 'identical' if same else 'different content'


def main():
    fresh = Path(sys.argv[1]).resolve()
    reference = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else Path(__file__).resolve().parents[1]
    failures = 0
    for folder in ('models', 'results', 'checks'):
        for ref in sorted((reference / folder).glob('*')):
            new = fresh / folder / ref.name
            if not new.exists():
                print(f'MISSING  {folder}/{ref.name}')
                failures += 1
                continue
            ok, detail = compare(ref, new)
            failures += not ok
            print(f"{'ok      ' if ok else 'DIFFERS '} {folder}/{ref.name}: {detail}")
    print('all equal' if not failures else f'{failures} files differ or are missing')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
