#!/usr/bin/env bash
# Run the whole pipeline in order. Outputs go below $EVIDENCE_OUTPUT (default: this repository).
# Usage: bash run_all.sh            (rewrite models/, results/, checks/ and figures/ in place)
#        EVIDENCE_OUTPUT=/some/dir bash run_all.sh   (write a fresh copy elsewhere)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1

run() {
    echo "== $1"
    start=$(date +%s)
    "$PY" -B "scripts/$1"
    echo "   $1 done in $(( $(date +%s) - start )) s"
}

run queues.py            # daily queues and starts
run fit_models.py        # answer model of each fold
run count.py             # label free count of admissible questions and of differences between criteria
run budget_replay.py     # replay with shared budget, every criterion, prefix rule check
run replay_summary.py    # realistic summary and rebuild of every replay cell
run value.py             # value per question, ordinary windows, calibration
run sensitivity.py       # assignment by EDV, successive questions, weaker smoothing
run assumption_check.py  # answer strength inside source alarms against all source windows
run power.py             # attacks needed to tell two criteria apart (model based)
run power_summary.py
run robustness.py        # bootstrap, starts and smoothing variants, predictive intervals, fitted ratios
run minutes.py           # values in analyst minutes
run claims.py            # ranges over the realistic region quoted in the text
run figure_data.py       # small CSV inputs of the figures
run make_figures.py      # figures (needs matplotlib)
run tables.py            # tables of the paper as Markdown
