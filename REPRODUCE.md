# Reproducing the experiments

This document explains what this repository reproduces, which data it starts from, how to run it and how each output relates to the paper *Towards Active Inference Triage Agents: Budgeted Evidence Gathering in Provenance-Based Intrusion Detection*. Section, figure and table numbers refer to the current version of the manuscript. Items prefixed with S belong to the [supplementary material](https://qwertyjacob.github.io/evidence_pids/), including the shared daily budget analysis in [Section S6.3](https://qwertyjacob.github.io/evidence_pids/#supp-budget).

## 1. Overview

The paper replays the daily alarm queues of a published provenance based detector, Velox, on five public collections (CADETS E3, ClearScope E3 and THEIA E3 from the DARPA Transparent Computing programme, host 051 of DARPA OpTC, and ATLASv2). Every alarm is one process on one host during a window of 15 minutes. Before an alarm is escalated or dismissed, an automated triage stage may ask one evidence question about it: is the **file**, **network** or **process** activity of the window new to the host? Each question costs a price `c`, a daily budget `B` limits how many questions are asked across the queue, and a criterion (expected decision value, information gain, random choice or a fixed order) chooses which question to ask. The outcome is scored by the decision loss `D = r M + F`, with `M` missed attack windows, `F` false escalations and `r` the cost ratio.

The repository starts from one table per collection and split that lists every window with its Velox score, its category and the recorded answer to each of the three questions, together with the labels of the windows (Section 3 below). From there, every step of the paper is reproduced:

1. the daily queues and the starting probability (start) of their alarms;
2. the answer model of each fold, fitted on the source collections;
3. the label free count of alarms on which a question can change the decision;
4. the replay with a shared budget for every queue, cost ratio, price, budget and criterion;
5. the value of a question, its calibration, sensitivity, uncertainty and robustness analyses;
6. the number of attacks needed to tell two criteria apart;
7. every figure and table of the paper.

All scripts are in `scripts/`; `run_all.sh` runs them in order. A complete run takes a few minutes (Section 4).

### Where each result of the paper comes from

| Paper | Content | Script | Files in `results/` (and `figures/`) |
| --- | --- | --- | --- |
| Figure 1 | Worked example of Section 3.4 | `make_figures.py` (no data) | `figures/f0_example.*` |
| Figure 2 | The process studied | copied from `assets/` | `figures/f1_process.*` |
| Table 1 | The collections | `tables.py` | `tables.md` (T1) |
| Table S1 | Queues and starts | `queues.py`, `tables.py` | `starts.csv`, `queue_membership.parquet`, `tables.md` (T2) |
| Table 2 | Prices of costly actions | not computed (literature) | |
| Figures S1 to S3 | Where alarms start (Section S1) | `figure_data.py`, `make_figures.py` | `fig_f2_zones.csv`, `fig_f2_starts.csv`, `figures/f2_starts*` |
| Table 3, Figure S4 | Alarms with an admissible question (Section 5.1) | `count.py` | `count.csv`, `count_windows.csv`, `fig_f3.csv`, `figures/f3_admissible.*` |
| Table 4 | Differences between the criteria (Section 5.2) | `count.py` | `count.csv`, `count_pairs.parquet`, `count_by_alarm.parquet` |
| Figures S5 to S11; Tables S7 and S8 | The shared budget (Section S6.3) | `budget_replay.py`, `replay_summary.py` | `replay.parquet`, `replay.csv`, `replay_realistic.csv`, `by_alarm.parquet`, `fig_f4.csv`, `figures/f4_budget*` |
| Sections S6.3 and S4 | Several questions per alarm, assignment by EDV, stronger smoothing | `sensitivity.py` | `sensitivity.parquet`, `sensitivity.csv` |
| Figure 3, Figures S12 and S13; Table S10 | What a question is worth (Section 5.3) | `value.py`, `robustness.py` | `value.csv`, `value.parquet`, `value_windows.csv`, `robustness_bootstrap.csv`, `robustness_predictive.csv`, `fig_f5.csv`, `figures/f5_value*` |
| Table S3 | Value under other starts and weaker smoothing | `robustness.py` | `robustness_sensitivity.csv`, `robustness_starts.csv` |
| Section 5.3, text | Values in analyst minutes | `minutes.py` | `minutes.csv` |
| Figure 4 | Calibration of the model on alarms (Section 5.4) | `value.py`, `robustness.py` | `calibration.csv`, `fig_f6.csv`, `figures/f6_calibration.*` |
| Table A1 | Answer strength inside alarms (Sections 5.4 and A.3) | `assumption_check.py` | `assumption_check.csv`, `checks/assumption_check.json` |
| Table 5 | The deployment loop (Section 5.4) | not computed (protocol) | |
| Table 6 | Time to each level of model health (Section 5.5) | arithmetic from the cited attack rates and the paper's equations | |
| Table S5 | Attacks needed to rank two criteria (Sections 5.5 and S5) | `power.py`, `power_summary.py` | `power.csv`, `power_realistic.csv` |
| Table S8 | Replay at r = 10 and c = 0.01 | `tables.py` | `replay.parquet` |
| Table S11 | Results of each tested collection | `robustness.py` | `robustness_per_group.csv` |
| Table S2 | Fitted likelihood ratios | `robustness.py` | `robustness_fitted_lr.csv` |
| Table S9 | Decision loss of the six fixed orders | `robustness.py` | `robustness_per_group.csv` |
| Table S4 | Starts under three estimates | `robustness.py` | `robustness_starts.csv` |
| Table S6 | Blocks of alarms | `tables.py` | `by_alarm.parquet` |
| Text | Ranges over the realistic region quoted in the text | `claims.py` | `claims.csv` |

`tables.py` writes the computed tables to `results/tables.md`. Codes such as T1 and A1 in that file identify analysis outputs; manuscript table numbers are listed above.

## 2. Requirements

Python 3.11 or later with numpy, pandas, pyarrow and matplotlib (matplotlib is needed only for the figures):

```
pip install -r requirements.txt
```

The results in this repository were produced with Python 3.13, numpy 2.5, pandas 3.0, pyarrow 25.0 and matplotlib 3.11. No GPU, database or network access is needed.

## 3. The data

All inputs are in `data/` (2.5 MB).

### 3.1 Window tables: `data/windows/<collection>/<split>.parquet`

One row per window, for the validation (`val`) and test (`test`) splits of each collection. A window is one process (`subject_id`, a node of the provenance graph) during the 15 minutes that start at `window_start_ns` (UTC, nanoseconds).

| Column | Meaning |
| --- | --- |
| `subject_id` | Process identifier within the collection |
| `window_start_ns` | Start of the window, UTC nanoseconds |
| `velox_score` | Largest Velox score of the process within the window; empty when Velox scored nothing |
| `category` | `unscored`, `low` (score below the 99th percentile `q99` of the validation scores), `high` (from `q99` to the validation maximum `vmax`) or `alert` (above `vmax`) |
| `warm_up` | True for windows in the first 24 hours of the records of the collection, which have no full history |
| `file`, `network`, `process` | Recorded answer to each question: `empty` (no such activity in the window), `known` (everything involved was seen on the host more than 24 hours earlier), `novel` (something involved was first seen on the host within the last 24 hours) or `unavailable` (the question cannot be answered on this collection) |

For example, a window whose process opened only files already seen two days earlier, contacted no network address and started a process first seen one hour earlier has the answers `known`, `empty` and `novel`. The process question is `unavailable` on CADETS E3, so its alarms have two questions instead of three.

`data/velox_thresholds.csv` gives, for each collection, the Velox run, the training epoch (11), the number of validation scores and the thresholds `q99` and `vmax`. `scripts/queues.py` checks that every stored category equals the threshold rule applied to the stored score.

### 3.2 Labels: `data/labels/<collection>/<split>.parquet`

Same rows and order as the window tables, with `label` (1 attack window, 0 benign, -1 unknown) and `attack_groups` (identifiers of the annotated attacks the window belongs to, numbered within each collection). A window is an attack window when its process took part in an annotated attack and has at least one event within the period of that attack. The label is unknown when the process took part in an attack but has no event within its period (90 test windows). Labels are read only to fit the starts and the answer model on source collections, and to score outcomes after a replay; the choice of questions never uses them.

### 3.3 Folds: `data/folds.json`

The collections form three groups: DARPA TC (CADETS E3, ClearScope E3, THEIA E3), OpTC (host 051) and ATLASv2. Each fold tests one group (`target_evaluation`, test split) and fits the starts and the answer model on the validation and test splits of the other two (`fitting`). Fold identifiers (`velox_holdout_DARPA_TC`, `velox_holdout_OpTC`, `velox_holdout_ATLASv2`) appear in the result files.

### 3.4 How the inputs were produced

The window tables are derived from public data with steps that are not part of this repository:

* **Detector scores.** Velox was trained and evaluated with the PIDSMaker framework, with its default configuration, one run per collection (run identifiers in `data/velox_thresholds.csv`). The score of a window is the largest edge loss of its process over the Velox batches that end within the window, at epoch 11, the last evaluated epoch of each run, fixed without looking at labels.
* **Answers.** The three answers are lookups in the event history of the host, computed from the events of each collection as loaded into the PIDSMaker database, up to the end of each window, so no future event is used.
* **Labels.** The attack processes and the periods of the annotated attacks come from the ground truth distributed with PIDSMaker for these datasets, which follows the official reports of the DARPA TC third engagement, DARPA OpTC and ATLASv2.

Starting from these tables keeps the repository small and independent of the raw datasets, which are large and distributed under their own terms.

## 4. Running

From the repository root:

```
bash run_all.sh
```

This rewrites `models/`, `results/`, `checks/` and `figures/` in place, so `git diff` shows any change against the shipped versions. To keep the shipped files untouched, write the run to another folder and compare:

```
EVIDENCE_OUTPUT=/tmp/evidence_run bash run_all.sh
python scripts/compare_results.py /tmp/evidence_run
```

`compare_results.py` compares every model, result and check file of the fresh run with the shipped one: Markdown and JSON must be identical (ignoring time stamps), CSV and parquet files must hold the same rows with numbers equal to a relative tolerance of 1e-9. Figures are not compared, because PNG and SVG bytes depend on the matplotlib version. A fresh run of this repository reproduces every shipped model and result file.

Each script can also be run on its own, as `python scripts/<name>.py`, once the files it reads exist. A different Python interpreter can be chosen with `PYTHON=/path/to/python bash run_all.sh`.

Indicative running times of one run (single process):

| Step | Time |
| --- | --- |
| `queues.py`, `fit_models.py` | about 10 s |
| `count.py` | about 15 s |
| `budget_replay.py`, `replay_summary.py` | about 1.5 min |
| `value.py` | about 25 s |
| `sensitivity.py` | about 3.5 min |
| `robustness.py` | about 20 s |
| all other steps | about 40 s |
| whole pipeline | about 7 min |

## 5. The pipeline step by step

### 5.1 Queues and starts: `queues.py`

On each UTC day of each collection and split, the `N` windows with the highest Velox score form the daily queue of `N` alarms a day, for `N` = 1, 3, 10, 30 and 100; ties are broken by process identifier, then window start. The alert queue holds every window of category `alert`. A queue covers all hosts of its collection.

Every alarm of a tested queue starts at the same probability of attack, its start, estimated from the queue built by the same rule on the source collections of the fold:

`p_N = (a_N + 1) / (n_N + 2)`,

with `n_N` labelled source alarms and `a_N` attack windows among them. The `+1` and `+2` add one hypothetical window to each class (Laplace smoothing). For example, 0 attack windows among 27 source alarms give a start of 1/29, about 3.4 percent. Outputs: `queues.csv` and `day_counts.csv` (label free census), `queue_membership.parquet` (every alarm of every queue), `starts.csv`.

### 5.2 The answer model: `fit_models.py`

For each fold, the model counts how often each combination of answers occurs among attack and benign source windows (outside the warm up), separately for each category of the detector score and for each set of answerable questions (all three, or file and network only). Small tables are pulled towards a pooled rate:

`L_y(e) = (1 - λ) n_y(e) / N_y + λ B(e)`, with `λ = τ / (N + τ)` and `τ = 10,000`,

where `n_y(e)` counts the windows of class `y` with answers `e`, `N_y` the windows of class `y`, `N = N_1 + N_0`, and `B(e)` the product of the pooled frequencies of the answers in `e`. With `N = 30,000` windows, for instance, `λ = 0.25`. A category keeps its own prior only with at least 5 attack and 100 benign source windows, and its own tables only with at least 20 attack and 100 benign windows; an alert without its own prior is counted as `high`. Output: `models/<fold>.json`, read by every later step. `answer_model.py` holds the model and the belief view that every criterion queries (`ShiftedBelief`: the model's likelihoods applied from the start of the alarm's queue rather than from the category prior).

### 5.3 The label free count: `count.py`

For every tested alarm, cost ratio `r` (1, 2, 5, 10, 20, 50, 100) and price `c` (0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3), the script computes the expected decision value (EDV) and the information gain (IG) of each question from the alarm's start, the admissible questions (`EDV > c`), the question each criterion would ask and the decision it would take on the recorded answer. No label is read. It also counts, for all test windows at the prior of their category, how many have an admissible question. Outputs: `count_by_alarm.parquet` (one row per alarm, `r` and `c`), `count.csv` (per queue, `r`, `c` and budget), `count_pairs.parquet` (differences between every pair of criteria), `count_windows.csv`.

### 5.4 The replay: `budget_replay.py` and `replay_summary.py`

The replay visits the alarms of each collection day from the top of the queue. An alarm with an admissible question receives one while the daily budget lasts; the criterion chooses which; the alarm is escalated when its probability of attack, after the answer or at its start, reaches `1/(1 + r)`. The criteria are EDV, IG, random choice (ten seeds, averaged) and the six fixed orders of the three questions; the references are no question and all questions (every available answer, with no budget or filter).

The engine runs every criterion without a budget limit and records, for each alarm, its rank among the alarms of its day with an admissible question; a budget `B` then questions exactly the first `B` of them. This prefix rule is checked against explicit runs at every finite budget (10,059,840 comparisons). `replay_summary.py` rebuilds every value of `replay.parquet` from the per alarm records (470,400 values) and summarises the realistic region. Outputs: `by_alarm.parquet`, `replay.parquet` (every fold, collection, queue, `r`, `c`, `B` and criterion; DARPA TC also pooled as `DARPA_TC`), `replay.csv` (the extract at `r = 10`), `replay_realistic.csv`.

### 5.5 Sensitivity of the replay: `sensitivity.py`

One default is varied at a time: alarms of a day questioned in decreasing order of their largest admissible EDV instead of queue order; successive questions on the same alarm while some unasked question stays admissible; and stronger smoothing, `τ` = 100,000 and 1,000,000. Outputs: `sensitivity.parquet`, `sensitivity.csv`.

### 5.6 The value of a question: `value.py`

The realised value per question of a criterion is `v = (D_none - D) / Q`, the decision loss saved per question asked, in false escalation units; the model's expected value per question is the mean EDV of the questions asked, and comparing the two measures calibration. The script also replays every ordinary test window at the prior of its category, with and without the admissibility filter. Outputs: `value.parquet`, `value.csv`, `value_windows.csv`, `calibration.csv`.

### 5.7 Uncertainty and robustness: `robustness.py`

* a bootstrap over collection days (2,000 resamples) of the value per question;
* starts with half a window per class, `(a_N + 0.5) / (n_N + 1)`, and starts from the source test splits only;
* the model's 95 percent predictive interval of the realised value, drawing labels and answers from the model (2,000 draws);
* weaker smoothing, `τ` = 100 and 1,000;
* the fitted likelihood ratio of every single answer, with the prior and tables each category uses;
* the results of each tested collection, the six fixed orders and the spread over random seeds.

Outputs: `robustness_*.csv`.

### 5.8 Model assumption, power and minutes

* `assumption_check.py` compares, on the source collections only, how often each answer occurs among attack and benign windows inside the source alarm queues with how often it occurs among all `high` windows (and, for alerts, all windows), with Wilson 95 percent intervals.
* `power.py` estimates, under the model, how many alarms and distinct attacks a queue would need for a two sided test at 5 percent with power 0.8 to tell two criteria apart; a Monte Carlo draw checks the closed form. `power_summary.py` summarises it over the realistic cells.
* `minutes.py` converts values into analyst minutes for `m` = 10, 20 and 40 minutes per false escalation.

### 5.9 Text ranges, figures and tables

`claims.py` computes the ranges over the realistic region (queues of 1, 3, 10 and 30 alarms a day and the alert queue; `r` = 5, 10, 20; `c` = 0.001 to 0.3) that the text quotes. `figure_data.py` writes small CSV inputs for the figures, and computes the zones of Figure S1 (the starting probabilities from which some answer can move the decision across the threshold) from the fitted models. `make_figures.py` draws the figures and `tables.py` writes `results/tables.md`, with a cross-check of each value against a second route where one exists; a failed cross-check is printed as a mismatch.

To rebuild only the webpage plots, run `python scripts/make_web_figures.py`. This writes scalable SVG images and 600 dpi PNG downloads to `docs/figures/`, with larger labels and legends and bold tick labels. The paper figures in `figures/` are unchanged.

## 6. Determinism and checks

The pipeline is deterministic. Random choice of a question orders the questions of an alarm by the SHA256 digest of the alarm's key, the seed and the question, so it does not depend on the machine. The bootstrap, the predictive intervals and the Monte Carlo check of `power.py` use numpy generators with the fixed seed 20261024.

Every step writes its checks to `checks/`:

| File | What is checked |
| --- | --- |
| `count_check.json` | every criterion questions exactly the alarms with an admissible question; chosen questions are admissible; EDV picks a largest EDV; admissible counts never rise with the price |
| `replay_check.json` | the replay asks the same questions and takes the same decisions as the label free count; the prefix rule for finite budgets; the rebuild of every replay value from the per alarm records |
| `value_check.json` | every value per question rebuilt from two losses and a count of questions |
| `sensitivity_check.json` | each variant equals the default where it must (assignment by EDV without a budget limit, successive questions at `B = 1`, no question under any smoothing) |
| `power_check.json` | the closed form of the power analysis against a Monte Carlo draw |
| `minutes_check.json` | net value in minutes equals value minus price |
| `assumption_check.json` | summary of the interval comparisons of Table A1 |
| `robustness.json` | list of the robustness outputs |

## 7. Repository layout

```
README.md            short description
REPRODUCE.md         this document
run_all.sh           the whole pipeline in order
requirements.txt
assets/              process diagram of Figure 2 (HTML source, SVG and PNG exports)
data/                window tables, labels, folds, Velox thresholds (inputs)
models/              fitted answer model of each fold (output of fit_models.py)
results/             every result file and tables.md
checks/              checks written by the scripts
figures/             figures of the paper, PNG and SVG
scripts/             the code; common.py (paths and loaders), policies.py (decision rule and question scores),
                     answer_model.py (answer model and belief view), one script per step, compare_results.py
```
