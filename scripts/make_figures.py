"""Figures of the paper, from the small CSV inputs of scripts/figure_data.py.

Reads results/fig_*.csv and writes figures/*.png and *.svg below EVIDENCE_OUTPUT (default: the repository root).
The process diagram (Figure 2 of the paper) is authored in assets/f1_process.html; its SVG and PNG exports are
copied unchanged.

    python scripts/make_figures.py
"""
import os
from pathlib import Path
from shutil import copyfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("EVIDENCE_OUTPUT", BASE)).resolve()
RES, OUT = ROOT / "results", ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
PURPLE = "#9b4f96"
QCOL = {"1": YELLOW, "3": ORANGE, "10": BLUE, "30": AQUA, "100": MUTED, "alert": INK, "windows": PURPLE}
QLAB = {"1": "1 a day", "3": "3 a day", "10": "10 a day", "30": "30 a day", "100": "100 a day",
        "alert": "alert queue", "windows": "all test windows"}
CRIT = {"edv": ("EDV", "-", "o"), "ig": ("IG", "--", "s"), "random": ("random", ":", "^"),
        "order_fnp": ("fixed order (file, network, process)", "-.", "D")}
BUDGETS = ["1", "2", "3", "5", "10", "inf"]
PRICES = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3]
RS = [1, 2, 5, 10, 20, 50, 100]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def f0():
    # the worked example of Section 2.4: 10,000 windows at p = 6.4 percent, r = 10, one question with two answers
    n, p, t = 10_000, 0.064, 1 / 11
    att, oth = n * p, n * (1 - p)
    rows = [("before asking", 1.0, 1.0, "640 attacks + 9,360 others = 10,000"),
            ("answer $e_1$\nLR = 2", 0.4, 0.2, "256 (×0.4) + 1,872 (×0.2) = 2,128"),
            ("answer $e_2$\nLR = 0.75", 0.6, 0.8, "384 (×0.6) + 7,488 (×0.8) = 7,872")]
    light = "#a9c9ef"
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 2.4), sharey=True, gridspec_kw={"width_ratios": [3, 2], "wspace": 0.14})
    for y, (_, k1, k0, txt) in zip((2, 1, 0), rows):
        a, o = att * k1, oth * k0
        axa.barh(y, a, height=0.42, color=ORANGE, lw=0)
        axa.barh(y, o, left=a, height=0.42, color=light, lw=0)
        axa.text(0, y + 0.27, txt, fontsize=7, color=INK2, va="bottom")
        share = a / (a + o)
        axb.barh(y, 100 * share, height=0.42, color=ORANGE, lw=0)
        axb.barh(y, 25 - 100 * share, left=100 * share, height=0.42, color=light, lw=0)
        up = share >= t
        axb.text(25, y + 0.27, f"{100 * share:.1f}%: " + ("escalate" if up else "dismiss"), fontsize=7,
                 ha="right", va="bottom", color=INK if up else INK2, fontweight="bold" if up else "normal")
    axb.axvline(100 * t, color=INK, lw=1.0)
    axb.text(100 * t + 0.4, -0.55, "threshold 9.1%", fontsize=7, color=INK)
    axb.text(24.6, 2, "to 100%", fontsize=6.5, color=INK2, ha="right", va="center")
    axa.set_yticks([2, 1, 0], [r[0] for r in rows], fontsize=7.5)
    axa.set_xlim(0, n)
    axa.set_xticks([0, 2500, 5000, 7500, 10000], ["0", "2,500", "5,000", "7,500", "10,000"])
    axa.set_xlabel("windows giving each answer", fontsize=8)
    axb.set_xlim(0, 25)
    axb.set_xlabel("share within each bar (%)", fontsize=8)
    axa.set_title("(a) each answer keeps a share of each group", fontsize=8, loc="left")
    axb.set_title("(b) each bar rescaled to 100%", fontsize=8, loc="left")
    for ax in (axa, axb):
        ax.set_ylim(-0.7, 2.75)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.set_axisbelow(True)
    axb.spines["left"].set_visible(False)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=light)],
               labels=["attack windows", "other windows"], loc="upper center", ncol=2, fontsize=7.5,
               bbox_to_anchor=(0.5, 1.07))
    save(fig, "f0_example")


def f1():
    """Copy the process diagram and its browser exports from the assets."""
    for ext in ("svg", "png"):
        name = f"f1_process.{ext}"
        copyfile(BASE / "assets" / name, OUT / name)


def f2(r=10):
    z = pd.read_csv(RES / "fig_f2_zones.csv")
    s = pd.read_csv(RES / "fig_f2_starts.csv", dtype={"queue": str})
    folds = ["DARPA TC", "OpTC", "ATLASv2"]
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 4.6), sharex=True)
    t = 1 / (1 + r)
    marks = {"unscored": "u", "low": "l", "high": "h", "alert": "a"}
    for ax, fold in zip(axes, folds):
        zz = z.loc[(z.fold == fold) & (z.r == r)].set_index("category")
        ax.axvspan(zz.loc["any", "p_low"], zz.loc["any", "p_high"], color=BLUE, alpha=0.10, lw=0)
        ax.axvspan(zz.loc["high", "p_low"], zz.loc["high", "p_high"], color=BLUE, alpha=0.25, lw=0)
        ax.axvline(t, color=INK, lw=1.0)
        ax.set_ylim(0, 3)
        ax.set_yticks([])
        ax.grid(False)
        priors = {}
        for cat, letter in marks.items():
            priors.setdefault(round(float(zz.loc[cat, "prior"]), 12), []).append(cat)
        for p, cats in priors.items():
            inside = zz.loc[cats[0], "p_low"] <= p < zz.loc[cats[0], "p_high"]
            ax.plot(p, 0.8, "o", ms=7, mfc=MUTED if inside else "white", mec=INK2)
            ax.text(p, 0.2, "=".join(marks[c] for c in cats), ha="center", fontsize=7, color=INK2)
        ss = s.loc[s.fold == fold].sort_values("p_N")
        clusters = []
        for row in ss.itertuples(index=False):
            ax.plot(row.p_N, 1.9, "D", ms=6, color=QCOL[row.queue])
            name = "A" if row.queue == "alert" else row.queue
            if clusters and row.p_N / clusters[-1][-1][0] < 1.3:
                clusters[-1].append((row.p_N, name))
            else:
                clusters.append([(row.p_N, name)])
        for k, cl in enumerate(clusters):
            centre = float(np.exp(np.mean([np.log(v) for v, _ in cl])))
            ax.text(centre, 2.25 + 0.3 * (k % 2), ", ".join(n for _, n in cl), ha="center", fontsize=6.5, color=INK2,
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.6, alpha=0.9))
        ax.text(0.00045, 2.6, fold, fontsize=8.5, fontweight="bold", color=INK, va="center")
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(0.0004, 0.6)
    axes[-1].set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    axes[-1].set_xticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    axes[-1].set_xlabel("probability of attack before any question (logarithmic)")
    axes[-1].text(t * 1.06, 0.35, f"threshold {100 * t:.1f}% (r = {r})", fontsize=7, color=INK)
    save(fig, f"f2_starts_r{r}" if r != 10 else "f2_starts")


def f3():
    d = pd.read_csv(RES / "fig_f3.csv", dtype={"queue": str})
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    floor = 3e-6
    for ax, c in zip(axes, [0.0, 0.01]):
        for q in ["1", "3", "10", "30", "100", "alert", "windows"]:
            x = d.loc[(d.queue == q) & (d.c == c)].sort_values("r")
            y = x.share.clip(lower=floor)
            ls = "--" if q == "windows" else "-"
            ax.plot(x.r, y, ls, color=QCOL[q], lw=1.4, marker="o", ms=3, label=QLAB[q])
            zero = x.share == 0
            ax.plot(x.r[zero], [floor] * int(zero.sum()), "v", ms=4, color=QCOL[q], mfc="white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(RS)
        ax.set_xticklabels([str(r) for r in RS])
        ax.set_xlabel("cost ratio r")
        ax.set_title(f"price c = {c:g}", fontsize=9)
        ax.axvspan(5, 20, color=GRID, alpha=0.6, lw=0)
    axes[0].set_ylabel("share with an admissible question")
    axes[0].set_ylim(floor * 0.7, 1.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.15))
    save(fig, "f3_admissible")


def f4(r=10, c=0.01):
    d = pd.read_csv(RES / "fig_f4.csv", dtype={"queue": str, "B": str})
    queues = ["3", "10", "30", "alert"]
    fig, axes = plt.subplots(2, 4, figsize=(7.8, 4.0), sharex=True, gridspec_kw=dict(wspace=0.5))
    xs = np.arange(len(BUDGETS))
    for j, q in enumerate(queues):
        x = d.loc[(d.queue == q) & (d.r == r) & (d.c == c)]
        for i, metric in enumerate(["caught", "false"]):
            ax = axes[i, j]
            for crit, (lab, ls, mk) in CRIT.items():
                y = [x.loc[(x.criterion == crit) & (x.B == b), metric].sum() for b in BUDGETS]
                ax.plot(xs, y, ls, marker=mk, ms=3.5, lw=1.2, color=INK if crit == "edv" else (BLUE if crit == "ig" else (ORANGE if crit == "random" else AQUA)), label=lab)
            none = x.loc[x.criterion == "none", metric].sum()
            allq = x.loc[x.criterion == "all", metric].sum()
            ax.axhline(none, color=MUTED, lw=1.0, ls="--", label="no question")
            ax.axhline(allq, color=YELLOW, lw=1.0, ls="-", label="all questions")
            if i == 0:
                aw = int(x.loc[x.criterion == "none", "attack_windows"].sum())
                n = int(x.loc[x.criterion == "none", "labelled"].sum())
                ax.set_title(f"{QLAB[q]}\n{n} alarms\n{aw} attack windows", fontsize=7.2)
            ax.set_xticks(xs)
            ax.set_xticklabels(["1", "2", "3", "5", "10", "∞"])
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1.0))
            if i == 1:
                ax.set_xlabel("questions a day, B")
        axes[0, 0].set_ylabel("attack windows caught")
        axes[1, 0].set_ylabel("false escalations")
    axes[0, -1].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "f4_budget" if (r, c) == (10, 0.01) else f"f4_budget_r{r}_c{c:g}")


def f5(r=10, m=20):
    d = pd.read_csv(RES / "fig_f5.csv", dtype={"queue": str})
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    xs = np.arange(len(PRICES))
    for ax, crit in zip(axes, ["edv", "ig", "random"]):
        for q in ["1", "3", "10", "30", "alert", "windows"]:
            x = d.loc[(d.queue == q) & (d.r == r) & (d.criterion == crit)].set_index("c")
            y = [x.value_per_question.get(c, np.nan) for c in PRICES]
            ax.plot(xs, y, "--" if q == "windows" else "-", marker="o", ms=3, lw=1.3, color=QCOL[q], label=QLAB[q])
        ax.plot(xs, PRICES, color=INK, lw=0.8, ls=":", label="price c")
        ax.axhline(0, color=INK2, lw=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{c:g}" for c in PRICES], fontsize=7)
        ax.set_xlabel("price per question c")
        ax.set_title(CRIT[crit][0], fontsize=9)
    axes[0].set_ylabel("decision loss saved\n(false escalation units)")
    sec = axes[-1].secondary_yaxis("right", functions=(lambda v: v * m, lambda v: v / m))
    sec.set_ylabel(f"loss equivalent minutes\nper question (m = {m})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.16))
    save(fig, "f5_value" if r == 10 else f"f5_value_r{r}")


def f6():
    from matplotlib.lines import Line2D
    d = pd.read_csv(RES / "fig_f6.csv", dtype={"queue": str})
    d = d.loc[d.questions_known > 0]
    fig, ax = plt.subplots(figsize=(5.4, 3.9))
    shift = {"edv": -0.005, "ig": 0.0, "random": 0.005}      # small horizontal offsets against overplotting
    for q in ["1", "3", "10", "30", "alert", "windows"]:
        for crit, (lab, _, mk) in list(CRIT.items())[:3]:
            x = d.loc[(d.queue == q) & (d.criterion == crit)]
            xs = x.expected_value_per_question + shift[crit]
            if q != "windows":
                ax.vlines(xs, x.pred_lo, x.pred_hi, color=QCOL[q], lw=0.6, alpha=0.35)
            ax.plot(xs, x.value_per_question, mk, ms=4, ls="none", color=QCOL[q], alpha=0.85,
                    mfc=QCOL[q] if crit == "edv" else "white")
    ax.plot([0, 0.5], [0, 0.5], color=INK, lw=0.8, ls=":")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(-1.6, 3.6)
    ax.set_xlabel("expected value per question (model)")
    ax.set_ylabel("realised value per question (labels)")
    qh = [Line2D([], [], color=QCOL[q], marker="o", ls="none", ms=4, label=QLAB[q])
          for q in ["1", "3", "10", "30", "alert", "windows"]]
    ch = [Line2D([], [], color=INK2, marker=CRIT[c][2], ls="none", ms=4, mfc=INK2 if c == "edv" else "white",
                 label=CRIT[c][0]) for c in ["edv", "ig", "random"]]
    ch.append(Line2D([], [], color=INK2, lw=0.6, alpha=0.5, label="model's 95% interval"))
    first = ax.legend(handles=qh, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), title="colour: queue",
                      title_fontsize=6.5)
    ax.add_artist(first)
    ax.legend(handles=ch, fontsize=6.5, loc="lower left", bbox_to_anchor=(1.01, 0.0), title="marker: criterion",
              title_fontsize=6.5)
    save(fig, "f6_calibration")


def main():
    f0()
    f1()
    f2(10)
    f3()
    f4(10, 0.01)
    f5(10)
    f6()
    for r in (5, 20):           # appendix: the same figures at the other cost ratios of the region
        f2(r)
        f4(r, 0.01)
        f5(r)
    for c in (0.001, 0.03, 0.1, 0.3):   # appendix: F4 at the other prices of the region
        f4(10, c)
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
