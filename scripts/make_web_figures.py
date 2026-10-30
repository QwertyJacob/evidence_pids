"""Draw readable website plots from the shipped figure CSV inputs.

Run python scripts/make_web_figures.py from any directory. Only docs/figures/
is written. The paper figures and their generation settings are unchanged.
Each plot is exported as a scalable SVG and a 600 dpi PNG.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
RES, OUT = BASE / "results", BASE / "docs" / "figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
QCOL = {"1": YELLOW, "3": ORANGE, "10": BLUE, "30": AQUA,
        "100": MUTED, "alert": INK, "windows": "#9b4f96"}
QLAB = {"1": "1 a day", "3": "3 a day", "10": "10 a day", "30": "30 a day",
        "100": "100 a day", "alert": "alert queue", "windows": "all test windows"}
CRIT = {"edv": ("EDV", "-", "o", INK), "ig": ("IG", "--", "s", BLUE),
        "random": ("random", ":", "^", ORANGE),
        "order_fnp": ("fixed order (file, network, process)", "-.", "D", AQUA)}
BUDGETS = ["1", "2", "3", "5", "10", "inf"]
PRICES = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3]
RS = [1, 2, 5, 10, 20, 50, 100]
STYLE = {
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.labelsize": 15, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "axes.labelpad": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "xtick.major.width": 1.3, "ytick.major.width": 1.3,
    "xtick.major.size": 5, "ytick.major.size": 5,
    "legend.fontsize": 14, "legend.frameon": False,
    "legend.handlelength": 2.4, "legend.columnspacing": 1.6,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.18,
    "svg.fonttype": "path", "svg.hashsalt": "evidence-web-plots",
}


def save(fig, name):
    # Include secondary axes and logarithmic tick labels in the weight change.
    for axis in fig.findobj(match=matplotlib.axis.Axis):
        for label in axis.get_ticklabels():
            label.set_fontweight("bold")
        axis.get_offset_text().set_fontweight("bold")
    for extension in ("svg", "png"):
        path = OUT / f"{name}.{extension}"
        fig.savefig(path, metadata={"Date": None} if extension == "svg" else None)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def starts(r):
    zones = pd.read_csv(RES / "fig_f2_zones.csv")
    starts = pd.read_csv(RES / "fig_f2_starts.csv", dtype={"queue": str})
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 6.2), sharex=True)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.15, top=0.96, hspace=0.26)
    threshold = 1 / (1 + r)
    marks = {"unscored": "u", "low": "l", "high": "h", "alert": "a"}
    for ax, fold in zip(axes, ["DARPA TC", "OpTC", "ATLASv2"]):
        data = zones.loc[(zones.fold == fold) & (zones.r == r)].set_index("category")
        ax.axvspan(data.loc["any", "p_low"], data.loc["any", "p_high"], color=BLUE, alpha=0.10, lw=0)
        ax.axvspan(data.loc["high", "p_low"], data.loc["high", "p_high"], color=BLUE, alpha=0.25, lw=0)
        ax.axvline(threshold, color=INK, lw=1.5)
        ax.set_ylim(0, 3)
        ax.set_yticks([])
        ax.grid(False)
        priors = {}
        for category in marks:
            priors.setdefault(round(float(data.loc[category, "prior"]), 12), []).append(category)
        prior_clusters = []
        for probability, categories in sorted(priors.items()):
            inside = data.loc[categories[0], "p_low"] <= probability < data.loc[categories[0], "p_high"]
            ax.plot(probability, 0.8, "o", ms=9, mfc=MUTED if inside else "white", mec=INK2, mew=1.3)
            label = "=".join(marks[c] for c in categories)
            if prior_clusters and probability / prior_clusters[-1][-1][0] < 1.3:
                prior_clusters[-1].append((probability, label))
            else:
                prior_clusters.append([(probability, label)])
        for cluster in prior_clusters:
            centre = float(np.exp(np.mean([np.log(value) for value, _ in cluster])))
            ax.text(centre, 0.12, ", ".join(label for _, label in cluster),
                    ha="center", fontsize=13, color=INK)
        clusters = []
        for row in starts.loc[starts.fold == fold].sort_values("p_N").itertuples(index=False):
            ax.plot(row.p_N, 1.9, "D", ms=8, color=QCOL[row.queue])
            name = "A" if row.queue == "alert" else row.queue
            if clusters and row.p_N / clusters[-1][-1][0] < 1.3:
                clusters[-1].append((row.p_N, name))
            else:
                clusters.append([(row.p_N, name)])
        for index, cluster in enumerate(clusters):
            centre = float(np.exp(np.mean([np.log(value) for value, _ in cluster])))
            ax.text(centre, 2.25 + 0.3 * (index % 2), ", ".join(name for _, name in cluster),
                    ha="center", fontsize=13, color=INK,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1, alpha=0.95))
        ax.text(0.00045, 2.6, fold, fontsize=14, fontweight="bold", color=INK, va="center")
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(0.0004, 0.6)
    axes[-1].set_xticks([0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    axes[-1].set_xticklabels(["0.1%", "0.3%", "1%", "3%", "10%", "30%"])
    axes[-1].xaxis.set_minor_formatter(NullFormatter())
    axes[-1].set_xlabel("probability of attack before any question (logarithmic)\n"
                          f"decision threshold {100 * threshold:.1f}% (r = {r})")
    save(fig, "f2_starts" if r == 10 else f"f2_starts_r{r}")


def admissible():
    data = pd.read_csv(RES / "fig_f3.csv", dtype={"queue": str})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.3), sharey=True)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.31, top=0.90, wspace=0.18)
    floor = 3e-6
    for ax, price in zip(axes, [0.0, 0.01]):
        for queue in ["1", "3", "10", "30", "100", "alert", "windows"]:
            values = data.loc[(data.queue == queue) & (data.c == price)].sort_values("r")
            ax.plot(values.r, values.share.clip(lower=floor), "--" if queue == "windows" else "-",
                    color=QCOL[queue], lw=1.9, marker="o", ms=5, label=QLAB[queue])
            zero = values.share == 0
            ax.plot(values.r[zero], [floor] * int(zero.sum()), "v", ms=6,
                    color=QCOL[queue], mfc="white", mew=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(RS)
        ax.set_xticklabels([str(r) for r in RS])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, _: rf"$\mathbf{{10^{{{int(round(np.log10(value)))}}}}}$"))
        ax.set_xlabel("cost ratio r")
        ax.set_title(f"price c = {price:g}", pad=12)
        ax.axvspan(5, 20, color=GRID, alpha=0.6, lw=0)
    axes[0].set_ylabel("share with an admissible question")
    axes[0].set_ylim(floor * 0.7, 1.5)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.54, 0), labelspacing=0.9)
    save(fig, "f3_admissible")


def budget(r, price):
    data = pd.read_csv(RES / "fig_f4.csv", dtype={"queue": str, "B": str})
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 6.8), sharex=True)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.26, top=0.86, wspace=0.44, hspace=0.24)
    xs = np.arange(len(BUDGETS))
    for column, queue in enumerate(["3", "10", "30", "alert"]):
        values = data.loc[(data.queue == queue) & (data.r == r) & (data.c == price)]
        for row, metric in enumerate(["caught", "false"]):
            ax = axes[row, column]
            for criterion, (label, linestyle, marker, colour) in CRIT.items():
                ys = [values.loc[(values.criterion == criterion) & (values.B == b), metric].sum()
                      for b in BUDGETS]
                ax.plot(xs, ys, linestyle, marker=marker, ms=5, lw=1.8, color=colour, label=label)
            ax.axhline(values.loc[values.criterion == "none", metric].sum(),
                       color=MUTED, lw=1.5, ls="--", label="no question")
            ax.axhline(values.loc[values.criterion == "all", metric].sum(),
                       color=YELLOW, lw=1.5, label="all questions")
            if row == 0:
                attacks = int(values.loc[values.criterion == "none", "attack_windows"].sum())
                alarms = int(values.loc[values.criterion == "none", "labelled"].sum())
                ax.set_title(f"{QLAB[queue]}\n{alarms} alarms\n{attacks} attack windows", fontsize=13, pad=13)
            ax.set_xticks(xs)
            ax.set_xticklabels(["1", "2", "3", "5", "10", "∞"])
            ax.set_ylim(bottom=0, top=max(ax.get_ylim()[1], 1.0))
            if row == 1:
                ax.set_xlabel("questions a day, B", fontsize=13)
    axes[0, 0].set_ylabel("attack windows\ncaught")
    axes[1, 0].set_ylabel("false escalations")
    handles, labels = axes[0, -1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.54, 0.01), labelspacing=0.9)
    save(fig, "f4_budget" if (r, price) == (10, 0.01) else f"f4_budget_r{r}_c{price:g}")


def value(r, minutes=20):
    data = pd.read_csv(RES / "fig_f5.csv", dtype={"queue": str})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.3), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.88, bottom=0.35, top=0.90, wspace=0.18)
    xs = np.arange(len(PRICES))
    queues = ["1", "3", "10", "30", "alert"] + (["windows"] if r != 10 else [])
    for ax, criterion in zip(axes, ["edv", "ig", "random"]):
        for queue in queues:
            values = data.loc[(data.queue == queue) & (data.r == r) &
                              (data.criterion == criterion)].set_index("c")
            ys = [values.value_per_question.get(price, np.nan) for price in PRICES]
            ax.plot(xs, ys, "--" if queue == "windows" else "-", marker="o", ms=5,
                    lw=1.9, color=QCOL[queue], label=QLAB[queue])
        ax.plot(xs, PRICES, color=INK, lw=1.3, ls=":", label="price c")
        ax.axhline(0, color=INK2, lw=0.9)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{price:g}" for price in PRICES], rotation=60, ha="right")
        ax.set_xlabel("price per question c")
        ax.set_title(CRIT[criterion][0], pad=12)
    axes[0].set_ylabel("decision loss saved\n(false escalation units)")
    secondary = axes[-1].secondary_yaxis("right", functions=(lambda v: v * minutes, lambda v: v / minutes))
    secondary.set_ylabel(f"loss equivalent minutes\nper question (m = {minutes})")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.49, 0), labelspacing=0.9)
    save(fig, "f5_value" if r == 10 else f"f5_value_r{r}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        admissible()
        for ratio in (10, 5, 20):
            starts(ratio)
            budget(ratio, 0.01)
            value(ratio)
        for price in (0.001, 0.03, 0.1, 0.3):
            budget(10, price)
    print("Wrote 14 website plots as SVG and 600 dpi PNG files to", OUT)


if __name__ == "__main__":
    main()
