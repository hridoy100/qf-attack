"""Draw the detector figure from qf_results/embedding_analysis/detector_scores.csv."""
from pathlib import Path
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / os.environ.get("QF_OUTPUT_DIR", "qf_results") / "embedding_analysis"
WIDTH = 6.26
THRESHOLD = 0.10
FULL, MASKED, INK, MUTED, GRID, NEUTRAL = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df", "#8a8983"
COLOR = {"untargeted": FULL, "targeted": MASKED, "clean": INK}
SCENE = {0: "Bicycle scene", 1: "Apple scene"}
METHOD = {"clean": "Clean image", "random": "Random (1)", "greedy": "Greedy",
          "genetic": "Genetic", "budget_random": "Random (budget)"}
ORDER = [("clean", "clean"), ("untargeted", "random"), ("untargeted", "greedy"),
         ("untargeted", "genetic"), ("untargeted", "budget_random"), ("targeted", "random"),
         ("targeted", "greedy"), ("targeted", "genetic"), ("targeted", "budget_random")]

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "text.parse_math": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6, "axes.titlesize": 9.5,
    "axes.titlecolor": INK, "axes.titlelocation": "left", "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "figure.dpi": 300, "savefig.facecolor": "white",
})

rows = list(csv.DictReader((DATA / "detector_scores.csv").open()))
fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95), sharey=True, layout="constrained")
for case, ax in enumerate(axes):
    labels = []
    for y, (mode, method) in enumerate(reversed(ORDER)):
        r = next(x for x in rows if int(x["case_id"]) == case and x["mode"] == mode and x["method"] == method)
        labels.append(METHOD[method])
        tgt, ret = float(r["target_confidence"]), float(r["retain_confidence"])
        ax.scatter(tgt, y, s=34, marker="o", color=COLOR[mode], edgecolors="white", linewidths=0.5, zorder=3)
        ax.scatter(ret, y, s=26, marker="x", color=NEUTRAL, linewidths=1.1, zorder=2)
    ax.axvline(THRESHOLD, color=MUTED, ls=(0, (4, 3)), lw=0.8)
    ax.text(THRESHOLD + 0.01, 9.15, "detection threshold", fontsize=7.6, color=MUTED, va="center")
    ax.set_yticks(range(9), labels, fontsize=8)
    ax.set_ylim(-0.6, 9.5)
    ax.set_xlim(0, 0.9)
    ax.set_title(f"{SCENE[case]}: target \"{rows and next(x for x in rows if int(x['case_id'])==case)['target_query']}\"")
    ax.set_xlabel("Detector confidence")
    ax.grid(axis="x", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
handles = [Line2D([], [], ls="", marker="o", color=FULL, ms=5, label="Target, full objective"),
           Line2D([], [], ls="", marker="o", color=MASKED, ms=5, label="Target, masked objective"),
           Line2D([], [], ls="", marker="o", color=INK, ms=5, label="Target, clean image"),
           Line2D([], [], ls="", marker="x", color=NEUTRAL, ms=5, mew=1.1, label="Retained object")]
fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=8, columnspacing=1.2, handletextpad=0.3)
fig.savefig(DATA / "fig_detector.png")
print("wrote", DATA / "fig_detector.png")
