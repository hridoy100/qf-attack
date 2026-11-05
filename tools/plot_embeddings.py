"""Draw the report figures from qf_results/. Sized to print at full text width."""
from pathlib import Path
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / os.environ.get("QF_OUTPUT_DIR", "qf_results")
DATA = RUN / "embedding_analysis"
WIDTH = 6.26  # inches: the report's text width

# Colour means the same thing in every figure: blue = full, orange = masked.
FULL, MASKED = "#2a78d6", "#eb6834"
INK, MUTED, GRID, CLOUD, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#d3d2cc", "#8a8983"
COLOR = {"untargeted": FULL, "targeted": MASKED}
OBJECTIVE = {"untargeted": "Full", "targeted": "Masked"}
METHOD = {"random": "Random (1)", "greedy": "Greedy", "genetic": "Genetic", "budget_random": "Random (budget)"}
MARKER = {"random": "o", "greedy": "s", "genetic": "^", "budget_random": "D"}
SCENE = {0: "Bicycle scene", 1: "Apple scene"}
TARGET_POSITION = {0: 3, 1: 6}

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "font.serif": ["Times New Roman", "Times", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"], "text.parse_math": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.linewidth": 0.6, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "axes.titlecolor": INK, "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "figure.dpi": 300, "savefig.facecolor": "white",
})

metrics = list(csv.DictReader((DATA / "suffix_embedding_metrics.csv").open(encoding="utf-8")))
attack_rows = list(csv.DictReader((RUN / "attack_results.csv").open(encoding="utf-8")))
summary = json.loads((DATA / "summary.json").read_text())
similarity = json.loads((DATA / "similarity_map.json").read_text())
tokens = json.loads((DATA / "tokens.json").read_text(encoding="utf-8"))
positions = list(csv.DictReader((DATA / "distance_by_position.csv").open()))


def row_for(case, mode, method):
    return next(r for r in metrics if int(r["case_id"]) == case and r["mode"] == mode and r["method"] == method)


def tidy(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def scene_grid(case):
    """Clean image beside the 2 x 4 attacked images, with suffix and masked/full cosine."""
    fig = plt.figure(figsize=(WIDTH, 2.72))
    grid = fig.add_gridspec(2, 5, left=0.002, right=0.998, top=0.885, bottom=0.02, wspace=0.05, hspace=0.36)

    def show(ax, name, title):
        ax.imshow(Image.open(RUN / "images" / f"{case:02d}_27_{name}.png").convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=7.8, loc="center", pad=2.5, linespacing=1.05)

    show(fig.add_subplot(grid[0, 0]), "clean_clean", "Clean prompt\nno suffix")
    note = fig.add_subplot(grid[1, 0])
    note.axis("off")
    note.text(0.0, 0.84, "Top row: full objective", fontsize=8, va="top", transform=note.transAxes)
    note.text(0.0, 0.66, "Bottom row: masked\nobjective", fontsize=8, va="top", transform=note.transAxes)
    note.text(0.0, 0.36, "Above each image:\nmethod, suffix and\nobjective cosine", fontsize=8, va="top", color=MUTED, transform=note.transAxes)
    for r, (mode, column) in enumerate((("untargeted", "full_cosine"), ("targeted", "masked_cosine"))):
        for c, method in enumerate(METHOD):
            row = row_for(case, mode, method)
            show(fig.add_subplot(grid[r, c + 1]), f"{mode}_{method}",
                 f"{METHOD[method]}\n{row['suffix']}  {float(row[column]):.3f}")
    fig.savefig(DATA / f"fig_images_{'bicycle' if case == 0 else 'apple'}.png")
    plt.close(fig)


def token_change():
    """One worked example per scene, token by token: what changes and what the mask selects."""
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.65), sharey="row", layout="constrained",
                             gridspec_kw={"height_ratios": [1, 1]})
    for case in (0, 1):
        names = tokens[f"{case}/targeted/greedy"]
        row = row_for(case, "targeted", "greedy")
        distance = np.array([float(r["l2_distance"]) for r in positions if int(r["case_id"]) == case
                             and r["mode"] == "targeted" and r["method"] == "greedy"])
        selected = np.array(summary[case]["mask_fraction_by_position"]) * 100
        n = len(names)
        clean_end = summary[case]["clean_tokens"] - 1  # first position the suffix can reach
        labels = [{"<|startoftext|>": "[start]", "<|endoftext|>": "[end]"}.get(t, t) for t in names]
        labels.append(f"[padding]\n{77 - n} positions")
        x = np.arange(n + 1)
        change = np.append(distance[:n], distance[n:].mean())
        chosen = np.append(selected[:n], selected[n:].mean())
        top, bottom = axes[0][case], axes[1][case]

        top.bar(x, change, width=0.78, color=MASKED)
        top.set_title(f"{SCENE[case]}: suffix {row['suffix']} (Greedy, masked)")
        top.axvspan(-0.5, clean_end - 0.5, color="#f1f0ec", zorder=0, lw=0)
        top.text((clean_end - 1) / 2, 19, "prompt tokens\nunchanged", ha="center", va="center", fontsize=8.5, color=INK)
        top.set_ylim(0, 50)
        top.set_yticks([0, 10, 20, 30, 40])
        share = float(row["share_of_squared_change_after_end_token"]) * 100
        top.text(n + 0.45, 44, f"Padding holds {share:.1f}% of the change", ha="right", va="center", fontsize=8.5, color=INK)
        bottom.bar(x, chosen, width=0.78, color=NEUTRAL)
        bottom.axvspan(-0.5, clean_end - 0.5, color="#f1f0ec", zorder=0, lw=0)
        bottom.set_ylim(0, 60)
        bottom.set_yticks([0, 20, 40, 60])
        bottom.set_xticks(x, labels, rotation=62, ha="right", rotation_mode="anchor", fontsize=8)
        bottom.text(n, chosen[-1] + 2, f"{chosen[-1]:.0f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
        target = TARGET_POSITION[case]
        for tick in bottom.get_xticklabels():
            if tick.get_text() == names[target]:
                tick.set_fontweight("bold")
        bottom.annotate("target word:\n0% selected", (target, 0.5), xytext=(target, 30), ha="center", va="center",
                        fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkB=0))
        top.set_xticks(x, [])
        top.set_xlim(-0.6, n + 0.6)
        bottom.set_xlim(-0.6, n + 0.6)
        tidy(top)
        tidy(bottom)
    axes[0][0].set_ylabel("Change from clean\n(L2 distance)")
    axes[1][0].set_ylabel("Mask dimensions\nselected (%)")
    fig.savefig(DATA / "fig_token_change.png")
    plt.close(fig)


def similarity_plane():
    """Both axes are measured cosines, so nothing is projected or distorted."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.5), layout="constrained")
    for case, ax in enumerate(axes):
        m = similarity[str(case)]
        base = m["clean_to_target_free"]
        ax.scatter(m["random_to_clean"], m["random_to_target_free"], s=7, color=CLOUD, linewidths=0, zorder=1)
        ax.plot([1, base], [base, 1], ls=(0, (4, 3)), color=NEUTRAL, lw=0.9, zorder=1)
        ax.plot([0.56, 1], [base, base], color=NEUTRAL, lw=0.6, zorder=1)
        ax.scatter([1], [base], s=70, marker="*", color=INK, zorder=4, linewidths=0)
        ax.scatter([base], [1], s=52, marker="P", color=INK, edgecolors="white", linewidths=0.5, zorder=4)
        ax.annotate(f"Clean\nprompt\n{base:.3f}", (1, base), xytext=(7, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=8.5, color=INK)
        ax.annotate(f"\"{m['target_free_prompt']}\"", (base, 1), xytext=(0, 6), textcoords="offset points",
                    ha="center", fontsize=8.5, color=INK)
        ax.annotate("direction of the\ntarget-free prompt", (0.5 * (1 + base), 0.5 * (1 + base)), xytext=(7, 5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.2, color=MUTED)
        for r in (x for x in metrics if int(x["case_id"]) == case):
            x_value, y_value = float(r["full_cosine"]), float(r["cosine_to_retained_only"])
            ax.scatter(x_value, y_value, s=26, marker=MARKER[r["method"]], color=COLOR[r["mode"]],
                       edgecolors="white", linewidths=0.5, zorder=3)
            if r["mode"] == "targeted" and r["method"] == "greedy":
                place = dict(xytext=(-5, -1), ha="right", va="center") if case == 0 else dict(xytext=(0, 6), ha="center")
                ax.annotate(r["suffix"], (x_value, y_value), textcoords="offset points", fontsize=8, color=INK,
                            zorder=4, **place)
        ax.set_xlim(0.56, 1.13)
        ax.set_xticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_ylim(0.38, 1.08)
        ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_title(f"{SCENE[case]}: \"{m['prompt']}\"", fontsize=9)
        ax.set_xlabel("Cosine to the clean prompt")
        tidy(ax, "both")
    axes[0].set_ylabel("Cosine to the target-free prompt")
    handles = [Line2D([], [], ls="", marker="o", color=CLOUD, ms=4, label="500 random suffixes"),
               Line2D([], [], ls="", marker="s", color=FULL, ms=5, label="Full objective"),
               Line2D([], [], ls="", marker="s", color=MASKED, ms=5, label="Masked objective")]
    handles += [Line2D([], [], ls="", marker=MARKER[k], color=MUTED, ms=4.5, label=METHOD[k]) for k in METHOD]
    fig.legend(handles=handles, loc="outside lower center", ncols=4, frameon=False, fontsize=8.2, columnspacing=1.3, handletextpad=0.2)
    fig.savefig(DATA / "fig_similarity_plane.png")
    plt.close(fig)


def searches_vs_random():
    """Each row is one way of choosing a suffix; bands show typical random suffixes."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.95), sharey=True, layout="constrained")
    objectives = (("untargeted", "full_cosine", "random_full_cosines", 0.17),
                  ("targeted", "masked_cosine", "random_masked_cosines", -0.17))
    for case, ax in enumerate(axes):
        budgets = [int(r["unique_candidates"]) for r in attack_rows if r["method"] == "genetic"]
        low, high = min(budgets), max(budgets)
        labels = ["Typical random\n(middle 90%)", "Random (1)\n1 suffix", f"Genetic\n{low}–{high} suffixes",
                  f"Random (budget)\n{low}–{high} suffixes", "Greedy\n306 suffixes", "Best random\n500 suffixes"]
        for mode, column, key, shift in objectives:
            values = np.array(summary[case][key])
            color = COLOR[mode]
            q05, q95 = np.quantile(values, [.05, .95])
            ax.plot([q05, q95], [5 + shift] * 2, color=color, lw=3.2, solid_capstyle="butt", alpha=0.45)
            ax.text(q95 + 0.007, 5 + shift, f"{q05:.2f}–{q95:.2f}", va="center", fontsize=7.8, color=INK)
            points = [(4, row_for(case, mode, "random")[column]), (3, row_for(case, mode, "genetic")[column]),
                      (2, row_for(case, mode, "budget_random")[column]), (1, row_for(case, mode, "greedy")[column]),
                      (0, values.min())]
            for y, value in points:
                value = float(value)
                ax.scatter(value, y + shift, s=20, color=color, zorder=3, linewidths=0)
                ax.text(value + 0.007, y + shift, f"{value:.3f}", va="center", fontsize=7.8, color=INK)
        ax.set_yticks(range(6), labels[::-1], fontsize=8, linespacing=1.0)
        ax.set_ylim(-0.55, 5.55)
        ax.set_xlim(0.54, 1.0)
        ax.set_title(SCENE[case])
        ax.set_xlabel("Cosine to clean (lower = larger change)")
        tidy(ax, "x")
        ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], ls="", marker="o", color=FULL, ms=4.5, label="Full objective"),
               Line2D([], [], ls="", marker="o", color=MASKED, ms=4.5, label="Masked objective")]
    fig.legend(handles=handles, loc="outside lower center", ncols=2, frameon=False, fontsize=8.2)
    fig.savefig(DATA / "fig_searches_vs_random.png")
    plt.close(fig)


def text_vs_image():
    """Does a larger change in the text embedding give a larger drop in the image score?"""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.55), layout="constrained")
    panels = (("untargeted", "full_cosine", "full_drop", "Full objective", "Drop in score for the original prompt"),
              ("targeted", "masked_cosine", "target_drop", "Masked objective", "Drop in score for the target"))
    # Symmetric y-limit covering every plotted drop, so no point is clipped.
    largest = max(abs(float(r[y_key])) for mode, _, y_key, _, _ in panels for r in metrics if r["mode"] == mode)
    y_lim = max(0.065, np.ceil(largest * 1.1 / 0.005) * 0.005)
    for ax, (mode, x_key, y_key, title, label) in zip(axes, panels):
        rows = [r for r in metrics if r["mode"] == mode]
        x = np.array([float(r[x_key]) for r in rows])
        y = np.array([float(r[y_key]) for r in rows])
        rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        ax.axhline(0, color=MUTED, lw=0.7, zorder=1)
        for r, x_value, y_value in zip(rows, x, y):
            ax.scatter(x_value, y_value, s=28, marker=MARKER[r["method"]],
                       color=COLOR[mode] if int(r["case_id"]) == 0 else "white",
                       edgecolors=COLOR[mode], linewidths=1, zorder=3)
        ax.set_title(f"{title}: rank correlation {rho:+.2f}".replace("-", "−"))
        ax.set_xlabel("Text cosine (further left = larger text change)")
        ax.set_ylabel(label, fontsize=8.5)
        ax.set_xlim(0.56, 0.89)
        ax.set_ylim(-y_lim, y_lim)
        ax.text(0.885, 0.89 * y_lim, "image score fell", ha="right", va="center", fontsize=8.2, color=MUTED)
        ax.text(0.885, -0.89 * y_lim, "image score rose", ha="right", va="center", fontsize=8.2, color=MUTED)
        tidy(ax, "both")
    handles = [Line2D([], [], ls="", marker="o", mfc=NEUTRAL, mec=NEUTRAL, ms=5, label="Bicycle scene"),
               Line2D([], [], ls="", marker="o", mfc="white", mec=NEUTRAL, ms=5, label="Apple scene")]
    handles += [Line2D([], [], ls="", marker=MARKER[k], color=MUTED, ms=4.5, label=METHOD[k]) for k in METHOD]
    fig.legend(handles=handles, loc="outside lower center", ncols=6, frameon=False, fontsize=8.2, columnspacing=1.1, handletextpad=0.2)
    fig.savefig(DATA / "fig_text_vs_image.png")
    plt.close(fig)


has_images = all(r["full_drop"] != "" for r in metrics)
if has_images:
    scene_grid(0)
    scene_grid(1)
token_change()
similarity_plane()
searches_vs_random()
if has_images:
    text_vs_image()
else:
    print("text-only run: skipped the image grids and the text-vs-image figure")
print("wrote figures to", DATA)
