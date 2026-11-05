"""Build a standalone Colab notebook from the reviewed Python implementation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(source):
    cells.append(dict(cell_type="markdown", metadata={}, source=source.splitlines(True)))


def code(source):
    cells.append(dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                      source=source.splitlines(True)))


md(r"""# Query-Free Adversarial Attack against Stable Diffusion

An independent research implementation of **Zhuang, Zhang and Liu (2023)**,
[A Pilot Study of Query-Free Adversarial Attack against Stable Diffusion](https://arxiv.org/abs/2303.16378).
[Authors' reference implementation](https://github.com/OPTML-Group/QF-Attack).

**Run:** upload this `.ipynb` at [Google Colab](https://colab.research.google.com/), select
**Runtime > Change runtime type > T4 GPU**, and set **Runtime Version to 2026.07**
(Python 3.12), then run the cells in order. The pinned installation targets Python 3.12.
[Google's runtime-version instructions](https://research.google.com/colaboratory/runtime-version-faq.html).
Model weights require several GB of disk space. This notebook is self-contained;
you do not need to upload `qf_attack.py` or the PDF. Keep a fresh runtime for package installation.
If Colab requests a restart after installation, restart and continue from cell 2.

The default experiment attacks two benign scenes using a five-character suffix
(plus a separating space). Search uses only the local CLIP text encoder. Diffusion
is called **after search**, with identical initial noise across clean/attacked conditions.
The model's default safety checker remains enabled; filtered outputs are logged and excluded in pairs.

**Scope:** equations (1)-(2), genetic and coordinate-greedy search, one-sample random,
and a genetic-budget-matched random control. PGD is not reproduced. The fixed
contexts, operator choices and extended dataset are declared deviations from the paper.
The saved outputs are actual run measurements, never copies of the paper's scores.
""")
code('''# 1. Install known versions before importing the model libraries.
import importlib.util, subprocess, sys
try:
    IN_COLAB = importlib.util.find_spec("google.colab") is not None
except ModuleNotFoundError:
    IN_COLAB = False
if IN_COLAB:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Select Runtime > Change runtime type > Runtime Version 2026.07 (Python 3.12), then rerun.")
    packages = ''' + repr([line for line in (ROOT / "requirements-colab.txt").read_text().splitlines() if line]) + '''
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])
else:
    print("Local execution: use requirements-colab.txt; installation cell skipped.")
''')
md(r"""## Objective and targeted mask

For the complete CLIP final hidden-state tensor $H(x)\in\mathbb{R}^{77\times768}$,
the untargeted objective is $\min_s\cos(\mathrm{vec}H(x),\mathrm{vec}H(x+\text{space}+s))$.
The suffix $s$ contains exactly five characters from the reference alphabet.

For ten fixed context pairs, define $d_i=H(c_i+\text{and}+t)-H(c_i)$ and
$I_j=\mathbf{1}\{|\sum_i\mathrm{sign}(d_{ij})|>0.9n\}$.
The targeted objective minimizes cosine after multiplying **both** tensors by $I$.
With $n=10$, the strict threshold requires unanimous signs. An empty mask raises
an error instead of changing the threshold. Token states are flattened including
EOS padding, matching the authors' implementation. Image evaluation later uses
CLIP's projected, normalized pooled vectors, a different representation.

This objective does not enforce background preservation or even a direction of
concept removal. Evaluate target and retained concepts separately and inspect images.
""")
code("# 2. Write the standalone implementation (no GitHub clone required).\nfrom pathlib import Path\n" +
     "Path('qf_attack.py').write_text(" + repr((ROOT / "qf_attack.py").read_text()) + ", encoding='utf-8')\n" +
     "import qf_attack as qf\nprint('Implementation loaded; alphabet size:', len(qf.ALPHABET))\n")
code('''# 3. Experiment settings. The environment override is only for local verification.
import os
PROFILE = "demo"  # @param ["demo", "extended"]
RUN_IMAGES = True  # @param {type:"boolean"}
OUTPUT_DIR = "qf_results"  # @param {type:"string"}
SEED = 2026  # @param {type:"integer"}

factory = qf.Config if PROFILE == "demo" else qf.Config.full
cfg = factory(seed=SEED, run_images=RUN_IMAGES, output_dir=OUTPUT_DIR)
if os.environ.get("QF_TEXT_ONLY") == "1":
    cfg.run_images = False
if os.environ.get("QF_SD_REVISION"):
    cfg.sd_revision = os.environ["QF_SD_REVISION"]
print(cfg)
print("Selected device:", qf.choose_device(cfg.device))
print("Planned images:", cfg.n_prompts * len(cfg.image_seeds) * 9 if cfg.run_images else 0)
# Optional Hugging Face authentication, only if model access requires it:
# from huggingface_hub import notebook_login
# notebook_login()  # use the interactive widget; do not paste tokens in source code
''')
md("""## Search: zero diffusion queries

`demo`: 2 prompts, 10 GA generations, population 20, mutation probability 0.3,
one greedy sweep, one paired image seed and 20 denoising steps (18 images).
`extended`: 20 declared prompts, 50 generations, 50 denoising steps and 10 image
seeds per prompt (1,800 images across 9 conditions). This larger run may exceed
a free Colab session. Neither profile recovers the authors' exact 20-prompt dataset.

The matched random control receives exactly as many **unique candidate** encoder
evaluations as that case's genetic search. Mask construction, clean-reference
encoding and diagnostic rescoring are separately included in total encoder counts.
Searches have independent saved RNG seeds. No diffusion result selects a suffix.
""")
code('''# 4. Run both untargeted and targeted search; checkpoint rows after every method.
attacks = qf.run_search(cfg)
import pandas as pd
attack_df = pd.DataFrame(attacks)
display(attack_df[["case_id", "mode", "method", "suffix", "objective_cosine",
                   "full_cosine", "masked_cosine", "unique_candidates", "search_seconds"]])
assert (attack_df.diffusion_queries_during_search == 0).all()
assert attack_df.suffix.map(len).eq(5).all()
''')
code('''# 5. Plot measured encoder objectives. These are NOT image CLIP scores.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, mode in zip(axes, ("untargeted", "targeted")):
    part = attack_df[attack_df["mode"] == mode]
    table = part.pivot(index="case_id", columns="method", values="objective_cosine")
    table.plot.bar(ax=ax)
    ax.set(title=mode, xlabel="Prompt ID", ylabel="Text-text cosine (lower is better)")
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(Path(cfg.output_dir) / "encoder_objectives.png", dpi=180)
plt.show()
''')
md("""## Paired image generation and evaluation

Every clean/attack condition for a scene and seed receives a clone of the **same
initial latent tensor**. The latent hash, scheduler configuration, model commit,
guidance scale, resolution, sampling steps and safety flags are saved.

Scores are **raw CLIP cosine**, not probabilities or 100-scaled CLIPScore.
Untargeted scores compare the image with the **original clean prompt**. Targeted
scores use `This is a photo of <target>`. Retained-content scores use the parallel
retained phrase. Positive paired drop = clean score minus attacked score.
A target score drop alone does not prove object removal. Avoid defining a success
rate from an arbitrary score threshold. With fewer than five independent prompts,
confidence intervals are intentionally omitted; with more prompts, bootstrap at
the prompt level, averaging seeds first.
""")
code('''# 6. Generate and score images AFTER the encoder search.
if cfg.run_images:
    image_rows = qf.generate_images(cfg, attacks)
    image_summary = qf.evaluate_images(cfg, image_rows)
    display(pd.DataFrame(image_summary))
else:
    print("Text-only run: image generation and image CLIP scores were not measured.")
''')
code('''# 7. Inspect paired image grids, including random controls.
from IPython.display import Image, Markdown, display
for path in sorted(Path(cfg.output_dir).glob("grid_*.png")):
    display(Markdown("**" + path.name + "**"))
    display(Image(filename=str(path)))
''')
code('''# 8. Generate the analysis directly from this run and download all evidence.
report_path = qf.write_run_report(cfg)
display(Markdown(report_path.read_text()))
import shutil
archive = shutil.make_archive(cfg.output_dir + "_bundle", "zip", cfg.output_dir)
print("Saved:", archive)
if IN_COLAB:
    from google.colab import files
    files.download(archive)
''')
md("""## Read the outputs

- `run_report.md`: actual results and execution status for this run.
- `attack_results.csv`: suffixes, both cosine diagnostics, budgets, timings and seeds.
- `search_traces.json`: convergence histories and scored candidates.
- `mask_*.npy`, `mask_statistics.json`: targeted masks and selected-coordinate ratios.
- `manifest.json`: model commits, software versions, dataset, device and code hash.
- `image_scores.csv`: original/target/retained image CLIP cosine and safety flags.
- `paired_drops.csv`, `image_summary.csv`: paired differences and prompt-level summaries.
- `image_manifest.csv`, `scheduler.json`, `images/`, `grid_*.png`: generation evidence.

Keep random and optimized suffixes in the report, including negative or null results.
For claims of selective removal, add blinded human or independent detector labels
for target presence and retained-content preservation; this notebook does not
pretend CLIP similarity is an object detector. Encoder and evaluator share a model
family, so an independent evaluator is a useful follow-up.

Paper Table 1 reports untargeted genetic mean 0.203 vs. clean 0.277 and targeted
genetic mean 0.186 vs. clean 0.229. These are **published reference values**, not
expected outputs or measurements from this notebook. PGD is outside this implementation.
""")

optimization_path = ROOT / "Attack_Optimization.md"
if optimization_path.exists():
    md(optimization_path.read_text(encoding="utf-8"))

for i, cell in enumerate(cells):
    cell["id"] = f"qf-cell-{i:02d}"
nb = dict(cells=cells, metadata=dict(
    colab=dict(name="QF_Attack_Colab.ipynb", provenance=[]), accelerator="GPU",
    kernelspec=dict(display_name="Python 3", language="python", name="python3"),
    language_info=dict(name="python", version="3.12")), nbformat=4, nbformat_minor=5)
(ROOT / "QF_Attack_Colab.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Built", ROOT / "QF_Attack_Colab.ipynb")
