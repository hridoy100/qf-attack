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
Model weights require several GB of disk space. The code lives in the repository's
`qf_attack.py` and `tools/`: run the notebook from a checkout of
[the repository](https://github.com/hridoy100/qf-attack), or upload only this `.ipynb`
and cell 2 clones the repository. Keep a fresh runtime for package installation.
If Colab requests a restart after installation, restart and continue from cell 2.

The default experiment attacks two benign scenes using a five-character suffix
(plus a separating space). Search uses only the local CLIP text encoder. Diffusion
is called **after search**, with identical initial noise across clean/attacked conditions.
The model's default safety checker remains enabled; filtered outputs are logged and excluded in pairs.

**Scope:** equations (1)-(2), genetic and coordinate-greedy search, one-sample random,
a genetic-budget-matched random control, and a gradient-based PGD baseline
(encoder-only; no images are generated for its suffixes). For the `demo` profile
it also runs the report's embedding analysis and an OWL-ViT object-detector check.
The fixed contexts, operator choices and extended dataset are declared deviations from the paper.
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
code('''# 2. Load the implementation (qf_attack.py) and analysis scripts (tools/) from the repository.
import os, warnings
from pathlib import Path
REPO_URL = "https://github.com/hridoy100/qf-attack.git"
if not Path("qf_attack.py").exists():
    # Only the notebook is here (for example, uploaded to Colab): fetch the code.
    if not Path("qf-attack/qf_attack.py").exists():
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "qf-attack"], check=True)
    os.chdir("qf-attack")
sys.path.insert(0, str(Path.cwd()))
print("Working directory:", Path.cwd())
# Without ipywidgets, tqdm warns and falls back to text progress bars; that is fine here.
warnings.filterwarnings("ignore", message="IProgress not found")
import qf_attack as qf
print("Implementation loaded; alphabet size:", len(qf.ALPHABET))
''')
code('''# 3. Experiment settings. The environment override is only for local verification.
import os
PROFILE = "demo"  # @param ["demo", "extended"]
RUN_IMAGES = True  # @param {type:"boolean"}
# qf_results/ in the repository holds the committed run. For a fresh run, use a new name:
# the same folder reuses its existing images, or stops if the image settings differ.
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
md(r"""## PGD: gradient-based baseline (encoder-only)

The paper also reports a gradient attack (PGD). Each of the five suffix slots holds
a relaxed distribution over the alphabet; Gumbel-softmax draws near-discrete samples
and a sign-gradient step maximizes $1/\cos$ for 100 iterations. The final discrete
suffix is scored with the same flattened-embedding cosine as the other methods, so
it is directly comparable to the Table above. This is a reproduction of a published
baseline, not a stronger attack. No diffusion model is involved and no images are
generated for PGD suffixes. It runs on the GPU when available, falling back to CPU
on Apple MPS, where Gumbel-softmax is unsupported.
""")
code('''# 5b. PGD (gradient-based) attack on the text embedding. Encoder-only; no images.
import dataclasses
import numpy as np
import torch
import torch.nn.functional as F
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

PGD_ITERS, PGD_BATCH, PGD_ETA, PGD_SEED = 100, 16, 1.0, SEED
_dev = qf.choose_device(cfg.device)
pgd_device = "cpu" if _dev.type == "mps" else str(_dev)  # Gumbel-softmax unsupported on MPS
pgd_cfg = dataclasses.replace(cfg, device=pgd_device)
pgd_oracle = qf.TextOracle(pgd_cfg)
_tm = pgd_oracle.model.text_model
_A = qf.ALPHABET

def _tok_pos(text):
    ids = pgd_oracle.tokenizer(text, padding="max_length", truncation=True,
                               max_length=77, return_tensors="pt").input_ids.to(pgd_device)
    with torch.no_grad():
        return _tm.embeddings(input_ids=ids)

def _forward(embeds):
    m = _create_4d_causal_attention_mask((1, embeds.shape[1]), embeds.dtype, device=embeds.device)
    out = _tm.encoder(inputs_embeds=embeds, causal_attention_mask=m)[0]
    return _tm.final_layer_norm(out)

def _candidate_embeds():
    rows = [_tok_pos((ch + " ") * 75)[0] for ch in _A]  # [77, dim] per character
    return torch.stack(rows, dim=1)                     # [77, C, dim]

def _run_pgd(prompt, mask, length, iters, batch, eta, seed):
    torch.manual_seed(seed)
    candidate = _candidate_embeds().to(pgd_device)
    orig = _tok_pos(prompt)
    with torch.no_grad():
        clean_out = _forward(orig).flatten()
    mask_vec = mask.to(pgd_device) if mask is not None else torch.ones_like(clean_out)
    clean_ref = F.normalize(clean_out * mask_vec, dim=0)
    num_word = len(prompt.split(" "))
    z = torch.zeros(77, 1, device=pgd_device)
    z[num_word + 1:num_word + length + 1] = 1.0
    u = torch.full((77, len(_A)), 1.0 / len(_A), device=pgd_device, requires_grad=True)
    base = orig[0]
    def embed_from(onehot):
        picked = torch.einsum("pc,pcd->pd", onehot, candidate)
        return ((1 - z) * base + z * picked).unsqueeze(0)
    def masked_cos(out):
        v = F.normalize(out.flatten() * mask_vec, dim=0)
        return torch.dot(v, clean_ref)
    def suffix_of(sample):
        idx = sample[num_word + 1:num_word + length + 1].argmax(dim=1)
        return "".join(_A[i] for i in idx.tolist())
    best_suffix, best_loss = None, -float("inf")
    lr = eta / np.sqrt(iters)
    for _ in range(iters):
        samples = [F.gumbel_softmax(u, tau=0.5, hard=True, dim=-1) for _ in range(batch)]
        losses = [1.0 / masked_cos(_forward(embed_from(s))) for s in samples]
        ranked = sorted(zip(losses, samples), key=lambda x: x[0].item(), reverse=True)
        if ranked[0][0].item() > best_loss:
            best_loss = ranked[0][0].item()
            best_suffix = suffix_of(ranked[0][1].detach())
        total = sum(loss for loss, _ in ranked[:5])
        if u.grad is not None:
            u.grad.zero_()
        total.backward()
        with torch.no_grad():
            u += lr * torch.sign(u.grad)
    return best_suffix

pgd_rows = []
for case_id, (prompt, target, retain) in enumerate(qf.CASES[:cfg.n_prompts]):
    mask = torch.from_numpy(np.load(Path(cfg.output_dir) / f"mask_{case_id:02d}.npy")).float()
    reference = pgd_oracle.encode([prompt]).flatten()
    for mode in ("untargeted", "targeted"):
        mode_mask = None if mode == "untargeted" else mask
        suffix = _run_pgd(prompt, mode_mask, cfg.suffix_length, PGD_ITERS, PGD_BATCH, PGD_ETA,
                          PGD_SEED + 1000 * case_id + (mode == "targeted"))
        h = pgd_oracle.encode([prompt + " " + suffix]).flatten()
        full = float(F.cosine_similarity(reference, h, dim=0))
        masked = float(F.cosine_similarity(reference * mask, h * mask, dim=0))
        pgd_rows.append(dict(case_id=case_id, prompt=prompt, mode=mode, method="pgd", suffix=suffix,
                             objective_cosine=masked if mode == "targeted" else full,
                             full_cosine=full, masked_cosine=masked, iters=PGD_ITERS, batch=PGD_BATCH))
        print(f"{case_id} {mode:10s} pgd {suffix!r} full={full:.4f} masked={masked:.4f}", flush=True)

pgd_df = pd.DataFrame(pgd_rows)
pgd_df.to_csv(Path(cfg.output_dir) / "pgd_results.csv", index=False)
display(pgd_df[["case_id", "mode", "suffix", "objective_cosine", "full_cosine", "masked_cosine"]])
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
md("""## Report analysis: embedding, detector and figures

This cell runs the repository's `tools/` scripts to reproduce the analysis and
Figures 1-7 of the accompanying report.
The **embedding analysis** re-encodes every recorded suffix, checks that it
reproduces the recorded cosines, and measures where the change falls (token
positions, padding, the targeted mask) and how the suffixes compare with 500
random ones. The **detector check** scores every generated image with OWL-ViT,
an open-vocabulary object detector not used anywhere in the attack, to see
whether the target object is still present (threshold 0.10). The detector
model is downloaded on first use.

The scripts are written for the `demo` profile's two scenes and image seed 27,
so they are skipped for `extended`. In a text-only run the detector and the
two image-dependent figures are skipped. Outputs go to `<output dir>/embedding_analysis/`.
""")
code('''# 7b. Run the embedding analysis, the OWL-ViT detector check and the report figures (tools/).
import subprocess

def run_tool(name):
    env = dict(os.environ, QF_OUTPUT_DIR=cfg.output_dir)
    result = subprocess.run([sys.executable, "tools/" + name], env=env, capture_output=True, text=True)
    print(f"--- {name}\\n{result.stdout[-1500:]}")
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed:\\n{result.stderr[-3000:]}")

analysis_dir = Path(cfg.output_dir) / "embedding_analysis"
if PROFILE != "demo":
    print("Skipped: the report analysis is written for the demo profile's two scenes.")
else:
    run_tool("embedding_analysis.py")
    if cfg.run_images:
        run_tool("detector_scoring.py")
    run_tool("plot_embeddings.py")
    if cfg.run_images:
        run_tool("plot_detector.py")
        display(pd.read_csv(analysis_dir / "detector_summary.csv"))
    for name in ("fig_images_bicycle", "fig_images_apple", "fig_token_change", "fig_similarity_plane",
                 "fig_searches_vs_random", "fig_text_vs_image", "fig_detector"):
        path = analysis_dir / (name + ".png")
        if path.exists():
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
- `pgd_results.csv`: gradient-based PGD suffixes and their text-embedding cosines.
- `search_traces.json`: convergence histories and scored candidates.
- `mask_*.npy`, `mask_statistics.json`: targeted masks and selected-coordinate ratios.
- `manifest.json`: model commits, software versions, dataset, device and code hash.
- `image_scores.csv`: original/target/retained image CLIP cosine and safety flags.
- `paired_drops.csv`, `image_summary.csv`: paired differences and prompt-level summaries.
- `image_manifest.csv`, `scheduler.json`, `images/`, `grid_*.png`: generation evidence.
- `embedding_analysis/`: per-suffix embedding metrics, distances by token position,
  the 500-random-suffix reference, OWL-ViT detector scores and the report figures.

Keep random and optimized suffixes in the report, including negative or null results.
CLIP similarity is not an object detector. The OWL-ViT check above gives a
separate measure of target presence, but OWL-ViT also has a CLIP-based backbone,
so blinded human labels or a detector with no shared backbone would strengthen
claims of selective removal, especially for the retained content.

Paper Table 1 reports untargeted genetic mean 0.203 vs. clean 0.277 and targeted
genetic mean 0.186 vs. clean 0.229. These are **published reference values**, not
expected outputs or measurements from this notebook. The PGD cell above reproduces
the paper's gradient baseline at the encoder level only; its cosines are not directly
comparable to Table 1, which reports image CLIP scores.
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
