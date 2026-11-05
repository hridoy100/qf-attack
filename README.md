# QF-Attack paper analysis and Colab reproduction

The supplied paper is **A Pilot Study of Query-Free Adversarial Attack against
Stable Diffusion**, Zhuang, Zhang and Liu (2023), [arXiv:2303.16378v2](https://arxiv.org/abs/2303.16378v2).

Open `QF_Attack_Colab.ipynb` by uploading it at [Google Colab](https://colab.research.google.com/).
Select **Runtime > Change runtime type > T4 GPU** and **Runtime Version 2026.07
(Python 3.12)**, then run the cells in order. See [Google's runtime-version
instructions](https://research.google.com/colaboratory/runtime-version-faq.html).
The notebook contains its own implementation and does not require any other
workspace file. Use a fresh runtime for dependency installation; restart if Colab
requests it. The image models require several GB of downloads.

The default demo runs two scenes, both targeted and untargeted objectives,
genetic and greedy attacks, one-sample random and genetic-budget-matched random
controls. It generates 18 images using one paired seed and nine conditions at
20 denoising steps. The extended profile uses 50 denoising steps.
Search makes zero diffusion calls. Diffusion runs only afterward for evaluation.
The five-character budget excludes the separating space.

The final cell downloads a ZIP with the measured report, suffixes, search traces,
masks, image scores, paired differences, grids and model/software provenance.
Set `RUN_IMAGES=False` for encoder-only results. The `extended` profile uses
20 declared scenes, 50 genetic generations and ten image seeds; it generates 1,800 images and can
exceed a free Colab session.

## Running on an HPC cluster (e.g. Argon)

Run the image stage on a GPU node; the search, PGD, analysis, detector and
report steps are CPU-only. The models (~5 GB) are downloaded from Hugging Face,
so the node needs network access. Do not set `HF_HUB_OFFLINE`: the run resolves
each model's exact commit through the Hugging Face API, which fails in offline
mode even when the models are cached.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-colab.txt reportlab
python qf_attack.py --profile demo      # search + 18 images (needs CUDA)
python tools/validate_results.py
python tools/pgd_attack.py --iters 100 --batch 16
python tools/embedding_analysis.py
python tools/plot_embeddings.py
python tools/detector_scoring.py        # downloads OWL-ViT
python tools/plot_detector.py
python tools/build_paper.py             # output/pdf/QF_Attack_Report.pdf
python tools/build_latex.py             # QF_Attack_Report.tex
```

`reportlab` is required by `build_paper.py` and is not in the pinned Colab list,
so it is installed explicitly above. Steps must run in this order: the analysis,
detector and report steps read the outputs written by `qf_attack.py`.

This is an independent implementation of the paper's objectives and discrete
search methods, with a text-domain reproduction of the paper's PGD baseline in
`tools/pgd_attack.py` (encoder-only; no images are generated for its suffixes).
Genetic operator details, fixed mask
contexts and the larger prompt dataset differ from the authors' experiments.
The PDF analysis separates published values from actual measurements. Do not
treat encoder cosine changes as image-level attack success or infer reliable
target removal from CLIP score alone.

Files:

- `QF_Attack_Report.md`, `QF_Attack_Report.tex`, `output/pdf/QF_Attack_Report.pdf`: the report.
- `QF_Attack_Colab.ipynb`: clean, self-contained notebook for Colab (includes the PGD cell).
- `QF_Attack_Executed.ipynb`: the same notebook with local text-only execution outputs.
- `QF_Attack_Analysis.md`, `output/pdf/QF_Attack_Analysis.pdf`: earlier analysis report.
- `Attack_Optimization.md`: proposed improvements (not implemented or tested); also appended to the notebook.
- `qf_attack.py`: standalone implementation.
- `tools/`: PGD, embedding analysis, detector scoring, figures, report and notebook builders.
- `requirements-colab.txt`: Colab dependency pins.
- `qf_results/`: actual measurements and generation evidence.

Local command after installing dependencies:

```bash
python qf_attack.py --profile demo
python qf_attack.py --profile demo --text-only
python tools/validate_results.py
```

The recorded local environment uses Apple MPS; the Colab installation cell was
skipped locally. Hosted Colab execution has not been performed. CPU-only mode
supports encoder experiments; the image stage requires a GPU.

To regenerate the notebook from an edited implementation:

```bash
python tools/build_notebook.py
```

Reference implementation: [OPTML-Group/QF-Attack](https://github.com/OPTML-Group/QF-Attack).
