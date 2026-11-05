"""Build the paper analysis from the notebook's measured CSV/JSON evidence."""
from pathlib import Path
import csv
import json
import html
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf"
OUT.mkdir(parents=True, exist_ok=True)
RUN = ROOT / "qf_results"
manifest = json.loads((RUN / "manifest.json").read_text())
attacks = list(csv.DictReader((RUN / "attack_results.csv").open()))
masks = json.loads((RUN / "mask_statistics.json").read_text())
image_summary = list(csv.DictReader((RUN / "image_summary.csv").open())) if manifest["images_evaluated"] else []
image_scores = list(csv.DictReader((RUN / "image_scores.csv").open())) if manifest["images_evaluated"] else []
paired = list(csv.DictReader((RUN / "paired_drops.csv").open())) if manifest["images_evaluated"] else []
cfg = manifest["config"]

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image, KeepTogether)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.utils import ImageReader

NAVY = colors.HexColor("#183146")
TEAL = colors.HexColor("#137c83")
GRAY = colors.HexColor("#526171")
PALE = colors.HexColor("#edf4f6")
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="ReportTitle", fontName="Helvetica-Bold", fontSize=26,
                         leading=29, textColor=NAVY, spaceAfter=15))
styles.add(ParagraphStyle(name="Deck", fontName="Helvetica", fontSize=12,
                         leading=17, textColor=GRAY, spaceAfter=14))
styles.add(ParagraphStyle(name="Section", fontName="Helvetica-Bold", fontSize=17,
                         leading=21, textColor=NAVY, spaceAfter=10))
styles.add(ParagraphStyle(name="Sub", fontName="Helvetica-Bold", fontSize=10.5,
                         leading=14, textColor=TEAL, spaceBefore=8, spaceAfter=4))
styles.add(ParagraphStyle(name="Copy", fontName="Helvetica", fontSize=9.5,
                         leading=13.3, textColor=NAVY, spaceAfter=8))
styles.add(ParagraphStyle(name="SmallCopy", fontName="Helvetica", fontSize=8.1,
                         leading=11, textColor=GRAY, spaceAfter=6))
styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=8.2,
                         leading=10.5, textColor=NAVY))
styles.add(ParagraphStyle(name="HeaderCell", fontName="Helvetica-Bold", fontSize=8.2,
                         leading=10.5, textColor=colors.white))
styles.add(ParagraphStyle(name="Equation", fontName="Courier", fontSize=8.5,
                         leading=13, textColor=NAVY, backColor=PALE,
                         borderPadding=7, spaceBefore=4, spaceAfter=10))
story = []
markdown = []


def p(text, style="Copy"):
    story.append(Paragraph(text, styles[style]))


def title(text):
    p(text, "Section")
    markdown.extend(["", "## " + text, ""])


def prose(text, style="Copy"):
    p(text, style)
    import re
    markdown.extend([re.sub("<[^>]+>", "", text), ""])


def table(headers, rows, widths, small=False):
    data = [[Paragraph(html.escape(str(h)), styles["HeaderCell"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(html.escape(str(v)), styles["Cell"]) for v in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4 if small else 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 if small else 6),
        ("LINEBELOW", (0, -1), (-1, -1), .5, colors.HexColor("#ccd9df")),
    ]))
    story.extend([t, Spacer(1, 9)])
    markdown.append("| " + " | ".join(headers) + " |")
    markdown.append("|" + "---|" * len(headers))
    for row in rows:
        markdown.append("| " + " | ".join(str(v).replace("|", "&#124;") for v in row) + " |")
    markdown.append("")


def newpage():
    story.append(PageBreak())


def fmt(value, signed=False):
    if value in (None, ""):
        return "n/a"
    return format(float(value), "+.4f" if signed else ".4f")


def method_name(value):
    return {"random": "Random (1)", "budget_random": "Random (budget)",
            "genetic": "Genetic", "greedy": "Greedy"}[value]


markdown += ["# Query-free adversarial attacks: paper analysis and notebook evidence", "",
             "Prepared 4 November 2026. Notebook executed locally; ready for Google Colab upload.", ""]
p("Query-free adversarial attacks", "ReportTitle")
p("Paper analysis and notebook evidence", "Deck")
prose("<b>Source:</b> Haomin Zhuang, Yihua Zhang and Sijia Liu. <i>A Pilot Study of Query-Free "
      "Adversarial Attack against Stable Diffusion</i>, CVPR Workshop 2023, arXiv:2303.16378v2. "
      "The supplied eight-page PDF is the primary source. Prepared 4 November 2026.", "SmallCopy")
title("1. What the paper establishes")
prose("The study attacks a text-to-image system through its text conditioning interface. "
      "It appends a short, visible character string to an otherwise unchanged prompt, "
      "searching for text embeddings that differ from the original. Its main contribution "
      "is evidence that encoder-level perturbations can transfer into substantial changes "
      "in Stable Diffusion output without running diffusion during attack construction.")
prose("<b>Threat model.</b> The attacker can evaluate a locally accessible CLIP text encoder "
      "and modify the prompt. The victim is Stable Diffusion v1.4. Query-free refers specifically "
      "to the diffusion pipeline during search; encoder evaluations and post-search image "
      "generation still incur computation. This is a text-side attack on a multimodal system, "
      "not a joint text-and-image-input perturbation or a demonstration of safety-filter bypass.")
prose("<b>Targeted setting.</b> A mask selects embedding coordinates whose changes are consistent "
      "when a target phrase is added across contexts. Minimizing cosine only on these coordinates "
      "aims to suppress target-related image content. The loss does not explicitly preserve other "
      "objects and does not impose a signed direction toward a target-free representation.")
p("Published results - Table 1, PDF p. 4", "Sub")
table(["Method", "Untargeted image cosine", "Target-concept image cosine"], [
    ["No attack", "0.277 +/- 0.022", "0.229 +/- 0.030"],
    ["Random", "0.271 +/- 0.021", "0.223 +/- 0.037"],
    ["Greedy", "0.255 +/- 0.039", "0.204 +/- 0.037"],
    ["Genetic", "0.203 +/- 0.042", "0.186 +/- 0.040"],
    ["PGD", "0.226 +/- 0.041", "0.189 +/- 0.041"],
], [110, 195, 195], small=True)
prose("These are the paper's measurements, not this notebook's results. Each method is reported "
      "over 20 prompts with 10 images per prompt. Genetic search reduces the reported means by "
      "0.074 (26.7%) for untargeted alignment and 0.043 (18.8%) for target-concept alignment. "
      "Those percentages describe relative score reductions, not attack success rates.", "SmallCopy")

newpage()
title("2. How the notebook implements the attack")
prose("Let h(x) flatten the CLIP final hidden states to 59,136 values (77 tokens x 768 dimensions). "
      "The full sequence includes EOS padding, following the authors' reference implementation. "
      "This representation differs from CLIP's projected pooled text vector used for image scoring.")
p("Untargeted: min_s cosine(h(x), h(x + ' ' + s))<br/>"
  "Targeted:   min_s cosine(h(x) * I, h(x + ' ' + s) * I)", "Equation")
prose("For each context pair, d_i = h(context_i + ' and ' + target) - h(context_i). "
      "The mask is I_j = 1 when abs(sum_i sign(d_ij)) &gt; 0.9n. With n = 10 and a strict "
      "inequality, all ten nonzero signs must agree. An empty mask raises an error. "
      "The 62-character alphabet matches the reference set after removing its duplicated '*'; "
      "a five-character suffix has 62^5 = 916,132,832 possibilities.")
table(["Component", "Implementation and scope"], [
    ["Genetic", f"{cfg['generations']} generations, {cfg['population']} candidates, mutation probability 0.3. Two elites, 3-way tournament, one-point crossover, one-character mutation."],
    ["Greedy", "One coordinate sweep across all characters at each of five suffix positions, starting from a random five-character suffix."],
    ["Controls", "One random suffix; plus uniform random search matched to the genetic method's unique candidate evaluations separately per prompt and objective."],
    ["Target mask", "Ten fixed paired contexts supplied in code; no language-model API required. Concept phrase includes the article, e.g. 'a plate'."],
    ["PGD", "Reported in the paper but not implemented in this deliverable. No completeness or exact Table 1 replication claim."],
], [100, 400], small=True)
prose("<b>Measured run.</b> Two prompts: 'A black bicycle against a brick wall' "
      "(remove bicycle, retain wall) and 'A red apple on a plate' (remove plate, retain apple). "
      f"Search seed: {cfg['seed']}. Encoder device: {manifest['device']}. The notebook was executed "
      "locally through its actual cells; hosted Google Colab execution was not performed.")
prose(f"<b>Image protocol.</b> {len(cfg['image_seeds'])} paired {'seed' if len(cfg['image_seeds']) == 1 else 'seeds'} per scene ({', '.join(map(str,cfg['image_seeds']))}); "
      f"{cfg['steps']} inference steps, guidance {cfg['guidance_scale']}, {cfg['width']} x {cfg['height']} images. "
      "The saved scheduler is the checkpoint's PNDM configuration. The demo's 20 steps and one image seed "
      "are reductions for local validation; the extended profile uses 50 steps and ten seeds. Initial noise is drawn once on CPU "
      "and cloned for every condition; matching hashes are verified. The default safety checker remains enabled.")
prose("The extended profile provides 20 declared scenes and 50 GA generations; 12 scene prompts are "
      "new additions. It is not the authors' exact dataset. Different mask contexts, operators, "
      "sampling details and model-weight precision can change outcomes.", "SmallCopy")

newpage()
title("3. Measured encoder results")
prose("Lower objective cosine means a larger change in the selected text representation. "
      "Untargeted and targeted rows use different coordinate sets and should not be compared "
      "as if they were the same scale. The clean self-cosine is 1.0000.")
data = []
for r in attacks:
    data.append([r["case_id"], "Full" if r["mode"] == "untargeted" else "Masked",
                 method_name(r["method"]), r["suffix"], fmt(r["objective_cosine"]), r["unique_candidates"]])
table(["Case", "Objective", "Method", "Suffix", "Cosine", "Candidates"], data,
      [35, 60, 118, 80, 90, 117], small=True)
prose("Case 0 = bicycle/wall; case 1 = apple/plate. The separating space is outside the "
      "five-character suffix budget. Candidate counts exclude mask construction, clean-reference "
      "encoding and post-search diagnostics.", "SmallCopy")
prose("<b>Finding:</b> Genetic search beats budget-matched random search for the apple scene "
      "on both objectives, while the matched random control beats genetic search for the bicycle "
      "scene on both objectives. Greedy search has the lowest targeted objective on both scenes, "
      "but it evaluates 306 unique candidates per run, more than the 94-106 used by this short GA. "
      "This two-scene test does not establish a general ranking of optimizers.")
prose("The selected mask contains " + "; ".join(
    f"{m['selected']:,}/{m['total']:,} coordinates ({100*m['fraction']:.1f}%) for {m['target']}" for m in masks) +
      ". The mask is not especially sparse in this run. Sign consistency measures agreement, not "
      "effect magnitude or a causal role in object rendering.")
prose(f"The saved search manifest records {manifest['total_texts_encoded']:,} text encodings in "
      f"{manifest['total_forward_batches']:,} forward batches and zero diffusion calls. "
      f"Search-loop duration was {manifest['elapsed_seconds']:.1f} seconds, excluding initial model load "
      "and downloads; this is a local timing observation, not a hardware-independent speed claim.", "SmallCopy")

newpage()
title("4. Measured image results")
if image_summary:
    prose(f"The image stage generated and scored {manifest['image_count']} images: two prompts x "
          f"{len(cfg['image_seeds'])} {'seed' if len(cfg['image_seeds']) == 1 else 'seeds'} x nine conditions. Filtered images: {manifest['filtered_count']}. "
          "All results below use raw CLIP cosine. Positive drop means clean minus attacked; negative "
          "drop means alignment increased. Values average paired differences across prompts and seeds.")
    data = []
    for r in image_summary:
        data.append(["Full prompt" if r["mode"] == "untargeted" else "Target concept",
                     method_name(r["method"]), r["valid_pairs"], fmt(r["mean_drop"], True),
                     fmt(r["retain_mean_drop"], True)])
    table(["Scored content", "Method", "Pairs", "Mean drop", "Retained drop"], data,
          [110, 118, 45, 108, 119], small=True)
    prose("The retained-concept score is a diagnostic: a large positive drop suggests collateral "
          "semantic change. It does not impose a guarantee of preservation. No confidence intervals "
          "or success rate are claimed from only two independent scenes.", "SmallCopy")
    for mode in ("untargeted", "targeted"):
        part = [r for r in image_summary if r["mode"] == mode and r["mean_drop"]]
        winner = max(part, key=lambda r: float(r["mean_drop"]))
        genetic_row = next(r for r in part if r["method"] == "genetic")
        budget_row = next(r for r in part if r["method"] == "budget_random")
        label = "Original-prompt alignment" if mode == "untargeted" else "Target-concept alignment"
        prose(f"<b>{label}.</b> Largest measured drop: {method_name(winner['method'])}, "
              f"{fmt(winner['mean_drop'], True)}. Genetic: {fmt(genetic_row['mean_drop'], True)}; "
              f"matched random: {fmt(budget_row['mean_drop'], True)}. This comparison is descriptive, "
              "with only two scenes and one search initialization per case and method.")
    prose("<b>Why text loss is insufficient.</b> The encoder objective and image alignment operate on "
          "different representations. In the inspected case-0 targeted greedy image, the bicycle and wall "
          "remain recognizable despite masked text cosine falling to 0.6010. A strong text-space perturbation can leave an object visible, "
          "replace it with a related object, or alter style and composition. Conversely, a smaller "
          "text-space shift may change the generated scene. The grids on the next pages expose these "
          "differences rather than treating every lower score as successful object removal.")
    prose("<b>Evaluation limit.</b> The attack and evaluator share the CLIP model family. A score "
          "drop can partly reflect encoder sensitivity rather than independent evidence of semantic "
          "failure. Blinded human labels or an independent object detector would be needed to quantify "
          "target removal with retained-content preservation.")
else:
    prose("Image generation and scoring were not executed. The available measurements establish "
          "encoder-space changes only. They cannot substantiate image-level attack success or object removal.")

if image_scores:
    for case in range(cfg["n_prompts"]):
        newpage()
        title(f"5.{case+1}. Paired visual evidence: case {case}")
        seed = cfg["image_seeds"][0]
        prompt, target, retain = manifest["cases"][case]
        prose(f"<b>Clean prompt:</b> {html.escape(prompt)}<br/>"
              f"<b>Target to remove:</b> {html.escape(target)}; "
              f"<b>content to retain:</b> {html.escape(retain)}. Seed {seed}.")
        path = RUN / f"grid_{case:02d}_{seed}.png"
        story.append(Image(str(path), width=450, height=506.25))
        story.append(Spacer(1, 8))
        prose("Reading order: clean; untargeted random, greedy, genetic and matched random; "
              "then targeted random, greedy, genetic and matched random. All nine images use "
              "the same initial noise. Full-resolution images are included in the results bundle. "
              "No manual target-presence labels are claimed.", "SmallCopy")
        if case == 1:
            prose("Qualitative observation: the untargeted genetic image changes the real fruit "
                  "seen in the clean output into an apple motif on tableware. This is a change in "
                  "rendered content, while recognizable apple and plate concepts can remain.", "SmallCopy")
        markdown.extend([f"![Case {case} paired evidence](qf_results/grid_{case:02d}_{seed}.png)", ""])

newpage()
opt_start = len(markdown)
title("6. Optimizing the attack: features and objective")
prose("<b>Status: proposed follow-up experiments.</b> These changes have not been implemented or "
      "tested in the delivered attack code. The aim is greater image-level effectiveness under the "
      "same five-character suffix constraint. In this attack, text is the editable input and the "
      "generated image is the outcome to evaluate; there is no input image whose pixels are optimized.")
prose("<b>First priority: optimize the intended outcome.</b> Our targeted genetic run reduced mean "
      "target alignment by 0.0120, but retained-content alignment fell by 0.0516. The bicycle also "
      "remained visible in the targeted greedy example despite a masked text cosine of 0.6010. "
      "These observations motivate improving the loss before simply increasing the search budget. "
      "They do not establish which proposed replacement will work best.")
table(["Important part", "What to examine and why"], [
    ["Target noun and attributes", "Separate 'bicycle' from 'black', or 'plate' from 'red apple'. Distinguish object removal from a color or style change. Use controlled phrase-removal or replacement probes to estimate sensitivity."],
    ["Retained nouns and relations", "Protect 'brick wall' and 'red apple'. Inspect relations such as 'against' and 'on': changing a relation may alter layout while leaving both objects present. These probes diagnose importance; they do not edit the base prompt in the suffix-only attack."],
    ["Suffix tokens and boundaries", "Measure the actual CLIP tokenization of each complete prompt. Five characters can become a different number of BPE tokens. Track suffix token IDs, boundary effects and truncation; keep the suffix at the end for a comparable threat model."],
    ["Embedding coordinates", "Compare the original padded 77-token representation with EOS pooling or explicitly aligned content-token features. Recompute concept masks for each representation. A large padding-state shift need not imply an object change."],
    ["Target and retained image regions", "Inspect bicycle frame/wheels versus wall texture and geometry; plate rim/surface versus apple shape, color and count. Include background and object placement, not just the target crop. These are evaluation cues, not established causal rankings."],
], [112, 388], small=True)
prose("<b>A proposed directional, multi-objective loss.</b> Let z(x) be a normalized text representation "
      "and let x_minus be a grammatical caption describing the desired remaining scene: for example, "
      "'A red apple' when removing the plate. It is a loss reference, not the actual prompt sent to diffusion. "
      "For x_adv = x + space + suffix, consider:")
equation = "L = ||z(x_adv) - z(x_minus)||^2 + lambda * ||P_keep (z(x_adv) - z(x))||^2"
p("L = ||z(x_adv) - z(x_minus)||^2<br/>"
  "    + lambda * ||P_keep (z(x_adv) - z(x))||^2", "Equation")
markdown.extend(["```text", equation, "```", ""])
prose("P_keep is a candidate projector onto retained-concept directions, estimated from separate "
      "paired text examples in the same representation space. Lambda controls the preservation penalty. "
      "This is a hypothesis: target and retained semantics can overlap, and a caption without the target "
      "does not guarantee its absence in an image. Compare the original loss, the directional term alone "
      "and the combined loss on held-out prompts.", "SmallCopy")

newpage()
title("6.1. Search, image feedback and validation")
table(["Optimization type", "Proposed change and comparison"], [
    ["Discrete search", "Test GA diversity/restarts, mutation schedules, coordinate refinement and beam search. Compare all methods against uniform random search at the same unique-candidate budget. More generations alone do not demonstrate a better optimizer."],
    ["Feature and mask selection", "Use more varied context pairs, test sign stability together with change magnitude, and validate masks on unseen contexts. Ablate target noun, attributes and relations separately before assigning importance."],
    ["Gradient-assisted proposals", "A continuous relaxation could propose token edits, followed by projection to legal character strings and rescoring through the real tokenizer. Discrete tokenization is not directly differentiable. Five-token solutions need not satisfy a five-character limit."],
    ["Robust optimization", "Test whether a suffix works across held-out contexts or compatible encoder surrogates. Repeated search initializations assess search stability; repeated image seeds assess downstream reliability. Report these separately."],
    ["Compute efficiency", "Batch encoder evaluations and cache duplicates. Record unique candidates, forward batches and elapsed time. Speed improvements are distinct from improvements in attack success."],
], [112, 388], small=True)
prose("Gradient-based hard-prompt search is an established research direction (Wen et al., "
      "<i>Hard Prompts Made Easy</i>, 2023; reference below). Adapting it to this character constraint "
      "would be a new implementation and evaluation, not a result of the current notebook.", "SmallCopy")
prose("<b>Where image information belongs.</b> For a strictly query-free search, generated images "
      "are used only after a suffix is fixed. Evaluate target presence with an independent detector "
      "or blinded human review, and inspect retained objects and background separately. An object mask "
      "can identify the target region; compare outside an expanded target region to allow necessary "
      "changes to shadows and occlusions. Region similarity is a preservation proxy, not proof.")
prose("If image feedback selects suffixes, loss weights or restarts, explicitly call that a "
      "query-assisted extension and count all diffusion calls. A possible objective minimizes target "
      "presence plus a weighted penalty for changes to retained objects and background. The weights "
      "can be tuned on development prompts, or candidates can be compared on a Pareto frontier showing "
      "the trade-off between removal and preservation. Attention maps may help diagnose associations, "
      "but they are not causal evidence or an object-presence test.")
prose("<b>A practical experiment order.</b> First compare the three text losses at a fixed budget "
      "(for example, 1,000 unique candidates per prompt and method). Next ablate representation and "
      "mask construction, then compare search strategies and restarts within that same budget. Freeze "
      "choices on a development set before evaluating new prompts and seeds. If development tuning "
      "uses generated images, report those queries even when final suffix generation uses text only.")
prose("For targeted success, first verify that the clean image contains the target; then require "
      "its removal together with preservation of the intended remaining scene. Report baseline failures "
      "and the eligible denominator separately. Define any detector thresholds and preservation tolerances "
      "before testing. Keep resolution, denoising steps and guidance fixed across methods. Scale beyond "
      "the two-scene demo to more prompts, several search initializations and multiple paired image seeds; "
      "report prompt-level uncertainty, costs and failure cases. These tests establish whether an "
      "optimization improves the image outcome rather than only the encoder surrogate.")
(ROOT / "Attack_Optimization.md").write_text("\n".join(markdown[opt_start:]) +
    "\nReference: [Wen et al., Hard Prompts Made Easy](https://arxiv.org/abs/2302.03668).\n", encoding="utf-8")

newpage()
title("7. Assessment and reproducibility")
prose("<b>Strengths of the paper.</b> A clear modular vulnerability, a simple diffusion-free "
      "search loss, character-level constraints, multiple search approaches and paired-seed visual "
      "examples. The WALL-E case study checks a plausible alternative explanation for one suffix.")
prose("<b>What remains unproven.</b> The 20-prompt benchmark is small. CLIP cosine is not an object "
      "presence label, and the targeted objective includes no preservation term. Sign-vote masks "
      "depend on chosen contexts, token alignment and padding, and are coordinate-basis dependent. "
      "The experiments do not establish transfer to newer diffusion architectures or multilingual prompts. "
      "Selection over trials and optimizer budgets require explicit accounting for a fair comparison.")
prose("<b>Conclusion from this implementation.</b> The code confirms that short suffixes can "
      "substantially reduce the specified CLIP text-space objectives without querying diffusion. "
      + ("The paired image measurements provide additional, limited evidence of changes in text-image "
         "alignment; they do not demonstrate reliable selective removal. " if image_summary else "") +
      "The measured comparison with budget-matched random search is mixed, so the paper's optimizer "
      "ranking is not established by this smaller run. This is a method reproduction, not an exact replication.")
if image_summary:
    unt = next(r for r in image_summary if r["mode"] == "untargeted" and r["method"] == "genetic")
    tar = next(r for r in image_summary if r["mode"] == "targeted" and r["method"] == "genetic")
    prose(f"Specifically, untargeted genetic search has a mean original-prompt score drop of "
          f"{fmt(unt['mean_drop'], True)} (alignment increased). Targeted genetic search reduces "
          f"target alignment by {fmt(tar['mean_drop'])}, while retained-content alignment falls by "
          f"{fmt(tar['retain_mean_drop'])}. These results do not demonstrate reliable selective removal. "
          "The short search and reduced image sampling settings limit comparison with the paper.")
p("Run it in Google Colab", "Sub")
prose("1. Upload QF_Attack_Colab.ipynb at colab.research.google.com.<br/>"
      "2. Select T4 GPU and Runtime Version 2026.07 (Python 3.12); execute cells in order.<br/>"
      "3. Keep RUN_IMAGES=True for the complete evaluation. Select 'extended' only for a larger run.<br/>"
      "4. Download the automatically generated results ZIP and run_report.md. "
      "Expect fresh suffixes or scores if the device, software or configuration differs.")
p("Evidence and exact versions", "Sub")
prose("Deliverables: standalone Colab notebook; locally executed notebook; qf_attack.py; "
      "requirements-colab.txt; qf_results_bundle.zip; CSV metrics; masks; search traces; paired grids; "
      "manifest and scheduler configuration. tools/validate_results.py checks mask threshold behavior, "
      "budgets, suffix constraints, objective consistency, convergence, latent pairing, paired score "
      "arithmetic and notebook execution status.", "SmallCopy")
prose("<b>Observed software:</b> Python " + manifest["python"] + "; " +
      "; ".join(f"{k} {v}" for k,v in manifest["versions"].items()) +
      ". Colab dependencies are pinned separately; differences are recorded, not hidden.", "SmallCopy")
prose("<b>SD commit:</b> " + cfg["sd_revision"] + "<br/><b>CLIP commit:</b> " +
      (manifest.get("clip_revision") or "not evaluated") +
      "<br/><b>Image weights:</b> fp16 checkpoint variant; " +
      manifest.get("generation_dtype", "not generated") + " inference on " +
      manifest.get("generation_device", "n/a") + ".", "SmallCopy")
p("Primary references", "Sub")
for label, link in [
    ("Paper, arXiv:2303.16378v2 (supplied PDF, especially pp. 3-6)", "https://arxiv.org/abs/2303.16378v2"),
    ("Authors' QF-Attack code and original notebook", "https://github.com/OPTML-Group/QF-Attack"),
    ("Stable Diffusion v1.4 checkpoint and model card", "https://huggingface.co/CompVis/stable-diffusion-v1-4"),
    ("CLIP ViT-L/14 checkpoint and model card", "https://huggingface.co/openai/clip-vit-large-patch14"),
    ("Google Colab runtime versions and Python 3.12 runtime selection", "https://research.google.com/colaboratory/runtime-version-faq.html"),
    ("Wen et al., Hard Prompts Made Easy (gradient-based hard-prompt optimization)", "https://arxiv.org/abs/2302.03668"),
]:
    p(f'<link href="{link}" color="#137c83">{html.escape(label)}</link>', "SmallCopy")
    markdown.extend([f"[{label}]({link})", ""])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#d9e3e7"))
    canvas.line(56, 41, 556, 41)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(56, 27, "QF-Attack | Paper analysis + measured notebook evidence | 4 Nov 2026")
    canvas.drawRightString(556, 27, str(doc.page))
    canvas.restoreState()


doc = SimpleDocTemplate(str(OUT / "QF_Attack_Analysis.pdf"), pagesize=(612, 792),
                        leftMargin=56, rightMargin=56, topMargin=45, bottomMargin=56,
                        title="Query-free adversarial attacks: paper analysis and notebook evidence",
                        author="Research reproduction")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
(ROOT / "QF_Attack_Analysis.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
print(OUT / "QF_Attack_Analysis.pdf")
