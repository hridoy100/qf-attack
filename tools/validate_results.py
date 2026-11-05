"""Check substantive invariants on the real experiment, not attack efficacy."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import torch
from qf_attack import derive_mask

# At n=10 and epsilon=.9: 9 agreements and 1 opposing sign must fail.
plus = torch.ones(10, 1, 2)
plus[0, 0, 1] = -1
mask = derive_mask(plus, torch.zeros_like(plus), .9)
assert mask.tolist() == [1., 0.]
assert torch.equal(mask, derive_mask(torch.zeros_like(plus), plus, .9))
try:
    derive_mask(torch.zeros_like(plus), torch.zeros_like(plus), .9)
    raise AssertionError("empty mask accepted")
except ValueError:
    pass

out = ROOT / "qf_results"
manifest = json.loads((out / "manifest.json").read_text())
assert manifest["diffusion_queries_during_search"] == 0
assert manifest["text_encoder_shape"] == [1, 77, 768]
assert manifest["source_sha256"] == hashlib.sha256((ROOT / "qf_attack.py").read_bytes()).hexdigest()
rows = list(csv.DictReader((out / "attack_results.csv").open()))
assert len(rows) == manifest["config"]["n_prompts"] * 8
for r in rows:
    assert len(r["suffix"]) == 5
    assert r["attacked_prompt"] == r["prompt"] + " " + r["suffix"]
    assert np.isfinite(float(r["objective_cosine"]))
    assert -1.00001 <= float(r["objective_cosine"]) <= 1.00001
    if r["mode"] == "untargeted":
        assert abs(float(r["objective_cosine"]) - float(r["full_cosine"])) < 1e-5
    else:
        assert abs(float(r["objective_cosine"]) - float(r["masked_cosine"])) < 1e-5
    if r["method"] == "genetic":
        control = next(c for c in rows if c["case_id"] == r["case_id"]
                       and c["mode"] == r["mode"] and c["method"] == "budget_random")
        assert control["unique_candidates"] == r["unique_candidates"]
traces = json.loads((out / "search_traces.json").read_text())
for trace in traces.values():
    history = [x["best_loss"] for x in trace["history"]]
    assert all(b <= a + 1e-6 for a, b in zip(history, history[1:]))
if manifest["images_evaluated"]:
    images = list(csv.DictReader((out / "image_manifest.csv").open()))
    assert len(images) == manifest["config"]["n_prompts"] * len(manifest["config"]["image_seeds"]) * 9
    for row in images:
        digest = row["image_source_sha256"]
        source = out / "generation_sources" / (digest + ".py")
        assert source.exists() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
    for i in range(manifest["config"]["n_prompts"]):
        for seed in manifest["config"]["image_seeds"]:
            group = [r for r in images if int(r["case_id"]) == i and int(r["seed"]) == seed]
            assert len({r["latent_sha256"] for r in group}) == 1
    scores = list(csv.DictReader((out / "image_scores.csv").open()))
    clean = {(r["case_id"], r["seed"]): r for r in scores if r["mode"] == "clean"}
    pairs = list(csv.DictReader((out / "paired_drops.csv").open()))
    for pair in pairs:
        attacked = next(r for r in scores if all(r[k] == pair[k] for k in ("case_id", "seed", "mode", "method")))
        base = clean[(pair["case_id"], pair["seed"])]
        assert attacked["filtered"] != "True" and base["filtered"] != "True"
        for metric in ("full", "target", "retain"):
            assert abs(float(pair[metric + "_drop"]) -
                       (float(base[metric + "_score"]) - float(attacked[metric + "_score"]))) < 1e-7
for name in ("QF_Attack_Colab.ipynb", "QF_Attack_Executed.ipynb"):
    nb = nbformat.read(ROOT / name, as_version=4)
    nbformat.validate(nb)
    if "Executed" in name:
        assert all(cell.get("execution_count") is not None for cell in nb.cells if cell.cell_type == "code")
        assert not any(output.output_type == "error" for cell in nb.cells if cell.cell_type == "code" for output in cell.outputs)
clean = nbformat.read(ROOT / "QF_Attack_Colab.ipynb", as_version=4)
executed = nbformat.read(ROOT / "QF_Attack_Executed.ipynb", as_version=4)
assert len(clean.cells) == len(executed.cells)
assert all(a.source == b.source for a, b in zip(clean.cells, executed.cells))
print("PASS: masks, objectives, suffix constraints, query budgets, convergence, notebook schema and available image invariants")
