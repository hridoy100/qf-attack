"""Verify cached-image reuse preserves evidence and rejects mismatched settings."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import qf_attack as qf

manifest = json.loads((ROOT / "qf_results/manifest.json").read_text())
cfg = qf.Config(**manifest["config"])
attacks = list(csv.DictReader((ROOT / "qf_results/attack_results.csv").open()))
for row in attacks:
    row["case_id"] = int(row["case_id"])
paths = sorted((ROOT / "qf_results/images").glob("*.png"))
before = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}

original_steps = cfg.steps
cfg.steps += 1
try:
    qf.generate_images(cfg, attacks)
    raise AssertionError("Mismatched generation settings were accepted")
except ValueError as exc:
    assert "settings differ" in str(exc)
cfg.steps = original_steps

rows = qf.generate_images(cfg, attacks)
assert len(rows) == len(paths) == cfg.n_prompts * len(cfg.image_seeds) * 9
after = {p: (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
assert before == after, "Cached images were regenerated or modified"
print("PASS: settings mismatch rejected; every image reused without modifying its bytes or timestamp")
