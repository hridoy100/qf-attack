"""Score the already-generated images for object presence with OWL-ViT.

This is an evaluation step. It reads the 18 images produced by the earlier run
and reports how confidently an open-vocabulary detector finds the target object
and the retained object in each one. It does not change or generate any attack.
Output: qf_results/embedding_analysis/detector_scores.csv and detector_summary.csv
"""
from pathlib import Path
import csv
import json
import os

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

import torch
from PIL import Image
from transformers import OwlViTForObjectDetection, OwlViTProcessor

RUN = ROOT / os.environ.get("QF_OUTPUT_DIR", "qf_results")
OUT = RUN / "embedding_analysis"
MODEL = "google/owlvit-base-patch32"
THRESHOLD = 0.10  # presence cut-off on OWL-ViT confidence

# Per scene: the target object to remove and the content to retain, as detector queries.
SCENES = {
    0: dict(target="a bicycle", retain="a brick wall"),
    1: dict(target="a plate", retain="an apple"),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    processor = OwlViTProcessor.from_pretrained(MODEL)
    model = OwlViTForObjectDetection.from_pretrained(MODEL).eval().to(device)

    rows = []
    for row in csv.DictReader((RUN / "image_scores.csv").open()):
        case = int(row["case_id"])
        queries = [SCENES[case]["target"], SCENES[case]["retain"]]
        image = Image.open(ROOT / row["image_path"]).convert("RGB")
        inputs = processor(text=[queries], images=image, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        target_size = torch.tensor([image.size[::-1]], device=device)
        result = processor.post_process_object_detection(outputs, threshold=0.0,
                                                          target_sizes=target_size)[0]
        best = {0: 0.0, 1: 0.0}
        for score, label in zip(result["scores"].tolist(), result["labels"].tolist()):
            best[label] = max(best[label], score)
        rows.append({
            "case_id": case, "seed": row["seed"], "mode": row["mode"], "method": row["method"],
            "target_query": queries[0], "retain_query": queries[1],
            "target_confidence": round(best[0], 4), "retain_confidence": round(best[1], 4),
            "target_detected": int(best[0] >= THRESHOLD), "retain_detected": int(best[1] >= THRESHOLD),
        })

    with (OUT / "detector_scores.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Compare each attacked image with the clean image of the same scene and seed.
    clean = {(r["case_id"], r["seed"]): r for r in rows if r["mode"] == "clean"}
    summary = []
    for mode in ("untargeted", "targeted"):
        for method in ("random", "greedy", "genetic", "budget_random"):
            group = [r for r in rows if r["mode"] == mode and r["method"] == method]
            if not group:
                continue
            target_drops, retain_drops, removed = [], [], 0
            for r in group:
                base = clean[(r["case_id"], r["seed"])]
                target_drops.append(base["target_confidence"] - r["target_confidence"])
                retain_drops.append(base["retain_confidence"] - r["retain_confidence"])
                if base["target_detected"] and not r["target_detected"]:
                    removed += 1
            summary.append({
                "mode": mode, "method": method, "images": len(group),
                "mean_target_confidence_drop": round(sum(target_drops) / len(target_drops), 4),
                "mean_retain_confidence_drop": round(sum(retain_drops) / len(retain_drops), 4),
                "target_removed": removed,
                "target_present_in_clean": sum(clean[(r["case_id"], r["seed"])]["target_detected"] for r in group),
            })
    with (OUT / "detector_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("model:", MODEL, "threshold:", THRESHOLD)
    print("clean images:")
    for (c, s), r in sorted(clean.items()):
        print(f"  case {c} seed {s}: {r['target_query']} {r['target_confidence']} / "
              f"{r['retain_query']} {r['retain_confidence']}")
    print("wrote", OUT / "detector_scores.csv")


if __name__ == "__main__":
    main()
