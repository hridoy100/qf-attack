"""Measure how the recorded suffixes move the CLIP text embeddings.

Uses the same encoder, revision and flattening as qf_attack.py, and stops if the
recorded objective cosines cannot be reproduced. No diffusion model is used.
Output: qf_results/embedding_analysis/
"""
from pathlib import Path
import csv
import json
import os
import random

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer

RUN = ROOT / os.environ.get("QF_OUTPUT_DIR", "qf_results")
OUT = RUN / "embedding_analysis"
SD_ID = "CompVis/stable-diffusion-v1-4"
N_RANDOM = 500
TOLERANCE = 1e-3


def cos(a, b):
    return float(F.cosine_similarity(a.flatten(), b.flatten(), dim=0))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RUN / "manifest.json").read_text())
    revision = manifest["config"]["sd_revision"]
    alphabet = manifest["alphabet"]
    attacks = list(csv.DictReader((RUN / "attack_results.csv").open(encoding="utf-8")))
    # Image drops exist only after image generation; a text-only run leaves those columns empty.
    drops = {}
    if (RUN / "paired_drops.csv").exists():
        drops = {(r["case_id"], r["mode"], r["method"]): r
                 for r in csv.DictReader((RUN / "paired_drops.csv").open())}

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = CLIPTokenizer.from_pretrained(SD_ID, subfolder="tokenizer", revision=revision)
    model = CLIPTextModel.from_pretrained(SD_ID, subfolder="text_encoder", revision=revision,
                                          torch_dtype=torch.float32, use_safetensors=True).eval().to(device)

    @torch.inference_mode()
    def encode(texts):
        parts = []
        for start in range(0, len(texts), 32):
            ids = tokenizer(texts[start:start + 32], padding="max_length", truncation=False,
                            max_length=77, return_tensors="pt").input_ids.to(device)
            parts.append(model(input_ids=ids).last_hidden_state.float().cpu())
        return torch.cat(parts)

    def n_tokens(text):
        return len(tokenizer(text)["input_ids"])

    def tokens_of(text):
        ids = tokenizer(text)["input_ids"]
        return [t.replace("</w>", "") for t in tokenizer.convert_ids_to_tokens(ids)]

    rows, positions, similarity_map, summary, tokens = [], [], {}, [], {}
    for case, (prompt, target, retain) in enumerate(manifest["cases"]):
        mask = torch.from_numpy(np.load(RUN / f"mask_{case:02d}.npy")).float().reshape(77, 768)
        clean = encode([prompt])[0]
        retain_only = encode([retain[0].upper() + retain[1:]])[0]
        clean_len = n_tokens(prompt)  # includes start and end tokens
        case_rows = [r for r in attacks if int(r["case_id"]) == case]
        attacked = encode([r["attacked_prompt"] for r in case_rows])

        for r, h in zip(case_rows, attacked):
            full = cos(clean, h)
            masked = cos(clean * mask, h * mask)
            if abs(full - float(r["full_cosine"])) > TOLERANCE or abs(masked - float(r["masked_cosine"])) > TOLERANCE:
                raise SystemExit(f"Could not reproduce recorded cosine for {r['attacked_prompt']!r}: "
                                 f"{full:.5f}/{masked:.5f} vs {r['full_cosine']}/{r['masked_cosine']}")
            length = n_tokens(r["attacked_prompt"])
            tokens[f"{case}/{r['mode']}/{r['method']}"] = tokens_of(r["attacked_prompt"])
            distance = (h - clean).norm(dim=1)
            # The suffix starts where the clean prompt's end token used to be.
            first_changed = clean_len - 1
            drop = drops.get((r["case_id"], r["mode"], r["method"]))
            rows.append({
                "case_id": case, "mode": r["mode"], "method": r["method"], "suffix": r["suffix"],
                "suffix_tokens": length - clean_len,
                "attacked_tokens": length,
                "full_cosine": full, "masked_cosine": masked,
                "max_distance_before_suffix": float(distance[:first_changed].max()),
                "share_of_squared_change_after_end_token": float(
                    (distance[length:] ** 2).sum() / (distance ** 2).sum()),
                "cosine_to_retained_only": cos(h, retain_only),
                "full_drop": float(drop["full_drop"]) if drop else "",
                "target_drop": float(drop["target_drop"]) if drop else "",
                "retain_drop": float(drop["retain_drop"]) if drop else "",
            })
            for pos, value in enumerate(distance.tolist()):
                positions.append({"case_id": case, "mode": r["mode"], "method": r["method"],
                                  "position": pos, "l2_distance": value})

        # Reference cloud: random suffixes, to place the attacks in context.
        rng = random.Random(2026 + case)
        random_suffixes = ["".join(rng.choices(alphabet, k=5)) for _ in range(N_RANDOM)]
        cloud = encode([prompt + " " + s for s in random_suffixes])
        cloud_cos = F.cosine_similarity(cloud.flatten(1), clean.flatten()[None], dim=1).numpy()
        cloud_masked = F.cosine_similarity((cloud * mask).flatten(1), (clean * mask).flatten()[None], dim=1).numpy()

        cloud_to_free = F.cosine_similarity(cloud.flatten(1), retain_only.flatten()[None], dim=1).numpy()
        similarity_map[case] = {
            "prompt": prompt, "target_free_prompt": retain[0].upper() + retain[1:],
            "clean_tokens": tokens_of(prompt),
            "clean_to_target_free": cos(clean, retain_only),
            "random_to_clean": cloud_cos.tolist(),
            "random_to_target_free": cloud_to_free.tolist(),
        }

        norm = clean.norm(dim=1) ** 2
        layout = mask.mean(dim=1)
        summary.append({
            "case_id": case, "prompt": prompt, "clean_tokens": clean_len,
            "padding_positions": 77 - clean_len,
            "padding_share_of_squared_norm": float(norm[clean_len:].sum() / norm.sum()),
            "cosine_clean_vs_retained_only": cos(clean, retain_only),
            "mask_fraction_prompt_positions": float(layout[:clean_len].mean()),
            "mask_fraction_padding_positions": float(layout[clean_len:].mean()),
            "mask_fraction_by_position": layout.tolist(),
            "random_full_cosine_mean": float(cloud_cos.mean()),
            "random_full_cosine_p05": float(np.quantile(cloud_cos, .05)),
            "random_full_cosine_min": float(cloud_cos.min()),
            "random_masked_cosine_mean": float(cloud_masked.mean()),
            "random_masked_cosine_min": float(cloud_masked.min()),
            "random_full_cosines": cloud_cos.tolist(),
            "random_masked_cosines": cloud_masked.tolist(),
        })

    for name, data in (("suffix_embedding_metrics.csv", rows), ("distance_by_position.csv", positions)):
        with (OUT / name).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / "similarity_map.json").write_text(json.dumps(similarity_map))
    (OUT / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False))
    (OUT / "summary.json").write_text(json.dumps(summary))
    print("reproduced all", len(rows), "recorded cosines within", TOLERANCE)


if __name__ == "__main__":
    main()
