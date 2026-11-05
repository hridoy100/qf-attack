"""Text-domain PGD attack for the QF-Attack reproduction (encoder-only).

Reproduces the gradient-based attack from Zhuang et al. (the PGD row of their
Table 1), following their reference implementation: a relaxed distribution over
the candidate characters at each suffix token-slot, Gumbel-softmax near-discrete
selection, and a sign-gradient step maximising 1/cos (equivalently minimising
cosine) for 100 iterations. The final discrete suffix is then scored with the
same flattened-embedding cosine used for the other methods, so the number is
directly comparable to attack_results.csv.

This is a reproduction of a published baseline, not a new or stronger attack.
No diffusion model is involved. Output: qf_results/pgd_results.csv
"""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from qf_attack import ALPHABET, CASES, Config, TextOracle, choose_device

RUN = ROOT / "qf_results"


def causal_mask(seq_len, dtype, device):
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    return _create_4d_causal_attention_mask((1, seq_len), dtype, device=device)


class PGD:
    def __init__(self, oracle):
        self.model = oracle.model          # CLIPTextModel, fp32
        self.tokenizer = oracle.tokenizer
        self.device = oracle.device
        self.text_model = self.model.text_model

    def token_pos_embeds(self, text):
        ids = self.tokenizer(text, padding="max_length", truncation=True,
                             max_length=77, return_tensors="pt").input_ids.to(self.device)
        with torch.no_grad():
            return self.text_model.embeddings(input_ids=ids)  # token + position, [1, 77, dim]

    def forward_from_embeds(self, embeds):
        mask = causal_mask(embeds.shape[1], embeds.dtype, embeds.device)
        out = self.text_model.encoder(inputs_embeds=embeds, causal_attention_mask=mask)[0]
        return self.text_model.final_layer_norm(out)

    def candidate_embeds(self):
        # For each candidate character, its token+position embedding at every position.
        rows = []
        for ch in ALPHABET:
            rows.append(self.token_pos_embeds((ch + " ") * 75)[0])  # [77, dim]
        return torch.stack(rows, dim=1)  # [77, n_char, dim]

    def run(self, prompt, mask, length, iters, batch, eta, seed):
        torch.manual_seed(seed)
        n_char = len(ALPHABET)
        candidate = self.candidate_embeds().to(self.device)          # [77, C, dim]
        orig_embeds = self.token_pos_embeds(prompt)                  # [1, 77, dim]
        with torch.no_grad():
            clean_out = self.forward_from_embeds(orig_embeds).flatten()
        mask_vec = (mask.to(self.device) if mask is not None
                    else torch.ones_like(clean_out))
        clean_ref = F.normalize(clean_out * mask_vec, dim=0)

        num_word = len(prompt.split(" "))
        z = torch.zeros(77, 1, device=self.device)
        z[num_word + 1:num_word + length + 1] = 1.0                   # suffix token-slots

        u = torch.full((77, n_char), 1.0 / n_char, device=self.device, requires_grad=True)
        base = orig_embeds[0]                                        # [77, dim]

        def embed_from(u_onehot):
            picked = torch.einsum("pc,pcd->pd", u_onehot, candidate)  # [77, dim]
            return ((1 - z) * base + z * picked).unsqueeze(0)

        def masked_cos(out):
            v = F.normalize(out.flatten() * mask_vec, dim=0)
            return torch.dot(v, clean_ref)

        def suffix_of(sample):
            idx = sample[num_word + 1:num_word + length + 1].argmax(dim=1)
            return "".join(ALPHABET[i] for i in idx.tolist())

        best_suffix, best_loss = None, -float("inf")
        lr = eta / np.sqrt(iters)
        for _ in range(iters):
            # Draw Gumbel-softmax samples (straight-through, so gradient reaches u),
            # keep the most damaging ones, and ascend 1/cos on their summed loss.
            samples = [F.gumbel_softmax(u, tau=0.5, hard=True, dim=-1) for _ in range(batch)]
            losses = [1.0 / masked_cos(self.forward_from_embeds(embed_from(s))) for s in samples]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    # Demo profile (2 prompts). Gumbel-softmax is unsupported on Apple MPS, so fall back to CPU there.
    device = choose_device(args.device)
    cfg = Config(device="cpu" if device.type == "mps" else str(device))
    cfg.sd_revision = json.loads((RUN / "manifest.json").read_text())["config"]["sd_revision"]
    oracle = TextOracle(cfg)
    pgd = PGD(oracle)

    rows = []
    for case_id, (prompt, target, retain) in enumerate(CASES[:cfg.n_prompts]):
        mask = torch.from_numpy(np.load(RUN / f"mask_{case_id:02d}.npy")).float()
        reference = oracle.encode([prompt]).flatten()
        for mode in ("untargeted", "targeted"):
            mode_mask = None if mode == "untargeted" else mask
            suffix = pgd.run(prompt, mode_mask, cfg.suffix_length,
                             args.iters, args.batch, args.eta,
                             args.seed + 1000 * case_id + (mode == "targeted"))
            h = oracle.encode([prompt + " " + suffix]).flatten()
            full = float(F.cosine_similarity(reference, h, dim=0))
            masked = float(F.cosine_similarity(reference * mask, h * mask, dim=0))
            rows.append(dict(case_id=case_id, prompt=prompt, target=target, retain=retain,
                             mode=mode, method="pgd", suffix=suffix,
                             attacked_prompt=prompt + " " + suffix,
                             objective_cosine=masked if mode == "targeted" else full,
                             full_cosine=full, masked_cosine=masked,
                             iters=args.iters, batch=args.batch))
            print(f"{case_id} {mode:10s} pgd {suffix!r} "
                  f"full={full:.4f} masked={masked:.4f}", flush=True)
            with (RUN / "pgd_results.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    print("wrote", RUN / "pgd_results.csv")


if __name__ == "__main__":
    main()
