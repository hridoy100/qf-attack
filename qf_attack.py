"""QF-Attack research reimplementation; also embedded in the Colab notebook.

Paper: https://arxiv.org/abs/2303.16378 (Zhuang et al., 2023).
Independent implementation of equations (1)-(2), not the authors' code.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path.cwd() / ".hf-cache"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import CLIPTextModel, CLIPTokenizer

SD_ID = "CompVis/stable-diffusion-v1-4"
CLIP_ID = "openai/clip-vit-large-patch14"
# Same character set as the reference code, with its duplicate '*' removed.
ALPHABET = tuple(dict.fromkeys(list("·~!@#$%^&*()=-*+.<>?,';:|\\/")
                               + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")))
CONTEXTS = [
    "A bird flying above a river", "A boat floating on a lake",
    "A teapot resting on a table", "A train crossing the countryside",
    "A butterfly sitting on a flower", "A lantern glowing beside a window",
    "A cabin standing beside a forest", "A cloud drifting above a mountain",
    "A fountain flowing in a garden", "A bridge stretching across a stream",
]
# First eight scene prompts are transcribed from the paper's figures. Remaining
# twelve are declared additions, not a recovered copy of the authors' dataset.
CASES = [
    ("A black bicycle against a brick wall", "a black bicycle", "a brick wall"),
    ("A red apple on a plate", "a plate", "a red apple"),
    ("A white swan on a lake", "a lake", "a white swan"),
    ("A purple and blue butterfly on a leaf", "a leaf", "a butterfly"),
    ("A yellow and black bumblebee on a flower", "a flower", "a bumblebee"),
    ("A red and white picnic blanket with a basket", "a basket", "a picnic blanket"),
    ("A yellow sunflower in a field", "a field", "a yellow sunflower"),
    ("A white snowflake on a blue background", "a white snowflake", "a blue background"),
    ("A brown dog beside a wooden fence", "a wooden fence", "a brown dog"),
    ("A blue teapot on a kitchen table", "a blue teapot", "a kitchen table"),
    ("A white boat beside a lighthouse", "a lighthouse", "a white boat"),
    ("A green frog on a lily pad", "a lily pad", "a green frog"),
    ("A red train crossing a stone bridge", "a stone bridge", "a red train"),
    ("A yellow duck beside a pond", "a pond", "a yellow duck"),
    ("A glass vase containing pink flowers", "a glass vase", "pink flowers"),
    ("A black cat sitting on a sofa", "a sofa", "a black cat"),
    ("A silver car beside a red barn", "a red barn", "a silver car"),
    ("A white mug beside an open book", "an open book", "a white mug"),
    ("A sailboat below a cloudy sky", "a sailboat", "a cloudy sky"),
    ("A brown horse beside a green tree", "a green tree", "a brown horse"),
]


@dataclass
class Config:
    profile: str = "demo"
    seed: int = 2026
    suffix_length: int = 5
    generations: int = 10
    population: int = 20
    mutation_probability: float = 0.3
    greedy_passes: int = 1
    mask_threshold: float = 0.9
    batch_size: int = 8
    n_prompts: int = 2
    image_seeds: tuple = (27,)
    steps: int = 20
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    run_images: bool = True
    output_dir: str = "qf_results"
    device: str = "auto"
    sd_revision: str | None = None
    clip_revision: str | None = None

    @classmethod
    def full(cls, **kwargs):
        return cls(**dict(dict(profile="extended", generations=50, n_prompts=20,
                              image_seeds=tuple(range(27, 37)), steps=50), **kwargs))


def choose_device(requested="auto"):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_revision(model_id, revision):
    # Record immutable commits; pass the saved value to repeat the same weights.
    if revision and len(revision) == 40:
        return revision
    return HfApi().model_info(model_id, revision=revision or "main").sha


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def derive_mask(with_concept, without_concept, threshold=0.9):
    if with_concept.shape != without_concept.shape or with_concept.ndim != 3:
        raise ValueError("Expected matching [contexts, 77, hidden] tensors")
    if not 0 <= threshold < 1:
        raise ValueError("Mask threshold must lie in [0, 1)")
    votes = torch.sign(with_concept - without_concept).sum(dim=0)
    mask = votes.abs() > threshold * with_concept.shape[0]
    if not mask.any():
        raise ValueError("Empty concept mask. Change contexts; do not silently relax epsilon.")
    return mask.flatten().float()


class TextOracle:
    """Flatten all 77 final hidden states, including EOS padding, as reference code does.

    This is NOT the normalized, projected pooled vector used for image CLIP scores.
    """
    def __init__(self, cfg):
        self.device = choose_device(cfg.device)
        self.batch_size = cfg.batch_size
        self.tokenizer = CLIPTokenizer.from_pretrained(
            SD_ID, subfolder="tokenizer", revision=cfg.sd_revision)
        self.model = CLIPTextModel.from_pretrained(
            SD_ID, subfolder="text_encoder", revision=cfg.sd_revision,
            torch_dtype=torch.float32, use_safetensors=True).eval().to(self.device)
        self.model.requires_grad_(False)
        self.texts_encoded = 0
        self.forward_batches = 0

    def validate(self, texts):
        lengths = [len(ids) for ids in self.tokenizer(texts, truncation=False)["input_ids"]]
        if max(lengths) > self.tokenizer.model_max_length:
            raise ValueError("Prompt exceeds CLIP context; refusing silent truncation")

    @torch.inference_mode()
    def encode(self, texts):
        self.validate(texts)
        parts = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            tokens = self.tokenizer(batch, padding="max_length", truncation=False,
                                    max_length=self.tokenizer.model_max_length,
                                    return_tensors="pt")
            # No padding attention mask, matching the SD v1.x conditioning path.
            h = self.model(input_ids=tokens.input_ids.to(self.device)).last_hidden_state
            parts.append(h.float().cpu())
            self.texts_encoded += len(batch)
            self.forward_batches += 1
        return torch.cat(parts)

    def concept_mask(self, concept, threshold):
        with_concept = self.encode([s + " and " + concept for s in CONTEXTS])
        without = self.encode(CONTEXTS)
        return derive_mask(with_concept, without, threshold)


class Objective:
    def __init__(self, oracle, prompt, mask=None, length=5):
        self.oracle, self.prompt, self.length = oracle, prompt, length
        # Reserve enough tokens for worst-case suffix segmentation and a separator.
        if len(oracle.tokenizer(prompt)["input_ids"]) + 2 * length + 2 > 77:
            raise ValueError("Base prompt too long for a five-character attack")
        self.reference = oracle.encode([prompt]).flatten()
        self.mask = torch.ones_like(self.reference) if mask is None else mask
        if self.mask.shape != self.reference.shape or not self.mask.any():
            raise ValueError("Invalid mask")
        self.unit_reference = F.normalize(self.reference * self.mask, dim=0)
        self.cache = {}
        self.history = []
        self.requests = 0

    def score(self, suffixes):
        self.requests += len(suffixes)
        for suffix in suffixes:
            if len(suffix) != self.length or any(c not in ALPHABET for c in suffix):
                raise ValueError("Suffix violates the character constraint")
        missing = list(dict.fromkeys(s for s in suffixes if s not in self.cache))
        for start in range(0, len(missing), self.oracle.batch_size):
            part = missing[start:start + self.oracle.batch_size]
            h = self.oracle.encode([self.prompt + " " + s for s in part]).flatten(1)
            vals = F.normalize(h * self.mask, dim=1) @ self.unit_reference
            self.cache.update(zip(part, vals.tolist()))
        return [self.cache[s] for s in suffixes]

    def log(self, iteration):
        suffix = min(self.cache, key=self.cache.get)
        self.history.append(dict(iteration=iteration, best_loss=self.cache[suffix],
                                 unique_candidates=len(self.cache), suffix=suffix))

    def best(self):
        return min(self.cache, key=self.cache.get)


def random_suffix(rng, length):
    return "".join(rng.choices(ALPHABET, k=length))


def single_random(obj, rng, cfg):
    obj.score([random_suffix(rng, cfg.suffix_length)])
    obj.log(0)
    return obj.best()


def greedy(obj, rng, cfg):
    current = random_suffix(rng, cfg.suffix_length)
    obj.score([current])
    obj.log(0)
    for sweep in range(cfg.greedy_passes):
        for pos in range(cfg.suffix_length):
            candidates = [current[:pos] + c + current[pos + 1:] for c in ALPHABET]
            vals = obj.score(candidates)
            current = candidates[int(np.argmin(vals))]
            obj.log(sweep * cfg.suffix_length + pos + 1)
    return obj.best()


def genetic(obj, rng, cfg):
    if cfg.population < 4 or cfg.suffix_length < 2:
        raise ValueError("GA requires population >=4 and suffix length >=2")
    population = [random_suffix(rng, cfg.suffix_length) for _ in range(cfg.population)]
    for generation in range(cfg.generations + 1):
        scores = obj.score(population)
        ranked = sorted(zip(scores, population))
        obj.log(generation)
        if generation == cfg.generations:
            break
        # Explicit implementation choices: two elites, three-way tournaments,
        # one-point crossover, and one mutated character with probability 0.3.
        def parent():
            return min(rng.sample(ranked, 3))[1]
        next_population = [item[1] for item in ranked[:2]]
        while len(next_population) < cfg.population:
            a, b = parent(), parent()
            cut = rng.randrange(1, cfg.suffix_length)
            child = a[:cut] + b[cut:]
            if rng.random() < cfg.mutation_probability:
                pos = rng.randrange(cfg.suffix_length)
                child = child[:pos] + rng.choice(ALPHABET) + child[pos + 1:]
            next_population.append(child)
        population = next_population
    return obj.best()


def budget_random(obj, rng, cfg, budget):
    # Same number of unique candidate encoder evaluations as this prompt's GA.
    while len(obj.cache) < budget:
        n = min(cfg.batch_size, budget - len(obj.cache))
        candidates = []
        while len(candidates) < n:
            s = random_suffix(rng, cfg.suffix_length)
            if s not in obj.cache and s not in candidates:
                candidates.append(s)
        obj.score(candidates)
        obj.log(len(obj.cache))
    return obj.best()


def run_search(cfg):
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not 1 <= cfg.n_prompts <= len(CASES):
        raise ValueError("n_prompts outside the available dataset")
    cfg.sd_revision = resolve_revision(SD_ID, cfg.sd_revision)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
    oracle = TextOracle(cfg)
    started = time.perf_counter()
    rows, traces, mask_stats = [], {}, []
    for case_id, (prompt, target, retain) in enumerate(CASES[:cfg.n_prompts]):
        mask = oracle.concept_mask(target, cfg.mask_threshold)
        np.save(out / f"mask_{case_id:02d}.npy", mask.numpy())
        mask_stats.append(dict(case_id=case_id, target=target, selected=int(mask.sum()),
                               total=mask.numel(), fraction=float(mask.mean())))
        for mode in ("untargeted", "targeted"):
            mode_mask = None if mode == "untargeted" else mask
            ga_budget = None
            for method_id, method in enumerate(("random", "greedy", "genetic", "budget_random")):
                obj = Objective(oracle, prompt, mode_mask, cfg.suffix_length)
                rng_seed = cfg.seed + 1000 * case_id + 100 * (mode == "targeted") + method_id
                rng = random.Random(rng_seed)
                before_batches = oracle.forward_batches
                t = time.perf_counter()
                if method == "budget_random":
                    suffix = budget_random(obj, rng, cfg, ga_budget)
                else:
                    suffix = {"random": single_random, "greedy": greedy,
                              "genetic": genetic}[method](obj, rng, cfg)
                elapsed = time.perf_counter() - t
                search_batches = oracle.forward_batches - before_batches
                if method == "genetic":
                    ga_budget = len(obj.cache)
                h = oracle.encode([prompt + " " + suffix]).flatten()
                full_cosine = float(F.cosine_similarity(obj.reference, h, dim=0))
                masked_cosine = float(F.cosine_similarity(obj.reference * mask, h * mask, dim=0))
                row = dict(case_id=case_id, prompt=prompt, target=target, retain=retain,
                           mode=mode, method=method, suffix=suffix, attacked_prompt=prompt + " " + suffix,
                           objective_cosine=obj.cache[suffix], full_cosine=full_cosine,
                           masked_cosine=masked_cosine, candidate_requests=obj.requests,
                           unique_candidates=len(obj.cache), forward_batches=search_batches,
                           diffusion_queries_during_search=0, rng_seed=rng_seed,
                           search_seconds=elapsed)
                rows.append(row)
                traces[f"{case_id}_{mode}_{method}"] = dict(history=obj.history,
                                                            candidate_scores=obj.cache)
                print(f"{case_id} {mode:10s} {method:13s} {suffix!r} "
                      f"loss={obj.cache[suffix]:.5f} candidates={len(obj.cache)}", flush=True)
                save_csv(out / "attack_results.csv", rows)
    save_json(out / "search_traces.json", traces)
    save_json(out / "mask_statistics.json", mask_stats)
    versions = {p: importlib.metadata.version(p) for p in
                ("torch", "transformers", "diffusers", "accelerate", "huggingface-hub", "numpy")}
    manifest = dict(config=asdict(cfg), device=str(oracle.device), platform=platform.platform(),
                    python=platform.python_version(), versions=versions,
                    text_encoder_shape=list(oracle.encode([CASES[0][0]]).shape),
                    total_texts_encoded=oracle.texts_encoded,
                    total_forward_batches=oracle.forward_batches,
                    elapsed_seconds=time.perf_counter() - started,
                    diffusion_queries_during_search=0, images_evaluated=False,
                    alphabet="".join(ALPHABET), contexts=CONTEXTS,
                    cases=CASES[:cfg.n_prompts],
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save_json(out / "manifest.json", manifest)
    del oracle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return rows


def generate_images(cfg, attack_rows):
    """Post-search only. Every condition uses the exact same saved initial latent."""
    from diffusers import StableDiffusionPipeline
    from PIL import Image, ImageDraw
    device = choose_device(cfg.device)
    if device.type == "cpu":
        raise RuntimeError("Image generation needs a GPU. Set run_images=False for text-only mode.")
    out = Path(cfg.output_dir)
    (out / "images").mkdir(exist_ok=True)
    dtype = torch.float16
    settings = dict(model=SD_ID, revision=cfg.sd_revision, steps=cfg.steps,
                    guidance_scale=cfg.guidance_scale, height=cfg.height, width=cfg.width,
                    image_seeds=list(cfg.image_seeds), cases=[list(c) for c in CASES[:cfg.n_prompts]],
                    dtype=str(dtype), device=str(device), attention="default SDPA",
                    versions={p: importlib.metadata.version(p) for p in ("torch", "diffusers", "transformers")},
                    conditions=[dict(case_id=int(r["case_id"]), mode=r["mode"],
                                     method=r["method"], attacked_prompt=r["attacked_prompt"])
                                for r in attack_rows])
    settings_path = out / "generation_settings.json"
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cached = {}
    if settings_path.exists():
        previous = json.loads(settings_path.read_text())
        if previous["settings"] != settings:
            raise ValueError("Existing image settings differ. Use a new output directory.")
        if (out / "image_manifest.csv").exists():
            for r in csv.DictReader((out / "image_manifest.csv").open()):
                r["filtered"] = {"True": True, "False": False, "": None}[r["filtered"]]
                r["image_source_sha256"] = r.get("image_source_sha256") or previous["initial_source_sha256"]
                cached[(int(r["case_id"]), int(r["seed"]), r["mode"], r["method"])] = r
    else:
        if (out / "image_manifest.csv").exists():
            raise ValueError("Existing images lack generation settings. Use a new output directory.")
        save_json(settings_path, dict(settings=settings, initial_source_sha256=source_sha))
    (out / "generation_sources").mkdir(exist_ok=True)
    (out / "generation_sources" / (source_sha + ".py")).write_bytes(Path(__file__).read_bytes())
    pipe = StableDiffusionPipeline.from_pretrained(
        SD_ID, revision=cfg.sd_revision, variant="fp16", torch_dtype=dtype,
        use_safetensors=True)
    # PyTorch 2 uses memory-efficient SDPA. Attention slicing replaces that
    # processor and can severely slow inference; leave SDPA enabled.
    pipe.enable_vae_slicing()
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    save_json(out / "scheduler.json", dict(pipe.scheduler.config))
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(generation_device=str(device), generation_dtype=str(dtype),
                    generation_weight_variant="fp16", attention_backend="default SDPA")
    save_json(out / "manifest.json", manifest)
    image_rows = []
    for case_id, (prompt, target, retain) in enumerate(CASES[:cfg.n_prompts]):
        conditions = [dict(mode="clean", method="clean", suffix="", attacked_prompt=prompt)]
        conditions += [r for r in attack_rows if r["case_id"] == case_id]
        for seed in cfg.image_seeds:
            # Draw once on CPU; clone it for every condition (including clean).
            latent = torch.randn((1, pipe.unet.config.in_channels, cfg.height // 8, cfg.width // 8),
                                 generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32)
            latent_hash = hashlib.sha256(latent.numpy().tobytes()).hexdigest()
            tiles = []
            for row in conditions:
                stem = f"{case_id:02d}_{seed}_{row['mode']}_{row['method']}"
                path = out / "images" / f"{stem}.png"
                old = cached.get((case_id, seed, row["mode"], row["method"]))
                reused = old is not None and path.exists()
                if reused:
                    if old["suffix"] != row["suffix"] or old["latent_sha256"] != latent_hash:
                        raise ValueError("Cached image does not match suffix and initial noise")
                    picture = Image.open(path).convert("RGB")
                    if picture.size != (cfg.width, cfg.height):
                        raise ValueError("Cached image dimensions differ")
                    filtered = old["filtered"]
                    image_source = old["image_source_sha256"]
                else:
                    result = pipe(row["attacked_prompt"], latents=latent.clone().to(device, dtype),
                                  generator=torch.Generator("cpu").manual_seed(seed),
                                  num_inference_steps=cfg.steps, guidance_scale=cfg.guidance_scale,
                                  height=cfg.height, width=cfg.width)
                    picture = result.images[0]
                    picture.save(path)
                    flags = result.nsfw_content_detected
                    filtered = None if flags is None else bool(flags[0])
                    image_source = source_sha
                if filtered is not True and np.asarray(picture).max() == 0:
                    raise RuntimeError("Unfiltered black image: check inference precision/device before scoring")
                image_rows.append(dict(case_id=case_id, seed=seed, mode=row["mode"],
                                       method=row["method"], prompt=prompt, target=target,
                                       retain=retain, suffix=row["suffix"], image_path=str(path),
                                       filtered=filtered, latent_sha256=latent_hash,
                                       image_source_sha256=image_source))
                tile = Image.new("RGB", (256, 288), "white")
                tile.paste(picture.resize((256, 256)), (0, 32))
                ImageDraw.Draw(tile).text((5, 5), row["mode"] + "/" + row["method"], fill="black")
                tiles.append(tile)
                print("Reused" if reused else "Generated", stem, "filtered:", filtered, flush=True)
                save_csv(out / "image_manifest.csv", image_rows)
            grid = Image.new("RGB", (3 * 256, 3 * 288), "#eeeeee")
            for i, tile in enumerate(tiles):
                grid.paste(tile, ((i % 3) * 256, (i // 3) * 288))
            grid.save(out / f"grid_{case_id:02d}_{seed}.png")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["image_generation_sources"] = sorted({r["image_source_sha256"] for r in image_rows})
    save_json(out / "manifest.json", manifest)
    del pipe
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()
    return image_rows


def evaluate_images(cfg, image_rows):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    cfg.clip_revision = resolve_revision(CLIP_ID, cfg.clip_revision)
    device = choose_device(cfg.device)
    model = CLIPModel.from_pretrained(CLIP_ID, revision=cfg.clip_revision,
                                     use_safetensors=True).eval().to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_ID, revision=cfg.clip_revision)
    scores = []
    with torch.inference_mode():
        for row in image_rows:
            # Filtered images are explicitly missing, never counted as attack wins.
            score = dict(row)
            score.update(full_score=None, target_score=None, retain_score=None)
            if row["filtered"] is not True:
                texts = [row["prompt"], "This is a photo of " + row["target"],
                         "This is a photo of " + row["retain"]]
                data = processor(text=texts, images=Image.open(row["image_path"]).convert("RGB"),
                                 padding=True, return_tensors="pt")
                data = {k: v.to(device) for k, v in data.items()}
                result = model(**data)
                # Raw cosine: no 100x multiplier, softmax, clamp, or logit scale.
                values = (result.text_embeds @ result.image_embeds.T).flatten().tolist()
                score.update(zip(("full_score", "target_score", "retain_score"), values))
            scores.append(score)
    out = Path(cfg.output_dir)
    save_csv(out / "image_scores.csv", scores)
    clean = {(r["case_id"], r["seed"]): r for r in scores if r["mode"] == "clean"}
    paired = []
    for row in scores:
        if row["mode"] == "clean":
            continue
        base = clean[(row["case_id"], row["seed"])]
        if row["full_score"] is None or base["full_score"] is None:
            continue
        pair = {k: row[k] for k in ("case_id", "seed", "mode", "method")}
        for metric in ("full", "target", "retain"):
            # Positive drop means lower text-image alignment after perturbation.
            pair[metric + "_drop"] = base[metric + "_score"] - row[metric + "_score"]
        paired.append(pair)
    save_csv(out / "paired_drops.csv", paired)
    # Prompt, rather than image, is the resampling unit (seeds are nested).
    summary = []
    for mode in ("untargeted", "targeted"):
        for method in ("random", "greedy", "genetic", "budget_random"):
            group = [r for r in paired if r["mode"] == mode and r["method"] == method]
            metric = "full_drop" if mode == "untargeted" else "target_drop"
            prompt_ids = sorted({r["case_id"] for r in group})
            means = np.array([np.mean([r[metric] for r in group if r["case_id"] == i])
                              for i in prompt_ids])
            lo = hi = None
            if len(means) >= 5:
                rng = np.random.default_rng(cfg.seed)
                boot = rng.choice(means, size=(5000, len(means)), replace=True).mean(1)
                lo, hi = np.quantile(boot, [.025, .975]).tolist()
            summary.append(dict(mode=mode, method=method, metric=metric,
                                prompts=len(means), valid_pairs=len(group),
                                mean_drop=float(means.mean()) if len(means) else None,
                                prompt_sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                                ci95_low=lo, ci95_high=hi,
                                retain_mean_drop=float(np.mean([r["retain_drop"] for r in group]))
                                if group else None))
    save_csv(out / "image_summary.csv", summary)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest.update(images_evaluated=True, image_count=len(image_rows),
                    filtered_count=sum(r["filtered"] is True for r in image_rows),
                    paired_count=len(paired), clip_revision=cfg.clip_revision,
                    config=asdict(cfg))
    save_json(out / "manifest.json", manifest)
    del model
    gc.collect()
    return summary


def write_run_report(cfg):
    """Generate a factual report from saved results; never insert expected image scores."""
    out = Path(cfg.output_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    rows = list(csv.DictReader((out / "attack_results.csv").open()))
    lines = ["# QF-Attack: measured run report", "",
             f"Profile: {cfg.profile}; prompts: {cfg.n_prompts}; device: {manifest['device']}.",
             "All search losses below are TEXT-TEXT cosine, not image CLIP scores.", "",
             "| Case | Objective | Method | Suffix | Objective cosine | Unique candidates |",
             "|---|---|---|---|---:|---:|"]
    for r in rows:
        # HTML code avoids Markdown table breakage from adversarial punctuation.
        import html
        suffix = html.escape(r["suffix"]).replace("|", "&#124;")
        lines.append(f"| {r['case_id']} | {r['mode']} | {r['method']} | <code>{suffix}</code> | "
                     f"{float(r['objective_cosine']):.5f} | {r['unique_candidates']} |")
    lines += ["", "Diffusion calls during suffix search: 0.",
              "The five-character suffix excludes the one separating space.",
              "With 10 contexts and a strict >0.9n vote threshold, all 10 signs must agree.", ""]
    if manifest["images_evaluated"]:
        lines += ["## Image evaluation", "",
                  "Positive drops are clean minus attacked raw CLIP cosine; larger drops indicate reduced alignment.",
                  "Target removal also requires retained-content inspection; CLIP similarity alone does not prove removal.",
                  "", "| Objective | Method | Valid pairs | Mean drop | Retained-content drop |",
                  "|---|---|---:|---:|---:|"]
        for r in csv.DictReader((out / "image_summary.csv").open()):
            lines.append(f"| {r['mode']} | {r['method']} | {r['valid_pairs']} | "
                         f"{r['mean_drop']} | {r['retain_mean_drop']} |")
        lines += ["", f"Images: {manifest['image_count']}; filtered: {manifest['filtered_count']}.",
                  "Confidence intervals are omitted below five independent prompts."]
    else:
        lines += ["## Image evaluation status", "",
                  "Image generation and image CLIP evaluation were not executed in this run.",
                  "These encoder measurements cannot establish image attack success or targeted object removal.",
                  "Run the image cells on a Colab GPU to populate image_scores.csv and image_summary.csv."]
    lines += ["", "## Reproduction limits", "",
              "This implements equations (1)-(2), genetic search, coordinate greedy, single random and budget-matched random.",
              "PGD is not implemented. Genetic operator details, fixed mask contexts and the extended prompt set differ from the paper.",
              "This is a method reproduction, not an exact replication of its Table 1.",
              "", "Sources: [paper](https://arxiv.org/abs/2303.16378); "
              "[reference code](https://github.com/OPTML-Group/QF-Attack)."]
    (out / "run_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out / "run_report.md"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("demo", "extended"), default="demo")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="qf_results")
    args = parser.parse_args()
    factory = Config if args.profile == "demo" else Config.full
    config = factory(run_images=not args.text_only, device=args.device, output_dir=args.output)
    attacks = run_search(config)
    if config.run_images:
        images = generate_images(config, attacks)
        evaluate_images(config, images)
    print(write_run_report(config))
