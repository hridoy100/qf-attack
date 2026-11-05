
## 6. Extension: Optimizing the attack

Status: I am proposing follow-up experiments. These changes have not been implemented or tested in the delivered attack code. The aim is greater image-level effectiveness under the same 5-character suffix constraint. In this attack, text is the editable input and the generated image is the outcome to evaluate; there is no input image whose pixels are optimized.

First priority: optimize the intended outcome. Our targeted genetic run reduced mean target alignment by 0.0120, but retained-content alignment fell by 0.0516. The bicycle also remained visible in the targeted greedy example despite a masked text cosine of 0.6010. These observations motivate improving the loss before simply increasing the search budget. They do not establish which proposed replacement will work best.

| Important part | What to examine and why |
|---|---|
| Target noun and attributes | Separate 'bicycle' from 'black', or 'plate' from 'red apple'. Distinguish object removal from a color or style change. Use controlled phrase-removal or replacement probes to estimate sensitivity. |
| Retained nouns and relations | Protect 'brick wall' and 'red apple'. Inspect relations such as 'against' and 'on': changing a relation may alter layout while leaving both objects present. These probes diagnose importance; they do not edit the base prompt in the suffix-only attack. |
| Suffix tokens and boundaries | Measure the actual CLIP tokenization of each complete prompt. Five characters can become a different number of BPE tokens. Track suffix token IDs, boundary effects and truncation; keep the suffix at the end for a comparable threat model. |
| Target and retained image regions | Inspect bicycle frame/wheels versus wall texture and geometry; plate rim/surface versus apple shape, color and count. Include background and object placement, not just the target crop. These are evaluation cues, not established causal rankings. |

A proposed directional, multi-objective loss. Let z(x) be a normalized text representation and let x_minus be a grammatical caption describing the desired remaining scene: for example, 'A red apple' when removing the plate. It is a loss reference, not the actual prompt sent to diffusion. For x_adv = x + space + suffix, consider:

```text
L = ||z(x_adv) - z(x_minus)||^2 + lambda * ||P_keep (z(x_adv) - z(x))||^2
```

P_keep is a candidate projector onto retained-concept directions, estimated from separate paired text examples in the same representation space. Lambda controls the preservation penalty. This is a hypothesis: target and retained semantics can overlap, and a caption without the target does not guarantee its absence in an image. Compare the original loss, the directional term alone and the combined loss on held-out prompts.


## 6.1. Search, image feedback and validation

| Optimization type | Proposed change and comparison |
|---|---|
| Discrete search | Test GA diversity/restarts, mutation schedules, coordinate refinement and beam search. Compare all methods against uniform random search at the same unique-candidate budget. More generations alone do not demonstrate a better optimizer. |
| Feature and mask selection | Use more varied context pairs, test sign stability together with change magnitude, and validate masks on unseen contexts. Ablate target noun, attributes and relations separately before assigning importance. |

Gradient-based hard-prompt search is an established research direction (Wen et al., Hard Prompts Made Easy, 2023; reference below). Adapting it to this character constraint would be a new implementation and evaluation, not a result of the current notebook.

Where image information belongs. For a strictly query-free search, generated images are used only after a suffix is fixed. Evaluate target presence with an independent detector or blinded human review, and inspect retained objects and background separately. An object mask can identify the target region; compare outside an expanded target region to allow necessary changes to shadows and occlusions. Region similarity is a preservation proxy, not proof.

If image feedback selects suffixes, loss weights or restarts, explicitly call that a query-assisted extension and count all diffusion calls. A possible objective minimizes target presence plus a weighted penalty for changes to retained objects and background. The weights can be tuned on development prompts, or candidates can be compared on a Pareto frontier showing the trade-off between removal and preservation. Attention maps may help diagnose associations, but they are not causal evidence or an object-presence test.

A practical experiment order. First compare the three text losses at a fixed budget (for example, 1,000 unique candidates per prompt and method). Next ablate representation and mask construction, then compare search strategies and restarts within that same budget. Freeze choices on a development set before evaluating new prompts and seeds. If development tuning uses generated images, report those queries even when final suffix generation uses text only.

For targeted success, first verify that the clean image contains the target; then require its removal together with preservation of the intended remaining scene. Report baseline failures and the eligible denominator separately. Define any detector thresholds and preservation tolerances before testing. Keep resolution, denoising steps and guidance fixed across methods. Scale beyond the two-scene demo to more prompts, several search initializations and multiple paired image seeds; report prompt-level uncertainty, costs and failure cases. These tests establish whether an optimization improves the image outcome rather than only the encoder surrogate.

Reference: [Wen et al., Hard Prompts Made Easy](https://arxiv.org/abs/2302.03668).
