# QF-Attack: measured run report

Profile: demo; prompts: 2; device: mps.
All search losses below are TEXT-TEXT cosine, not image CLIP scores.

| Case | Objective | Method | Suffix | Objective cosine | Unique candidates |
|---|---|---|---|---:|---:|
| 0 | untargeted | random | <code>^FF1%</code> | 0.86276 | 1 |
| 0 | untargeted | greedy | <code>FC·!U</code> | 0.67223 | 306 |
| 0 | untargeted | genetic | <code>&#124;ZPV&gt;</code> | 0.69346 | 106 |
| 0 | untargeted | budget_random | <code>E2&#124;,D</code> | 0.69014 | 106 |
| 0 | targeted | random | <code>.7P&lt;Y</code> | 0.85095 | 1 |
| 0 | targeted | greedy | <code>DJE$8</code> | 0.60104 | 306 |
| 0 | targeted | genetic | <code>61PV&#x27;</code> | 0.74112 | 105 |
| 0 | targeted | budget_random | <code>QMT$9</code> | 0.70867 | 105 |
| 1 | untargeted | random | <code>H3X4\</code> | 0.82539 | 1 |
| 1 | untargeted | greedy | <code>.Y1$)</code> | 0.72309 | 306 |
| 1 | untargeted | genetic | <code>CT7#S</code> | 0.65914 | 103 |
| 1 | untargeted | budget_random | <code>+&#124;SOX</code> | 0.70216 | 103 |
| 1 | targeted | random | <code>#5)CT</code> | 0.75769 | 1 |
| 1 | targeted | greedy | <code>GP38I</code> | 0.59528 | 306 |
| 1 | targeted | genetic | <code>Y!KD8</code> | 0.71072 | 94 |
| 1 | targeted | budget_random | <code>X27QH</code> | 0.74572 | 94 |

Diffusion calls during suffix search: 0.
The five-character suffix excludes the one separating space.
With 10 contexts and a strict >0.9n vote threshold, all 10 signs must agree.

## Image evaluation

Positive drops are clean minus attacked raw CLIP cosine; larger drops indicate reduced alignment.
Target removal also requires retained-content inspection; CLIP similarity alone does not prove removal.

| Objective | Method | Valid pairs | Mean drop | Retained-content drop |
|---|---|---:|---:|---:|
| untargeted | random | 2 | 0.002903267741203308 | 0.010200604796409607 |
| untargeted | greedy | 2 | -0.02236771583557129 | -0.0072832778096199036 |
| untargeted | genetic | 2 | -0.02772088348865509 | 0.00842209905385971 |
| untargeted | budget_random | 2 | 0.029608607292175293 | 0.019840985536575317 |
| targeted | random | 2 | -0.012225925922393799 | -0.00640704482793808 |
| targeted | greedy | 2 | -0.0032863765954971313 | 0.011839881539344788 |
| targeted | genetic | 2 | 0.011972464621067047 | 0.051649920642375946 |
| targeted | budget_random | 2 | -0.014364682137966156 | -0.0073309168219566345 |

Images: 18; filtered: 0.
Confidence intervals are omitted below five independent prompts.

## Reproduction limits

This implements equations (1)-(2), genetic search, coordinate greedy, single random and budget-matched random.
PGD is not implemented. Genetic operator details, fixed mask contexts and the extended prompt set differ from the paper.
This is a method reproduction, not an exact replication of its Table 1.

Sources: [paper](https://arxiv.org/abs/2303.16378); [reference code](https://github.com/OPTML-Group/QF-Attack).
