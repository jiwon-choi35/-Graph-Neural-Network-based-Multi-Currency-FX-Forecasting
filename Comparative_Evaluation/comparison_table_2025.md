| 모형 | 2024 RRSE US | KR | JP | 2025 RRSE US | KR | JP | 2025 RAE US | KR | JP | 검증 출처 | 평가 출처 | 비고 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive-last | 0.697 | 0.707 | 0.419 | 0.642 | 0.446 | 0.310 | 0.645 | 0.435 | 0.299 | original_saved metric (posthoc CSV reproduces it) | original_saved |  |
| Drift | 0.763 | 0.715 | 0.382 | 0.652 | 0.398 | 0.411 | 0.680 | 0.361 | 0.382 | original_saved metric (posthoc CSV reproduces it) | original_saved |  |
| AR (diff, p=7/7/1) | 0.201 | 0.281 | 0.333 | 0.144 | 0.117 | 0.156 | 0.149 | 0.112 | 0.147 | new_run 2026-10-15 (ar_baseline.py) | new_run 2026-10-15 | audit 1015; validation-selected order, no post-processing |
| LSTM | 0.362 | 0.424 | 0.402 | 0.667 | 0.281 | 0.395 | 0.698 | 0.272 | 0.369 | original_saved metric (reproduction CSV, diff <=1.6e-8) | original_saved | top-2 seed mean, alpha=1 |
| GRU | 0.314 | 0.335 | 0.361 | 0.847 | 0.343 | 0.565 | 0.849 | 0.351 | 0.564 | new_run 2026-10 (original residual pipeline) | new_run 2026-10 | not a result of the 9/30 preregistered run; config 5, top-2 seeds 3;5, alpha=1 |
| TCN (causal) | 0.338 | 0.349 | 0.353 | 0.707 | 0.600 | 0.324 | 0.733 | 0.644 | 0.319 | original_saved metric | original_saved |  |
| Transformer | 0.341 | 0.331 | 0.341 | 0.697 | 0.252 | 0.328 | 0.705 | 0.296 | 0.304 | original_saved metric | original_saved | includes validation-selected persistence blend 0.5 |
| MTGNN | 0.224 | 0.398 | 0.545 | 0.518 | 0.215 | 0.224 | 0.530 | 0.194 | 0.216 | original_saved CSV | original_saved | no post-processing |
| MTGNN + Eq.(7), val alpha | 0.224 | 0.380 | 0.371 | 0.518 | 0.180 | 0.263 | 0.530 | 0.169 | 0.254 | posthoc on original_saved CSV | posthoc on original_saved | alpha US/KR/JP = 0.000/0.167/0.672 |
| B-MTGNN (Eq.7, val alpha) | 0.407 | 0.471 | 0.397 | 0.316 | 0.213 | 0.295 | 0.324 | 0.233 | 0.285 | reproduction (ep57, CPU, MC seed 1) | reproduction | alpha 0.377/0.426/0.804 |
| B-MTGNN (w/o Eq.7) | 0.532 | 0.613 | 0.682 | 0.437 | 0.299 | 0.294 | 0.405 | 0.308 | 0.269 | reproduction | reproduction |  |
| B-MTGNN (dropout off, single pass, val alpha) | 0.423 | 0.470 | 0.396 | 0.329 | 0.211 | 0.295 | 0.339 | 0.231 | 0.284 | new computation (ep57 weights) | new computation | alpha re-selected on dropout-off 2024 predictions: 0.403/0.438/0.803 |

각주용

| 항목 | 구간 | US | KR | JP | 출처 |
|---|---|---|---|---|---|
| Linear extrapolation 2y[t-1]-y[t-2] (no estimation) | val | 0.277 (RAE 0.251) | 0.350 (RAE 0.400) | 0.340 (RAE 0.269) | posthoc (rule, no selection) |
| Linear extrapolation 2y[t-1]-y[t-2] (no estimation) | test | 0.204 (RAE 0.198) | 0.134 (RAE 0.130) | 0.159 (RAE 0.150) | posthoc (rule, no selection) |
| Transformer without persistence blend (alpha=1) | test | 1.023 (RAE 1.031) | 0.516 (RAE 0.582) | 0.387 (RAE 0.399) | posthoc on original_saved (baseline_audit.py, 1015) |
| B-MTGNN retrain seeds 1-5 (w/o Eq.7) | test | 0.466 ± 0.224 (RAE 0.422) | 0.367 ± 0.146 (RAE 0.388) | 0.283 ± 0.038 (RAE 0.258) | new computation on ablation full_s1..s5 checkpoints |
| B-MTGNN retrain seeds 1-5 (w/o Eq.7) | val | 0.511 ± 0.063 (RAE 0.508) | 0.591 ± 0.076 (RAE 0.647) | 0.647 ± 0.114 (RAE 0.503) | new computation on ablation full_s1..s5 checkpoints |
| B-MTGNN retrain seeds 1-5 (val_alpha, per-seed val alpha) | test | 0.318 ± 0.110 (RAE 0.290) | 0.198 ± 0.075 (RAE 0.221) | 0.284 ± 0.014 (RAE 0.278) | new computation on ablation full_s1..s5 checkpoints |
| B-MTGNN retrain seeds 1-5 (val_alpha, per-seed val alpha) | val | 0.427 ± 0.099 (RAE 0.392) | 0.454 ± 0.043 (RAE 0.469) | 0.383 ± 0.029 (RAE 0.348) | new computation on ablation full_s1..s5 checkpoints |
