# Baseline Model Audit

All source paths and code locations refer to the original experimental workspace. Artifact paths identify earlier local snapshots, not files included here. A yes entry means that the model row appears in the reported baseline table; it does not mean its source or artifacts are supplied. Recorded commands are not executed. Filenames and metadata do not establish authorship or personal contributions.

| Model | Included in reported baseline table? | Input scope | Prediction target | Level/direct or delta-to-level | Uses previous actual level? | Persistence/Naive-last blending? | Evaluation loop | Recursive feedback? | Seed/config/checkpoint | Prediction CSV path | Metric CSV path | Reproduce command | Risk/notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Naive-last | yes | target-only | US/KR/JP | level baseline | yes | no extra | append actual after prediction | no | fixed; no model checkpoint | artifacts/naive_last/predictions_test_*.csv | artifacts/naive_last/test_metrics.csv | shared robustness --run (reference only) | original suite also runs excluded models |
| Drift | yes | target-only | US/KR/JP | level + observed average drift | yes | no extra | append actual after prediction | no | fixed; no model checkpoint | artifacts/drift/predictions_test_*.csv | artifacts/drift/test_metrics.csv | shared robustness --run (reference only) | drift estimate updates with observed history |
| LSTM-33var residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level | yes | no extra; alpha=1 | observed-array window slices | no | w6/h24/L1/do0.1/lr0.0005/Huber; seeds 5,2; epochs 10,9; top2 mean | artifacts/lstm_33var/predictions_test_*.csv | artifacts/lstm_33var/test_metrics.csv | residual stability --run | selected weights not persisted |
| TCN-33var causal residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level | yes | no extra; alpha=1 | observed-array window slices | no | w3/k3/ch16/do0.1/lr0.0005/Huber; seed5/epoch36 | artifacts/tcn_33var_causal/predictions_test_*.csv | artifacts/tcn_33var_causal/test_metrics.csv | TCN stability --run | weights not persisted; replaces older non-stability row |
| Transformer-33var residual_delta | yes | 33-variable | delta for US/KR/JP | delta-to-level + final blend | yes | yes; alpha=0.5 | observed-array window slices | no | w6/h8/L1/do0/lr0.001/Huber; seed5/epoch42 | artifacts/transformer_33var/predictions_test_*.csv | artifacts/transformer_33var/test_metrics.csv | residual stability --run | final score includes blending; weights not persisted |
| Deterministic MTGNN | yes; dependency-qualified | graph-33-variable | 33-node level; report US/KR/JP | direct level | yes, via history window | no; anchor/debias none | observed series[t-24:t] | no | seed1/w24/30 epochs; selected28; checkpoint_best.pt available locally | artifacts/mtgnn_deterministic/predictions_test_*.csv | artifacts/mtgnn_deterministic/test_metrics.csv | --arm mtgnn_deterministic --train-full / --eval-test | mean validation RRSE selection; external backbone; no weight copied |

## Code Evidence

- Naive/Drift: `run_baseline_robustness_20260923.py:74-88`. Prediction precedes `observed.append(actual)`.
- Neural input/delta: `run_neural_baseline_preregistered_strengthening_20260930.py:169-185`. Level restoration uses actual[t-1]; `anchored()` is at 189-191.
- Selected LSTM/Transformer test path: `run_neural_residual_stability_20260930.py:269-286`. Selected member predictions are averaged before applying the recorded alpha.
- Causal TCN: `run_tcn_stability_check_20261001.py:75-112,262-275`. Left causal padding, actual-last level restoration, and the recorded selected alpha.
- Deterministic MTGNN: `run_mtg_nn_matched.py:72-77,608-622`. Single eval/no_grad forward with an input slice excluding the current month.
- Historical `artifacts/<model>/selected_config.json` records point to original configuration sources. The selected settings above are documentary records; those snapshots are not included.

## Reproducibility Boundaries

The previous audit recorded the existence of source code and saved artifacts. No scripts or model inference are run during this documentation change.

The previous local prediction snapshots were copies of saved results. The existing verification report records comparisons of copy hashes, calendar/schema/finite values, and saved metrics against tables. Its PASS is not a standalone execution or reproduction guarantee.

Shared original runners include other arms and may lack per-model CLI selection. Do not invent unsupported `--models` or `--targets` options, disable output protection, or remove existing results. A separate reproduction requires the original dependencies and execution layout.
