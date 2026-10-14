# Reproducibility Boundaries and Source References

This document records the source inputs, runner entry points, and limitations behind the reported baseline results. Paths and commands refer to the original FX baseline experimental workspace, not an executable layout supplied by this directory. No scripts or model inference are run as part of this documentation change.

## Saved Verification Evidence

The original verifier compared saved RRSE/Max scores with the source table and inspected prediction schema, twelve-month dates, finite values, and copy hashes. It did not compute new prediction-based RRSE or load checkpoints. The [existing verification record](manifests/official_artifact_verification.json) preserves that previous local inspection.

Verification tools and artifact snapshots are not included. The saved PASS is not a standalone reproduction guarantee. Builder/verifier improvements are outside this documentation-only change.

## Environment Boundary

The previously recorded artifact verification environment was Python 3.11.9. An exact historical model environment lock was not saved. The original requirements list was a minimum dependency list, not a complete environment lock. No dependencies are installed or added here.

## Frozen Inputs

Original neural/statistical baseline data path:

`external_team_repo_20260922_newzip/Cyber-trend-forecasting-main 3/B-MTGNN/data/sm_data.csv`

Frozen SHA-256: `5086946995c5e2da176c17a73b94ebdbdb6c2e75f2a59b5cc5c1d932031132dd`.

The deterministic runner references `external_team_repo/Cyber-trend-forecasting-main 3/B-MTGNN/`, including `net.py`, `layer.py`, `util.py`, data, and graph inputs. Its recorded data hash is identical. The graph hash is documented in [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md). Data and external-code redistribution permissions and the original dependency layout are required. Local `data/sm_data.csv` is not an interchangeable substitute.

## Original Runner Entry Points

These files are references only and are not added. Preserve their original root-relative layout when arranging a separate reproduction environment.

| Required file | Role |
|---|---|
| run_baseline_robustness_20260923.py | Naive-last/Drift entry point within a shared multi-model suite |
| run_neural_residual_stability_20260930.py | LSTM/Transformer residual prediction and seed-ensemble entry point |
| run_neural_baseline_preregistered_strengthening_20260930.py | Shared neural model, array, restoration, and metric imports |
| run_tcn_stability_check_20261001.py | Causal TCN entry point |
| run_mtg_nn_matched.py | Deterministic MTGNN validation selection and locked-test entry point |

| Models | Recorded original command | Limitation |
|---|---|---|
| Naive-last / Drift | python run_baseline_robustness_20260923.py --run | Also runs excluded statistical/neural models; no CLI restricts execution to these two rows |
| LSTM / Transformer | python run_neural_residual_stability_20260930.py --run | Also runs other models/scopes; no CLI restricts execution to these two rows |
| causal TCN | python run_tcn_stability_check_20261001.py --run | Separate TCN suite; refuses execution if its output directory exists |
| Deterministic training selection | python run_mtg_nn_matched.py --arm mtgnn_deterministic --train-full --epochs 30 --seed 1 | Requires original frozen dependencies and a separate new execution environment |
| Deterministic locked test | python run_mtg_nn_matched.py --arm mtgnn_deterministic --eval-test | Original checkpoint/config contain local paths; moved files may not work unchanged |

These commands are not executed here. They must not be rerun to select configurations with better test scores. Existing output protection must not be disabled, and existing results must not be deleted.

## Historical Runtime Dependencies

For report generation, the residual runner reads `runs/reviewer_matched_20260930/neural_baseline_preregistered_strengthening/final_test_metrics.csv`.

The causal TCN runner reads `runs/reviewer_matched_20260930/neural_residual_stability/final_test_metrics.csv` and `predictions/tcn_33_variable_residual_delta_{US,KR,JP}.csv` under the same original output directory.

Unmodified runners require these earlier comparison files at their recorded locations. They are not added here. Prediction files contain observed targets, so sharing requires permission.

## Reproducibility Boundaries

- The documents support inspection of evaluation conditions, saved scores, source mapping, and existing verification evidence, not standalone model execution.
- Earlier local metric/prediction/config copies are not included. Artifact paths in tables are historical references.
- Naive/Drift definitions and observed-history boundaries are identified by source references.
- Neural selected weights were not persisted. Recorded seeds, configurations, and grids do not guarantee identical values after fresh training.
- The original deterministic selected checkpoint is recorded at `runs/reviewer_matched_20260919/full_valid_select/mtgnn_deterministic/checkpoint_best.pt`; weights are not included.
