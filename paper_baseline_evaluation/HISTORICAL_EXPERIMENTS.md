# Historical Experiments and Excluded Results

This document summarizes the recorded differences between earlier experiments and the reported baseline evaluation conditions. Source results, earlier predictions, historical/excluded manifests, and external repository copies are references only and are not added. Exclusion reflects documentation scope or evaluation conditions, not inferred code ownership.

## Spring Experiment History

The February 2026 ARIMA, LSTM-delta, TCN-delta, and Transformer-delta experiments used local data. The initial neural delta outputs were restored using the previous actual level before level evaluation, rather than being scored directly as deltas. The recorded results primarily covered KR/JP; US scores were unavailable.

Some early results used monthly actual-update. Some rolling results after February 19 fed predictions back recursively. Previously recorded issues include teacher-forced CSVs classified as recursive results and rankings based on test RMSE.

These historical observations differ from the frozen-data results. The original `data_raw/data.csv` was reported missing from the experimental workspace during the earlier audit, so identical-data reproduction cannot be claimed. Spring values are not substituted into the current baseline table or Table 3. The original historical manifest recorded archive-only references; earlier experiment files are not changed.

Source audit reference: `runs/baseline_level_check/spring_2026_baseline_history_audit_20261008.md`. That audit is not added because it contains local absolute-path links. Dates were based on filesystem metadata without identified experiment commits and are not evidence of authorship.

## Excluded Models and Rows

| Model/category | Reason | Existing source kept |
|---|---|---|
| ARIMA / Expanded ARIMA | Outside the six-model table; statistical/contextual/historical reference only | runs/run_arima/ and runs/reviewer_matched_20260923/baseline_robustness/expanded_arima/ |
| ETS | Outside the current table scope | runs/reviewer_matched_20260923/baseline_robustness/ets/ |
| Ridge-33var | Outside the current table scope | runs/reviewer_matched_20260923/baseline_robustness/ridge_33var/ |
| B-MTGNN / Clean B-MTGNN | Implementation, MC dropout, anchor settings, and official model selection are outside this baseline documentation; no authorship attribution | Original source comparison table and run directories remain unchanged; scores are not copied here |
| GRU / FX-only / selected-small | Outside the reported baseline table scope | Original neural results remain available; earlier snapshots filtered designated 33var arms |
| Older TCN-33var row | Replaced in the reported table by the recorded causal stability result | Earlier sources/results are preserved; no new training or selection is performed |
| Recursive/fixed-origin/2026 forecasts | Different evaluation problems from monthly one-step results | Original spring audit and forecast directories remain unchanged |

The original excluded-file manifest is referenced at `manifests/excluded_large_files_manifest.csv` in the experimental workspace; it is not included. An excluded label does not mean an original file was deleted. Source comparison rows remain unchanged. Data/code redistribution and public-release permissions require separate review.
