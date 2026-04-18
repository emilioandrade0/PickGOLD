# Full Game Volatility Normalization Compare (2026-04-15)

| scenario | acc | pub_acc | pub_cov | roi/bet | priced | vol_test | vol_high_test | decision | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| vol_norm_off | 0.578231 | 0.513889 | 0.122449 | 0.036978 | 37 | 0.0000 | 0.0000 | baseline | reference |
| vol_norm_on | 0.581633 | 0.515625 | 0.108844 | -0.007697 | 35 | 0.3000 | 0.2383 | promote_global_accuracy | global_accuracy_up |

## Guardrails
- global_accuracy_improved: accuracy > baseline (objetivo primario)
- no_accuracy_regression: accuracy >= baseline
- no_published_accuracy_regression: published_accuracy >= baseline
- roi_non_drop: published_roi_per_bet >= baseline
- priced_non_drop: published_priced_picks >= baseline