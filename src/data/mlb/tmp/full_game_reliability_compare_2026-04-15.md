# Full Game Reliability Compare (2026-04-15)

| scenario | acc | pub_acc | pub_cov | roi/bet | priced | rel_split | rel_shift | decision | reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| reliability_off | 0.581633 | 0.515625 | 0.108844 | -0.007697 | 35 | 0.0000 | nan | baseline | reference |
| reliability_on | 0.556122 | 0.575000 | 0.068027 | 0.070746 | 21 | 1.0000 | nan | reject | accuracy_drop;priced_drop |

## Guardrails
- global_accuracy_improved: accuracy > baseline (objetivo primario)
- no_accuracy_regression: accuracy >= baseline
- no_published_accuracy_regression: published_accuracy >= baseline
- roi_non_drop: published_roi_per_bet >= baseline
- priced_non_drop: published_priced_picks >= baseline