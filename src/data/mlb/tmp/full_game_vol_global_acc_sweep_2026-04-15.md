# Full Game Volatility Global Accuracy Sweep (2026-04-15)

| scenario | acc | pub_acc | pub_cov | roi/bet | priced | beta_mean | decision | reason |
|---|---:|---:|---:|---:|---:|---:|---|---|
| beta_neg_008 | 0.581633 | 0.515625 | 0.108844 | -0.007697 | 35 | -0.0145 | promote_global_accuracy | global_accuracy_up |
| beta_neg_012 | 0.579932 | 0.525424 | 0.100340 | 0.049758 | 30 | -0.0153 | promote_global_accuracy | global_accuracy_up |
| baseline_off | 0.578231 | 0.513889 | 0.122449 | 0.036978 | 37 | 0.0000 | baseline | reference |
| vol_norm_only | 0.578231 | 0.525424 | 0.100340 | 0.049758 | 30 | 0.0000 | reject | global_accuracy_not_up |
| beta_neg_005 | 0.578231 | 0.507937 | 0.107143 | -0.007697 | 35 | -0.0009 | reject | global_accuracy_not_up |
| beta_pos_003 | 0.569728 | 0.508475 | 0.100340 | 0.033514 | 29 | 0.0033 | reject | global_accuracy_not_up |
| beta_neg_003 | 0.568027 | 0.507692 | 0.110544 | -0.004578 | 37 | -0.0022 | reject | global_accuracy_not_up |
| beta_pos_008 | 0.568027 | 0.516667 | 0.102041 | 0.049773 | 32 | 0.0116 | reject | global_accuracy_not_up |
| beta_pos_005 | 0.562925 | 0.516129 | 0.105442 | 0.049773 | 32 | 0.0055 | reject | global_accuracy_not_up |
| beta_pos_012 | 0.562925 | 0.515152 | 0.112245 | -0.004578 | 37 | 0.0153 | reject | global_accuracy_not_up |

## Objective
- maximize full_game global accuracy (not published metrics)