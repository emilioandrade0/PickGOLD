# Liga MX Market Reactivation Recommendations

Decision rule: prioritize temporal accuracy and calibration improvements from baseline to v3.

| Market | Baseline Acc | V3 Acc | Delta Acc | Baseline LogLoss | V3 LogLoss | Decision |
|---|---:|---:|---:|---:|---:|---|
| full_game | 0.5607 | 0.5730 | +0.0123 | 1.1315 | 1.1341 | WATCHLIST |
| ht_result | 0.9988 | 0.9975 | -0.0012 | 0.0048 | 0.0055 | KEEP_FROZEN |
| over_25 | 0.7043 | 0.7092 | +0.0049 | 0.7252 | 0.7388 | WATCHLIST |
| btts | 0.6613 | 0.6736 | +0.0123 | 0.7169 | 0.7048 | WATCHLIST |