# Liga MX Market Reactivation Recommendations

Decision rule: prioritize temporal accuracy and calibration improvements from baseline to v3.

| Market | Baseline Acc | V3 Acc | Delta Acc | Baseline LogLoss | V3 LogLoss | Decision |
|---|---:|---:|---:|---:|---:|---|
| full_game | 0.5400 | 0.4876 | -0.0524 | 1.3635 | 1.3593 | KEEP_FROZEN |
| over_25 | 0.5038 | 0.5276 | +0.0238 | 1.0609 | 1.0963 | KEEP_FROZEN |
| btts | 0.5333 | 0.4971 | -0.0362 | 0.9947 | 1.0464 | KEEP_FROZEN |