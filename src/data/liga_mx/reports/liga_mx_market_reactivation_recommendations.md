# Liga MX Market Reactivation Recommendations

Decision rule: prioritize temporal accuracy and calibration improvements from baseline to v3.

| Market | Baseline Acc | V3 Acc | Delta Acc | Baseline LogLoss | V3 LogLoss | Decision |
|---|---:|---:|---:|---:|---:|---|
| full_game | 0.5429 | 0.4971 | -0.0457 | 1.3545 | 1.3435 | KEEP_FROZEN |
| over_25 | 0.5200 | 0.5371 | +0.0171 | 1.0401 | 1.0737 | KEEP_FROZEN |
| btts | 0.5314 | 0.4971 | -0.0343 | 0.9933 | 1.0356 | KEEP_FROZEN |