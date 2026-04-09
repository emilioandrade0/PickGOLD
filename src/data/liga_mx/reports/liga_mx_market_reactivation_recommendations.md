# Liga MX Market Reactivation Recommendations

Decision rule: prioritize temporal accuracy and calibration improvements from baseline to v3.

| Market | Baseline Acc | V3 Acc | Delta Acc | Baseline LogLoss | V3 LogLoss | Decision |
|---|---:|---:|---:|---:|---:|---|
| full_game | 0.5081 | 0.4865 | -0.0216 | 1.3848 | 1.4024 | KEEP_FROZEN |
| over_25 | 0.5351 | 0.5297 | -0.0054 | 1.0500 | 1.0675 | KEEP_FROZEN |
| btts | 0.5405 | 0.4973 | -0.0432 | 0.9646 | 1.0059 | KEEP_FROZEN |