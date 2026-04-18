# Full Game Baseline Comparative Audit

- Old detail: src\data\mlb\tmp\full_game_detail_segment_off.csv
- New detail: src\data\mlb\tmp\full_game_detail_segment_on.csv
- Features: C:\Users\andra\Desktop\NBA GOLD\src\data\mlb\processed\model_ready_features_mlb.csv
- Rows compared: 588

## Overall
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 588 | 340 | 328 | -12 | 57.82% | 55.78% | -2.04 |

## Heavy Favorites vs Balanced
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heavy_favorite | 172 | 95 | 92 | -3 | 55.23% | 53.49% | -1.74 |
| mid_gap | 249 | 139 | 133 | -6 | 55.82% | 53.41% | -2.41 |
| balanced | 167 | 106 | 103 | -3 | 63.47% | 61.68% | -1.80 |

## High/Mid/Low Strength
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | 76 | 51 | 47 | -4 | 67.11% | 61.84% | -5.26 |
| mid | 52 | 30 | 30 | 0 | 57.69% | 57.69% | +0.00 |
| low | 460 | 259 | 251 | -8 | 56.30% | 54.57% | -1.74 |

## Home/Away Favorite
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| home_favorite | 21 | 14 | 14 | 0 | 66.67% | 66.67% | +0.00 |
| away_favorite | 558 | 322 | 310 | -12 | 57.71% | 55.56% | -2.15 |
| unknown | 9 | 4 | 4 | 0 | 44.44% | 44.44% | +0.00 |

## Confidence Tiers (Old Baseline Buckets)
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.65+ | 7 | 4 | 4 | 0 | 57.14% | 57.14% | +0.00 |
| 0.60-0.65 | 33 | 15 | 15 | 0 | 45.45% | 45.45% | +0.00 |
| 0.56-0.60 | 122 | 60 | 60 | 0 | 49.18% | 49.18% | +0.00 |
| 0.53-0.56 | 147 | 88 | 88 | 0 | 59.86% | 59.86% | +0.00 |
| <=0.53 | 279 | 173 | 161 | -12 | 62.01% | 57.71% | -4.30 |

## Confidence Tiers (New Baseline Buckets)
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.65+ | 4 | 2 | 2 | 0 | 50.00% | 50.00% | +0.00 |
| 0.60-0.65 | 46 | 25 | 25 | 0 | 54.35% | 54.35% | +0.00 |
| 0.56-0.60 | 107 | 57 | 57 | 0 | 53.27% | 53.27% | +0.00 |
| 0.53-0.56 | 155 | 87 | 85 | -2 | 56.13% | 54.84% | -1.29 |
| <=0.53 | 276 | 169 | 159 | -10 | 61.23% | 57.61% | -3.62 |

CSV: src\data\mlb\tmp\full_game_segment_on_vs_off_segments.csv
JSON: src\data\mlb\tmp\full_game_segment_on_vs_off_summary.json