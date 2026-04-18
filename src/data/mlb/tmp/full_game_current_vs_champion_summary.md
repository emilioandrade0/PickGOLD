# Full Game Baseline Comparative Audit

- Old detail: src\data\mlb\tmp\full_game_detail_segment_off.csv
- New detail: src\data\mlb\walkforward\full_game\walkforward_predictions_detail.csv
- Features: C:\Users\andra\Desktop\NBA GOLD\src\data\mlb\processed\model_ready_features_mlb.csv
- Rows compared: 588

## Overall
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| overall | 588 | 340 | 312 | -28 | 57.82% | 53.06% | -4.76 |

## Heavy Favorites vs Balanced
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| heavy_favorite | 172 | 95 | 87 | -8 | 55.23% | 50.58% | -4.65 |
| mid_gap | 249 | 139 | 128 | -11 | 55.82% | 51.41% | -4.42 |
| balanced | 167 | 106 | 97 | -9 | 63.47% | 58.08% | -5.39 |

## High/Mid/Low Strength
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high | 76 | 51 | 47 | -4 | 67.11% | 61.84% | -5.26 |
| mid | 52 | 30 | 27 | -3 | 57.69% | 51.92% | -5.77 |
| low | 460 | 259 | 238 | -21 | 56.30% | 51.74% | -4.57 |

## Home/Away Favorite
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| home_favorite | 21 | 14 | 14 | 0 | 66.67% | 66.67% | +0.00 |
| away_favorite | 558 | 322 | 294 | -28 | 57.71% | 52.69% | -5.02 |
| unknown | 9 | 4 | 4 | 0 | 44.44% | 44.44% | +0.00 |

## Confidence Tiers (Old Baseline Buckets)
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.65+ | 7 | 4 | 4 | 0 | 57.14% | 57.14% | +0.00 |
| 0.60-0.65 | 33 | 15 | 15 | 0 | 45.45% | 45.45% | +0.00 |
| 0.56-0.60 | 122 | 60 | 60 | 0 | 49.18% | 49.18% | +0.00 |
| 0.53-0.56 | 147 | 88 | 86 | -2 | 59.86% | 58.50% | -1.36 |
| <=0.53 | 279 | 173 | 147 | -26 | 62.01% | 52.69% | -9.32 |

## Confidence Tiers (New Baseline Buckets)
| segment | rows | old_hits | new_hits | delta_hits | old_acc | new_acc | delta_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.65+ | 7 | 4 | 4 | 0 | 57.14% | 57.14% | +0.00 |
| 0.60-0.65 | 37 | 15 | 15 | 0 | 40.54% | 40.54% | +0.00 |
| 0.56-0.60 | 111 | 55 | 55 | 0 | 49.55% | 49.55% | +0.00 |
| 0.53-0.56 | 127 | 67 | 68 | 1 | 52.76% | 53.54% | +0.79 |
| <=0.53 | 306 | 199 | 170 | -29 | 65.03% | 55.56% | -9.48 |

CSV: src\data\mlb\tmp\full_game_current_vs_champion_segments.csv
JSON: src\data\mlb\tmp\full_game_current_vs_champion_summary.json