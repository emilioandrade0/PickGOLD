# Liga MX Full_Game Accuracy Improvement Report

Generated: 2026-04-05 18:22:40

## Summary

| Metric | Value |
|--------|-------|
| Baseline Accuracy | 50.81% |
| Improved Accuracy (v3) | 54.27% |
| Improvement | +3.46% (+6.80%) |
| Target Accuracy | 67.00% |
| Gap Remaining | 12.73% |

## Version Comparison

| Version | Accuracy | CV Folds | Features | Approach |
|---------|----------|----------|----------|----------|
| v1 Baseline | 50.81% | N/A | 127 | Two-stage baseline |
| v2 CV Tuned | 52.12% | 6 | 127 | Improved hyperparams |
| v3 Enhanced | 54.27% | 8 | 129 | Baseline + V3 draws |
| v4 CatBoost | 53.04% | 10 | 129 | CatBoost + XGB |

## Selected Version: v3 Enhanced

**Accuracy: 54.27%** (+6.80% vs baseline)

### Features Added
- draw_equilibrium_index
- match_parity_elo
- match_parity_goal_diff
- draw_pressure_avg
- draw_pressure_surface_avg
- xg_balance_index
- diff_locality_proxy

### Impact on Other Markets
✓ NO IMPACT - Models saved only to models_selective/full_game/
✓ Baseline models remain in models/ directory
✓ over_25: 53.51% (unchanged)
✓ btts: 54.05% (unchanged)

## Deployment

1. Models already saved to: data/liga_mx/models_selective/full_game/
2. Update selective plan: data/liga_mx/reports/liga_mx_selective_upgrade_plan.csv
3. Run: predict_today_liga_mx.py (uses resolve_market_model_dir)
4. Predictions will automatically use new v3 models for full_game only

## Gap Analysis: Why Not 67%?

Target: 67.00% (-12.73% gap)

### Challenges
- Draw prediction is 3x harder than win/loss (multiclass imbalance)
- Only 23.1% of games are draws (minority class)
- Dataset size: 445 games (limited for deep learning)

### To Reach 67%
- Class weighting or separate draw model
- Additional domain features (injury data, form, motivation)
- Deep learning with temporal patterns
- Larger dataset (1000+ games)
