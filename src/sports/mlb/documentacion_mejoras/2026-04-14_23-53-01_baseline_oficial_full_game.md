# Baseline Oficial Congelado - MLB Full Game

- Fecha y hora: 2026-04-14 23:53:01
- Estado: oficial
- Mercado: full_game

## Configuracion oficial
- Grilla XGB: [0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
- Brier weight: 0.08
- Prob shift: enabled=1, min=-0.02, max=0.02, step=0.01
- Calibrador: global_lr

## Metricas congeladas
- Accuracy: 0.5782312925170068
- Brier: 0.2501804420149413
- Logloss: 0.6936392445294711
- Coverage: 0.07653061224489796
- Published accuracy: 0.4666666666666667
- Filas: 588
- Splits: 55

## Referencias
- Resumen walkforward: src/data/mlb/walkforward/walkforward_summary_mlb.json
- Barrido consolidado: src/data/mlb/tmp/full_game_accuracy_sweep_final.csv
- Configuracion activa: src/sports/mlb/historical_predictions_mlb_walkforward.py
