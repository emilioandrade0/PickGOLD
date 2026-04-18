# Iteracion de Mejora - Busqueda Fina Full Game

- Fecha y hora: 2026-04-14 23:59:26
- Mercado: full_game
- Objetivo: superar accuracy 0.5782312925 sin regresion

## Acciones ejecutadas
1. Baseline oficial congelado en:
   - src/sports/mlb/documentacion_mejoras/baseline_oficial_full_game.json
2. Barrido fino alrededor de la grilla ganadora con 7 escenarios.
3. Limpieza de artefactos temporales en src/data/mlb/tmp.

## Resultado de busqueda fina
Archivo consolidado:
- src/data/mlb/tmp/full_game_accuracy_fine_search_2026-04-14.csv

Top resultados por accuracy:
- fine_grid_c: 0.578231292517007
- fine_brier_006: 0.578231292517007
- fine_brier_010: 0.578231292517007
- baseline_locked: 0.578231292517007

Conclusión:
- No se encontró mejora por encima del baseline oficial.
- Se mantiene baseline oficial sin cambios de configuración.

## Decision tecnica
- Mantener como referencia oficial:
  - XGB grid: [0.00, 0.20, 0.35, 0.50, 0.65, 0.80, 1.00]
  - Brier weight: 0.08
  - Prob shift: -0.02..0.02 step 0.01
  - Calibrador: global_lr

## Limpieza de tmp
Se eliminaron logs y CSV intermedios de experimentación.
Se conservaron solo:
- src/data/mlb/tmp/full_game_accuracy_sweep_final.csv
- src/data/mlb/tmp/full_game_accuracy_fine_search_2026-04-14.csv
