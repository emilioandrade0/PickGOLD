# Mejora MLB Full Game - EV/ROI Threshold Objective

- Fecha: 2026-04-15
- Mercado: full_game
- Objetivo: optimizar EV/ROI en picks publicados sin perder accuracy global walkforward

## Cambios implementados
- Se agrego objetivo de umbral por ROI en walkforward:
  - `NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE=roi`
- Se agrego control por edge EV minimo:
  - `NBA_MLB_FULL_GAME_ROI_MIN_EDGE`
- Se agrego control de calidad minima para calibracion ROI:
  - `NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY`
  - `NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS`
- Se agrego enriquecimiento de odds reales para calibracion/test usando:
  - `src/data/mlb/raw/mlb_advanced_history.csv`
- Se agregaron columnas nuevas al detalle walkforward para auditoria economica:
  - `pick_decimal_odds`
  - `pick_implied_prob`
  - `pick_ev_edge`
  - `priced_pick`
  - `return_per_unit`
- Se agregaron metricas ROI al summary:
  - `published_roi_per_bet`
  - `published_total_return_units`
  - `published_priced_picks`
  - `published_priced_coverage`
  - `published_mean_ev_edge`
  - `roi_threshold_splits`
  - `roi_threshold_split_rate`

## Resultado validado (roi_no_edge)
Configuracion principal usada:
- `NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE=roi`
- `NBA_MLB_FULL_GAME_ROI_MIN_EDGE=0.00`
- `NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY=0.45`
- `NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS=8`

Metricas full_game:
- accuracy: `0.5782312925170068`
- published_accuracy: `0.5138888888888888`
- published_coverage: `0.12244897959183673`
- published_roi_per_bet: `0.036977510211509024`
- published_total_return_units: `1.368167877825834`
- published_priced_picks: `37`
- published_priced_coverage: `0.06292517006802721`
- roi_threshold_splits: `14`
- roi_threshold_split_rate: `0.2545454545454545`

## Comparativa de escenarios
Archivo comparativo:
- `src/data/mlb/tmp/full_game_roi_objective_compare_2026-04-15.csv`

Lectura rapida:
- `roi_no_edge` mejoro ROI publicado vs referencia accuracy_cov, manteniendo accuracy global.
- `roi_edge_0015` fue demasiado estricto y degrado ROI/cobertura.

## Promocion a perfil baseline
Se promovio la configuracion ROI no-regresiva a:
- `tools/mlb_apply_profile.bat`

Y se agrego limpieza de nuevas variables en:
- `tools/mlb_clear_overrides.cmd`
