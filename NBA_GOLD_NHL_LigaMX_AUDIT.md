# Auditoría NHL + Liga MX

## NHL

### Hallazgos
- El dataset `nhl_advanced_history.csv` ya viene limpio en este zip: `completed = 1` en 1090/1090 juegos.
- `model_ready_features_nhl.csv` ya no trae la fuga grave anterior de juegos futuros 0-0.
- El pipeline de features NHL ya usa `.shift(1)` en rolling y H2H incremental. Esa parte está bastante mejor que la versión previa.
- Los pendientes reales están en la capa de evaluación/live:
  - `historical_predictions_nhl.py` solo evaluaba mercados binarios y entrenaba por fila, no por fecha.
  - `predict_today_nhl.py` estaba forzando `threshold = 0.5` en live para mercados binarios, ignorando el threshold guardado en metadata.
  - `train_models_nhl.py` importaba `train_test_split` pero realmente conviene seguir un split temporal explícito.

### Correcciones entregadas
- `src/predict_today_nhl.py`
  - usa el threshold de metadata de forma consistente
  - alinea la construcción de features con la lista real de columnas del modelo
- `src/historical_predictions_nhl.py`
  - reescrito a walk-forward por fecha
  - incluye `full_game`, `spread_2_5` y `home_over_2_5`
  - entrena con datos estrictamente anteriores a cada fecha
- `src/train_models_nhl.py`
  - ajustado para usar split temporal explícito en vez de depender del bloque anterior con `train_test_split`

## Liga MX

### Hallazgos
- `liga_mx_advanced_history.csv` también está limpio en este zip: 437/437 juegos completados.
- `feature_engineering_liga_mx.py` no muestra la fuga obvia que sí tenía la vieja versión de NHL: los rolling usan `shift(1)`.
- El problema principal está en evaluación y en live schedule:
  - `historical_predictions_liga_mx.py` estaba evaluando histórico con modelos ya entrenados, no con validación walk-forward.
  - `data_ingest_liga_mx.py` guardaba en `liga_mx_upcoming_schedule.csv` cualquier juego del día, incluso si ya estaba terminado.

### Correcciones entregadas
- `src/data_ingest_liga_mx.py`
  - ahora filtra `status_completed == 0` para que el upcoming del día no mezcle juegos ya finalizados
- `src/historical_predictions_liga_mx.py`
  - reescrito a walk-forward por fecha
  - genera métricas históricas prospectivas para `full_game`, `over_25` y `btts`

## Prioridad recomendada
1. Reemplazar estos scripts en tu proyecto.
2. Regenerar Liga MX upcoming.
3. Reentrenar NHL si quieres que el threshold/metadata quede totalmente alineado con el nuevo flujo.
4. Regenerar históricos NHL y Liga MX con los scripts walk-forward nuevos.
5. Ya con eso, revisar UI y picks live.

## Orden sugerido de ejecución

### NHL
1. `python src/feature_engineering_nhl.py`
2. `python src/train_models_nhl.py`
3. `python src/predict_today_nhl.py`
4. `python src/historical_predictions_nhl.py`

### Liga MX
1. `python src/data_ingest_liga_mx.py`
2. `python src/feature_engineering_liga_mx.py`
3. `python src/train_models_liga_mx.py` o el trainer que estés usando como principal
4. `python src/predict_today_liga_mx.py`
5. `python src/historical_predictions_liga_mx.py`
