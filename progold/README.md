# Progol Contrarian Lab

Aplicacion web local para analizar partidos de Progol con enfoque contrarian-inteligente.

## Enfoque

Esta app es una herramienta de apoyo analitico basada en lectura de distribucion publica de pronosticos. No predice ganadores con certeza.

## Funcionalidades principales

- Captura manual directa en tabla editable de 14 partidos (formato boleto).
- Validacion de porcentajes y alerta cuando no suman cerca de 100.
- Clasificacion automatica de partido:
  - empate_vivo
  - empate_ignorado_por_masa
  - local_sobrepopular
  - visita_sobrecomprada_moderada
  - visita_viva
  - favorito_estable / favorito_sobrejugado
  - partido_caotico
- Motor heuristico configurable (sin machine learning).
- Motor refactorizado por modulos (`pattern_detector`, `scoring_engine`, `recommendation_engine`, `explainability`, `calibration`).
- Scoring enriquecido por resultado y contexto: `score_1`, `score_X`, `score_2`, `score_riesgo`, `score_contrarian`, `score_estabilidad`.
- Tabla principal con columnas calculadas y recomendaciones.
- Recomendacion principal + doble oportunidad + nivel de confianza.
- Explicabilidad por partido.
- Modo debug por partido con patrones activados, ajustes de score y traza de decision.
- Vista boleto inspirada en quiniela Progol/Revancha.
- Vista analisis para lectura tecnica y detalle por partido.
- Importacion automatica desde captura (OCR) para leer equipos y porcentajes.
- Verificacion de picks contra resultados ESPN (acierto/fallo/pendiente) con coloreado visual en vista boleto.
- Panel de resumen de jornada.
- Calibracion historica lista para evaluar accuracy directo/doble y tasas por patron.
- Exportacion a CSV y Excel.
- Guardado/carga de sesion en JSON.
- Datos dummy para validar rapidamente la logica.

## Estructura

```
PRO-TIP/
├── app.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── default_rules.json
├── models/
│   ├── __init__.py
│   └── match_models.py
├── logic/
│   ├── __init__.py
│   ├── analyzer.py
│   ├── calibration.py
│   ├── classifier.py
│   ├── explainability.py
│   ├── pattern_detector.py
│   ├── recommendation_engine.py
│   ├── recommender.py
│   ├── scoring_engine.py
│   ├── scoring.py
│   └── validator.py
├── tests/
│   └── test_heuristic_engine.py
├── ui/
│   ├── __init__.py
│   └── components.py
└── utils/
    ├── __init__.py
    ├── dummy_data.py
  ├── exporter.py
  └── progol_ocr.py
```

## Requisitos

- Python 3.10+
- pip

## Como ejecutar

### Opcion rapida (Windows, recomendado)

Ejecuta el archivo `iniciar_progol.bat` con doble clic o desde terminal:

```bat
iniciar_progol.bat
```

Este launcher:

- crea `.venv` si no existe
- instala dependencias de `requirements.txt`
- inicia Streamlit usando el Python del entorno virtual

### Opcion manual

1. Crear entorno virtual (opcional pero recomendado):

```bash
python -m venv .venv
```

2. Activar entorno:

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

4. Ejecutar app:

```bash
streamlit run app.py
```

5. Abrir en navegador la URL local que muestra Streamlit.

## Flujo visual actual

1. En la parte superior define el nombre de jornada/concurso.
2. Elige modo:
  - `Vista boleto`: formato principal tipo quiniela con filas numeradas.
  - `Vista analisis`: vista tecnica con filtros y detalle explicable.
3. Captura directamente local, empate y visita en la tabla editable de 14 filas.
4. Opcional: usa `Importar desde captura (OCR)` para cargar automaticamente partidos desde imagen.
5. El sistema coloca los partidos detectados en el boleto desde la posicion elegida.
6. Ajusta manualmente cualquier fila que OCR no haya reconocido al 100%.
4. El sistema calcula automaticamente por fila:
  - recomendacion principal
  - doble oportunidad
  - confianza
  - alerta
7. Exporta resultados desde los botones de exportacion.

## Importar captura OCR

En el bloque `Importar desde captura (OCR)`:

1. Sube una o varias capturas (`png`, `jpg`, `jpeg`, `webp`).
2. Elige si leer `Progol (tabla superior)` o `Revancha (tabla inferior)`.
3. Define desde que partido cargar en el boleto.
4. Presiona `Leer captura(s) y cargar boleto`.

Opcional (en el mismo bloque):

- activa `Verificar picks contra ESPN`
- define `Fecha de partidos`
- selecciona las `Ligas ESPN` a consultar

Cuando ESPN tenga marcador final, el pick se marca como `Acierto` o `Fallo` y se colorea automaticamente en la vista boleto.

Recomendaciones para mejor lectura:

- usar capturas lo mas rectas posible
- evitar compresion excesiva o texto borroso
- si una fila sale incompleta, corregirla directamente en la tabla editable

Nota tecnica OCR:

- usa `rapidocr` + `onnxruntime`
- en la primera ejecucion OCR puede descargar modelos una sola vez

## Reglas heuristicas configurables

Los umbrales estan centralizados en:

- `config/default_rules.json`

Tambien puedes editarlos desde la barra lateral de la app y guardarlos nuevamente al archivo.

## Calibracion con historico

El modulo `logic/calibration.py` incluye `evaluate_historical_cases`, que recibe casos con:

- `pct_local`
- `pct_empate`
- `pct_visita`
- `resultado_real` (`1`, `X` o `2`)

Y devuelve:

- `accuracy_pick_directo`
- `accuracy_doble_oportunidad`
- `pattern_activation_matrix`
- `pattern_hit_rate`
- `pattern_combo_hit_rate`

## Escalabilidad futura

La arquitectura modular deja listo el camino para agregar:

- cuotas reales
- resultados historicos
- calibracion de heuristicas
- tracking de aciertos
- motor estadistico mas avanzado

## Nota de uso responsable

La salida de esta aplicacion es analitica y orientativa. No se debe interpretar como garantia de resultado.
