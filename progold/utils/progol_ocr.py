from __future__ import annotations

import io
import re
import statistics
import unicodedata
from dataclasses import dataclass
from datetime import date as dt_date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

try:
    from rapidocr import RapidOCR
except Exception:  # pragma: no cover - optional runtime dependency
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:  # pragma: no cover - optional runtime dependency
        RapidOCR = None


_OCR_ENGINE: Any = None

_PERCENT_RE = re.compile(r"(\d{1,2})[\.,](\d{2})")
_INT_RE = re.compile(r"^\d{1,2}$")
_DATE_NUMERIC_RE = re.compile(r"\b(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{2,4})\b")
_DATE_RANGE_SAME_MONTH_RE = re.compile(
    r"\b(?:DEL|DE)\s*(\d{1,2})\s*(?:AL|A)\s*(\d{1,2})\s*(?:DE\s+)?([A-Z\u00D1\u00DC]{3,12})\s*(\d{2,4})\b"
)
_DATE_RANGE_TWO_MONTHS_RE = re.compile(
    r"\b(?:DEL|DE)\s*(\d{1,2})\s*(?:DE\s+)?([A-Z\u00D1\u00DC]{3,12})\s*(?:AL|A)\s*(\d{1,2})\s*(?:DE\s+)?([A-Z\u00D1\u00DC]{3,12})\s*(\d{2,4})\b"
)
_DATE_TEXT_RE = re.compile(
    r"\b(\d{1,2})\s*(?:DE\s+)?([A-Z\u00D1\u00DC]{3,12})\s*(?:DE\s+)?(\d{2,4})\b"
)

_MONTH_MAP: Dict[str, int] = {
    "ENE": 1,
    "ENERO": 1,
    "FEB": 2,
    "FEBRERO": 2,
    "MAR": 3,
    "MARZO": 3,
    "ABR": 4,
    "ABRIL": 4,
    "MAY": 5,
    "MAYO": 5,
    "JUN": 6,
    "JUNIO": 6,
    "JUL": 7,
    "JULIO": 7,
    "AGO": 8,
    "AGOSTO": 8,
    "SEP": 9,
    "SEPT": 9,
    "SEPTIEMBRE": 9,
    "OCT": 10,
    "OCTUBRE": 10,
    "NOV": 11,
    "NOVIEMBRE": 11,
    "DIC": 12,
    "DICIEMBRE": 12,
}

_STOPWORDS = {
    "LOCAL",
    "EMPATE",
    "VISITA",
    "PORCENTAJES",
    "CONCURSO",
    "QUINIELA",
    "SENCILLA",
    "PROGOL",
    "REVANCHA",
    "A",
    "LA",
    "VENTA",
    "DEL",
    "DE",
    "HASTA",
    "LAS",
    "JUEGOS",
}


@dataclass
class OcrToken:
    text: str
    score: float
    cx: float
    cy: float
    x1: float
    y1: float
    x2: float
    y2: float


def ocr_runtime_status() -> Tuple[bool, str]:
    if RapidOCR is None:
        return (
            False,
            "OCR no disponible: instala rapidocr y onnxruntime para habilitar lectura automatica de capturas.",
        )
    return (
        True,
        "OCR listo: puedes subir capturas de Progol/Revancha para autollenado del boleto.",
    )


def _get_ocr_engine() -> Any:
    global _OCR_ENGINE
    if RapidOCR is None:
        return None
    if _OCR_ENGINE is None:
        _OCR_ENGINE = RapidOCR()
    return _OCR_ENGINE


def _normalize_text(text: str) -> str:
    cleaned = text.upper()
    cleaned = cleaned.replace("|", "I")
    cleaned = cleaned.replace("_", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _normalize_text_for_date(text: str) -> str:
    lowered = unicodedata.normalize("NFD", text.upper())
    lowered = "".join(ch for ch in lowered if unicodedata.category(ch) != "Mn")
    lowered = lowered.replace("|", "I")
    lowered = lowered.replace("_", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def _resolve_year(value: str) -> int:
    year = int(value)
    if year < 100:
        return 2000 + year
    return year


def _safe_date(year: int, month: int, day: int) -> Optional[dt_date]:
    try:
        resolved = dt_date(year, month, day)
    except ValueError:
        return None

    current_year = dt_date.today().year
    if resolved.year < 2020 or resolved.year > current_year + 1:
        return None
    return resolved


def _extract_capture_date(tokens: Sequence[OcrToken]) -> Tuple[Optional[dt_date], Optional[str]]:
    if not tokens:
        return None, None

    ordered = sorted(tokens, key=lambda item: (item.cy, item.cx))
    full_text = _normalize_text_for_date(" ".join(token.text for token in ordered))

    candidates: List[Tuple[int, dt_date, str]] = []

    for match in _DATE_RANGE_TWO_MONTHS_RE.finditer(full_text):
        _, _start_month_raw, end_day_raw, end_month_raw, year_raw = match.groups()
        end_month = _MONTH_MAP.get(end_month_raw)
        if end_month is None:
            continue
        candidate = _safe_date(_resolve_year(year_raw), end_month, int(end_day_raw))
        if candidate is not None:
            candidates.append((4, candidate, "rango con dos meses"))

    for match in _DATE_RANGE_SAME_MONTH_RE.finditer(full_text):
        _, end_day_raw, month_raw, year_raw = match.groups()
        month = _MONTH_MAP.get(month_raw)
        if month is None:
            continue
        candidate = _safe_date(_resolve_year(year_raw), month, int(end_day_raw))
        if candidate is not None:
            candidates.append((4, candidate, "rango semanal"))

    for match in _DATE_NUMERIC_RE.finditer(full_text):
        day_raw, month_raw, year_raw = match.groups()
        candidate = _safe_date(_resolve_year(year_raw), int(month_raw), int(day_raw))
        if candidate is not None:
            candidates.append((3, candidate, "formato numerico"))

    for match in _DATE_TEXT_RE.finditer(full_text):
        day_raw, month_raw, year_raw = match.groups()
        month = _MONTH_MAP.get(month_raw)
        if month is None:
            continue
        candidate = _safe_date(_resolve_year(year_raw), month, int(day_raw))
        if candidate is not None:
            candidates.append((2, candidate, "formato textual"))

    if not candidates:
        return None, None

    today = dt_date.today()
    selected = sorted(candidates, key=lambda item: (-item[0], abs((today - item[1]).days)))[0]
    return selected[1], selected[2]


def _box_bounds(box: Any) -> Tuple[float, float, float, float]:
    if isinstance(box, np.ndarray):
        box = box.tolist()

    points: List[Tuple[float, float]] = []
    if isinstance(box, Sequence):
        for point in box:
            if isinstance(point, Sequence) and len(point) >= 2:
                try:
                    points.append((float(point[0]), float(point[1])))
                except (TypeError, ValueError):
                    continue

    if not points:
        return 0.0, 0.0, 0.0, 0.0

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _tokens_from_ocr_result(raw_result: Any) -> List[OcrToken]:
    tokens: List[OcrToken] = []
    if raw_result is None:
        return tokens

    candidate_items: List[Tuple[Any, str, float]] = []

    # RapidOCR returns a RapidOCROutput object with boxes/txts/scores fields.
    if hasattr(raw_result, "boxes") and hasattr(raw_result, "txts"):
        boxes = getattr(raw_result, "boxes", None)
        txts = getattr(raw_result, "txts", None)
        scores = getattr(raw_result, "scores", None)

        if boxes is None or txts is None:
            return tokens

        if isinstance(boxes, np.ndarray):
            boxes = boxes.tolist()
        if isinstance(txts, tuple):
            txts = list(txts)
        elif isinstance(txts, np.ndarray):
            txts = txts.tolist()

        if isinstance(scores, tuple):
            scores = list(scores)
        elif isinstance(scores, np.ndarray):
            scores = scores.tolist()

        if not isinstance(boxes, Sequence) or not isinstance(txts, Sequence):
            return tokens

        score_list = scores if isinstance(scores, Sequence) else []
        for idx, (box, text_value) in enumerate(zip(boxes, txts)):
            score = 1.0
            if idx < len(score_list):
                try:
                    score = float(score_list[idx])
                except (TypeError, ValueError):
                    score = 1.0
            candidate_items.append((box, str(text_value), score))

    else:
        if isinstance(raw_result, tuple):
            raw_result = raw_result[0] if raw_result else []

        if not isinstance(raw_result, Sequence):
            return tokens

        for item in raw_result:
            if not isinstance(item, Sequence) or len(item) < 2:
                continue

            box = item[0]
            text_value = item[1]
            score = 1.0
            if len(item) >= 3:
                try:
                    score = float(item[2])
                except (TypeError, ValueError):
                    score = 1.0
            candidate_items.append((box, str(text_value), score))

    for box, raw_text, score in candidate_items:
        text = _normalize_text(raw_text)
        if not text:
            continue

        x1, y1, x2, y2 = _box_bounds(box)
        if x2 <= x1 or y2 <= y1:
            continue

        tokens.append(
            OcrToken(
                text=text,
                score=score,
                cx=(x1 + x2) / 2,
                cy=(y1 + y2) / 2,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )

    return tokens


def _image_variants(image_bytes: bytes) -> List[np.ndarray]:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    variants: List[np.ndarray] = []

    rgb = np.array(image)
    variants.append(rgb)

    gray = ImageOps.autocontrast(image.convert("L"))
    variants.append(np.array(gray))

    sharp = gray.filter(ImageFilter.SHARPEN)
    variants.append(np.array(sharp))

    bw = sharp.point(lambda px: 255 if px >= 165 else 0, mode="L")
    variants.append(np.array(bw))

    return variants


def _extract_percent_values(text: str) -> List[float]:
    compact = text.replace(" ", "")
    values: List[float] = []
    for whole, decimals in _PERCENT_RE.findall(compact):
        try:
            value = float(f"{int(whole)}.{decimals}")
        except ValueError:
            continue
        if 0.0 <= value <= 100.0:
            values.append(round(value, 2))

    dedup: List[float] = []
    for value in values:
        if value not in dedup:
            dedup.append(value)
    return dedup


def _strip_percent_text(text: str) -> str:
    without_pct = _PERCENT_RE.sub(" ", text)
    without_numbers = re.sub(r"\b\d{1,2}\b", " ", without_pct)
    cleaned = re.sub(r"[^A-Z0-9 .&'/-]", " ", without_numbers)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _is_valid_team_fragment(text: str) -> bool:
    if not text:
        return False
    if text in _STOPWORDS:
        return False
    if _INT_RE.fullmatch(text):
        return False
    if len(text) < 2:
        return False
    return bool(re.search(r"[A-Z]", text))


def _join_name_fragments(fragments: List[Tuple[float, str]]) -> str:
    if not fragments:
        return ""

    fragments = sorted(fragments, key=lambda pair: pair[0])
    parts: List[str] = []
    seen = set()
    for _, fragment in fragments:
        token = fragment.strip()
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        parts.append(token)

    return " ".join(parts).strip()


def _cluster_y_values(values: List[float], threshold: float = 17.0) -> List[float]:
    if not values:
        return []

    sorted_values = sorted(values)
    clusters: List[List[float]] = [[sorted_values[0]]]

    for value in sorted_values[1:]:
        if abs(value - clusters[-1][-1]) <= threshold:
            clusters[-1].append(value)
        else:
            clusters.append([value])

    return [float(statistics.mean(cluster)) for cluster in clusters]


def _choose_table_headers(
    tokens: List[OcrToken],
    section: str,
) -> Tuple[Tuple[float, float, float, float, float], List[str]]:
    notes: List[str] = []

    local_tokens = [t for t in tokens if "LOCAL" in t.text]
    empate_tokens = [t for t in tokens if "EMPATE" in t.text]
    visita_tokens = [t for t in tokens if "VISITA" in t.text]

    triples: List[Tuple[float, float, OcrToken, OcrToken, OcrToken]] = []

    for local in local_tokens:
        for empate in empate_tokens:
            for visita in visita_tokens:
                if not (local.cx < empate.cx < visita.cx):
                    continue
                y_spread = max(abs(local.cy - empate.cy), abs(empate.cy - visita.cy), abs(local.cy - visita.cy))
                if y_spread > 90:
                    continue
                header_y = (local.cy + empate.cy + visita.cy) / 3
                triples.append((header_y, y_spread, local, empate, visita))

    if not triples:
        if not tokens:
            return (0.0, 0.0, 0.0, 0.0, 0.0), ["No se detecto texto util en la captura."]

        max_x = max(t.x2 for t in tokens)
        max_y = max(t.y2 for t in tokens)
        notes.append("No se detectaron encabezados completos LOCAL/EMPATE/VISITA; se uso region aproximada.")
        return (max_x * 0.22, max_x * 0.50, max_x * 0.82, max_y * 0.16, max_y * 0.95), notes

    triples = sorted(triples, key=lambda item: item[0])
    selected = triples[-1] if section == "revancha" else triples[0]
    selected_y = selected[0]

    next_headers = [entry for entry in triples if entry[0] > selected_y + 120]
    max_y = max(t.y2 for t in tokens)

    if next_headers:
        y_end = next_headers[0][0] - 15
    else:
        y_end = max_y + 10

    local_x = selected[2].cx
    empate_x = selected[3].cx
    visita_x = selected[4].cx
    y_start = selected_y + 18

    return (local_x, empate_x, visita_x, y_start, y_end), notes


def _candidate_row_anchors(
    region_tokens: List[OcrToken],
    local_x: float,
    max_matches: int,
) -> List[Tuple[int, float]]:
    by_number: Dict[int, OcrToken] = {}

    for token in region_tokens:
        if not _INT_RE.fullmatch(token.text):
            continue

        number = int(token.text)
        if number < 1 or number > max_matches:
            continue
        if token.cx > local_x - 16:
            continue

        current = by_number.get(number)
        if current is None:
            by_number[number] = token
            continue

        if token.cx < current.cx or token.score > current.score:
            by_number[number] = token

    anchors = [(number, token.cy) for number, token in sorted(by_number.items(), key=lambda item: item[0])]

    if len(anchors) >= 6:
        return anchors

    percentage_y = []
    for token in region_tokens:
        if _extract_percent_values(token.text):
            percentage_y.append(token.cy)

    inferred_rows = _cluster_y_values(percentage_y)
    return [(idx + 1, y_value) for idx, y_value in enumerate(inferred_rows[:max_matches])]


def _pick_percentage(bucket: List[Tuple[float, float]]) -> float | None:
    if not bucket:
        return None
    sorted_bucket = sorted(bucket, key=lambda pair: pair[1])
    return sorted_bucket[0][0]


def _row_score(row: Dict[str, Any]) -> int:
    score = 0
    if row.get("local"):
        score += 2
    if row.get("visitante"):
        score += 2
    if row.get("pct_local") is not None:
        score += 1
    if row.get("pct_empate") is not None:
        score += 1
    if row.get("pct_visita") is not None:
        score += 1
    return score


def _extract_rows_from_tokens(
    tokens: List[OcrToken],
    section: str,
    max_matches: int,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not tokens:
        return [], ["OCR sin resultados en esta captura."]

    (local_x, empate_x, visita_x, y_start, y_end), notes = _choose_table_headers(tokens, section)
    mid_left = (local_x + empate_x) / 2
    mid_right = (empate_x + visita_x) / 2

    region_tokens = [token for token in tokens if y_start <= token.cy <= y_end]
    anchors = _candidate_row_anchors(region_tokens, local_x, max_matches)

    if not anchors:
        return [], notes + ["No se detectaron filas de partidos en la region de la tabla."]

    anchors = sorted(anchors, key=lambda pair: pair[1])

    if len(anchors) >= 2:
        gaps = [anchors[idx + 1][1] - anchors[idx][1] for idx in range(len(anchors) - 1) if anchors[idx + 1][1] > anchors[idx][1]]
        median_gap = statistics.median(gaps) if gaps else 36.0
    else:
        median_gap = 36.0

    row_tolerance = max(11.0, min(28.0, median_gap * 0.42))

    extracted: Dict[int, Dict[str, Any]] = {}
    for idx, (partido, anchor_y) in enumerate(anchors, start=1):
        if partido < 1 or partido > max_matches:
            partido = idx

        row_tokens = [token for token in region_tokens if abs(token.cy - anchor_y) <= row_tolerance]
        if not row_tokens:
            continue

        left_percentages: List[Tuple[float, float]] = []
        center_percentages: List[Tuple[float, float]] = []
        right_percentages: List[Tuple[float, float]] = []

        local_fragments: List[Tuple[float, str]] = []
        visitante_fragments: List[Tuple[float, str]] = []

        for token in sorted(row_tokens, key=lambda row_token: row_token.cx):
            for value in _extract_percent_values(token.text):
                if token.cx < mid_left:
                    left_percentages.append((value, abs(token.cx - local_x)))
                elif token.cx < mid_right:
                    center_percentages.append((value, abs(token.cx - empate_x)))
                else:
                    right_percentages.append((value, abs(token.cx - visita_x)))

            team_candidate = _strip_percent_text(token.text)
            if not _is_valid_team_fragment(team_candidate):
                continue

            if local_x <= token.cx < mid_left:
                local_fragments.append((token.cx, team_candidate))
            elif token.cx >= mid_right:
                visitante_fragments.append((token.cx, team_candidate))

        row_payload = {
            "partido": partido,
            "local": _join_name_fragments(local_fragments),
            "pct_local": _pick_percentage(left_percentages),
            "pct_empate": _pick_percentage(center_percentages),
            "pct_visita": _pick_percentage(right_percentages),
            "visitante": _join_name_fragments(visitante_fragments),
        }

        if not any(
            [
                row_payload["local"],
                row_payload["visitante"],
                row_payload["pct_local"] is not None,
                row_payload["pct_empate"] is not None,
                row_payload["pct_visita"] is not None,
            ]
        ):
            continue

        existing = extracted.get(partido)
        if existing is None or _row_score(row_payload) > _row_score(existing):
            extracted[partido] = row_payload

    rows = [extracted[number] for number in sorted(extracted) if 1 <= number <= max_matches]
    return rows, notes


def _variant_quality_score(rows: List[Dict[str, Any]]) -> int:
    complete = 0
    partial = 0
    for row in rows:
        has_names = bool(row.get("local")) and bool(row.get("visitante"))
        has_percents = all(
            row.get(key) is not None for key in ("pct_local", "pct_empate", "pct_visita")
        )
        if has_names and has_percents:
            complete += 1
        elif has_names or has_percents:
            partial += 1
    return (complete * 10) + partial


def extract_matches_with_date_from_capture(
    image_bytes: bytes,
    section: str = "progol",
    max_matches: int = 14,
) -> Tuple[List[Dict[str, Any]], List[str], Optional[dt_date]]:
    available, status_msg = ocr_runtime_status()
    if not available:
        return [], [status_msg], None

    engine = _get_ocr_engine()
    if engine is None:
        return [], ["No se pudo inicializar el motor OCR."], None

    variants = _image_variants(image_bytes)

    best_rows: List[Dict[str, Any]] = []
    best_notes: List[str] = []
    best_tokens: List[OcrToken] = []
    best_score = -1

    for variant_index, variant in enumerate(variants, start=1):
        try:
            raw_result = engine(variant)
        except Exception as exc:  # pragma: no cover - runtime dependency variability
            return [], [f"Error al ejecutar OCR: {exc}"], None

        if isinstance(raw_result, tuple):
            raw_result = raw_result[0]

        tokens = _tokens_from_ocr_result(raw_result)
        rows, notes = _extract_rows_from_tokens(tokens, section=section, max_matches=max_matches)
        score = _variant_quality_score(rows)

        if score > best_score:
            best_score = score
            best_rows = rows
            best_notes = notes + [f"Variante OCR elegida: {variant_index}."]
            best_tokens = tokens

    if not best_rows:
        return [], best_notes + ["No fue posible extraer filas validas desde la captura."], None

    detected_date, detection_source = _extract_capture_date(best_tokens)
    if detected_date is not None:
        source_label = detection_source or "OCR"
        best_notes.append(
            f"Fecha detectada ({source_label}): {detected_date.strftime('%d/%m/%Y')}"
        )

    if len(best_rows) < max_matches:
        best_notes.append(
            f"Se detectaron {len(best_rows)} filas; revisa manualmente las faltantes en la tabla editable."
        )

    return best_rows[:max_matches], best_notes, detected_date


def extract_matches_from_capture(
    image_bytes: bytes,
    section: str = "progol",
    max_matches: int = 14,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows, notes, _ = extract_matches_with_date_from_capture(
        image_bytes=image_bytes,
        section=section,
        max_matches=max_matches,
    )
    return rows, notes
