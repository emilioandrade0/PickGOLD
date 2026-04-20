from __future__ import annotations

import io
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from config import HeuristicConfig
from models import MatchInput

from .analyzer import analyze_match
from .calibration import evaluate_historical_cases

_DIRECT = {"1", "X", "2"}
_DOUBLE = {"1X", "X2", "12"}


@dataclass
class SimilarExample:
    local: str
    visitante: str
    pct_local: float
    pct_empate: float
    pct_visita: float
    resultado_real: str
    distance: float
    patterns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "local": self.local,
            "visitante": self.visitante,
            "pct_local": self.pct_local,
            "pct_empate": self.pct_empate,
            "pct_visita": self.pct_visita,
            "resultado_real": self.resultado_real,
            "distance": round(self.distance, 4),
            "patterns": self.patterns,
        }


@dataclass
class HistoricalSignal:
    sample_size: int
    min_required: int
    distribution: Dict[str, float]
    recommended: str
    confidence_score: float
    confidence_label: str
    avg_distance: float
    examples: List[SimilarExample]
    strategy_note: str

    @property
    def enough_samples(self) -> bool:
        return self.sample_size >= self.min_required

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_size": self.sample_size,
            "min_required": self.min_required,
            "enough_samples": self.enough_samples,
            "distribution": {k: round(v, 4) for k, v in self.distribution.items()},
            "recommended": self.recommended,
            "confidence_score": round(self.confidence_score, 4),
            "confidence_label": self.confidence_label,
            "avg_distance": round(self.avg_distance, 4),
            "strategy_note": self.strategy_note,
            "examples": [item.to_dict() for item in self.examples],
        }


@dataclass
class HybridDecision:
    final_recommendation: str
    final_direct_symbol: str
    final_double: str
    final_confidence: str
    decision_note: str
    decision_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_recommendation": self.final_recommendation,
            "final_direct_symbol": self.final_direct_symbol,
            "final_double": self.final_double,
            "final_confidence": self.final_confidence,
            "decision_note": self.decision_note,
            "decision_mode": self.decision_mode,
        }


def _safe_json_list(value: str | None) -> List[str]:
    if not value:
        return []
    try:
        payload = json.loads(value)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


def _normalize_name(raw: str) -> str:
    normalized = "".join(ch for ch in str(raw).strip().lower() if ch.isalnum())
    return normalized


def _pick_column(columns: Sequence[str], aliases: Sequence[str]) -> str | None:
    normalized = {_normalize_name(col): col for col in columns}
    for alias in aliases:
        found = normalized.get(_normalize_name(alias))
        if found:
            return found
    return None


def _normalize_result(value: Any) -> str:
    token = str(value or "").strip().upper()
    if token in _DIRECT:
        return token
    return ""


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _gap_top2(local: float, empate: float, visita: float) -> float:
    ordered = sorted([local, empate, visita], reverse=True)
    return ordered[0] - ordered[1]


def _draw_structure(local: float, empate: float, visita: float) -> float:
    min_side = min(local, visita)
    return empate - min_side


def _jaccard_distance(a: Sequence[str], b: Sequence[str]) -> float:
    set_a = set(a)
    set_b = set(b)
    if not set_a and not set_b:
        return 0.0
    if not set_a or not set_b:
        return 1.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    if union <= 0:
        return 1.0
    return 1.0 - (intersection / union)


def _score_to_label(score: float, sample_size: int, min_required: int) -> str:
    if sample_size < min_required:
        return "baja"
    if score >= 0.74:
        return "alta"
    if score >= 0.48:
        return "media"
    return "baja"


def _adjust_confidence(base: str, direction: int) -> str:
    scale = ["baja", "media", "alta"]
    current = str(base or "media").lower()
    idx = scale.index(current) if current in scale else 1
    idx = max(0, min(len(scale) - 1, idx + direction))
    return scale[idx]


def _double_from_pair(first: str, second: str) -> str:
    pair = {first, second}
    if pair == {"1", "X"}:
        return "1X"
    if pair == {"X", "2"}:
        return "X2"
    if pair == {"1", "2"}:
        return "12"
    return "-"


class HistoricalPatternEngine:
    def __init__(self, db_path: Path, min_required: int = 8, max_distance: float = 0.33, top_k: int = 40) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.min_required = max(3, int(min_required))
        self.max_distance = max(0.08, float(max_distance))
        self.top_k = max(8, int(top_k))
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _session(self) -> Any:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_matches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concurso TEXT,
                    fecha TEXT,
                    local TEXT NOT NULL,
                    visitante TEXT NOT NULL,
                    pct_local REAL NOT NULL,
                    pct_empate REAL NOT NULL,
                    pct_visita REAL NOT NULL,
                    resultado_real TEXT NOT NULL,
                    patrones_json TEXT NOT NULL,
                    source TEXT NOT NULL DEFAULT 'manual',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def total_matches(self) -> int:
        with self._session() as conn:
            row = conn.execute("SELECT COUNT(*) AS total FROM historical_matches").fetchone()
            return int(row["total"] if row else 0)

    def import_csv_bytes(self, content: bytes, source: str = "csv_upload") -> Dict[str, int]:
        df = pd.read_csv(io.BytesIO(content))
        rows = self._normalize_dataframe(df)
        inserted = self.insert_rows(rows, source=source)
        return {
            "received_rows": int(len(df.index)),
            "valid_rows": int(len(rows)),
            "inserted_rows": int(inserted),
            "skipped_rows": int(max(0, len(df.index) - len(rows))),
        }

    def insert_rows(self, rows: Iterable[Dict[str, Any]], source: str = "manual") -> int:
        prepared: List[Tuple[Any, ...]] = []
        for raw in rows:
            result = _normalize_result(raw.get("resultado_real"))
            if result not in _DIRECT:
                continue

            p1 = _to_float(raw.get("pct_local"))
            px = _to_float(raw.get("pct_empate"))
            p2 = _to_float(raw.get("pct_visita"))
            if p1 is None or px is None or p2 is None:
                continue

            local = str(raw.get("local", "LOCAL") or "LOCAL").strip()
            visitante = str(raw.get("visitante", "VISITA") or "VISITA").strip()
            if not local:
                local = "LOCAL"
            if not visitante:
                visitante = "VISITA"

            patterns = raw.get("patrones_activados") or raw.get("patterns") or []
            if isinstance(patterns, str):
                patterns = [item.strip() for item in patterns.split("|") if item.strip()]
            patterns_json = json.dumps([str(item) for item in patterns], ensure_ascii=True)

            prepared.append(
                (
                    str(raw.get("concurso", "") or ""),
                    str(raw.get("fecha", "") or ""),
                    local,
                    visitante,
                    float(p1),
                    float(px),
                    float(p2),
                    result,
                    patterns_json,
                    str(source or "manual"),
                )
            )

        if not prepared:
            return 0

        with self._session() as conn:
            conn.executemany(
                """
                INSERT INTO historical_matches (
                    concurso, fecha, local, visitante, pct_local, pct_empate, pct_visita,
                    resultado_real, patrones_json, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                prepared,
            )
        return len(prepared)

    def insert_rows_dedup(self, rows: Iterable[Dict[str, Any]], source: str = "manual") -> Dict[str, int]:
        prepared: List[Tuple[Any, ...]] = []
        skipped_invalid = 0
        skipped_duplicate = 0

        with self._session() as conn:
            for raw in rows:
                result = _normalize_result(raw.get("resultado_real"))
                if result not in _DIRECT:
                    skipped_invalid += 1
                    continue

                p1 = _to_float(raw.get("pct_local"))
                px = _to_float(raw.get("pct_empate"))
                p2 = _to_float(raw.get("pct_visita"))
                if p1 is None or px is None or p2 is None:
                    skipped_invalid += 1
                    continue

                local = str(raw.get("local", "LOCAL") or "LOCAL").strip() or "LOCAL"
                visitante = str(raw.get("visitante", "VISITA") or "VISITA").strip() or "VISITA"
                fecha = str(raw.get("fecha", "") or "")
                concurso = str(raw.get("concurso", "") or "")

                duplicate_row = conn.execute(
                    """
                    SELECT 1
                    FROM historical_matches
                    WHERE local = ?
                      AND visitante = ?
                      AND pct_local = ?
                      AND pct_empate = ?
                      AND pct_visita = ?
                      AND resultado_real = ?
                      AND fecha = ?
                    LIMIT 1
                    """,
                    (local, visitante, float(p1), float(px), float(p2), result, fecha),
                ).fetchone()
                if duplicate_row:
                    skipped_duplicate += 1
                    continue

                patterns = raw.get("patrones_activados") or raw.get("patterns") or []
                if isinstance(patterns, str):
                    patterns = [item.strip() for item in patterns.split("|") if item.strip()]
                patterns_json = json.dumps([str(item) for item in patterns], ensure_ascii=True)

                prepared.append(
                    (
                        concurso,
                        fecha,
                        local,
                        visitante,
                        float(p1),
                        float(px),
                        float(p2),
                        result,
                        patterns_json,
                        str(source or "manual"),
                    )
                )

            if prepared:
                conn.executemany(
                    """
                    INSERT INTO historical_matches (
                        concurso, fecha, local, visitante, pct_local, pct_empate, pct_visita,
                        resultado_real, patrones_json, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    prepared,
                )

        return {
            "inserted_rows": len(prepared),
            "skipped_invalid": skipped_invalid,
            "skipped_duplicate": skipped_duplicate,
        }

    def _normalize_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        cols = list(df.columns)
        col_p1 = _pick_column(cols, ["pct_local", "% local", "porcentaje_local", "p1"])
        col_px = _pick_column(cols, ["pct_empate", "% empate", "porcentaje_empate", "px"])
        col_p2 = _pick_column(cols, ["pct_visita", "% visita", "porcentaje_visita", "p2"])
        col_res = _pick_column(cols, ["resultado_real", "resultado", "real", "outcome"])

        if not (col_p1 and col_px and col_p2 and col_res):
            raise ValueError(
                "El CSV debe incluir columnas de porcentaje y resultado real: "
                "pct_local, pct_empate, pct_visita, resultado_real."
            )

        col_local = _pick_column(cols, ["local", "equipo_local", "home", "home_team"])
        col_visit = _pick_column(cols, ["visitante", "equipo_visitante", "away", "away_team"])
        col_fecha = _pick_column(cols, ["fecha", "date"])
        col_concurso = _pick_column(cols, ["concurso", "jornada", "ticket"])
        col_patterns = _pick_column(cols, ["patrones_activados", "patrones", "patterns"])

        normalized: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            normalized.append(
                {
                    "local": row[col_local] if col_local else "LOCAL",
                    "visitante": row[col_visit] if col_visit else "VISITA",
                    "pct_local": row[col_p1],
                    "pct_empate": row[col_px],
                    "pct_visita": row[col_p2],
                    "resultado_real": row[col_res],
                    "fecha": row[col_fecha] if col_fecha else "",
                    "concurso": row[col_concurso] if col_concurso else "",
                    "patrones_activados": row[col_patterns] if col_patterns else [],
                }
            )
        return normalized

    def _load_cases(self) -> List[Dict[str, Any]]:
        with self._session() as conn:
            rows = conn.execute(
                """
                SELECT local, visitante, pct_local, pct_empate, pct_visita, resultado_real, patrones_json
                FROM historical_matches
                WHERE resultado_real IN ('1', 'X', '2')
                """
            ).fetchall()

        cases: List[Dict[str, Any]] = []
        for row in rows:
            cases.append(
                {
                    "local": str(row["local"]),
                    "visitante": str(row["visitante"]),
                    "pct_local": float(row["pct_local"]),
                    "pct_empate": float(row["pct_empate"]),
                    "pct_visita": float(row["pct_visita"]),
                    "resultado_real": str(row["resultado_real"]),
                    "patrones_activados": _safe_json_list(row["patrones_json"]),
                }
            )
        return cases

    def find_similar_signal(self, match: MatchInput, pattern_tags: Sequence[str]) -> HistoricalSignal:
        cases = self._load_cases()
        if not cases:
            return HistoricalSignal(
                sample_size=0,
                min_required=self.min_required,
                distribution={"1": 0.0, "X": 0.0, "2": 0.0},
                recommended="",
                confidence_score=0.0,
                confidence_label="baja",
                avg_distance=1.0,
                examples=[],
                strategy_note="Sin base historica cargada.",
            )

        q_gap = _gap_top2(match.pct_local, match.pct_empate, match.pct_visita)
        q_draw_shape = _draw_structure(match.pct_local, match.pct_empate, match.pct_visita)
        query_patterns = [str(item) for item in pattern_tags]

        ranked: List[Tuple[float, Dict[str, Any]]] = []
        for candidate in cases:
            c_p1 = float(candidate["pct_local"])
            c_px = float(candidate["pct_empate"])
            c_p2 = float(candidate["pct_visita"])
            pct_dist = (
                abs(c_p1 - match.pct_local) + abs(c_px - match.pct_empate) + abs(c_p2 - match.pct_visita)
            ) / 300.0
            gap_dist = min(1.0, abs(_gap_top2(c_p1, c_px, c_p2) - q_gap) / 35.0)
            draw_dist = min(1.0, abs(_draw_structure(c_p1, c_px, c_p2) - q_draw_shape) / 26.0)
            pattern_dist = _jaccard_distance(query_patterns, candidate.get("patrones_activados", []))

            distance = (0.58 * pct_dist) + (0.2 * gap_dist) + (0.12 * draw_dist) + (0.1 * pattern_dist)
            ranked.append((distance, candidate))

        ranked.sort(key=lambda item: item[0])
        similar = [item for item in ranked if item[0] <= self.max_distance][: self.top_k]

        if len(similar) < self.min_required:
            similar = ranked[: min(self.top_k, max(self.min_required - len(similar), self.min_required))]

        if not similar:
            return HistoricalSignal(
                sample_size=0,
                min_required=self.min_required,
                distribution={"1": 0.0, "X": 0.0, "2": 0.0},
                recommended="",
                confidence_score=0.0,
                confidence_label="baja",
                avg_distance=1.0,
                examples=[],
                strategy_note="No se encontraron similares utilies.",
            )

        weighted = {"1": 0.0, "X": 0.0, "2": 0.0}
        avg_distance = 0.0
        examples: List[SimilarExample] = []

        for distance, candidate in similar:
            result = str(candidate["resultado_real"]).upper()
            if result not in _DIRECT:
                continue
            weight = 1.0 / (0.05 + distance)
            weighted[result] += weight
            avg_distance += distance

            if len(examples) < 6:
                examples.append(
                    SimilarExample(
                        local=str(candidate["local"]),
                        visitante=str(candidate["visitante"]),
                        pct_local=float(candidate["pct_local"]),
                        pct_empate=float(candidate["pct_empate"]),
                        pct_visita=float(candidate["pct_visita"]),
                        resultado_real=result,
                        distance=float(distance),
                        patterns=list(candidate.get("patrones_activados", [])),
                    )
                )

        total_weight = sum(weighted.values())
        if total_weight <= 0:
            distribution = {"1": 0.0, "X": 0.0, "2": 0.0}
            recommended = ""
        else:
            distribution = {key: weighted[key] / total_weight for key in ["1", "X", "2"]}
            recommended = sorted(distribution.items(), key=lambda item: item[1], reverse=True)[0][0]

        ordered = sorted(distribution.values(), reverse=True)
        top_prob = ordered[0] if ordered else 0.0
        second_prob = ordered[1] if len(ordered) > 1 else 0.0
        dominance = max(0.0, top_prob - second_prob)
        sample_factor = min(1.0, len(similar) / max(float(self.min_required * 2), 1.0))
        similarity_quality = max(0.0, 1.0 - (avg_distance / max(float(len(similar)) * self.max_distance, 1e-6)))
        score = (0.45 * sample_factor) + (0.35 * dominance) + (0.2 * similarity_quality)
        confidence_label = _score_to_label(score, len(similar), self.min_required)

        strategy_note = (
            "Muestra suficiente para ajuste historico."
            if len(similar) >= self.min_required
            else "Muestra baja: usar historico con cautela."
        )

        return HistoricalSignal(
            sample_size=len(similar),
            min_required=self.min_required,
            distribution=distribution,
            recommended=recommended,
            confidence_score=score,
            confidence_label=confidence_label,
            avg_distance=(avg_distance / len(similar)),
            examples=examples,
            strategy_note=strategy_note,
        )

    def build_hybrid_decision(
        self,
        heuristic_recommendation: str,
        heuristic_direct_symbol: str,
        heuristic_double: str,
        heuristic_confidence: str,
        historical_signal: HistoricalSignal,
    ) -> HybridDecision:
        reco = str(heuristic_recommendation or "").upper().strip()
        direct = str(heuristic_direct_symbol or "").upper().strip()
        double = str(heuristic_double or "").upper().strip()
        base_conf = str(heuristic_confidence or "media").lower()

        if direct not in _DIRECT:
            direct = "X"
        if double not in _DOUBLE:
            double = _double_from_pair(direct, "X")

        if not historical_signal.enough_samples or historical_signal.recommended not in _DIRECT:
            return HybridDecision(
                final_recommendation=reco if reco else direct,
                final_direct_symbol=direct,
                final_double=double if double in _DOUBLE else "-",
                final_confidence=base_conf,
                decision_note="Sin suficientes similares: se conserva heuristica.",
                decision_mode="heuristic_only",
            )

        hist = historical_signal.recommended
        agrees = hist in reco

        if agrees:
            boosted = _adjust_confidence(base_conf, +1)
            return HybridDecision(
                final_recommendation=reco if reco else direct,
                final_direct_symbol=direct,
                final_double=double if double in _DOUBLE else "-",
                final_confidence=boosted,
                decision_note=f"Historico y heuristica coinciden en {hist}: se incrementa confianza.",
                decision_mode="aligned",
            )

        conflict_double = _double_from_pair(direct, hist)
        if conflict_double in _DOUBLE:
            reduced = _adjust_confidence(base_conf, -1)
            return HybridDecision(
                final_recommendation=conflict_double,
                final_direct_symbol=hist,
                final_double=conflict_double,
                final_confidence=reduced,
                decision_note=(
                    f"Conflicto heuristica vs historico ({direct} vs {hist}): "
                    f"se protege con {conflict_double}."
                ),
                decision_mode="conflict_double",
            )

        reduced = _adjust_confidence(base_conf, -1)
        fallback = _double_from_pair(hist, "X")
        return HybridDecision(
            final_recommendation=fallback if fallback in _DOUBLE else hist,
            final_direct_symbol=hist,
            final_double=fallback if fallback in _DOUBLE else double,
            final_confidence=reduced,
            decision_note=f"Historico domina ({hist}) con ajuste conservador.",
            decision_mode="historical_override",
        )

    def metrics_report(self, cfg: HeuristicConfig, top_buckets: int = 12) -> Dict[str, Any]:
        cases = self._load_cases()
        if not cases:
            return {
                "total_cases": 0,
                "calibration": {},
                "bucket_accuracy": [],
                "frequent_patterns": [],
            }

        calibration_rows = [
            {
                "local": item["local"],
                "visitante": item["visitante"],
                "pct_local": item["pct_local"],
                "pct_empate": item["pct_empate"],
                "pct_visita": item["pct_visita"],
                "resultado_real": item["resultado_real"],
            }
            for item in cases
        ]
        calibration = evaluate_historical_cases(calibration_rows, cfg)
        bucket_counter: Dict[str, Dict[str, float]] = {}
        pattern_counter: Dict[str, int] = {}

        for case in cases:
            p1 = float(case["pct_local"])
            px = float(case["pct_empate"])
            p2 = float(case["pct_visita"])
            result = str(case["resultado_real"]).upper()
            if result not in _DIRECT:
                continue

            analysis = analyze_match(
                MatchInput(
                    local=str(case["local"]),
                    visitante=str(case["visitante"]),
                    pct_local=p1,
                    pct_empate=px,
                    pct_visita=p2,
                ),
                cfg,
                debug_mode=False,
            )

            direct_pick = str(analysis.recomendacion_principal).upper()
            direct_hit = 1.0 if direct_pick in _DIRECT and direct_pick == result else 0.0

            bucket = f"L{int(p1 // 5) * 5:02d}-E{int(px // 5) * 5:02d}-V{int(p2 // 5) * 5:02d}"
            stats = bucket_counter.setdefault(bucket, {"cases": 0.0, "direct_hits": 0.0})
            stats["cases"] += 1.0
            stats["direct_hits"] += direct_hit

            for name in analysis.patrones_activados:
                pattern_counter[name] = pattern_counter.get(name, 0) + 1

        bucket_rows: List[Dict[str, Any]] = []
        for bucket, stats in bucket_counter.items():
            cases_count = int(stats["cases"])
            if cases_count <= 0:
                continue
            bucket_rows.append(
                {
                    "bucket": bucket,
                    "cases": cases_count,
                    "accuracy_pick_directo": round(stats["direct_hits"] / cases_count, 4),
                }
            )

        bucket_rows = sorted(bucket_rows, key=lambda item: (-item["cases"], -item["accuracy_pick_directo"]))[
            : max(1, int(top_buckets))
        ]
        frequent_patterns = sorted(pattern_counter.items(), key=lambda item: item[1], reverse=True)[:15]

        return {
            "total_cases": len(cases),
            "calibration": calibration,
            "bucket_accuracy": bucket_rows,
            "frequent_patterns": [{"pattern": key, "activations": value} for key, value in frequent_patterns],
        }
