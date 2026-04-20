from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import load_rules
from logic.historical_patterns import HistoricalPatternEngine
from models import MatchInput


class HistoricalPatternsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp_dir.name) / "historical_test.db"
        self.cfg = load_rules()
        self.engine = HistoricalPatternEngine(
            db_path=self.db_path,
            min_required=6,
            max_distance=0.35,
            top_k=30,
        )

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _seed_rows(self) -> None:
        rows = [
            {"local": "A", "visitante": "B", "pct_local": 44.9, "pct_empate": 21.5, "pct_visita": 33.6, "resultado_real": "2"},
            {"local": "C", "visitante": "D", "pct_local": 45.1, "pct_empate": 22.0, "pct_visita": 32.9, "resultado_real": "2"},
            {"local": "E", "visitante": "F", "pct_local": 43.8, "pct_empate": 21.7, "pct_visita": 34.5, "resultado_real": "2"},
            {"local": "G", "visitante": "H", "pct_local": 46.2, "pct_empate": 20.8, "pct_visita": 33.0, "resultado_real": "2"},
            {"local": "I", "visitante": "J", "pct_local": 44.4, "pct_empate": 22.3, "pct_visita": 33.3, "resultado_real": "1"},
            {"local": "K", "visitante": "L", "pct_local": 44.0, "pct_empate": 21.9, "pct_visita": 34.1, "resultado_real": "2"},
            {"local": "M", "visitante": "N", "pct_local": 45.5, "pct_empate": 21.1, "pct_visita": 33.4, "resultado_real": "2"},
            {"local": "O", "visitante": "P", "pct_local": 44.2, "pct_empate": 21.4, "pct_visita": 34.4, "resultado_real": "X"},
        ]
        inserted = self.engine.insert_rows(rows, source="unit_test")
        self.assertEqual(inserted, len(rows))

    def test_find_similar_signal_detects_visit_pattern(self) -> None:
        self._seed_rows()
        signal = self.engine.find_similar_signal(
            MatchInput(local="Q", visitante="R", pct_local=44.85, pct_empate=21.57, pct_visita=33.58),
            pattern_tags=["visita_viva"],
        )
        self.assertGreaterEqual(signal.sample_size, 6)
        self.assertEqual(signal.recommended, "2")
        self.assertGreater(signal.distribution.get("2", 0.0), signal.distribution.get("1", 0.0))

    def test_hybrid_conflict_moves_to_double(self) -> None:
        self._seed_rows()
        signal = self.engine.find_similar_signal(
            MatchInput(local="Q", visitante="R", pct_local=44.85, pct_empate=21.57, pct_visita=33.58),
            pattern_tags=["visita_viva"],
        )
        decision = self.engine.build_hybrid_decision(
            heuristic_recommendation="1",
            heuristic_direct_symbol="1",
            heuristic_double="1X",
            heuristic_confidence="alta",
            historical_signal=signal,
        )
        self.assertIn(decision.final_recommendation, {"12", "X2"})
        self.assertIn(decision.final_confidence, {"media", "baja"})

    def test_metrics_report_has_expected_contract(self) -> None:
        self._seed_rows()
        report = self.engine.metrics_report(self.cfg)
        self.assertEqual(report.get("total_cases"), 8)
        self.assertIn("calibration", report)
        self.assertIn("bucket_accuracy", report)
        self.assertIn("frequent_patterns", report)

    def test_insert_rows_dedup_avoids_repeated_rows(self) -> None:
        row = {
            "local": "AA",
            "visitante": "BB",
            "pct_local": 44.85,
            "pct_empate": 21.57,
            "pct_visita": 33.58,
            "resultado_real": "2",
            "fecha": "2026-04-19",
        }
        first = self.engine.insert_rows_dedup([row], source="unit_test")
        second = self.engine.insert_rows_dedup([row], source="unit_test")

        self.assertEqual(first.get("inserted_rows"), 1)
        self.assertEqual(second.get("inserted_rows"), 0)
        self.assertEqual(second.get("skipped_duplicate"), 1)


if __name__ == "__main__":
    unittest.main()
