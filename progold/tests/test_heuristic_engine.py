from __future__ import annotations

import unittest

from config import load_rules
from logic import analyze_match, evaluate_historical_cases
from models import MatchInput


class HeuristicEngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = load_rules()

    def test_regla_empate_real_vs_empate_artificial(self) -> None:
        empate_real = analyze_match(
            MatchInput(local="A", visitante="B", pct_local=34.0, pct_empate=24.0, pct_visita=42.0),
            self.cfg,
            debug_mode=True,
        )
        empate_artificial = analyze_match(
            MatchInput(local="A", visitante="B", pct_local=52.0, pct_empate=18.0, pct_visita=30.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("empate_vivo", empate_real.patrones_activados)
        self.assertIn(empate_real.recomendacion_principal, {"X", "1X", "X2"})
        self.assertNotIn("empate_vivo", empate_artificial.patrones_activados)
        self.assertLess(empate_artificial.score_empate, max(empate_artificial.score_local, empate_artificial.score_visita))

    def test_regla_local_sobrepopular(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=47.0, pct_empate=23.0, pct_visita=30.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("local_sobrepopular", analysis.patrones_activados)
        self.assertNotEqual(analysis.recomendacion_principal, "1")
        self.assertTrue(analysis.sugerir_sorpresa)

    def test_regla_visita_viva_competitiva(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=38.0, pct_empate=23.0, pct_visita=39.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("visita_viva", analysis.patrones_activados)
        self.assertGreaterEqual(analysis.score_visita, analysis.score_local - 6.0)
        self.assertIn(analysis.doble_oportunidad, {"1X", "X2", "12"})

    def test_regla_visita_sobrecomprada_moderada(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=30.0, pct_empate=21.0, pct_visita=49.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("visita_sobrecomprada_moderada", analysis.patrones_activados)
        self.assertNotEqual(analysis.recomendacion_principal, "2")
        self.assertIn(analysis.doble_oportunidad, {"X2", "1X", "12"})

    def test_regla_favorito_realmente_estable(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=58.0, pct_empate=17.0, pct_visita=25.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("favorito_estable", analysis.patrones_activados)
        self.assertNotIn("favorito_sobrejugado", analysis.patrones_activados)
        self.assertEqual(analysis.recomendacion_principal, "1")
        self.assertIn(analysis.confianza, {"alta", "media"})

    def test_regla_partido_abierto_empate_castigado(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=39.0, pct_empate=18.0, pct_visita=43.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("partido_abierto_con_empate_castigado", analysis.patrones_activados)
        self.assertIn("12", {analysis.recomendacion_principal, analysis.doble_oportunidad})
        self.assertLess(analysis.score_empate, min(analysis.score_local, analysis.score_visita))

    def test_regla_partido_caotico(self) -> None:
        analysis = analyze_match(
            MatchInput(local="Local", visitante="Visita", pct_local=34.0, pct_empate=32.0, pct_visita=34.0),
            self.cfg,
            debug_mode=True,
        )

        self.assertIn("partido_caotico", analysis.patrones_activados)
        self.assertIn(analysis.recomendacion_principal, {"1X", "X2", "12"})
        self.assertEqual(analysis.confianza, "baja")

    def test_calibration_report_contract(self) -> None:
        historical = [
            {
                "local": "L1",
                "visitante": "V1",
                "pct_local": 47.0,
                "pct_empate": 23.0,
                "pct_visita": 30.0,
                "resultado_real": "2",
            },
            {
                "local": "L2",
                "visitante": "V2",
                "pct_local": 33.0,
                "pct_empate": 25.0,
                "pct_visita": 42.0,
                "resultado_real": "X",
            },
            {
                "local": "L3",
                "visitante": "V3",
                "pct_local": 55.0,
                "pct_empate": 19.0,
                "pct_visita": 26.0,
                "resultado_real": "1",
            },
        ]

        report = evaluate_historical_cases(historical, self.cfg)

        self.assertIn("accuracy_pick_directo", report)
        self.assertIn("accuracy_doble_oportunidad", report)
        self.assertIn("double_rescued_cases", report)
        self.assertIn("recommendation_mode_distribution", report)
        self.assertIn("confidence_distribution", report)
        self.assertIn("tipo_partido_distribution", report)
        self.assertIn("pattern_activation_matrix", report)
        self.assertIn("pattern_hit_rate", report)
        self.assertIn("pattern_combo_hit_rate", report)


if __name__ == "__main__":
    unittest.main()
