import json
from pathlib import Path
import pandas as pd

# --- RUTAS ---
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
RAW_DATA = BASE_DIR / "data" / "wnba" / "raw" / "wnba_advanced_history.csv"
HIST_PRED_DIR = BASE_DIR / "data" / "wnba" / "historical_predictions"
SNAPSHOT_DIR = BASE_DIR / "data" / "insights"


def _safe_float(value):
    try:
        num = pd.to_numeric(value, errors="coerce")
        return None if pd.isna(num) else float(num)
    except Exception:
        return None


def _calc_spread_hit(pick: dict, h_team: str, a_team: str, h_pts: float, a_pts: float):
    cached = pick.get("correct_spread", None)
    if isinstance(cached, bool):
        return cached

    spread_pick = str(pick.get("spread_pick", "") or "").strip()
    h_spread = _safe_float(pick.get("home_spread"))
    if not spread_pick or spread_pick == "N/A" or h_spread is None or h_spread == 0:
        return None

    if spread_pick.startswith(h_team):
        return (h_pts + h_spread) > a_pts
    if spread_pick.startswith(a_team):
        return (a_pts - h_spread) > h_pts
    return None


def _calc_total_hit(pick: dict, h_pts: float, a_pts: float):
    cached = pick.get("total_hit", None)
    if isinstance(cached, bool):
        return cached

    total_pick = str(pick.get("total_pick", "") or "").strip().upper()
    total_line = _safe_float(pick.get("odds_over_under"))
    if not total_pick or total_pick == "N/A" or total_line is None or total_line <= 0:
        return None

    total_pts = h_pts + a_pts
    if total_pick.startswith("OVER"):
        return total_pts > total_line
    if total_pick.startswith("UNDER"):
        return total_pts < total_line
    return None


def _write_snapshot(report: dict):
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_file = SNAPSHOT_DIR / "wnba_baseline_snapshot_2026-03-31.md"
    lines = [
        "# WNBA baseline snapshot",
        "",
        "Fecha: 2026-03-31",
        "",
        "## Accuracy global",
        f"- Partido completo: {report['acc_full']:.2f}% ({report['correct_full']}/{report['total_games']})",
        f"- Primer cuarto: {report['acc_q1']:.2f}% ({report['correct_q1']}/{report['total_q1']})",
        f"- Primera mitad: {report['acc_h1']:.2f}% ({report['correct_h1']}/{report['total_h1']})",
        f"- Handicap/Spread: {report['acc_spread']:.2f}% ({report['correct_spread']}/{report['total_spread']})",
        f"- Over/Under: {report['acc_total']:.2f}% ({report['correct_total']}/{report['total_total']})",
        "",
        "## Accuracy por tier (partido completo)",
    ]
    for tier_name, tier_block in report["tiers"].items():
        hits = tier_block["hits"]
        total = tier_block["total"]
        if total > 0:
            pct = (hits / total) * 100
            lines.append(f"- {tier_name}: {pct:.2f}% ({hits}/{total})")
        else:
            lines.append(f"- {tier_name}: sin predicciones")
    snapshot_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return snapshot_file

def evaluate_historical_predictions():
    print("📊 Calculando el Accuracy Base de tus modelos WNBA...\n")

    if not RAW_DATA.exists() or not HIST_PRED_DIR.exists():
        print("❌ Faltan los archivos CSV o la carpeta de predicciones históricas.")
        return

    # 1. Cargar resultados reales
    df_raw = pd.read_csv(RAW_DATA, dtype={"game_id": str})
    
    # Crear diccionario de resultados reales para búsqueda ultra rápida
    actual_results = {}
    for _, row in df_raw.iterrows():
        g_id = str(row["game_id"])
        h_team = row["home_team"]
        a_team = row["away_team"]
        h_pts = row["home_pts_total"]
        a_pts = row["away_pts_total"]
        h_q1 = row["home_q1"]
        a_q1 = row["away_q1"]
        h_q2 = row.get("home_q2", 0)
        a_q2 = row.get("away_q2", 0)
        
        winner = h_team if h_pts > a_pts else a_team
        
        if h_q1 > a_q1:
            q1_winner = h_team
        elif a_q1 > h_q1:
            q1_winner = a_team
        else:
            q1_winner = "TIE"

        h1_home = h_q1 + h_q2
        h1_away = a_q1 + a_q2
        if h1_home > h1_away:
            h1_winner = h_team
        elif h1_away > h1_home:
            h1_winner = a_team
        else:
            h1_winner = "TIE"
            
        actual_results[g_id] = {
            "winner": winner,
            "q1_winner": q1_winner,
            "h1_winner": h1_winner,
            "home_team": h_team,
            "away_team": a_team,
            "home_pts_total": h_pts,
            "away_pts_total": a_pts,
        }

    # 2. Contadores
    total_games = 0
    correct_full = 0
    correct_q1 = 0
    total_q1_evaluable = 0
    correct_h1 = 0
    total_h1_evaluable = 0
    correct_spread = 0
    total_spread_evaluable = 0
    correct_total = 0
    total_total_evaluable = 0
    
    # Contadores por Tier: [Aciertos, Total]
    tiers = {
        "ELITE": [0, 0],
        "PREMIUM": [0, 0],
        "STRONG": [0, 0],
        "NORMAL": [0, 0],
        "PASS": [0, 0]
    }

    # 3. Leer todos los JSONs generados
    json_files = list(HIST_PRED_DIR.glob("*.json"))
    if not json_files:
        print("📭 No hay archivos JSON en data/historical_predictions/")
        return

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                day_picks = json.load(f)
            except Exception:
                continue
                
        for pick in day_picks:
            g_id = str(pick.get("game_id"))
            if g_id not in actual_results:
                continue # Partido cancelado o no está en el CSV
                
            pred_winner = pick.get("full_game_pick")
            pred_tier = pick.get("full_game_tier", "PASS")
            pred_q1 = pick.get("q1_pick")
            pred_h1 = pick.get("h1_pick")
            
            real_winner = actual_results[g_id]["winner"]
            real_q1_winner = actual_results[g_id]["q1_winner"]
            real_h1_winner = actual_results[g_id]["h1_winner"]
            
            total_games += 1
            
            # Evaluar Partido Completo
            if pred_winner == real_winner:
                correct_full += 1
                if pred_tier in tiers:
                    tiers[pred_tier][0] += 1
            
            if pred_tier in tiers:
                tiers[pred_tier][1] += 1
                
            # Evaluar Q1: solo contar partidos Q1 evaluables (no-TIE)
            if real_q1_winner != "TIE":
                total_q1_evaluable += 1
                if pred_q1 == real_q1_winner:
                    correct_q1 += 1

            if real_h1_winner != "TIE":
                total_h1_evaluable += 1
                if pred_h1 == real_h1_winner:
                    correct_h1 += 1

            # Evaluar spread con fallback robusto
            cs = _calc_spread_hit(
                pick,
                actual_results[g_id]["home_team"],
                actual_results[g_id]["away_team"],
                actual_results[g_id]["home_pts_total"],
                actual_results[g_id]["away_pts_total"],
            )
            if cs is not None:
                total_spread_evaluable += 1
                if bool(cs):
                    correct_spread += 1

            # Evaluar total con fallback robusto
            th = _calc_total_hit(
                pick,
                actual_results[g_id]["home_pts_total"],
                actual_results[g_id]["away_pts_total"],
            )
            if th is not None:
                total_total_evaluable += 1
                if bool(th):
                    correct_total += 1

    # 4. Imprimir Reporte
    if total_games == 0:
        print("⚠️ No se encontraron cruces entre las predicciones y los resultados reales.")
        return
        
    acc_full = (correct_full / total_games) * 100
    acc_q1 = (correct_q1 / total_q1_evaluable) * 100 if total_q1_evaluable > 0 else 0
    acc_h1 = (correct_h1 / total_h1_evaluable) * 100 if total_h1_evaluable > 0 else 0
    acc_spread = (correct_spread / total_spread_evaluable) * 100 if total_spread_evaluable > 0 else 0
    acc_total = (correct_total / total_total_evaluable) * 100 if total_total_evaluable > 0 else 0
    
    print("======================================================")
    print(f"🏆 REPORTE DE ACCURACY BASE (MUESTRA: {total_games} JUEGOS)")
    print("======================================================")
    print(f"🏀 PARTIDO COMPLETO Global : {acc_full:.2f}% ({correct_full}/{total_games})")
    print(f"⏱️ PRIMER CUARTO Global    : {acc_q1:.2f}% ({correct_q1}/{total_q1_evaluable})")
    print(f"🕐 PRIMERA MITAD Global    : {acc_h1:.2f}% ({correct_h1}/{total_h1_evaluable})")
    print(f"📏 HANDICAP/SPREAD Global  : {acc_spread:.2f}% ({correct_spread}/{total_spread_evaluable})")
    print(f"🎯 OVER/UNDER Global       : {acc_total:.2f}% ({correct_total}/{total_total_evaluable})")
    print("-" * 54)
    print("📈 ACCURACY POR NIVEL DE CONFIANZA (PARTIDO COMPLETO)")
    
    for tier_name, (hits, total) in tiers.items():
        if total > 0:
            tier_acc = (hits / total) * 100
            print(f"   {tier_name.ljust(8)} : {tier_acc:.2f}% ({hits}/{total})")
        else:
            print(f"   {tier_name.ljust(8)} : Sin predicciones")
    report = {
        "acc_full": acc_full,
        "acc_q1": acc_q1,
        "acc_h1": acc_h1,
        "acc_spread": acc_spread,
        "acc_total": acc_total,
        "correct_full": correct_full,
        "correct_q1": correct_q1,
        "correct_h1": correct_h1,
        "correct_spread": correct_spread,
        "correct_total": correct_total,
        "total_games": total_games,
        "total_q1": total_q1_evaluable,
        "total_h1": total_h1_evaluable,
        "total_spread": total_spread_evaluable,
        "total_total": total_total_evaluable,
        "tiers": {
            tier_name: {"hits": hits, "total": total}
            for tier_name, (hits, total) in tiers.items()
        },
    }
    snapshot_file = _write_snapshot(report)
    print("======================================================")
    print(f"📝 Snapshot guardado en: {snapshot_file}")

if __name__ == "__main__":
    evaluate_historical_predictions()
