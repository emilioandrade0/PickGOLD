import json
from pathlib import Path

import pandas as pd


def _first_existing(paths):
    for p in paths:
        pp = Path(p)
        if pp.exists():
            return pp
    return None


SPORT_CONFIG = {
    "euroleague": {
        "label": "EuroLeague",
        "raw_candidates": [
            "src/data/euroleague/raw/euroleague_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/euroleague/historical_predictions",
            "src/data/euroleague/predictions",
        ],
    },
    "kbo": {
        "label": "KBO",
        "raw_candidates": [
            "src/data/kbo/raw/kbo_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/kbo/historical_predictions",
            "src/data/kbo/predictions",
        ],
    },
    "laliga": {
        "label": "LaLiga",
        "raw_candidates": [
            "src/data/laliga/raw/laliga_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/laliga/historical_predictions",
            "src/data/laliga/predictions",
        ],
    },
    "ligamx": {
        "label": "Liga MX",
        "raw_candidates": [
            "src/data/liga_mx/raw/liga_mx_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/liga_mx/historical_predictions",
            "src/data/liga_mx/predictions",
        ],
    },
    "mlb": {
        "label": "MLB",
        "raw_candidates": [
            "src/data/mlb/raw/mlb_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/mlb/historical_predictions",
            "src/data/mlb/predictions",
        ],
    },
    "nba": {
        "label": "NBA",
        "raw_candidates": [
            "src/data/raw/nba_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/historical_predictions",
            "src/data/predictions",
        ],
    },
    "ncaa_baseball": {
        "label": "NCAA Baseball",
        "raw_candidates": [
            "src/data/ncaa_baseball/raw/ncaa_baseball_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/ncaa_baseball/historical_predictions",
            "src/data/ncaa_baseball/predictions",
        ],
    },
    "nhl": {
        "label": "NHL",
        "raw_candidates": [
            "src/data/nhl/raw/nhl_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/nhl/historical_predictions",
            "src/data/nhl/predictions",
        ],
    },
    "triple_a": {
        "label": "Triple-A",
        "raw_candidates": [
            "src/data/triple_a/raw/triple_a_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/triple_a/historical_predictions",
            "src/data/triple_a/predictions",
        ],
    },
    "bundesliga": {
        "label": "Bundesliga",
        "raw_candidates": [
            "src/data/bundesliga/raw/bundesliga_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/bundesliga/historical_predictions",
            "src/data/bundesliga/predictions",
        ],
    },
    "ligue1": {
        "label": "Ligue 1",
        "raw_candidates": [
            "src/data/ligue1/raw/ligue1_advanced_history.csv",
        ],
        "pred_dir_candidates": [
            "src/data/ligue1/historical_predictions",
            "src/data/ligue1/predictions",
        ],
    },
}


def _resolve_paths(base_dir: Path, sport_key: str):
    cfg = SPORT_CONFIG[sport_key]
    raw = _first_existing([base_dir / p for p in cfg["raw_candidates"]])
    pred_dir = _first_existing([base_dir / p for p in cfg["pred_dir_candidates"]])
    return raw, pred_dir


def _safe_pct(value):
    try:
        return float(value) * 100.0
    except Exception:
        return 0.0


def _format_ratio_from_pct(pct_value: float, total: int):
    if total <= 0:
        return "0/0"
    hits = int(round((pct_value / 100.0) * total))
    return f"{hits}/{total}"


def _to_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    txt = str(value).strip().lower()
    if txt in {"1", "true", "yes", "si", "hit", "acierto"}:
        return True
    if txt in {"0", "false", "no", "fallo", "miss"}:
        return False
    return None


def _safe_rows_from_json(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        games = payload.get("games")
        if isinstance(games, list):
            return games
    return []


def _parse_over_under_pick(pick):
    if pick is None:
        return None
    txt = str(pick).upper()
    if "OVER" in txt:
        return True
    if "UNDER" in txt:
        return False
    return None


def _parse_btts_pick(pick):
    if pick is None:
        return None
    txt = str(pick).upper()
    if "YES" in txt or "SI" in txt:
        return True
    if "NO" in txt:
        return False
    return None


def _parse_yes_no_actual(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    txt = str(value).strip().upper()
    if txt in {"YES", "SI", "TRUE", "OVER", "1"}:
        return True
    if txt in {"NO", "FALSE", "UNDER", "0"}:
        return False
    return None


def _parse_team_result_value(value):
    if value is None:
        return None
    txt = str(value).strip().upper()
    if not txt:
        return None
    if txt in {"DRAW", "EMPATE", "TIE"}:
        return "DRAW"
    return txt


def _evaluate_total_pick_text(pick_text, total_value, fallback_line):
    p = str(pick_text or "").strip().upper()
    if not p:
        return None
    try:
        line = float(fallback_line)
    except Exception:
        line = None
    if line is None:
        return None
    if "OVER" in p:
        return total_value > line
    if "UNDER" in p:
        return total_value < line
    return None


def _evaluate_ligamx_markets(base_dir: Path, label: str, pred_dir: Path):
    print(f"Calculando accuracy base para {label}...")
    if pred_dir is None or not pred_dir.exists():
        print(f"ERROR: No encontre carpeta de predicciones para {label}.")
        return True

    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No hay JSONs en {pred_dir}.")
        return True

    counters = {
        "full_base": [0, 0],
        "full_adj": [0, 0],
        "ou_base": [0, 0],
        "ou_adj": [0, 0],
        "btts_base": [0, 0],
        "btts_adj": [0, 0],
        "corners_base": [0, 0],
        "corners_adj": [0, 0],
        "ht_base": [0, 0],
        "ht_adj": [0, 0],
    }
    tiers = {}
    total_rows = 0
    with_actual = 0

    for jf in json_files:
        rows = _safe_rows_from_json(jf)
        for r in rows:
            if not isinstance(r, dict):
                continue
            total_rows += 1

            # Detect if row has resolvable actuals.
            has_actual = any(
                r.get(k) is not None
                for k in [
                    "actual_result",
                    "actual_ht_result",
                    "actual_home_score",
                    "actual_away_score",
                    "over_actual",
                    "btts_actual",
                    "corners_actual",
                ]
            )
            if not has_actual:
                continue
            with_actual += 1

            # Full game base/adjusted
            fg_base_hit = _to_bool(r.get("correct_full_game_base"))
            if fg_base_hit is None:
                pred = _parse_team_result_value(r.get("full_game_pick"))
                actual = _parse_team_result_value(r.get("actual_result"))
                if pred is not None and actual is not None:
                    fg_base_hit = pred == actual
            if fg_base_hit is not None:
                counters["full_base"][1] += 1
                counters["full_base"][0] += int(fg_base_hit)
                tier = str(r.get("full_game_tier", "PASS")).upper()
                if tier not in tiers:
                    tiers[tier] = [0, 0]
                tiers[tier][1] += 1
                tiers[tier][0] += int(fg_base_hit)

            fg_adj_hit = _to_bool(r.get("correct_full_game_adjusted"))
            if fg_adj_hit is None:
                pred = _parse_team_result_value(r.get("recommended_pick"))
                actual = _parse_team_result_value(r.get("actual_result"))
                if pred is not None and actual is not None:
                    fg_adj_hit = pred == actual
            if fg_adj_hit is not None:
                counters["full_adj"][1] += 1
                counters["full_adj"][0] += int(fg_adj_hit)

            # Over/Under goals
            ou_base_hit = _to_bool(r.get("correct_total_base"))
            if ou_base_hit is None:
                pred = _parse_over_under_pick(r.get("total_pick"))
                actual = _parse_yes_no_actual(r.get("over_actual"))
                if pred is not None and actual is not None:
                    ou_base_hit = pred == actual
            if ou_base_hit is not None:
                counters["ou_base"][1] += 1
                counters["ou_base"][0] += int(ou_base_hit)

            ou_adj_hit = _to_bool(r.get("correct_total_adjusted"))
            if ou_adj_hit is None:
                pred = _parse_over_under_pick(r.get("total_recommended_pick"))
                actual = _parse_yes_no_actual(r.get("over_actual"))
                if pred is not None and actual is not None:
                    ou_adj_hit = pred == actual
            if ou_adj_hit is not None:
                counters["ou_adj"][1] += 1
                counters["ou_adj"][0] += int(ou_adj_hit)

            # BTTS
            btts_base_hit = _to_bool(r.get("correct_btts_base"))
            if btts_base_hit is None:
                pred = _parse_btts_pick(r.get("btts_pick"))
                actual = _parse_yes_no_actual(r.get("btts_actual"))
                if pred is not None and actual is not None:
                    btts_base_hit = pred == actual
            if btts_base_hit is not None:
                counters["btts_base"][1] += 1
                counters["btts_base"][0] += int(btts_base_hit)

            btts_adj_hit = _to_bool(r.get("correct_btts_adjusted"))
            if btts_adj_hit is None:
                pred = _parse_btts_pick(r.get("btts_recommended_pick"))
                actual = _parse_yes_no_actual(r.get("btts_actual"))
                if pred is not None and actual is not None:
                    btts_adj_hit = pred == actual
            if btts_adj_hit is not None:
                counters["btts_adj"][1] += 1
                counters["btts_adj"][0] += int(btts_adj_hit)

            # Corners O/U
            corners_base_hit = _to_bool(r.get("correct_corners_base"))
            if corners_base_hit is None:
                pred = _parse_over_under_pick(r.get("corners_pick"))
                actual = _parse_yes_no_actual(r.get("corners_actual"))
                if pred is not None and actual is not None:
                    corners_base_hit = pred == actual
            if corners_base_hit is not None:
                counters["corners_base"][1] += 1
                counters["corners_base"][0] += int(corners_base_hit)

            corners_adj_hit = _to_bool(r.get("correct_corners_adjusted"))
            if corners_adj_hit is None:
                pred = _parse_over_under_pick(r.get("corners_recommended_pick"))
                actual = _parse_yes_no_actual(r.get("corners_actual"))
                if pred is not None and actual is not None:
                    corners_adj_hit = pred == actual
            if corners_adj_hit is not None:
                counters["corners_adj"][1] += 1
                counters["corners_adj"][0] += int(corners_adj_hit)

            # Medio tiempo (si existe data en JSON)
            ht_base_hit = _to_bool(r.get("correct_ht_base"))
            if ht_base_hit is not None:
                counters["ht_base"][1] += 1
                counters["ht_base"][0] += int(ht_base_hit)

            ht_adj_hit = _to_bool(r.get("correct_ht_adjusted"))
            if ht_adj_hit is not None:
                counters["ht_adj"][1] += 1
                counters["ht_adj"][0] += int(ht_adj_hit)

            ht_actual = _parse_team_result_value(
                r.get("actual_ht_result") or r.get("halftime_actual_result") or r.get("h1_actual_result")
            )
            ht_base_pick = _parse_team_result_value(
                r.get("ht_pick") or r.get("half_time_pick") or r.get("h1_pick")
            )
            if ht_base_hit is None and ht_actual is not None and ht_base_pick is not None:
                counters["ht_base"][1] += 1
                counters["ht_base"][0] += int(ht_base_pick == ht_actual)

            ht_adj_pick = _parse_team_result_value(
                r.get("ht_recommended_pick") or r.get("half_time_recommended_pick") or r.get("h1_recommended_pick")
            )
            if ht_adj_hit is None and ht_actual is not None and ht_adj_pick is not None:
                counters["ht_adj"][1] += 1
                counters["ht_adj"][0] += int(ht_adj_pick == ht_actual)

    def _pct(h, t):
        return (100.0 * h / t) if t > 0 else None

    full_total = counters["full_base"][1]
    if full_total == 0:
        print("No hay cruces evaluables entre picks y resultados.")
        return True

    print("=" * 66)
    print(f"REPORTE ACCURACY BASE - {label} (MUESTRA FULL GAME: {full_total} JUEGOS)")
    print("=" * 66)
    print(f"Filas JSON revisadas : {total_rows}")
    print(f"Filas con resultado  : {with_actual}")
    print("-" * 66)
    fg_base = _pct(*counters["full_base"])
    fg_adj = _pct(*counters["full_adj"])
    print(f"FULL GAME Base       : {fg_base:.2f}% ({counters['full_base'][0]}/{counters['full_base'][1]})")
    if fg_adj is None:
        print("FULL GAME Adjusted   : N/A")
    else:
        print(f"FULL GAME Adjusted   : {fg_adj:.2f}% ({counters['full_adj'][0]}/{counters['full_adj'][1]})")

    for key, title in [
        ("ou_base", "OVER/UNDER GOLES Base"),
        ("ou_adj", "OVER/UNDER GOLES Adjusted"),
        ("btts_base", "BTTS Base"),
        ("btts_adj", "BTTS Adjusted"),
        ("corners_base", "CORNERS O/U Base"),
        ("corners_adj", "CORNERS O/U Adjusted"),
        ("ht_base", "RESULTADO HT Base"),
        ("ht_adj", "RESULTADO HT Adjusted"),
    ]:
        h, t = counters[key]
        p = _pct(h, t)
        if p is None:
            print(f"{title.ljust(22)}: N/A")
        else:
            print(f"{title.ljust(22)}: {p:.2f}% ({h}/{t})")

    print("-" * 66)
    print("ACCURACY POR TIER (FULL GAME BASE)")
    for tier_name in sorted(tiers.keys()):
        h, t = tiers[tier_name]
        tier_acc = (100.0 * h / t) if t > 0 else 0.0
        print(f"   {tier_name.ljust(8)} : {tier_acc:.2f}% ({h}/{t})")
    print("=" * 66)
    return True


def _evaluate_nhl_markets(base_dir: Path, label: str, raw_path: Path, pred_dir: Path):
    print(f"Calculando accuracy base para {label}...")
    if pred_dir is None or not pred_dir.exists():
        print(f"ERROR: No encontre carpeta de predicciones para {label}.")
        return True
    if raw_path is None or not raw_path.exists():
        print(f"ERROR: No encontre RAW para {label}.")
        return True

    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No hay JSONs en {pred_dir}.")
        return True

    raw_df = pd.read_csv(raw_path, dtype={"game_id": str})
    p1_map = {}
    if {"game_id", "home_p1_goals", "away_p1_goals"}.issubset(raw_df.columns):
        for _, rr in raw_df.iterrows():
            gid = str(rr.get("game_id") or "").strip()
            if not gid:
                continue
            hp1 = rr.get("home_p1_goals")
            ap1 = rr.get("away_p1_goals")
            if pd.notna(hp1) and pd.notna(ap1):
                p1_map[gid] = int(hp1) + int(ap1)

    counters = {
        "ml": [0, 0],
        "spread": [0, 0],
        "total": [0, 0],
        "q1": [0, 0],
        "home_over": [0, 0],
    }
    tiers = {}
    consensus_ml = {
        "STRONG": [0, 0],
        "NEUTRAL": [0, 0],
        "WEAK": [0, 0],
    }
    meta_bucket_ml = {
        "ELITE": [0, 0],
        "STRONG": [0, 0],
        "NORMAL": [0, 0],
        "LOW": [0, 0],
    }
    meta_range_ml = {
        ">=0.66": [0, 0],
        "0.60-0.66": [0, 0],
        "0.55-0.60": [0, 0],
        "<0.55": [0, 0],
        "missing": [0, 0],
    }
    market_alignment_ml = {
        "aligned": [0, 0],
        "neutral": [0, 0],
        "conflicted": [0, 0],
    }

    def _safe_prob(value):
        try:
            out = float(value)
        except Exception:
            return None
        if out != out:
            return None
        if out < 0.0 or out > 1.0:
            return None
        return out

    def _resolve_meta_score(row):
        score = _safe_prob(row.get("full_game_meta_score"))
        if score is not None:
            return score
        conf = row.get("full_game_meta_confidence")
        try:
            conf_f = float(conf)
        except Exception:
            return None
        if conf_f != conf_f:
            return None
        if conf_f > 1.0:
            conf_f = conf_f / 100.0
        if conf_f < 0.0 or conf_f > 1.0:
            return None
        return conf_f

    def _resolve_meta_bucket(row):
        bucket = str(row.get("full_game_meta_bucket", "") or "").strip().upper()
        if bucket in meta_bucket_ml:
            return bucket
        score = _resolve_meta_score(row)
        if score is None:
            return "LOW"
        if score >= 0.67:
            return "ELITE"
        if score >= 0.60:
            return "STRONG"
        if score >= 0.55:
            return "NORMAL"
        return "LOW"

    def _resolve_meta_range(row):
        score = _resolve_meta_score(row)
        if score is None:
            return "missing"
        if score >= 0.66:
            return ">=0.66"
        if score >= 0.60:
            return "0.60-0.66"
        if score >= 0.55:
            return "0.55-0.60"
        return "<0.55"

    for jf in json_files:
        rows = _safe_rows_from_json(jf)
        for r in rows:
            if not isinstance(r, dict):
                continue

            ml_hit = _to_bool(r.get("moneyline_correct") if "moneyline_correct" in r else r.get("correct"))
            if ml_hit is not None:
                counters["ml"][1] += 1
                counters["ml"][0] += int(ml_hit)
                tier = str(r.get("full_game_tier", r.get("tier", "PASS"))).upper()
                if tier not in tiers:
                    tiers[tier] = [0, 0]
                tiers[tier][1] += 1
                tiers[tier][0] += int(ml_hit)

                consensus_key = str(r.get("consensus_signal", "NEUTRAL") or "NEUTRAL").strip().upper()
                if consensus_key not in consensus_ml:
                    consensus_ml[consensus_key] = [0, 0]
                consensus_ml[consensus_key][1] += 1
                consensus_ml[consensus_key][0] += int(ml_hit)

                meta_bucket_key = _resolve_meta_bucket(r)
                meta_bucket_ml[meta_bucket_key][1] += 1
                meta_bucket_ml[meta_bucket_key][0] += int(ml_hit)

                meta_range_key = _resolve_meta_range(r)
                meta_range_ml[meta_range_key][1] += 1
                meta_range_ml[meta_range_key][0] += int(ml_hit)

                align_key = str(r.get("market_ml_alignment", "neutral") or "neutral").strip().lower()
                if align_key not in market_alignment_ml:
                    align_key = "neutral"
                market_alignment_ml[align_key][1] += 1
                market_alignment_ml[align_key][0] += int(ml_hit)

            spread_hit = _to_bool(r.get("correct_spread"))
            if spread_hit is not None:
                counters["spread"][1] += 1
                counters["spread"][0] += int(spread_hit)

            total_hit = _to_bool(r.get("correct_total") if "correct_total" in r else r.get("total_correct"))
            if total_hit is not None:
                counters["total"][1] += 1
                counters["total"][0] += int(total_hit)

            home_hit = _to_bool(r.get("home_over_correct"))
            if home_hit is not None:
                counters["home_over"][1] += 1
                counters["home_over"][0] += int(home_hit)

            q1_hit = _to_bool(r.get("q1_correct") if "q1_correct" in r else r.get("q1_hit"))
            if q1_hit is None:
                gid = str(r.get("game_id") or "").strip()
                q1_total = p1_map.get(gid)
                q1_pick = str(r.get("q1_pick") or "").strip()
                q1_line = r.get("q1_line", 1.5)
                if q1_total is not None:
                    q1_hit = _to_bool(_evaluate_total_pick_text(q1_pick, q1_total, q1_line))
            if q1_hit is not None:
                counters["q1"][1] += 1
                counters["q1"][0] += int(q1_hit)

    if counters["ml"][1] == 0:
        print("No hay cruces evaluables entre picks y resultados.")
        return True

    def _pct(h, t):
        return (100.0 * h / t) if t > 0 else None

    print("=" * 66)
    print(f"REPORTE ACCURACY BASE - {label} (MUESTRA ML: {counters['ml'][1]} JUEGOS)")
    print("=" * 66)
    for key, title in [
        ("ml", "MONEYLINE"),
        ("spread", "HANDICAP -1.5"),
        ("total", "OVER/UNDER GOLES"),
        ("q1", "PRIMER PERIODO O/U 1.5"),
        ("home_over", "GOLES LOCAL O/U 2.5"),
    ]:
        h, t = counters[key]
        p = _pct(h, t)
        if p is None:
            print(f"{title.ljust(24)}: N/A")
        else:
            print(f"{title.ljust(24)}: {p:.2f}% ({h}/{t})")

    print("-" * 66)
    print("ACCURACY POR TIER (MONEYLINE)")
    for tier_name in sorted(tiers.keys()):
        h, t = tiers[tier_name]
        tier_acc = (100.0 * h / t) if t > 0 else 0.0
        print(f"   {tier_name.ljust(8)} : {tier_acc:.2f}% ({h}/{t})")

    print("-" * 66)
    print("ACCURACY MONEYLINE POR CONSENSUS_SIGNAL")
    for signal_name in ["STRONG", "NEUTRAL", "WEAK"]:
        h, t = consensus_ml.get(signal_name, [0, 0])
        p = _pct(h, t)
        if p is None:
            print(f"   {signal_name.ljust(8)} : N/A")
        else:
            print(f"   {signal_name.ljust(8)} : {p:.2f}% ({h}/{t})")

    extra_signals = sorted(
        signal for signal in consensus_ml.keys() if signal not in {"STRONG", "NEUTRAL", "WEAK"}
    )
    for signal_name in extra_signals:
        h, t = consensus_ml[signal_name]
        p = _pct(h, t)
        if p is None:
            print(f"   {signal_name.ljust(8)} : N/A")
        else:
            print(f"   {signal_name.ljust(8)} : {p:.2f}% ({h}/{t})")

    strong_h, strong_t = consensus_ml.get("STRONG", [0, 0])
    if strong_t > 0 and counters["ml"][1] > 0:
        strong_acc = 100.0 * strong_h / strong_t
        global_acc = 100.0 * counters["ml"][0] / counters["ml"][1]
        print(f"   DELTA STRONG vs GLOBAL: {strong_acc - global_acc:+.2f} pp")

    print("-" * 66)
    print("FULL_GAME META BUCKETS (MONEYLINE)")
    global_ml_total = counters["ml"][1]
    for bucket_name in ["ELITE", "STRONG", "NORMAL", "LOW"]:
        h, t = meta_bucket_ml[bucket_name]
        p = _pct(h, t)
        if p is None:
            print(f"   {bucket_name.ljust(8)} : N/A")
            continue
        coverage = (100.0 * t / global_ml_total) if global_ml_total > 0 else 0.0
        print(f"   {bucket_name.ljust(8)} : {p:.2f}% ({h}/{t}) | Cobertura {coverage:.2f}%")

    print("-" * 66)
    print("FULL_GAME META SCORE RANGES (MONEYLINE)")
    for range_name in [">=0.66", "0.60-0.66", "0.55-0.60", "<0.55", "missing"]:
        h, t = meta_range_ml[range_name]
        p = _pct(h, t)
        if p is None:
            print(f"   {range_name.ljust(10)} : N/A")
            continue
        coverage = (100.0 * t / global_ml_total) if global_ml_total > 0 else 0.0
        print(f"   {range_name.ljust(10)} : {p:.2f}% ({h}/{t}) | Cobertura {coverage:.2f}%")

    print("-" * 66)
    print("MARKET ML ALIGNMENT (MONEYLINE)")
    for align_name in ["aligned", "neutral", "conflicted"]:
        h, t = market_alignment_ml[align_name]
        p = _pct(h, t)
        if p is None:
            print(f"   {align_name.upper().ljust(10)} : N/A")
            continue
        coverage = (100.0 * t / global_ml_total) if global_ml_total > 0 else 0.0
        print(f"   {align_name.upper().ljust(10)} : {p:.2f}% ({h}/{t}) | Cobertura {coverage:.2f}%")
    print("=" * 66)
    return True


def _evaluate_mlb_from_walkforward(base_dir: Path):
    summary_path = base_dir / "src" / "data" / "mlb" / "walkforward" / "walkforward_summary_mlb.json"
    if not summary_path.exists():
        return False

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    full_game = summary.get("full_game") or {}
    yrfi = summary.get("yrfi") or {}
    f5 = summary.get("f5") or {}
    totals = summary.get("totals") or {}
    run_line = summary.get("run_line") or {}
    if not full_game:
        return False

    total_rows = int(full_game.get("rows", 0) or 0)
    full_acc = _safe_pct(full_game.get("accuracy", 0.0))
    full_pub_acc = _safe_pct(full_game.get("published_accuracy", 0.0))
    full_pub_cov = _safe_pct(full_game.get("published_coverage", 0.0))
    yrfi_acc = _safe_pct(yrfi.get("accuracy", 0.0))
    yrfi_pub_acc = _safe_pct(yrfi.get("published_accuracy", 0.0))
    yrfi_pub_cov = _safe_pct(yrfi.get("published_coverage", 0.0))
    f5_acc = _safe_pct(f5.get("accuracy", 0.0))
    f5_pub_acc = _safe_pct(f5.get("published_accuracy", 0.0))
    f5_pub_cov = _safe_pct(f5.get("published_coverage", 0.0))
    totals_acc = _safe_pct(totals.get("accuracy", 0.0))
    totals_pub_acc = _safe_pct(totals.get("published_accuracy", 0.0))
    totals_pub_cov = _safe_pct(totals.get("published_coverage", 0.0))
    run_line_acc = _safe_pct(run_line.get("accuracy", 0.0))
    run_line_pub_acc = _safe_pct(run_line.get("published_accuracy", 0.0))
    run_line_pub_cov = _safe_pct(run_line.get("published_coverage", 0.0))

    print("=" * 54)
    print(f"REPORTE ACCURACY BASE - MLB (WALK-FORWARD, MUESTRA: {total_rows} JUEGOS)")
    print("=" * 54)
    print(f"FULL GAME Global    : {full_acc:.2f}% ({_format_ratio_from_pct(full_acc, total_rows)})")
    print(f"FULL GAME Publicado : {full_pub_acc:.2f}% | Cobertura {full_pub_cov:.2f}%")
    print(f"YRFI Global         : {yrfi_acc:.2f}%")
    print(f"YRFI Publicado      : {yrfi_pub_acc:.2f}% | Cobertura {yrfi_pub_cov:.2f}%")
    print(f"F5 Global           : {f5_acc:.2f}%")
    print(f"F5 Publicado        : {f5_pub_acc:.2f}% | Cobertura {f5_pub_cov:.2f}%")
    if totals:
        print(f"TOTALS Global       : {totals_acc:.2f}%")
        print(f"TOTALS Publicado    : {totals_pub_acc:.2f}% | Cobertura {totals_pub_cov:.2f}%")
    if run_line:
        print(f"RUN LINE Global     : {run_line_acc:.2f}%")
        print(f"RUN LINE Publicado  : {run_line_pub_acc:.2f}% | Cobertura {run_line_pub_cov:.2f}%")
    print("-" * 54)
    print("BUCKETS PUBLICADOS (FULL GAME)")
    for bucket_name, bucket in (full_game.get("published_confidence_buckets") or {}).items():
        published_rows = int(bucket.get("published_rows", 0) or 0)
        if published_rows <= 0:
            continue
        bucket_acc = _safe_pct(bucket.get("published_accuracy", 0.0))
        print(f"   {bucket_name.ljust(8)} : {bucket_acc:.2f}% ({_format_ratio_from_pct(bucket_acc, published_rows)})")
    print("=" * 54)
    print(f"Fuente: {summary_path}")
    return True


def _parse_actual_row(row: pd.Series):
    home_team = str(row.get("home_team", ""))
    away_team = str(row.get("away_team", ""))

    winner = None
    score_pairs = [
        ("home_pts_total", "away_pts_total"),
        ("home_runs_total", "away_runs_total"),
        ("home_score", "away_score"),
    ]
    for h_col, a_col in score_pairs:
        if h_col in row and a_col in row and pd.notna(row[h_col]) and pd.notna(row[a_col]):
            h_val = float(row[h_col])
            a_val = float(row[a_col])
            if h_val > a_val:
                winner = home_team
            elif a_val > h_val:
                winner = away_team
            else:
                winner = "DRAW"
            break

    q1_type = None
    q1_value = None

    if "home_q1" in row and "away_q1" in row and pd.notna(row["home_q1"]) and pd.notna(row["away_q1"]):
        q1_type = "team"
        hq = float(row["home_q1"])
        aq = float(row["away_q1"])
        if hq > aq:
            q1_value = home_team
        elif aq > hq:
            q1_value = away_team
        else:
            q1_value = "TIE"
    elif "home_r1" in row and "away_r1" in row and pd.notna(row["home_r1"]) and pd.notna(row["away_r1"]):
        q1_type = "yrfi"
        q1_value = "YRFI" if (float(row["home_r1"]) + float(row["away_r1"])) > 0 else "NRFI"
    elif "home_p1_goals" in row and "away_p1_goals" in row and pd.notna(row["home_p1_goals"]) and pd.notna(row["away_p1_goals"]):
        q1_type = "ou15"
        q1_value = "OVER" if (float(row["home_p1_goals"]) + float(row["away_p1_goals"])) > 1.5 else "UNDER"

    return {"winner": winner, "q1_type": q1_type, "q1_value": q1_value}


def _is_q1_hit(pred_q1, q1_type, q1_value):
    if pred_q1 is None or q1_type is None or q1_value is None:
        return None

    pred_str = str(pred_q1).strip().upper()
    if q1_type == "team":
        if q1_value == "TIE":
            return None
        return pred_str == str(q1_value).strip().upper()
    if q1_type == "yrfi":
        if "YRFI" in pred_str:
            return q1_value == "YRFI"
        if "NRFI" in pred_str:
            return q1_value == "NRFI"
        return None
    if q1_type == "ou15":
        if "OVER" in pred_str:
            return q1_value == "OVER"
        if "UNDER" in pred_str:
            return q1_value == "UNDER"
        return None
    return None


def evaluate_for_sport(sport_key: str):
    if sport_key not in SPORT_CONFIG:
        print(f"ERROR: sport_key no soportado: {sport_key}")
        return

    base_dir = Path(__file__).resolve().parent.parent.parent
    label = SPORT_CONFIG[sport_key]["label"]

    # MLB usa walk-forward como referencia seria; los JSON historicos legacy pueden inflar el baseline.
    if sport_key == "mlb" and _evaluate_mlb_from_walkforward(base_dir):
        return

    if sport_key == "ligamx":
        _, pred_dir = _resolve_paths(base_dir, sport_key)
        _evaluate_ligamx_markets(base_dir=base_dir, label=label, pred_dir=pred_dir)
        return

    if sport_key == "nhl":
        raw_path, pred_dir = _resolve_paths(base_dir, sport_key)
        _evaluate_nhl_markets(base_dir=base_dir, label=label, raw_path=raw_path, pred_dir=pred_dir)
        return

    raw_path, pred_dir = _resolve_paths(base_dir, sport_key)

    print(f"Calculando accuracy base para {label}...")
    if raw_path is None or not raw_path.exists():
        print(f"ERROR: No encontre RAW para {label}.")
        return
    if pred_dir is None or not pred_dir.exists():
        print(f"ERROR: No encontre carpeta de predicciones para {label}.")
        return

    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        print(f"ERROR: No hay JSONs en {pred_dir}.")
        return

    df_raw = pd.read_csv(raw_path, dtype={"game_id": str})
    if "game_id" not in df_raw.columns:
        print("ERROR: El RAW no tiene game_id.")
        return

    actual = {}
    for _, row in df_raw.iterrows():
        gid = str(row.get("game_id"))
        parsed = _parse_actual_row(row)
        actual[gid] = parsed

    total_full = 0
    hit_full = 0
    total_q1 = 0
    hit_q1 = 0
    tiers = {}

    for jf in json_files:
        try:
            picks = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(picks, list):
            continue

        for p in picks:
            gid = str(p.get("game_id"))
            if gid not in actual:
                continue

            pred_full = p.get("full_game_pick", p.get("moneyline_pick"))
            pred_q1 = p.get("q1_pick")
            pred_tier = str(p.get("full_game_tier", p.get("tier", "PASS"))).upper()

            a = actual[gid]
            real_winner = a["winner"]
            if real_winner is not None and pred_full is not None:
                total_full += 1
                if pred_tier not in tiers:
                    tiers[pred_tier] = [0, 0]
                tiers[pred_tier][1] += 1

                if str(pred_full).strip().upper() == str(real_winner).strip().upper():
                    hit_full += 1
                    tiers[pred_tier][0] += 1

            q1_hit = _is_q1_hit(pred_q1, a["q1_type"], a["q1_value"])
            if q1_hit is not None:
                total_q1 += 1
                hit_q1 += int(q1_hit)

    if total_full == 0:
        print("No hay cruces evaluables entre picks y resultados.")
        return

    acc_full = 100.0 * hit_full / total_full
    acc_q1 = (100.0 * hit_q1 / total_q1) if total_q1 > 0 else None

    print("=" * 54)
    print(f"REPORTE ACCURACY BASE - {label} (MUESTRA: {total_full} JUEGOS)")
    print("=" * 54)
    print(f"FULL GAME Global : {acc_full:.2f}% ({hit_full}/{total_full})")
    if acc_q1 is None:
        print("Q1 Global        : N/A (sin mercado comparable)")
    else:
        print(f"Q1 Global        : {acc_q1:.2f}% ({hit_q1}/{total_q1})")
    print("-" * 54)
    print("ACCURACY POR TIER (FULL GAME)")
    for tier_name in sorted(tiers.keys()):
        h, t = tiers[tier_name]
        tier_acc = (100.0 * h / t) if t > 0 else 0.0
        print(f"   {tier_name.ljust(8)} : {tier_acc:.2f}% ({h}/{t})")
    print("=" * 54)
