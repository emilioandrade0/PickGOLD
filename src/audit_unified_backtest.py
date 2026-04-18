import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

try:
    from .external_odds_overrides import apply_overrides_to_events
except Exception:
    from external_odds_overrides import apply_overrides_to_events

BASE_DIR = Path(__file__).resolve().parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SPORTS = {
    "nba": {
        "historical_dir": BASE_DIR / "data" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "raw" / "nba_advanced_history.csv",
        "raw_type": "nba",
    },
    "mlb": {
        "historical_dir": BASE_DIR / "data" / "mlb" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv",
        "raw_type": "baseball",
    },
    "lmb": {
        "historical_dir": BASE_DIR / "data" / "lmb" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "lmb" / "raw" / "lmb_advanced_history.csv",
        "raw_type": "baseball",
    },
    "kbo": {
        "historical_dir": BASE_DIR / "data" / "kbo" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "kbo" / "raw" / "kbo_advanced_history.csv",
        "raw_type": "baseball",
    },
    "nhl": {
        "historical_dir": BASE_DIR / "data" / "nhl" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv",
        "raw_type": "soccer_like",
    },
    "liga_mx": {
        "historical_dir": BASE_DIR / "data" / "liga_mx" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv",
        "raw_type": "soccer_like",
    },
    "laliga": {
        "historical_dir": BASE_DIR / "data" / "laliga" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "laliga" / "raw" / "laliga_advanced_history.csv",
        "raw_type": "soccer_like",
    },
    "euroleague": {
        "historical_dir": BASE_DIR / "data" / "euroleague" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv",
        "raw_type": "nba",
    },
    "ncaa_baseball": {
        "historical_dir": BASE_DIR / "data" / "ncaa_baseball" / "historical_predictions",
        "raw_file": BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_advanced_history.csv",
        "raw_type": "baseball",
    },
}


def _read_json_events(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("games"), list):
        return data["games"]
    if isinstance(data, list):
        return data
    return []


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        if int(v) in (0, 1):
            return bool(int(v))
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "si", "acierto", "win", "won"}:
            return True
        if s in {"0", "false", "no", "fallo", "lose", "lost"}:
            return False
    return None


def _prob_from_confidence(conf):
    try:
        c = float(conf)
    except Exception:
        return None
    if c <= 0:
        return None
    return min(max(c / 100.0, 1e-6), 1 - 1e-6)


def _bucket_from_conf(conf):
    if conf is None:
        return "unknown"
    c = float(conf)
    if c < 55:
        return "50-54.9"
    if c < 60:
        return "55-59.9"
    if c < 65:
        return "60-64.9"
    if c < 70:
        return "65-69.9"
    if c < 75:
        return "70-74.9"
    if c < 80:
        return "75-79.9"
    return "80+"


def _extract_line(text, fallback=None):
    m = re.search(r"(\d+(?:\.\d+)?)", str(text or ""))
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    try:
        fb = float(fallback)
        if fb > 0:
            return fb
    except Exception:
        pass
    return None


def _winner_from_score(home, away, hs, aw):
    if hs > aw:
        return home
    if aw > hs:
        return away
    return "TIE"


def _eval_team_pick(pick: str, home: str, away: str, winner: str):
    if not pick:
        return None
    p = str(pick).strip().upper()
    h = str(home).strip().upper()
    a = str(away).strip().upper()
    w = str(winner).strip().upper()

    if p in {"DRAW", "TIE", "EMPATE", "X"}:
        return w in {"DRAW", "TIE", "EMPATE", "X"}

    if w in {"DRAW", "TIE", "EMPATE", "X"}:
        return False

    if p in {"HOME WIN", "HOME", "LOCAL", "1"}:
        return h == w
    if p in {"AWAY WIN", "AWAY", "VISITANTE", "VISITOR", "2"}:
        return a == w
    if p in {h, a}:
        return p == w

    if "HOME" in p or "LOCAL" in p:
        return h == w
    if "AWAY" in p or "VISITOR" in p or "VISITANTE" in p:
        return a == w
    if h and h in p:
        return h == w
    if a and a in p:
        return a == w

    return None


def _eval_yrfi(pick: str, home_r1: int, away_r1: int):
    p = str(pick or "").strip().upper()
    if not p:
        return None
    rfi = (int(home_r1) + int(away_r1)) > 0
    if p == "YRFI":
        return rfi
    if p == "NRFI":
        return not rfi
    return None


def _eval_over_under_pick(pick: str, observed_total: int, fallback_line=None):
    p = str(pick or "").strip().upper()
    if not p:
        return None
    line = _extract_line(p, fallback=fallback_line)
    if line is None:
        return None
    if "OVER" in p:
        return observed_total > line
    if "UNDER" in p:
        return observed_total < line
    return None


def _american_odds_to_decimal(value):
    try:
        odds = float(value)
    except Exception:
        return None
    if odds == 0:
        return None
    if odds > 0:
        return 1.0 + (odds / 100.0)
    return 1.0 + (100.0 / abs(odds))


def _decimal_odds_normalized(value):
    try:
        odds = float(value)
    except Exception:
        return None
    if odds <= 1.0:
        return None
    return odds


def _to_decimal_odds(value):
    dec = _american_odds_to_decimal(value)
    if dec is not None:
        return dec
    return _decimal_odds_normalized(value)


def _extract_market_odds(event: dict, market: str):
    odds_keys = {
        "full_game": [
            "closing_moneyline_odds", "closing_ml_odds", "closing_odds_ml", "moneyline_odds",
            "ml_odds", "odds_ml", "odds_moneyline", "opening_moneyline_odds", "opening_ml_odds",
        ],
        "spread": ["closing_spread_odds", "spread_odds", "odds_spread_price", "opening_spread_odds"],
        "total_goals_55": ["closing_total_odds", "total_odds", "odds_total_price", "opening_total_odds"],
        "total": ["closing_total_odds", "total_odds", "odds_total_price", "opening_total_odds"],
        "q1": ["closing_q1_odds", "closing_yrfi_odds", "q1_odds", "yrfi_odds", "nrfi_odds", "opening_q1_odds", "opening_yrfi_odds"],
        "f5": ["closing_f5_odds", "f5_odds", "opening_f5_odds"],
        "home_over": ["closing_home_over_odds", "home_over_odds", "opening_home_over_odds"],
        "corners": ["closing_corners_odds", "corners_odds", "opening_corners_odds"],
        "btts": ["closing_btts_odds", "btts_odds", "opening_btts_odds"],
    }

    for key in odds_keys.get(market, []):
        if key not in event:
            continue
        decimal = _to_decimal_odds(event.get(key))
        if decimal is not None:
            return float(decimal), key
    return None, None


def _build_lookup(raw_file: Path, raw_type: str):
    if not raw_file.exists():
        return {}

    df = pd.read_csv(raw_file)
    lookup = {}

    for _, row in df.iterrows():
        gid = str(row.get("game_id", ""))
        if not gid:
            continue

        home = str(row.get("home_team", ""))
        away = str(row.get("away_team", ""))

        if raw_type == "nba":
            hs = int(row.get("home_pts_total", 0) or 0)
            aw = int(row.get("away_pts_total", 0) or 0)
            hq1 = int(row.get("home_q1", 0) or 0)
            aq1 = int(row.get("away_q1", 0) or 0)
            hf5 = None
            af5 = None
            total_corners = None
        elif raw_type == "baseball":
            hs = int(row.get("home_runs_total", 0) or 0)
            aw = int(row.get("away_runs_total", 0) or 0)
            hq1 = int(row.get("home_r1", 0) or 0)
            aq1 = int(row.get("away_r1", 0) or 0)
            hf5 = int(row.get("home_runs_f5", 0) or 0)
            af5 = int(row.get("away_runs_f5", 0) or 0)
            total_corners = None
        else:
            hs = int(row.get("home_score", 0) or 0)
            aw = int(row.get("away_score", 0) or 0)
            hq1 = None
            aq1 = None
            hf5 = None
            af5 = None
            corners = row.get("total_corners")
            total_corners = int(corners) if pd.notna(corners) else None

        winner = _winner_from_score(home, away, hs, aw)
        q1_winner = None
        if hq1 is not None and aq1 is not None:
            q1_winner = _winner_from_score(home, away, hq1, aq1)

        lookup[gid] = {
            "date": str(row.get("date", "")),
            "home": home,
            "away": away,
            "home_score": hs,
            "away_score": aw,
            "winner": winner,
            "home_q1": hq1,
            "away_q1": aq1,
            "q1_winner": q1_winner,
            "home_f5": hf5,
            "away_f5": af5,
            "total_corners": total_corners,
        }

    return lookup


def _append_market(out, market, hit, event, prob=None, conf_key=None):
    if hit is None:
        return
    conf = event.get(conf_key) if conf_key else None
    if prob is None:
        prob = _prob_from_confidence(conf)

    decimal_odds, odds_source = _extract_market_odds(event, market)
    return_per_unit = None
    if decimal_odds is not None:
        return_per_unit = (decimal_odds - 1.0) if bool(hit) else -1.0

    out.append(
        {
            "market": market,
            "hit": int(bool(hit)),
            "prob": prob,
            "confidence": conf,
            "decimal_odds": decimal_odds,
            "odds_source": odds_source,
            "return_per_unit": return_per_unit,
        }
    )


def _extract_market_rows(sport: str, event: dict, game_lookup: dict):
    out = []
    gid = str(event.get("game_id", ""))
    game = game_lookup.get(gid, {})

    home = str(event.get("home_team") or game.get("home") or "")
    away = str(event.get("away_team") or game.get("away") or "")
    hs = int(event.get("home_score") or game.get("home_score") or 0)
    aw = int(event.get("away_score") or game.get("away_score") or 0)
    winner = game.get("winner") or _winner_from_score(home, away, hs, aw)

    fg_hit = _to_bool(event.get("correct_full_game_adjusted"))
    if fg_hit is None:
        fg_hit = _to_bool(event.get("correct_full_game"))
    if fg_hit is None:
        fg_hit = _to_bool(event.get("full_game_hit"))
    if fg_hit is None:
        fg_hit = _eval_team_pick(event.get("full_game_pick") or event.get("recommended_pick"), home, away, winner)

    fg_prob = event.get("full_game_calibrated_prob_pick")
    if fg_prob is None:
        fg_prob = event.get("full_game_model_prob_pick")
    if fg_prob is None:
        p_home = event.get("full_game_calibrated_prob_home")
        if p_home is None:
            p_home = event.get("full_game_model_prob_home")
        try:
            p_home = float(p_home)
            side_is_away = _eval_team_pick(event.get("full_game_pick") or event.get("recommended_pick"), home, away, away)
            fg_prob = (1.0 - p_home) if side_is_away else p_home
        except Exception:
            fg_prob = None

    _append_market(out, "full_game", fg_hit, event, prob=fg_prob, conf_key="full_game_confidence")

    q1_hit = _to_bool(event.get("q1_hit"))
    if q1_hit is None:
        q1_pick = str(event.get("q1_pick") or "").strip().upper()
        if q1_pick in {"YRFI", "NRFI"}:
            q1_hit = _eval_yrfi(q1_pick, game.get("home_q1", 0), game.get("away_q1", 0))
        elif game.get("q1_winner") is not None:
            q1_hit = _eval_team_pick(q1_pick, home, away, game.get("q1_winner"))
    _append_market(out, "q1", q1_hit, event, conf_key="q1_confidence")

    spread_hit = _to_bool(event.get("correct_spread"))
    if spread_hit is None:
        spread_hit = _eval_team_pick(event.get("spread_pick"), home, away, winner)
    spread_market_name = "total_goals_55" if sport == "nhl" else "spread"
    _append_market(out, spread_market_name, spread_hit, event, conf_key="spread_confidence")

    total_hit = _to_bool(event.get("correct_total_adjusted"))
    if total_hit is None:
        total_hit = _to_bool(event.get("correct_total"))
    if total_hit is None:
        total_pick = event.get("total_recommended_pick") or event.get("total_pick")
        total_hit = _eval_over_under_pick(total_pick, hs + aw, fallback_line=event.get("odds_over_under"))
    _append_market(out, "total", total_hit, event, conf_key="total_confidence")

    btts_hit = _to_bool(event.get("correct_btts_adjusted"))
    if btts_hit is None:
        btts_hit = _to_bool(event.get("correct_btts"))
    if btts_hit is None:
        btts_pick = str(event.get("btts_recommended_pick") or event.get("btts_pick") or "").upper()
        if btts_pick:
            both_scored = hs > 0 and aw > 0
            if "YES" in btts_pick:
                btts_hit = both_scored
            elif "NO" in btts_pick:
                btts_hit = not both_scored
    _append_market(out, "btts", btts_hit, event, conf_key="btts_confidence")

    corners_hit = _to_bool(event.get("correct_corners_adjusted"))
    if corners_hit is None:
        corners_hit = _to_bool(event.get("correct_corners_base"))
    if corners_hit is None:
        corners_pick = event.get("corners_recommended_pick") or event.get("corners_pick")
        total_corners = game.get("total_corners")
        if corners_pick and total_corners is not None:
            corners_hit = _eval_over_under_pick(corners_pick, int(total_corners), fallback_line=event.get("corners_line"))
    _append_market(out, "corners", corners_hit, event, conf_key="corners_confidence")

    f5_hit = _to_bool(event.get("correct_home_win_f5"))
    if f5_hit is None:
        f5_hit = _to_bool(event.get("correct_f5"))
    if f5_hit is None:
        f5_pick = event.get("f5_pick") or event.get("assists_pick")
        hf5 = game.get("home_f5")
        af5 = game.get("away_f5")
        if f5_pick and hf5 is not None and af5 is not None:
            f5_winner = _winner_from_score(home, away, int(hf5), int(af5))
            f5_hit = _eval_team_pick(f5_pick, home, away, f5_winner)
    _append_market(out, "f5", f5_hit, event, conf_key="extra_f5_confidence")

    home_over_hit = _to_bool(event.get("correct_home_over"))
    if home_over_hit is None:
        home_over_hit = _to_bool(event.get("correct_home_total"))
    if home_over_hit is None:
        home_over_pick = event.get("home_over_pick")
        if home_over_pick:
            home_over_hit = _eval_over_under_pick(home_over_pick, hs)
    _append_market(out, "home_over", home_over_hit, event, conf_key="home_over_confidence")

    return out


def _summarize_rows(rows):
    if not rows:
        return {}

    picks = len(rows)
    hits = sum(r["hit"] for r in rows)
    acc = hits / picks if picks else 0.0

    probs = [r["prob"] for r in rows if r["prob"] is not None]
    brier = None
    logloss = None
    if probs:
        aligned = [(r["hit"], r["prob"]) for r in rows if r["prob"] is not None]
        brier = sum((p - y) ** 2 for y, p in aligned) / len(aligned)
        eps = 1e-6
        logloss = -sum((y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps))) for y, p in aligned) / len(aligned)

    buckets = defaultdict(lambda: {"picks": 0, "hits": 0})
    for r in rows:
        b = _bucket_from_conf(r.get("confidence"))
        buckets[b]["picks"] += 1
        buckets[b]["hits"] += int(r["hit"])

    bucket_rows = []
    for b, vals in sorted(buckets.items(), key=lambda x: x[0]):
        bucket_rows.append(
            {
                "bucket": b,
                "picks": vals["picks"],
                "hits": vals["hits"],
                "accuracy": (vals["hits"] / vals["picks"]) if vals["picks"] else 0.0,
            }
        )

    suspicious = bool(picks >= 40 and acc >= 0.85)

    priced_rows = [r for r in rows if r.get("return_per_unit") is not None]
    priced_picks = len(priced_rows)
    priced_coverage_pct = (priced_picks / picks) if picks > 0 else 0.0
    total_return_units = float(sum(r.get("return_per_unit", 0.0) for r in priced_rows)) if priced_rows else 0.0
    roi_per_bet = (total_return_units / priced_picks) if priced_picks > 0 else None
    yield_pct = (roi_per_bet * 100.0) if roi_per_bet is not None else None

    max_drawdown_units = None
    if priced_rows:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for r in priced_rows:
            equity += float(r.get("return_per_unit") or 0.0)
            peak = max(peak, equity)
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        max_drawdown_units = max_dd

    return {
        "picks": picks,
        "hits": hits,
        "accuracy": acc,
        "brier": brier,
        "logloss": logloss,
        "priced_picks": priced_picks,
        "priced_coverage_pct": priced_coverage_pct,
        "total_return_units": total_return_units,
        "roi_per_bet": roi_per_bet,
        "yield_pct": yield_pct,
        "max_drawdown_units": max_drawdown_units,
        "suspicious_high_accuracy": suspicious,
        "buckets": bucket_rows,
    }


def run_unified_audit():
    result = {
        "generated_at": datetime.now().isoformat(),
        "sports": {},
    }

    flat_rows = []

    for sport, cfg in SPORTS.items():
        hist_dir = cfg["historical_dir"]
        if not hist_dir.exists():
            result["sports"][sport] = {"markets": {}, "files": 0, "events": 0}
            continue

        lookup = _build_lookup(cfg["raw_file"], cfg["raw_type"])
        files = sorted([p for p in hist_dir.glob("*.json") if p.name != "_summary.json"])

        market_rows = defaultdict(list)
        events_count = 0

        for file_path in files:
            try:
                events = _read_json_events(file_path)
            except Exception:
                continue

            events = apply_overrides_to_events(sport, str(file_path.stem), events)

            for event in events:
                events_count += 1
                for row in _extract_market_rows(sport, event, lookup):
                    market_rows[row["market"]].append(row)

        market_summary = {}
        for market, rows in market_rows.items():
            summary = _summarize_rows(rows)
            market_summary[market] = summary
            flat_rows.append(
                {
                    "sport": sport,
                    "market": market,
                    "picks": summary.get("picks", 0),
                    "hits": summary.get("hits", 0),
                    "accuracy": summary.get("accuracy", 0.0),
                    "brier": summary.get("brier"),
                    "logloss": summary.get("logloss"),
                    "priced_picks": summary.get("priced_picks", 0),
                    "priced_coverage_pct": summary.get("priced_coverage_pct", 0.0),
                    "total_return_units": summary.get("total_return_units"),
                    "roi_per_bet": summary.get("roi_per_bet"),
                    "yield_pct": summary.get("yield_pct"),
                    "max_drawdown_units": summary.get("max_drawdown_units"),
                    "suspicious_high_accuracy": bool(summary.get("suspicious_high_accuracy", False)),
                }
            )

        result["sports"][sport] = {
            "files": len(files),
            "events": events_count,
            "markets": market_summary,
        }

    out_json = REPORTS_DIR / "unified_backtest_audit.json"
    out_csv = REPORTS_DIR / "unified_backtest_audit.csv"

    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)

    print(f"[OK] Unified audit JSON: {out_json}")
    print(f"[OK] Unified audit CSV : {out_csv}")


if __name__ == "__main__":
    run_unified_audit()
