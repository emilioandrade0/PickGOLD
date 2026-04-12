import json
import os
from collections import defaultdict
from pathlib import Path
import sys

SPORTS_ROOT = Path(__file__).resolve().parent.parent
if str(SPORTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SPORTS_ROOT))

from evaluate_baseline_common import evaluate_for_sport


SRC_ROOT = Path(__file__).resolve().parents[2]


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


NHL_FULL_GAME_PUBLISH_RULE = os.getenv("NHL_FULL_GAME_PUBLISH_RULE", "elite_or_strong").strip().lower()
NHL_FULL_GAME_PUBLISH_MIN_META_CONF = _env_int("NHL_FULL_GAME_PUBLISH_MIN_META_CONF", 62)
NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED = _env_bool("NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED", True)
NHL_FULL_GAME_HYBRID_V2_BUCKETS = {
    x.strip().upper()
    for x in os.getenv("NHL_FULL_GAME_HYBRID_V2_BUCKETS", "ELITE,STRONG,NORMAL").split(",")
    if x.strip()
}
NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED = _env_bool("NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED", False)


def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
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


def _safe_prob(value):
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out > 1.0:
        out = out / 100.0
    if out < 0.0 or out > 1.0:
        return None
    return out


def _normalize_bucket(value):
    bucket = str(value or "PASS").strip().upper()
    if bucket == "LOW":
        return "PASS"
    if bucket in {"ELITE", "STRONG", "NORMAL", "PASS"}:
        return bucket
    return "PASS"


def _resolve_meta_score(row):
    score = _safe_prob(row.get("full_game_meta_score"))
    if score is not None:
        return score
    score = _safe_prob(row.get("full_game_meta_confidence"))
    if score is not None:
        return score
    return _safe_prob(row.get("full_game_confidence"))


def _resolve_meta_range(meta_score):
    if meta_score is None:
        return "missing"
    if meta_score >= 0.66:
        return ">=0.66"
    if meta_score >= 0.60:
        return "0.60-0.66"
    if meta_score >= 0.55:
        return "0.55-0.60"
    return "<0.55"


def _infer_publish_from_rule(row):
    explicit = _to_bool(row.get("publish_full_game"))
    if explicit is not None:
        return explicit, "json_field"

    bucket = _normalize_bucket(row.get("full_game_meta_bucket"))
    alignment = str(row.get("market_ml_alignment", "neutral") or "neutral").strip().lower()
    if alignment not in {"aligned", "neutral", "conflicted"}:
        alignment = "neutral"

    if alignment == "conflicted" and NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED:
        return False, "rule_blocked_market_conflicted"

    meta_conf = row.get("full_game_meta_confidence", row.get("full_game_confidence", 0))
    try:
        meta_conf_int = int(float(meta_conf))
    except Exception:
        meta_conf_int = 0

    if NHL_FULL_GAME_PUBLISH_RULE == "elite_only":
        return bucket == "ELITE", "rule_fallback_elite_only"

    if NHL_FULL_GAME_PUBLISH_RULE == "elite_or_strong_min_conf":
        return (
            bucket in {"ELITE", "STRONG"} and meta_conf_int >= NHL_FULL_GAME_PUBLISH_MIN_META_CONF,
            "rule_fallback_elite_or_strong_min_conf",
        )

    return bucket in {"ELITE", "STRONG"}, "rule_fallback_elite_or_strong"


def _pct(hits, total):
    return (100.0 * hits / total) if total > 0 else None


def _print_extra_nhl_full_game_report():
    pred_dir = _first_existing(
        [
            SRC_ROOT / "data" / "nhl" / "historical_predictions",
            SRC_ROOT / "data" / "nhl" / "predictions",
        ]
    )
    if pred_dir is None:
        print("No se encontro carpeta de predicciones NHL para bloque meta extra.")
        return

    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        print("No hay JSONs NHL para bloque meta extra.")
        return

    global_hits = 0
    global_total = 0
    v2_global_hits = 0
    v2_global_total = 0
    hybrid_global_hits = 0
    hybrid_global_total = 0
    bucket_stats = defaultdict(lambda: [0, 0])
    v2_bucket_stats = defaultdict(lambda: [0, 0])
    hybrid_bucket_stats = defaultdict(lambda: [0, 0])
    alignment_stats = defaultdict(lambda: [0, 0])
    publish_stats = {True: [0, 0], False: [0, 0]}
    score_range_stats = defaultdict(lambda: [0, 0])
    top_subset_stats = {
        "elite_only": [0, 0],
        "elite_or_strong_non_conflicted": [0, 0],
    }

    for jf in json_files:
        for row in _safe_rows_from_json(jf):
            if not isinstance(row, dict):
                continue

            ml_hit = _to_bool(row.get("moneyline_correct") if "moneyline_correct" in row else row.get("correct"))
            if ml_hit is None:
                continue

            ml_hit_int = int(ml_hit)
            global_hits += ml_hit_int
            global_total += 1

            v2_hit = _to_bool(row.get("full_game_v2_hit"))
            if v2_hit is None:
                v2_pick = str(row.get("full_game_v2_pick") or "").strip().upper()
                v2_actual = str(row.get("moneyline_actual", row.get("actual_winner", "")) or "").strip().upper()
                if v2_pick and v2_actual:
                    v2_hit = (v2_pick == v2_actual)
            if v2_hit is not None:
                v2_hit_int = int(v2_hit)
                v2_global_hits += v2_hit_int
                v2_global_total += 1

                v2_bucket = _normalize_bucket(row.get("full_game_v2_bucket"))
                v2_bucket_stats[v2_bucket][0] += v2_hit_int
                v2_bucket_stats[v2_bucket][1] += 1

            hybrid_hit = _to_bool(row.get("full_game_hybrid_hit"))
            if hybrid_hit is None:
                hybrid_use_v2 = _to_bool(row.get("full_game_hybrid_use_v2"))
                if hybrid_use_v2 is None:
                    v2_bucket = _normalize_bucket(row.get("full_game_v2_bucket"))
                    alignment = str(row.get("market_ml_alignment", "neutral") or "neutral").strip().lower()
                    hybrid_use_v2 = v2_bucket in NHL_FULL_GAME_HYBRID_V2_BUCKETS
                    if NHL_FULL_GAME_HYBRID_BLOCK_CONFLICTED and alignment == "conflicted":
                        hybrid_use_v2 = False
                if hybrid_use_v2 and v2_hit is not None:
                    hybrid_hit = v2_hit
                else:
                    hybrid_hit = ml_hit

            if hybrid_hit is not None:
                hybrid_hit_int = int(hybrid_hit)
                hybrid_global_hits += hybrid_hit_int
                hybrid_global_total += 1
                hybrid_bucket = _normalize_bucket(row.get("full_game_v2_bucket")) if _to_bool(row.get("full_game_hybrid_use_v2")) else _normalize_bucket(row.get("full_game_meta_bucket"))
                hybrid_bucket_stats[hybrid_bucket][0] += hybrid_hit_int
                hybrid_bucket_stats[hybrid_bucket][1] += 1

            bucket = _normalize_bucket(row.get("full_game_meta_bucket"))
            bucket_stats[bucket][0] += ml_hit_int
            bucket_stats[bucket][1] += 1

            alignment = str(row.get("market_ml_alignment", "neutral") or "neutral").strip().lower()
            if alignment not in {"aligned", "neutral", "conflicted"}:
                alignment = "neutral"
            alignment_stats[alignment][0] += ml_hit_int
            alignment_stats[alignment][1] += 1

            meta_score = _resolve_meta_score(row)
            score_range = _resolve_meta_range(meta_score)
            score_range_stats[score_range][0] += ml_hit_int
            score_range_stats[score_range][1] += 1

            publish_flag, publish_reason = _infer_publish_from_rule(row)
            _ = publish_reason
            publish_stats[publish_flag][0] += ml_hit_int
            publish_stats[publish_flag][1] += 1

            if bucket == "ELITE":
                top_subset_stats["elite_only"][0] += ml_hit_int
                top_subset_stats["elite_only"][1] += 1

            if bucket in {"ELITE", "STRONG"} and alignment != "conflicted":
                top_subset_stats["elite_or_strong_non_conflicted"][0] += ml_hit_int
                top_subset_stats["elite_or_strong_non_conflicted"][1] += 1

    if global_total == 0:
        print("No hay juegos evaluables de moneyline en bloque meta extra.")
        return

    global_acc = _pct(global_hits, global_total)
    v2_global_acc = _pct(v2_global_hits, v2_global_total)
    hybrid_global_acc = _pct(hybrid_global_hits, hybrid_global_total)
    pub_h, pub_t = publish_stats[True]
    no_pub_h, no_pub_t = publish_stats[False]
    pub_acc = _pct(pub_h, pub_t)
    pub_cov = (100.0 * pub_t / global_total) if global_total > 0 else 0.0
    no_pub_acc = _pct(no_pub_h, no_pub_t)
    no_pub_cov = (100.0 * no_pub_t / global_total) if global_total > 0 else 0.0

    print("=" * 66)
    print("REPORTE EXTRA NHL FULL GAME META/PUBLICACION")
    print(
        "Regla publish fallback: "
        f"{NHL_FULL_GAME_PUBLISH_RULE} | min_conf={NHL_FULL_GAME_PUBLISH_MIN_META_CONF} "
        f"| exclude_conflicted={int(NHL_FULL_GAME_PUBLISH_EXCLUDE_CONFLICTED)}"
    )
    print("=" * 66)
    print(f"GLOBAL ACCURACY          : {global_acc:.2f}% ({global_hits}/{global_total})")
    if v2_global_acc is None:
        print("GLOBAL ACCURACY V2       : N/A (0/0)")
    else:
        print(f"GLOBAL ACCURACY V2       : {v2_global_acc:.2f}% ({v2_global_hits}/{v2_global_total})")
        print(f"DELTA GLOBAL V2-BASE     : {v2_global_acc - global_acc:+.2f} pp")
    if hybrid_global_acc is None:
        print("GLOBAL ACCURACY HYBRID   : N/A (0/0)")
    else:
        print(f"GLOBAL ACCURACY HYBRID   : {hybrid_global_acc:.2f}% ({hybrid_global_hits}/{hybrid_global_total})")
        print(f"DELTA GLOBAL HYBRID-BASE : {hybrid_global_acc - global_acc:+.2f} pp")
    if pub_acc is None:
        print("PUBLISHED ACCURACY       : N/A (0/0) | Cobertura 0.00%")
    else:
        print(f"PUBLISHED ACCURACY       : {pub_acc:.2f}% ({pub_h}/{pub_t}) | Cobertura {pub_cov:.2f}%")
    if no_pub_acc is None:
        print("NO PUBLISHED ACCURACY    : N/A (0/0) | Cobertura 0.00%")
    else:
        print(f"NO PUBLISHED ACCURACY    : {no_pub_acc:.2f}% ({no_pub_h}/{no_pub_t}) | Cobertura {no_pub_cov:.2f}%")
    if pub_acc is not None:
        print(f"DELTA PUBLISHED-GLOBAL   : {pub_acc - global_acc:+.2f} pp")

    print("-" * 66)
    print("FULL_GAME META BUCKETS")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        h, t = bucket_stats[bucket]
        p = _pct(h, t)
        cov = (100.0 * t / global_total) if global_total > 0 else 0.0
        if p is None:
            print(f"   {bucket.ljust(8)} : N/A | Cobertura {cov:.2f}%")
        else:
            print(f"   {bucket.ljust(8)} : {p:.2f}% ({h}/{t}) | Cobertura {cov:.2f}%")

    print("-" * 66)
    print("FULL_GAME V2 BUCKETS")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        h, t = v2_bucket_stats[bucket]
        p = _pct(h, t)
        cov = (100.0 * t / global_total) if global_total > 0 else 0.0
        if p is None:
            print(f"   {bucket.ljust(8)} : N/A | Cobertura {cov:.2f}%")
        else:
            print(f"   {bucket.ljust(8)} : {p:.2f}% ({h}/{t}) | Cobertura {cov:.2f}%")

    print("-" * 66)
    print("FULL_GAME HYBRID BUCKETS")
    for bucket in ["ELITE", "STRONG", "NORMAL", "PASS"]:
        h, t = hybrid_bucket_stats[bucket]
        p = _pct(h, t)
        cov = (100.0 * t / global_total) if global_total > 0 else 0.0
        if p is None:
            print(f"   {bucket.ljust(8)} : N/A | Cobertura {cov:.2f}%")
        else:
            print(f"   {bucket.ljust(8)} : {p:.2f}% ({h}/{t}) | Cobertura {cov:.2f}%")

    print("-" * 66)
    print("MARKET ML ALIGNMENT")
    for align in ["aligned", "neutral", "conflicted"]:
        h, t = alignment_stats[align]
        p = _pct(h, t)
        cov = (100.0 * t / global_total) if global_total > 0 else 0.0
        name = align.upper()
        if p is None:
            print(f"   {name.ljust(10)} : N/A | Cobertura {cov:.2f}%")
        else:
            print(f"   {name.ljust(10)} : {p:.2f}% ({h}/{t}) | Cobertura {cov:.2f}%")

    print("-" * 66)
    print("FULL_GAME META SCORE RANGES")
    for score_range in [">=0.66", "0.60-0.66", "0.55-0.60", "<0.55", "missing"]:
        h, t = score_range_stats[score_range]
        p = _pct(h, t)
        cov = (100.0 * t / global_total) if global_total > 0 else 0.0
        if p is None:
            print(f"   {score_range.ljust(10)} : N/A | Cobertura {cov:.2f}%")
        else:
            print(f"   {score_range.ljust(10)} : {p:.2f}% ({h}/{t}) | Cobertura {cov:.2f}%")

    print("-" * 66)
    print("SUBCONJUNTOS TOP")
    elite_h, elite_t = top_subset_stats["elite_only"]
    elite_p = _pct(elite_h, elite_t)
    elite_cov = (100.0 * elite_t / global_total) if global_total > 0 else 0.0
    if elite_p is None:
        print(f"   ELITE SOLO                 : N/A | Cobertura {elite_cov:.2f}%")
    else:
        print(f"   ELITE SOLO                 : {elite_p:.2f}% ({elite_h}/{elite_t}) | Cobertura {elite_cov:.2f}%")

    top_h, top_t = top_subset_stats["elite_or_strong_non_conflicted"]
    top_p = _pct(top_h, top_t)
    top_cov = (100.0 * top_t / global_total) if global_total > 0 else 0.0
    if top_p is None:
        print(f"   ELITE/STRONG NO CONFLICTED : N/A | Cobertura {top_cov:.2f}%")
    else:
        print(f"   ELITE/STRONG NO CONFLICTED : {top_p:.2f}% ({top_h}/{top_t}) | Cobertura {top_cov:.2f}%")
    print("=" * 66)


if __name__ == "__main__":
    evaluate_for_sport("nhl")
    _print_extra_nhl_full_game_report()
