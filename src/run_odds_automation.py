from __future__ import annotations

import os

from build_odds_overrides_template import build_template
from build_odds_bulk_entry_sheet import build_bulk_entry_sheet
from export_priority_odds_subset import run_export
from scrape_espn_odds_lines_all_sports import scrape_and_apply
from scrape_flashscore_lines_basketball import scrape_and_apply as scrape_flashscore_basketball
from scrape_flashscore_lines_baseball import scrape_and_apply as scrape_flashscore
from scrape_flashscore_lines_hockey import scrape_and_apply as scrape_flashscore_hockey
from scrape_flashscore_lines_soccer import scrape_and_apply as scrape_flashscore_soccer
from scrape_theoddsapi_lines_all_sports import scrape_and_apply as scrape_theoddsapi
from validate_odds_overrides import run_validation


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = str(os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int, low: int, high: int) -> int:
    raw = str(os.getenv(name, str(default)) or "").strip()
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(int(low), min(int(high), value))


def run_all(days_ahead_template: int = 21, days_ahead_priority: int = 10, priority_top_n: int = 600, bulk_limit_rows: int = 2500):
    print("[STEP] build template from predictions")
    build_template(days_ahead=days_ahead_template)

    print("[STEP] scrape ESPN odds + lines and apply to overrides")
    scrape_and_apply(days_ahead=days_ahead_priority, max_workers=8)

    if _env_enabled("FLASHSCORE_ENABLED", default=True):
        print("[STEP] scrape Flashscore basketball odds and apply to overrides")
        scrape_flashscore_basketball(days_ahead=3)

        print("[STEP] scrape Flashscore hockey odds and apply to overrides")
        scrape_flashscore_hockey(days_ahead=3)

        print("[STEP] scrape Flashscore baseball odds and apply to overrides")
        scrape_flashscore(days_ahead=2)

        print("[STEP] scrape Flashscore soccer odds and apply to overrides")
        scrape_flashscore_soccer(days_ahead=days_ahead_priority)
    else:
        print("[SKIP] FLASHSCORE_ENABLED is off; skipping Flashscore scrapers")

    if str(os.getenv("THEODDSAPI_KEY") or "").strip():
        print("[STEP] scrape TheOddsAPI odds + lines and apply to overrides")
        theodds_days = _env_int("THEODDSAPI_DAYS_AHEAD", default=min(int(days_ahead_priority), 3), low=1, high=7)
        try:
            scrape_theoddsapi(days_ahead=theodds_days)
        except Exception as exc:
            print(f"[WARN] TheOddsAPI step failed or timed out: {exc}")
            print("[WARN] Continuing pipeline with ESPN/Flashscore data")
    else:
        print("[SKIP] THEODDSAPI_KEY not set; skipping TheOddsAPI scraper")

    print("[STEP] validate overrides")
    run_validation()

    print("[STEP] build bulk entry sheet")
    build_bulk_entry_sheet(limit_rows=bulk_limit_rows)

    print("[STEP] export priority subsets")
    run_export(days_ahead=days_ahead_priority, top_n=priority_top_n)

    print("[OK] odds automation pipeline completed")


if __name__ == "__main__":
    run_all()
