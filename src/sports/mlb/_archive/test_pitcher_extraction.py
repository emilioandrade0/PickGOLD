"""
Lightweight tests for pitcher extraction logic (pure-Python replicas).

These tests avoid importing the full `data_ingest_mlb` module to keep
dependencies minimal during quick validation runs.
"""

import re


def normalize_pitcher_name(name: str) -> str:
    s = str(name) if name is not None else ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def extract_probable_pitchers(comp: dict):
    result = {
        "home_starting_pitcher": "",
        "away_starting_pitcher": "",
        "home_starting_pitcher_id": "",
        "away_starting_pitcher_id": "",
        "pitcher_source": "",
    }
    probables = comp.get("probables", [])
    if isinstance(probables, list) and probables:
        for p in probables:
            athlete = p.get("athlete") or {}
            home_away = (p.get("homeAway") or "").lower()
            pitcher_name = normalize_pitcher_name(athlete.get("displayName") or athlete.get("shortName") or p.get("displayName") or p.get("name") or "")
            pitcher_id = str(athlete.get("id") or p.get("id") or "")
            if home_away == "home":
                result["home_starting_pitcher"] = pitcher_name
                result["home_starting_pitcher_id"] = pitcher_id
                result["pitcher_source"] = "competition_probables"
            elif home_away == "away":
                result["away_starting_pitcher"] = pitcher_name
                result["away_starting_pitcher_id"] = pitcher_id
                result["pitcher_source"] = "competition_probables"
    return result


def _extract_pitchers_from_mlb_payload(payload: dict, home_abbr: str, away_abbr: str):
    out = {"home_starting_pitcher": "", "away_starting_pitcher": "", "home_starting_pitcher_id": "", "away_starting_pitcher_id": "", "pitcher_data_available": 0}
    if not isinstance(payload, dict):
        return out
    mlb = payload.get("mlb_boxscore") if isinstance(payload.get("mlb_boxscore"), dict) else payload

    # probablePitchers style
    prob = None
    for key in ("probablePitchers", "probables", "probablesList", "probables"):
        if key in mlb and isinstance(mlb[key], list):
            prob = mlb[key]
            break
    if prob:
        for item in prob:
            name = item.get("fullName") or item.get("displayName") or item.get("name") or ""
            team = ""
            t = item.get("team")
            if isinstance(t, dict):
                team = t.get("abbreviation", "")
            elif isinstance(t, str):
                team = t
            pid = str(item.get("id") or item.get("playerId") or "")
            if team.upper() == home_abbr.upper():
                out["home_starting_pitcher"] = normalize_pitcher_name(name)
                out["home_starting_pitcher_id"] = pid
            if team.upper() == away_abbr.upper():
                out["away_starting_pitcher"] = normalize_pitcher_name(name)
                out["away_starting_pitcher_id"] = pid

    if out["home_starting_pitcher"] and out["away_starting_pitcher"]:
        out["pitcher_data_available"] = 1

    return out


def test_extract_probable_pitchers():
    comp = {
        "probables": [
            {"athlete": {"displayName": "John Doe", "id": "123"}, "homeAway": "home", "team": {"abbreviation": "NYM", "id": "1"}},
            {"athlete": {"displayName": "Bob Roe", "id": "456"}, "homeAway": "away", "team": {"abbreviation": "BOS", "id": "2"}},
        ]
    }
    out = extract_probable_pitchers(comp)
    assert out["home_starting_pitcher"] == "John Doe"
    assert out["away_starting_pitcher"] == "Bob Roe"


def test_external_payload_extractor():
    payload = {"mlb_boxscore": {"probablePitchers": [{"fullName": "EP Home", "id": "200", "team": {"abbreviation": "NYM"}}, {"fullName": "EP Away", "id": "201", "team": {"abbreviation": "BOS"}}]}}
    out = _extract_pitchers_from_mlb_payload(payload, home_abbr="NYM", away_abbr="BOS")
    assert out["home_starting_pitcher"] == "EP Home"
    assert out["away_starting_pitcher"] == "EP Away"


def run():
    tests = [test_extract_probable_pitchers, test_external_payload_extractor]
    failures = 0
    for t in tests:
        try:
            t()
            print(f"OK: {t.__name__}")
        except AssertionError as e:
            failures += 1
            print(f"FAIL: {t.__name__} -> {e}")
    if failures:
        raise SystemExit(f"{failures} tests failed")
    print("All tests passed")


if __name__ == "__main__":
    run()
