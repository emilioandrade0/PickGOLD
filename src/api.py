from pathlib import Path
import json
import math
import os
from datetime import date, datetime, timedelta
from datetime import timedelta
import re
import uuid
import hashlib

import pandas as pd
import requests
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware

try:
    from .best_picks.daily import build_daily_best_picks
except ImportError:
    from best_picks.daily import build_daily_best_picks
try:
    from .weekday_insights.scoring import SportScoringConfig, build_weekday_scoring_summary
except ImportError:
    from weekday_insights.scoring import SportScoringConfig, build_weekday_scoring_summary
try:
    from .external_odds_overrides import apply_overrides_to_events
except ImportError:
    from external_odds_overrides import apply_overrides_to_events

BASE_DIR = Path(__file__).resolve().parent
NBA_RAW_HISTORY = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
MLB_RAW_HISTORY = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
NHL_RAW_HISTORY = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
LIGA_MX_RAW_HISTORY = BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv"
LALIGA_RAW_HISTORY = BASE_DIR / "data" / "laliga" / "raw" / "laliga_advanced_history.csv"
KBO_RAW_HISTORY = BASE_DIR / "data" / "kbo" / "raw" / "kbo_advanced_history.csv"
EUROLEAGUE_RAW_HISTORY = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv"
NCAA_BASEBALL_RAW_HISTORY = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_advanced_history.csv"
BEST_PICKS_SNAPSHOTS_DIR = BASE_DIR / "data" / "insights" / "best_picks"
BEST_PICKS_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PICKS_EXCLUDED_SPORTS = {"ncaa_baseball"}

ESPN_SCOREBOARD_URLS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "kbo": "https://site.api.espn.com/apis/site/v2/sports/baseball/kbo/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "liga_mx": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
    "laliga": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
}

SPORTS_CONFIG = {
    "nba": {
        "predictions_dir": BASE_DIR / "data" / "predictions",
        "historical_dir": BASE_DIR / "data" / "historical_predictions",
        "label": "NBA",
    },
    "mlb": {
        "predictions_dir": BASE_DIR / "data" / "mlb" / "predictions",
        "historical_dir": BASE_DIR / "data" / "mlb" / "historical_predictions",
        "label": "MLB",
    },
    "kbo": {
        "predictions_dir": BASE_DIR / "data" / "kbo" / "predictions",
        "historical_dir": BASE_DIR / "data" / "kbo" / "historical_predictions",
        "label": "KBO",
    },
    "nhl": {
        "predictions_dir": BASE_DIR / "data" / "nhl" / "predictions",
        "historical_dir": BASE_DIR / "data" / "nhl" / "historical_predictions",
        "label": "NHL",
    },
    "liga_mx": {
        "predictions_dir": BASE_DIR / "data" / "liga_mx" / "predictions",
        "historical_dir": BASE_DIR / "data" / "liga_mx" / "historical_predictions",
        "label": "Liga MX",
    },
    "laliga": {
        "predictions_dir": BASE_DIR / "data" / "laliga" / "predictions",
        "historical_dir": BASE_DIR / "data" / "laliga" / "historical_predictions",
        "label": "LaLiga EA Sports",
    },
    "euroleague": {
        "predictions_dir": BASE_DIR / "data" / "euroleague" / "predictions",
        "historical_dir": BASE_DIR / "data" / "euroleague" / "historical_predictions",
        "label": "EuroLeague",
    },
    "ncaa_baseball": {
        "predictions_dir": BASE_DIR / "data" / "ncaa_baseball" / "predictions",
        "historical_dir": BASE_DIR / "data" / "ncaa_baseball" / "historical_predictions",
        "label": "NCAA Baseball",
    },
}

app = FastAPI(title="NBA GOLD API")

frontend_origin_env = os.getenv("FRONTEND_ORIGIN", "*").strip()
if frontend_origin_env == "*":
    allowed_origins = ["*"]
    cors_allow_credentials = False
else:
    allowed_origins = [o.strip() for o in frontend_origin_env.split(",") if o.strip()]
    cors_allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "nba-gold-api",
    }


ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "emilio.andra.na@gmail.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")

users_store = [
    {
        "id": str(uuid.uuid4()),
        "name": "Admin",
        "email": ADMIN_EMAIL,
        "password": hashlib.sha256(ADMIN_PASSWORD.encode()).hexdigest(),
        "role": "admin",
        "status": "approved",
    }
]


@app.post("/api/register")
def register(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not name or not email or not password:
        return {"ok": False, "error": "Todos los campos son requeridos."}

    existing = next((u for u in users_store if u["email"] == email), None)
    if existing:
        return {"ok": False, "error": "El email ya está registrado."}

    role = "admin" if email == ADMIN_EMAIL else "user"
    status = "approved" if role == "admin" else "pending"

    user = {
        "id": str(uuid.uuid4()),
        "name": name,
        "email": email,
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "role": role,
        "status": status,
    }
    users_store.append(user)

    return {
        "ok": True,
        "user": {
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "status": user["status"],
        },
    }


@app.post("/api/login")
def login(payload: dict = Body(...)):
    email = str(payload.get("email", "")).strip().lower()
    password = hashlib.sha256(str(payload.get("password", "")).encode()).hexdigest()

    user = next(
        (u for u in users_store if u["email"] == email and u["password"] == password),
        None,
    )
    if not user:
        return {"ok": False, "error": "Credenciales incorrectas."}

    if user["status"] != "approved":
        return {
            "ok": True,
            "pending": True,
            "user": {
                "name": user["name"],
                "email": user["email"],
                "role": user["role"],
                "status": user["status"],
            },
            "message": "Usuario pendiente de aprobación.",
        }

    return {
        "ok": True,
        "pending": False,
        "user": {
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "status": user["status"],
        },
    }


@app.get("/api/pending-users")
def pending_users(admin_email: str):
    admin_email = (admin_email or "").strip().lower()
    admin = next(
        (u for u in users_store if u["email"] == admin_email and u["role"] == "admin"),
        None,
    )
    if not admin:
        return {"ok": False, "error": "No autorizado."}

    pending = [
        {"id": u["id"], "name": u["name"], "email": u["email"]}
        for u in users_store
        if u["status"] == "pending"
    ]
    return {"ok": True, "pending": pending}


@app.post("/api/approve-user")
def approve_user(payload: dict = Body(...)):
    admin_email = str(payload.get("admin_email", "")).strip().lower()
    user_id = str(payload.get("user_id", "")).strip()

    admin = next(
        (u for u in users_store if u["email"] == admin_email and u["role"] == "admin"),
        None,
    )
    if not admin:
        return {"ok": False, "error": "No autorizado."}

    user = next((u for u in users_store if u["id"] == user_id), None)
    if not user:
        return {"ok": False, "error": "Usuario no encontrado."}

    user["status"] = "approved"

    return {
        "ok": True,
        "user": {
            "name": user["name"],
            "email": user["email"],
            "role": user["role"],
            "status": user["status"],
        },
    }


def ensure_sport_exists(sport: str):
    if sport not in SPORTS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Deporte no soportado: {sport}")
    return SPORTS_CONFIG[sport]


def parse_date_str(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Formato de fecha inválido. Usa YYYY-MM-DD."
        ) from exc


def _kbo_source_date_from_local(date_str: str) -> str:
    # KBO source data is effectively one day ahead vs local display timezone.
    d = parse_date_str(date_str)
    return (d + timedelta(days=1)).strftime("%Y-%m-%d")


def _kbo_local_date_from_source(date_str: str) -> str:
    try:
        d = datetime.strptime(str(date_str), "%Y-%m-%d").date()
    except Exception:
        return str(date_str)
    return (d - timedelta(days=1)).strftime("%Y-%m-%d")


def _translate_event_dates_for_sport(sport: str, events: list):
    if sport != "kbo":
        return events

    translated = []
    for event in events:
        item = dict(event)
        if "date" in item:
            item["date"] = _kbo_local_date_from_source(item.get("date"))
        translated.append(item)
    return translated


def read_json_file(file_path: Path):
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"No existe el archivo: {file_path.name}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and isinstance(data.get("games"), list):
            return data["games"]

        if not isinstance(data, list):
            raise HTTPException(
                status_code=500,
                detail=f"El archivo {file_path.name} no contiene una lista válida de eventos."
            )

        return data

    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"JSON inválido en {file_path.name}: {str(e)}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error leyendo {file_path.name}: {str(e)}"
        ) from e


def _normalize_events_payload(events):
    if isinstance(events, list):
        return [e for e in events if isinstance(e, dict)]
    if isinstance(events, dict) and isinstance(events.get("games"), list):
        return [e for e in events.get("games") if isinstance(e, dict)]
    return []


def _sanitize_json_values(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_json_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_values(v) for v in value]
    return value


def evaluate_team_pick(pick: str, home_team: str, away_team: str, winner: str):
    if not pick:
        return None

    p = str(pick).strip().upper()
    home = str(home_team).strip().upper()
    away = str(away_team).strip().upper()
    w = str(winner).strip().upper()

    is_draw_winner = w in {"TIE", "DRAW", "EMPATE", "X"}

    if p in {"DRAW", "TIE", "EMPATE", "X"}:
        return is_draw_winner

    if is_draw_winner:
        return False

    if p in {"HOME WIN", "HOME", "LOCAL", "1"}:
        return home == w
    if p in {"AWAY WIN", "AWAY", "VISITOR", "2"}:
        return away == w
    if p in {home, away}:
        return p == w

    return None


def evaluate_mlb_q1_pick(pick: str, home_r1: int, away_r1: int):
    if not pick:
        return None

    p = str(pick).strip().upper()
    rfi = (int(home_r1) + int(away_r1)) > 0

    if p == "YRFI":
        return rfi
    if p == "NRFI":
        return not rfi

    return None


def build_results_lookup_for_sport(sport: str):
    if sport == "nba":
        file_path = NBA_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team",
            "home_pts_total", "away_pts_total", "home_q1", "away_q1",
        ]
    elif sport == "mlb":
        file_path = MLB_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team",
            "home_runs_total", "away_runs_total", "home_r1", "away_r1", "home_runs_f5", "away_runs_f5",
        ]
    elif sport == "kbo":
        file_path = KBO_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team",
            "home_runs_total", "away_runs_total", "home_r1", "away_r1", "home_runs_f5", "away_runs_f5",
        ]
    elif sport == "ncaa_baseball":
        file_path = NCAA_BASEBALL_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team",
            "home_runs_total", "away_runs_total", "home_r1", "away_r1", "home_runs_f5", "away_runs_f5",
        ]
    elif sport == "nhl":
        file_path = NHL_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team", "home_score", "away_score",
        ]
    elif sport == "liga_mx":
        file_path = LIGA_MX_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team", "home_score", "away_score", "home_corners", "away_corners", "total_corners",
        ]
    elif sport == "laliga":
        file_path = LALIGA_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team", "home_score", "away_score", "home_corners", "away_corners", "total_corners",
        ]
    elif sport == "euroleague":
        file_path = EUROLEAGUE_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team",
            "home_pts_total", "away_pts_total", "home_q1", "away_q1",
        ]
    else:
        return {}

    if not file_path.exists():
        return {}

    optional_status_cols = ["status_completed", "status_state", "completed"]
    try:
        header_cols = list(pd.read_csv(file_path, nrows=0).columns)
    except Exception:
        return {}

    selected_cols = [c for c in (use_cols + optional_status_cols) if c in header_cols]
    try:
        df = pd.read_csv(file_path, usecols=selected_cols)
    except Exception:
        return {}

    lookup = {}
    for _, row in df.iterrows():
        try:
            has_explicit_status = False
            # Skip games that are not completed/final in raw history to avoid false result matches.
            if "status_completed" in row and not pd.isna(row.get("status_completed")):
                has_explicit_status = True
                try:
                    if int(float(row.get("status_completed") or 0)) != 1:
                        continue
                except Exception:
                    pass
            if "completed" in row and not pd.isna(row.get("completed")):
                has_explicit_status = True
                try:
                    if int(float(row.get("completed") or 0)) != 1:
                        continue
                except Exception:
                    pass
            if "status_state" in row and not pd.isna(row.get("status_state")):
                state = str(row.get("status_state") or "").strip().lower()
                if state:
                    has_explicit_status = True
                if state and state not in {"post", "final", "completed"}:
                    continue

            game_id = str(row["game_id"])
            home_team = str(row["home_team"])
            away_team = str(row["away_team"])
            if sport in {"nba", "euroleague"}:
                home_score = int(row["home_pts_total"])
                away_score = int(row["away_pts_total"])
                home_q1_score = int(row["home_q1"])
                away_q1_score = int(row["away_q1"])
            elif sport in {"mlb", "kbo", "ncaa_baseball"}:
                home_score = int(row["home_runs_total"])
                away_score = int(row["away_runs_total"])
                home_q1_score = int(row["home_r1"])
                away_q1_score = int(row["away_r1"])
                home_f5_score = int(row["home_runs_f5"])
                away_f5_score = int(row["away_runs_f5"])
            else:
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                home_q1_score = None
                away_q1_score = None
                home_f5_score = None
                away_f5_score = None

            # Defensive guard: unresolved fixtures often appear as 0-0 in some raw feeds.
            if (not has_explicit_status) and home_score == 0 and away_score == 0:
                continue

            home_corners = None
            away_corners = None
            total_corners = None
            if sport in {"liga_mx", "laliga"}:
                home_corners = int(row["home_corners"]) if "home_corners" in row and not pd.isna(row["home_corners"]) else None
                away_corners = int(row["away_corners"]) if "away_corners" in row and not pd.isna(row["away_corners"]) else None
                total_corners = int(row["total_corners"]) if "total_corners" in row and not pd.isna(row["total_corners"]) else None

            if home_score > away_score:
                full_game_winner = home_team
            elif away_score > home_score:
                full_game_winner = away_team
            else:
                full_game_winner = "TIE"

            q1_winner = None
            if home_q1_score is not None and away_q1_score is not None:
                if home_q1_score > away_q1_score:
                    q1_winner = home_team
                elif away_q1_score > home_q1_score:
                    q1_winner = away_team
                else:
                    q1_winner = "TIE"

            lookup[game_id] = {
                "date": str(row["date"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_q1_score": home_q1_score,
                "away_q1_score": away_q1_score,
                "home_f5_score": home_f5_score,
                "away_f5_score": away_f5_score,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
                "full_game_winner": full_game_winner,
                "q1_winner": q1_winner,
            }
        except Exception:
            continue

    return lookup


def enrich_predictions_with_results(sport: str, events: list, lookup: dict | None = None, allow_live: bool = True):
    def _target_date_from_events(items: list[dict]):
        if not items:
            return None
        first = str((items[0] or {}).get("date", "") or "")[:10]
        try:
            datetime.strptime(first, "%Y-%m-%d")
            return first
        except Exception:
            return None

    def _fetch_live_lookup_espn(target_sport: str, target_date: str):
        base_url = ESPN_SCOREBOARD_URLS.get(target_sport)
        if not base_url:
            return {}
        try:
            yyyymmdd = datetime.strptime(target_date, "%Y-%m-%d").strftime("%Y%m%d")
            url = f"{base_url}?dates={yyyymmdd}&limit=500"
            payload = requests.get(
                url,
                timeout=20,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; NBA-GOLD/1.0)",
                    "Accept": "application/json",
                },
            ).json() or {}
        except Exception:
            return {}

        out = {}
        for event in payload.get("events") or []:
            try:
                game_id = str(event.get("id") or "").strip()
                comp = (event.get("competitions") or [{}])[0]
                competitors = comp.get("competitors") or []
                home = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "home"), None)
                away = next((c for c in competitors if str(c.get("homeAway", "")).lower() == "away"), None)
                if not game_id or not home or not away:
                    continue

                status = comp.get("status") or event.get("status") or {}
                stype = status.get("type") or {}
                state = str(stype.get("state") or "").strip().lower()
                if state not in {"pre", "in", "post"}:
                    state = "pre"

                out[game_id] = {
                    "status_state": state,
                    "status_description": str(stype.get("description") or status.get("displayClock") or "").strip(),
                    "status_detail": str(stype.get("shortDetail") or stype.get("detail") or "").strip(),
                    "status_completed": 1 if bool(stype.get("completed")) else 0,
                    "home_score": int(float(home.get("score", 0) or 0)),
                    "away_score": int(float(away.get("score", 0) or 0)),
                    "home_q1_score": _to_int_or_none((home.get("linescores") or [{}])[0].get("value")) if isinstance(home.get("linescores"), list) and len(home.get("linescores") or []) > 0 else None,
                    "away_q1_score": _to_int_or_none((away.get("linescores") or [{}])[0].get("value")) if isinstance(away.get("linescores"), list) and len(away.get("linescores") or []) > 0 else None,
                }
            except Exception:
                continue
        return out

    def _fetch_live_lookup_euroleague(items: list[dict], target_date: str):
        if not items:
            return {}

        out = {}
        for item in items:
            game_id = str(item.get("game_id") or "")
            m = re.match(r"^E(\d{4})-(\d+)$", game_id)
            if not m:
                continue
            season_code = m.group(1)
            gamecode = m.group(2)
            try:
                header_url = (
                    f"https://live.euroleague.net/api/Header?gamecode={gamecode}&seasoncode={season_code}"
                )
                header = requests.get(header_url, timeout=20).json() or {}
            except Exception:
                continue

            try:
                home_score = int(float(header.get("ScoreA", 0) or 0))
                away_score = int(float(header.get("ScoreB", 0) or 0))
                game_date = str(header.get("Date") or "")
                # Header date can come as dd/mm/yyyy.
                if "/" in game_date:
                    day, month, year = game_date.split("/")
                    game_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                completed = (home_score + away_score) > 0 and game_date <= target_date
                out[game_id] = {
                    "status_state": "post" if completed else "pre",
                    "status_description": "Final" if completed else "Scheduled",
                    "status_detail": "Final" if completed else "Scheduled",
                    "status_completed": 1 if completed else 0,
                    "home_score": home_score,
                    "away_score": away_score,
                }
            except Exception:
                continue

        return out

    def _fetch_live_lookup_ncaa(target_date: str):
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except Exception:
            return {}

        try:
            try:
                from . import data_ingest_ncaa_baseball as ncaa_ingest
            except Exception:
                import data_ingest_ncaa_baseball as ncaa_ingest

            cfg = ncaa_ingest._parse_drupal_scoreboard_settings(target_dt)
            contests = ncaa_ingest._fetch_contests_for_date(cfg, target_dt)
        except Exception:
            return {}

        out = {}
        for contest in contests or []:
            try:
                contest_id = str(int(contest.get("contestId") or 0))
                teams = contest.get("teams") or []
                if not contest_id or len(teams) < 2:
                    continue
                home = next((t for t in teams if bool(t.get("isHome"))), None)
                away = next((t for t in teams if not bool(t.get("isHome"))), None)
                if not home or not away:
                    continue

                state = str(contest.get("gameState") or "").upper()
                status_display = str(contest.get("statusCodeDisplay") or "").lower()
                completed = state == "F" or status_display in {"final", "post"}
                in_progress = state == "I"

                out[contest_id] = {
                    "status_state": "post" if completed else ("in" if in_progress else "pre"),
                    "status_description": "Final" if completed else ("In Progress" if in_progress else "Scheduled"),
                    "status_detail": str(contest.get("finalMessage") or "").strip(),
                    "status_completed": 1 if completed else 0,
                    "home_score": int(float(home.get("score", 0) or 0)),
                    "away_score": int(float(away.get("score", 0) or 0)),
                }
            except Exception:
                continue
        return out

    def _fetch_live_lookup(target_sport: str, items: list[dict], target_date: str):
        # Render runs in UTC; allow a one-day tolerance to avoid timezone drift
        # where local "today" events can appear as yesterday/tomorrow on server date.
        try:
            target_dt = datetime.strptime(str(target_date), "%Y-%m-%d").date()
        except Exception:
            return {}

        if abs((target_dt - date.today()).days) > 1:
            return {}
        if target_sport in ESPN_SCOREBOARD_URLS:
            return _fetch_live_lookup_espn(target_sport, target_date)
        if target_sport == "euroleague":
            return _fetch_live_lookup_euroleague(items, target_date)
        if target_sport == "ncaa_baseball":
            return _fetch_live_lookup_ncaa(target_date)
        return {}

    target_date = _target_date_from_events(events)
    live_lookup = _fetch_live_lookup(sport, events, target_date) if (allow_live and target_date) else {}

    if lookup is None:
        lookup = build_results_lookup_for_sport(sport)
    if not lookup:
        lookup = {}

    enriched = []

    def _event_has_completed_result(item: dict) -> bool:
        completed = item.get("status_completed")
        if completed is not None:
            try:
                return int(completed) == 1
            except Exception:
                pass

        state = str(item.get("status_state", "") or "").strip().lower()
        if state in {"post", "final", "completed"}:
            return True

        # Historical prediction files may not include explicit status markers.
        item_date = str(item.get("date", "") or "")[:10]
        if item_date:
            try:
                return datetime.strptime(item_date, "%Y-%m-%d").date() < date.today()
            except Exception:
                return False

        return False

    def _to_int_or_none(value):
        try:
            return int(value)
        except Exception:
            return None

    def _same_event_date(item_date, result_date) -> bool:
        item_str = str(item_date or "")[:10]
        result_str = str(result_date or "")[:10]
        return bool(item_str) and item_str == result_str

    def _to_bool_or_none(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and value in (0, 1):
            return bool(int(value))
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "si", "acierto", "win", "won"}:
                return True
            if v in {"false", "0", "no", "fallo", "lose", "lost"}:
                return False
        return None

    def _extract_line_from_pick(pick_text: str):
        m = re.search(r"(\d+(?:\.\d+)?)", str(pick_text or ""))
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    def _resolve_pick_line(pick_text: str, fallback_line):
        line = _extract_line_from_pick(pick_text)
        if line is not None:
            return line
        try:
            fb = float(fallback_line)
            return fb if fb > 0 else None
        except Exception:
            return None

    def _evaluate_total_pick(pick_text: str, total_value: int, fallback_line):
        p = str(pick_text or "").strip().upper()
        if not p or p in {"PENDIENTE", "N/A", "NAN"}:
            return None
        line = _resolve_pick_line(p, fallback_line)
        if line is None:
            return None
        if "OVER" in p:
            return total_value > line
        if "UNDER" in p:
            return total_value < line
        return None

    def _is_pending_pick(value) -> bool:
        v = str(value or "").strip().upper()
        return (not v) or v in {"PENDIENTE", "N/A", "NAN", "RECONSTRUIDO", "PASS", "PASAR"}

    def _resolve_spread_context(item: dict):
        spread_market = str(item.get("spread_market", "") or "").strip().upper()
        if spread_market and spread_market not in {"NO LINE", "N/A", "NAN"}:
            return True
        for key in ("closing_spread_line", "home_spread", "spread_abs"):
            try:
                if abs(float(item.get(key, 0) or 0)) > 0:
                    return True
            except Exception:
                continue
        return False

    def _synthesize_missing_market_picks(item: dict):
        spread_pick = item.get("spread_pick")
        full_game_pick = str(item.get("full_game_pick") or "").strip()
        if _is_pending_pick(spread_pick) and full_game_pick and _resolve_spread_context(item):
            item["spread_pick"] = full_game_pick

    def _winner_from_score(home_team: str, away_team: str, home_score: int, away_score: int):
        if home_score > away_score:
            return home_team
        if away_score > home_score:
            return away_team
        return "TIE"

    def _evaluate_team_like_pick(pick: str, home_team: str, away_team: str, winner: str):
        pick_text = str(pick or "").strip()
        if not pick_text:
            return None

        direct = evaluate_team_pick(pick_text, home_team, away_team, winner)
        if direct is not None:
            return direct

        p = pick_text.upper()
        home = str(home_team or "").strip().upper()
        away = str(away_team or "").strip().upper()
        win = str(winner or "").strip().upper()

        if "HOME" in p or "LOCAL" in p:
            return win == home
        if "AWAY" in p or "VISITOR" in p or "VISITANTE" in p:
            return win == away
        if home and home in p:
            return win == home
        if away and away in p:
            return win == away
        return None

    def _evaluate_btts_pick(pick_text: str, home_score: int, away_score: int):
        p = str(pick_text or "").strip().upper()
        if not p:
            return None
        yes = home_score > 0 and away_score > 0
        if "YES" in p or "SI" in p:
            return yes
        if "NO" in p:
            return not yes
        return None

        def _hydrate_precomputed_result_flags(item: dict) -> bool:
        # Some historical files (notably NHL) already include hit flags but no final score payload.
        full_game_hit = _to_bool_or_none(item.get("full_game_hit"))
        if full_game_hit is None:
            full_game_hit = _to_bool_or_none(item.get("correct_full_game_adjusted"))
        if full_game_hit is None:
            full_game_hit = _to_bool_or_none(item.get("correct_full_game"))
        if full_game_hit is None:
            full_game_hit = _to_bool_or_none(item.get("correct_full_game_base"))
        if full_game_hit is not None:
            item["full_game_hit"] = full_game_hit

        spread_hit = _to_bool_or_none(item.get("correct_spread"))
        total_hit = _to_bool_or_none(item.get("correct_total_adjusted"))
        if total_hit is None:
            total_hit = _to_bool_or_none(item.get("correct_total"))
        q1_hit = _to_bool_or_none(item.get("q1_hit"))

        has_any_hit = any(v is not None for v in (full_game_hit, spread_hit, total_hit, q1_hit))
        if not has_any_hit:
            return False

        home_score = item.get("home_score")
        away_score = item.get("away_score")
        home_team = item.get("home_team")
        away_team = item.get("away_team")

        if (
            item.get("final_score_text") in (None, "", "N/A")
            and home_score is not None
            and away_score is not None
            and home_team
            and away_team
        ):
            item["final_score_text"] = f"{away_team} {away_score} - {home_team} {home_score}"

        item["result_available"] = True
        item["status_state"] = "post"
        item["status_description"] = item.get("status_description") or "Final"
        item["status_completed"] = 1
        return True

    def _apply_market_hit_flags(
        item: dict,
        home_team: str,
        away_team: str,
        home_score: int,
        away_score: int,
        home_q1_score,
        away_q1_score,
        home_f5_score=None,
        away_f5_score=None,
        total_corners=None,
    ):
        full_game_winner = _winner_from_score(home_team, away_team, home_score, away_score)
        item["full_game_result_winner"] = full_game_winner
        item["final_score_text"] = f"{away_team} {away_score} - {home_team} {home_score}"

        item["full_game_hit"] = evaluate_team_pick(
            pick=str(item.get("full_game_pick", "")),
            home_team=home_team,
            away_team=away_team,
            winner=full_game_winner,
        )

        existing_q1 = _to_bool_or_none(item.get("q1_hit"))
        if existing_q1 is None:
            if sport in {"mlb", "kbo", "ncaa_baseball"} and home_q1_score is not None and away_q1_score is not None:
                item["q1_hit"] = evaluate_mlb_q1_pick(
                    pick=str(item.get("q1_pick", "")),
                    home_r1=int(home_q1_score),
                    away_r1=int(away_q1_score),
                )
            elif sport in {"nba", "euroleague"} and home_q1_score is not None and away_q1_score is not None:
                q1_winner = _winner_from_score(home_team, away_team, int(home_q1_score), int(away_q1_score))
                item["q1_result_winner"] = q1_winner
                item["q1_hit"] = evaluate_team_pick(
                    pick=str(item.get("q1_pick", "")),
                    home_team=home_team,
                    away_team=away_team,
                    winner=q1_winner,
                )
            elif sport == "nhl" and home_q1_score is not None and away_q1_score is not None:
                q1_pick = str(item.get("q1_pick", "") or "").strip()
                q1_line = item.get("q1_line", 1.5)
                q1_hit = _evaluate_total_pick(q1_pick, int(home_q1_score) + int(away_q1_score), q1_line)
                if q1_hit is not None:
                    item["q1_hit"] = q1_hit

        if sport == "nhl":
            spread_pick = str(item.get("spread_pick") or "").strip()
            spread_pick_upper = spread_pick.upper()

            if _to_bool_or_none(item.get("correct_spread")) is None:
                if not spread_pick or spread_pick_upper in {"PENDIENTE", "N/A", "NAN", "RECONSTRUIDO"}:
                    item["correct_spread"] = None
                elif "OVER" in spread_pick_upper or "UNDER" in spread_pick_upper:
                    item["correct_spread"] = None
                else:
                    item["correct_spread"] = _evaluate_team_like_pick(
                        pick=spread_pick,
                        home_team=home_team,
                        away_team=away_team,
                        winner=full_game_winner,
                    )
        else:
            if _to_bool_or_none(item.get("correct_spread")) is None:
                spread_pick = str(item.get("spread_pick") or "").strip()
                if spread_pick and spread_pick.upper() not in {"PENDIENTE", "N/A", "NAN", "RECONSTRUIDO"}:
                    item["correct_spread"] = _evaluate_team_like_pick(
                        pick=spread_pick,
                        home_team=home_team,
                        away_team=away_team,
                        winner=full_game_winner,
                    )

        total_existing = _to_bool_or_none(item.get("correct_total_adjusted"))
        if total_existing is None:
            total_existing = _to_bool_or_none(item.get("correct_total"))
        if total_existing is None:
            total_pick = str(item.get("total_recommended_pick") or item.get("total_pick") or "").strip()

            total_line = item.get("total_line")
            if total_line in (None, "", "nan"):
                total_line = item.get("odds_over_under")
            if sport == "nhl" and total_line in (None, "", "nan", 0, 0.0):
                total_line = 5.5

            total_hit = _evaluate_total_pick(total_pick, home_score + away_score, total_line)
            if total_hit is not None:
                item["correct_total"] = total_hit

        btts_existing = _to_bool_or_none(item.get("correct_btts_adjusted"))
        if btts_existing is None:
            btts_existing = _to_bool_or_none(item.get("correct_btts"))
        if btts_existing is None:
            btts_pick = str(item.get("btts_recommended_pick") or item.get("btts_pick") or "").strip()
            btts_hit = _evaluate_btts_pick(btts_pick, home_score, away_score)
            if btts_hit is not None:
                item["correct_btts"] = btts_hit

        corners_existing = _to_bool_or_none(item.get("correct_corners_adjusted"))
        if corners_existing is None:
            corners_existing = _to_bool_or_none(item.get("correct_corners_base"))
        if corners_existing is None:
            corners_pick = str(item.get("corners_recommended_pick") or item.get("corners_pick") or "").strip()
            if corners_pick and total_corners is not None:
                corners_hit = _evaluate_total_pick(corners_pick, int(total_corners), item.get("corners_line"))
                if corners_hit is not None:
                    item["correct_corners_adjusted"] = corners_hit

        f5_existing = _to_bool_or_none(item.get("correct_home_win_f5"))
        if f5_existing is None:
            f5_existing = _to_bool_or_none(item.get("correct_f5"))
        if f5_existing is None and home_f5_score is not None and away_f5_score is not None:
            f5_pick = str(item.get("assists_pick") or item.get("f5_pick") or "").strip()
            if f5_pick:
                f5_winner = _winner_from_score(home_team, away_team, int(home_f5_score), int(away_f5_score))
                f5_hit = _evaluate_team_like_pick(
                    pick=f5_pick,
                    home_team=home_team,
                    away_team=away_team,
                    winner=f5_winner,
                )
                if f5_hit is not None:
                    item["correct_f5"] = f5_hit

        home_over_existing = _to_bool_or_none(item.get("correct_home_over"))
        if home_over_existing is None:
            home_over_existing = _to_bool_or_none(item.get("correct_home_total"))
        if home_over_existing is None:
            home_over_pick = str(item.get("home_over_pick") or "").strip()
            if home_over_pick:
                home_hit = _evaluate_total_pick(home_over_pick, int(home_score), None)
                if home_hit is not None:
                    item["correct_home_over"] = home_hit
                
    for event in events:
        item = dict(event)
        _synthesize_missing_market_picks(item)
        game_id = str(item.get("game_id", ""))

        live = live_lookup.get(game_id)
        if live:
            item["status_state"] = live.get("status_state")
            item["status_description"] = live.get("status_description")
            item["status_detail"] = live.get("status_detail")
            item["status_completed"] = int(live.get("status_completed", 0) or 0)
            item["home_score"] = live.get("home_score", item.get("home_score"))
            item["away_score"] = live.get("away_score", item.get("away_score"))

        result = lookup.get(game_id)

        # Guard against accidental game_id collisions across different dates.
        if result and not _same_event_date(item.get("date"), result.get("date")):
            result = None

        if not result:
            # Fallback for same-day finals not yet present in historical CSV.
            if _event_has_completed_result(item):
                home_team = str(item.get("home_team", "") or "")
                away_team = str(item.get("away_team", "") or "")
                home_score = _to_int_or_none(item.get("home_score"))
                away_score = _to_int_or_none(item.get("away_score"))
                home_q1_score = _to_int_or_none(item.get("home_q1_score"))
                away_q1_score = _to_int_or_none(item.get("away_q1_score"))

                if home_score is not None and away_score is not None and home_team and away_team:
                    item["result_available"] = True
                    item["home_score"] = home_score
                    item["away_score"] = away_score

                    if home_q1_score is not None:
                        item["home_q1_score"] = home_q1_score
                    if away_q1_score is not None:
                        item["away_q1_score"] = away_q1_score

                    if home_score > away_score:
                        full_game_winner = home_team
                    elif away_score > home_score:
                        full_game_winner = away_team
                    else:
                        full_game_winner = "TIE"

                    item["full_game_result_winner"] = full_game_winner
                    item["final_score_text"] = f"{away_team} {away_score} - {home_team} {home_score}"
                    _apply_market_hit_flags(
                        item=item,
                        home_team=home_team,
                        away_team=away_team,
                        home_score=home_score,
                        away_score=away_score,
                        home_q1_score=home_q1_score,
                        away_q1_score=away_q1_score,
                        home_f5_score=None,
                        away_f5_score=None,
                        total_corners=None,
                    )

                    enriched.append(item)
                    continue

            if _hydrate_precomputed_result_flags(item):
                enriched.append(item)
                continue

            item["result_available"] = False
            enriched.append(item)
            continue

        item["result_available"] = True
        item["status_state"] = "post"
        item["status_description"] = "Final"
        item["status_completed"] = 1
        item["home_score"] = result["home_score"]
        item["away_score"] = result["away_score"]
        if result["home_q1_score"] is not None:
            item["home_q1_score"] = result["home_q1_score"]
        if result["away_q1_score"] is not None:
            item["away_q1_score"] = result["away_q1_score"]
        item["full_game_result_winner"] = result["full_game_winner"]
        if result["q1_winner"] is not None:
            item["q1_result_winner"] = result["q1_winner"]
        item["final_score_text"] = (
            f"{result['away_team']} {result['away_score']} - {result['home_team']} {result['home_score']}"
        )

        _apply_market_hit_flags(
            item=item,
            home_team=result["home_team"],
            away_team=result["away_team"],
            home_score=result["home_score"],
            away_score=result["away_score"],
            home_q1_score=result.get("home_q1_score"),
            away_q1_score=result.get("away_q1_score"),
            home_f5_score=result.get("home_f5_score"),
            away_f5_score=result.get("away_f5_score"),
            total_corners=result.get("total_corners"),
        )

        enriched.append(item)

    return enriched


def enrich_predictions_if_available(sport: str, events: list, lookup: dict | None = None):
    events = _normalize_events_payload(events)
    if sport in {"nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "euroleague", "ncaa_baseball"}:
        return enrich_predictions_with_results(sport, events, lookup=lookup)
    return events


def _event_market_hits(event: dict) -> dict:
    """Extract normalized hit flags per market from heterogeneous event schemas."""
    def _to_bool_or_none(value):
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            if value in (0, 1):
                return bool(int(value))
            return None
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "1", "yes", "si", "acierto"}:
                return True
            if v in {"false", "0", "no", "fallo"}:
                return False
        return None

    markets = {}

    full_game_hit = _to_bool_or_none(event.get("correct_full_game_adjusted"))
    if full_game_hit is None:
        full_game_hit = _to_bool_or_none(event.get("correct_full_game"))
    if full_game_hit is None:
        full_game_hit = _to_bool_or_none(event.get("full_game_hit"))
    if full_game_hit is None:
        full_game_hit = _to_bool_or_none(event.get("correct_full_game_base"))
    if full_game_hit is not None:
        markets["full_game"] = full_game_hit

    q1_hit = _to_bool_or_none(event.get("q1_hit"))
    if q1_hit is not None:
        markets["q1_or_yrfi"] = q1_hit

    spread_hit = _to_bool_or_none(event.get("correct_spread"))
    if spread_hit is not None:
        markets["spread"] = spread_hit

    total_hit = _to_bool_or_none(event.get("correct_total_adjusted"))
    if total_hit is None:
        total_hit = _to_bool_or_none(event.get("correct_total"))
    if total_hit is not None:
        markets["total"] = total_hit

    btts_hit = _to_bool_or_none(event.get("correct_btts_adjusted"))
    if btts_hit is None:
        btts_hit = _to_bool_or_none(event.get("correct_btts"))
    if btts_hit is not None:
        markets["btts"] = btts_hit

    return markets


def _resolve_picked_team(event: dict):
    pick = str(event.get("recommended_pick") or event.get("full_game_pick") or "").strip()
    if not pick:
        return None

    home_team = str(event.get("home_team") or "").strip()
    away_team = str(event.get("away_team") or "").strip()

    p = pick.upper()
    if p in {"DRAW", "TIE", "EMPATE", "X"}:
        return None
    if p in {"HOME WIN", "HOME", "LOCAL", "1"}:
        return home_team or None
    if p in {"AWAY WIN", "AWAY", "VISITANTE", "2"}:
        return away_team or None
    if pick == home_team:
        return home_team
    if pick == away_team:
        return away_team
    return None


def build_sport_insights_summary(sport: str):
    config = ensure_sport_exists(sport)
    hist_dir = config["historical_dir"]

    if not hist_dir.exists():
        return {
            "sport": sport,
            "label": config["label"],
            "total_events": 0,
            "market_insights": [],
            "team_insights": [],
        }

    files = sorted(hist_dir.glob("*.json"))
    if not files:
        return {
            "sport": sport,
            "label": config["label"],
            "total_events": 0,
            "market_insights": [],
            "team_insights": [],
        }

    market_stats = {}
    team_stats = {}
    total_events = 0
    lookup = build_results_lookup_for_sport(sport)

    for file_path in files:
        try:
            events = read_json_file(file_path)
        except Exception:
            continue

        events = enrich_predictions_if_available(sport, events, lookup=lookup)

        for event in events:
            total_events += 1

            market_hits = _event_market_hits(event)
            for market_name, hit in market_hits.items():
                stat = market_stats.setdefault(market_name, {"picks": 0, "hits": 0})
                stat["picks"] += 1
                stat["hits"] += int(bool(hit))

            fg_hit = market_hits.get("full_game")
            picked_team = _resolve_picked_team(event)
            if fg_hit is not None and picked_team:
                tstat = team_stats.setdefault(picked_team, {"picks": 0, "hits": 0})
                tstat["picks"] += 1
                tstat["hits"] += int(bool(fg_hit))

    market_insights = []
    for name, stat in market_stats.items():
        picks = stat["picks"]
        if picks <= 0:
            continue
        acc = stat["hits"] / picks
        market_insights.append(
            {
                "market": name,
                "picks": picks,
                "hits": stat["hits"],
                "accuracy": acc,
            }
        )

    market_insights = sorted(market_insights, key=lambda x: (x["accuracy"], x["picks"]), reverse=True)

    team_insights = []
    for team, stat in team_stats.items():
        picks = stat["picks"]
        if picks < 8:
            continue
        acc = stat["hits"] / picks
        team_insights.append(
            {
                "team": team,
                "picks": picks,
                "hits": stat["hits"],
                "accuracy": acc,
            }
        )

    team_insights = sorted(team_insights, key=lambda x: (x["accuracy"], x["picks"]), reverse=True)[:15]

    return {
        "sport": sport,
        "label": config["label"],
        "total_events": total_events,
        "market_insights": market_insights,
        "team_insights": team_insights,
    }


def build_tier_performance_summary():
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "euroleague", "ncaa_baseball"]
    tracked_tiers = ["ELITE", "PREMIUM", "STRONG"]
    rows = []

    lookups = {sport: build_results_lookup_for_sport(sport) for sport in sports}

    for sport in sports:
        config = ensure_sport_exists(sport)
        hist_dir = config["historical_dir"]
        if not hist_dir.exists():
            continue

        files = sorted(hist_dir.glob("*.json"))
        for file_path in files:
            try:
                events = read_json_file(file_path)
            except Exception:
                continue

            for event in events:
                if not isinstance(event, dict):
                    continue

                tier = str(event.get("full_game_tier", "") or "").strip().upper()
                if tier not in tracked_tiers:
                    continue

                game_id = str(event.get("game_id", "") or "")
                result = lookups.get(sport, {}).get(game_id)
                if not result:
                    continue

                event_date = str(event.get("date", "") or "")[:10]
                result_date = str(result.get("date", "") or "")[:10]
                if not event_date or event_date != result_date:
                    continue

                pick = str(event.get("full_game_pick") or event.get("recommended_pick") or "").strip()
                hit = evaluate_team_pick(
                    pick=pick,
                    home_team=result.get("home_team", ""),
                    away_team=result.get("away_team", ""),
                    winner=result.get("full_game_winner", ""),
                )
                if hit is None:
                    continue

                rows.append(
                    {
                        "tier": tier,
                        "sport": sport,
                        "sport_label": config["label"],
                        "hit": bool(hit),
                    }
                )

    def _summarize(row_subset):
        n = len(row_subset)
        if n <= 0:
            return {
                "sample_size": 0,
                "hits": 0,
                "accuracy": 0.0,
                "error_rate": 0.0,
                "ci95_low": 0.0,
                "ci95_high": 0.0,
            }

        hits = sum(1 for r in row_subset if r["hit"])
        acc = hits / n
        err = 1.0 - acc
        moe = 1.96 * ((acc * (1.0 - acc) / n) ** 0.5)

        return {
            "sample_size": n,
            "hits": hits,
            "accuracy": acc,
            "error_rate": err,
            "ci95_low": max(0.0, acc - moe),
            "ci95_high": min(1.0, acc + moe),
        }

    tier_summary = []
    by_sport = []

    for tier in tracked_tiers:
        tier_rows = [r for r in rows if r["tier"] == tier]
        summary = _summarize(tier_rows)
        tier_summary.append(
            {
                "tier": tier,
                **summary,
            }
        )

        for sport in sports:
            sport_rows = [r for r in tier_rows if r["sport"] == sport]
            if not sport_rows:
                continue
            sport_summary = _summarize(sport_rows)
            by_sport.append(
                {
                    "tier": tier,
                    "sport": sport,
                    "label": SPORTS_CONFIG[sport]["label"],
                    **sport_summary,
                }
            )

    by_sport = sorted(by_sport, key=lambda x: (x["tier"], -x["sample_size"], x["sport"]))

    return {
        "generated_at": datetime.now().isoformat(),
        "tiers": tier_summary,
        "by_sport": by_sport,
    }


def _to_prob(value):
    try:
        v = float(value)
    except Exception:
        return None
    if v != v:
        return None
    if 0.0 <= v <= 1.0:
        return v
    if 1.0 < v <= 100.0:
        return v / 100.0
    return None


def _event_prob_for_market(event: dict, market: str):
    keys_by_market = {
        "full_game": [
            "full_game_calibrated_prob_home",
            "full_game_model_prob_home",
            "full_game_calibrated_prob_pick",
            "full_game_model_prob_pick",
            "full_game_confidence",
            "recommended_confidence",
        ],
        "q1_yrfi": ["q1_calibrated_prob_yes", "q1_model_prob_yes", "q1_calibrated_prob_home", "q1_confidence"],
        "spread": ["spread_calibrated_prob_pick", "spread_model_prob_pick", "spread_confidence"],
        "total": ["total_adjusted_probability", "total_base_probability", "total_confidence"],
        "btts": ["btts_adjusted_probability", "btts_base_probability", "btts_confidence"],
        "f5": ["extra_f5_calibrated_prob_home", "extra_f5_model_prob_home", "extra_f5_confidence"],
        "home_over": ["home_over_calibrated_prob_pick", "home_over_model_prob_pick", "home_over_confidence"],
        "corners": ["corners_model_prob_over", "corners_confidence"],
    }

    for key in keys_by_market.get(market, []):
        if key in event:
            p = _to_prob(event.get(key))
            if p is not None:
                return p
    return None


def _event_hit_for_market(event: dict, market: str):
    def _to_hit(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)) and v in (0, 1):
            return bool(int(v))
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"true", "1", "yes", "si", "acierto"}:
                return True
            if s in {"false", "0", "no", "fallo"}:
                return False
        return None

    keys_by_market = {
        "full_game": ["correct_full_game_adjusted", "correct_full_game", "full_game_hit", "correct_full_game_base"],
        "q1_yrfi": ["q1_hit"],
        "spread": ["correct_spread"],
        "total": ["correct_total_adjusted", "correct_total"],
        "btts": ["correct_btts_adjusted", "correct_btts"],
        "f5": ["correct_home_win_f5", "correct_f5"],
        "home_over": ["correct_home_over", "correct_home_total"],
        "corners": ["correct_corners_adjusted", "correct_corners_base", "correct_corners"],
    }

    for key in keys_by_market.get(market, []):
        if key in event:
            hit = _to_hit(event.get(key))
            if hit is not None:
                return hit

    # Never infer hit/miss from score unless the event is clearly completed.
    status_completed = event.get("status_completed")
    try:
        if status_completed is not None and int(float(status_completed)) != 1:
            return None
    except Exception:
        pass

    status_state = str(event.get("status_state") or "").strip().lower()
    if status_state and status_state not in {"post", "final", "completed"}:
        return None

    # Fallback: infer result directly from final score when explicit flags are absent.
    try:
        home_team = str(event.get("home_team") or "").strip()
        away_team = str(event.get("away_team") or "").strip()
        home_score = int(float(event.get("home_score") or 0))
        away_score = int(float(event.get("away_score") or 0))
    except Exception:
        return None

        if market == "full_game":
        pick = str(event.get("recommended_pick") or event.get("full_game_pick") or "").strip()
        if not pick:
            return None
        if home_score > away_score:
            winner = home_team
        elif away_score > home_score:
            winner = away_team
        else:
            winner = "TIE"
        return evaluate_team_pick(pick=pick, home_team=home_team, away_team=away_team, winner=winner)

    if market in {"spread", "total"}:
        if market == "total":
            pick_text = str(
                event.get("total_recommended_pick")
                or event.get("total_pick")
                or event.get("spread_pick")
                or ""
            ).strip()
        else:
            pick_text = str(event.get("spread_pick") or "").strip()

        if not pick_text:
            return None

        pick_upper = pick_text.upper()
        total_points = home_score + away_score

        if "OVER" in pick_upper:
            m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
            line = float(m.group(1)) if m else float(event.get("odds_over_under") or 0.0)
            if line <= 0:
                return None
            return total_points > line

        if "UNDER" in pick_upper:
            m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
            line = float(m.group(1)) if m else float(event.get("odds_over_under") or 0.0)
            if line <= 0:
                return None
            return total_points < line

        if market == "spread":
            if "OVER" in pick_upper or "UNDER" in pick_upper:
                return None

            if home_score > away_score:
                winner = home_team
            elif away_score > home_score:
                winner = away_team
            else:
                winner = "TIE"
            return evaluate_team_pick(
                pick=pick_text,
                home_team=home_team,
                away_team=away_team,
                winner=winner,
            )

        return None

    if market == "btts":
        pick_text = str(event.get("btts_recommended_pick") or event.get("btts_pick") or "").strip().upper()
        if not pick_text:
            return None
        btts_yes = home_score > 0 and away_score > 0
        if "YES" in pick_text:
            return btts_yes
        if "NO" in pick_text:
            return not btts_yes

    if market == "q1_yrfi":
        q1_pick = str(event.get("q1_pick") or "").strip()
        if not q1_pick:
            return None
        home_q1 = event.get("home_q1_score")
        away_q1 = event.get("away_q1_score")
        try:
            if home_q1 is not None and away_q1 is not None:
                return evaluate_mlb_q1_pick(q1_pick, int(home_q1), int(away_q1))
        except Exception:
            return None

    if market == "corners":
        pick_text = str(event.get("corners_recommended_pick") or event.get("corners_pick") or "").strip().upper()
        if not pick_text:
            return None

        total_corners = event.get("total_corners")
        if total_corners is None:
            return None
        try:
            total_corners = int(float(total_corners))
        except Exception:
            return None

        m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
        line = None
        if m:
            try:
                line = float(m.group(1))
            except Exception:
                line = None
        if line is None:
            try:
                line = float(event.get("corners_line") or 0.0)
            except Exception:
                line = 0.0
        if line <= 0:
            return None

        if "OVER" in pick_text:
            return total_corners > line
        if "UNDER" in pick_text:
            return total_corners < line

    return None


def _brier_score(probs: list[float], outcomes: list[int]):
    if not probs:
        return None
    return float(sum((p - y) ** 2 for p, y in zip(probs, outcomes)) / len(probs))


def _log_loss(probs: list[float], outcomes: list[int]):
    if not probs:
        return None
    eps = 1e-12
    total = 0.0
    for p, y in zip(probs, outcomes):
        p = min(1.0 - eps, max(eps, p))
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return float(total / len(probs))


def build_probability_calibration_profiles():
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "euroleague"]
    markets = ["full_game", "q1_yrfi", "spread", "total", "btts", "f5", "home_over", "corners"]

    grouped = {}
    lookups = {sport: build_results_lookup_for_sport(sport) for sport in sports}

    for sport in sports:
        config = ensure_sport_exists(sport)
        hist_dir = config["historical_dir"]
        if not hist_dir.exists():
            continue

        files = sorted(hist_dir.glob("*.json"))
        for file_path in files:
            try:
                events = read_json_file(file_path)
            except Exception:
                continue

            events = enrich_predictions_if_available(sport, events, lookup=lookups.get(sport))

            for event in events:
                if not isinstance(event, dict):
                    continue

                for market in markets:
                    prob = _event_prob_for_market(event, market)
                    hit = _event_hit_for_market(event, market)
                    if prob is None or hit is None:
                        continue
                    grouped.setdefault((sport, market), []).append((float(prob), int(bool(hit))))

    profiles = {}
    for (sport, market), rows in grouped.items():
        rows = [(max(0.001, min(0.999, p)), y) for p, y in rows]
        if len(rows) < 30:
            continue

        bins = [
            {
                "min": round(i / 10.0, 4),
                "max": round((i + 1) / 10.0, 4),
                "count": 0,
                "sum_pred": 0.0,
                "sum_hit": 0.0,
            }
            for i in range(10)
        ]

        probs = []
        outcomes = []
        for p, y in rows:
            idx = min(9, max(0, int(p * 10)))
            b = bins[idx]
            b["count"] += 1
            b["sum_pred"] += p
            b["sum_hit"] += y
            probs.append(p)
            outcomes.append(y)

        out_bins = []
        for b in bins:
            if b["count"] <= 0:
                continue
            out_bins.append(
                {
                    "min": b["min"],
                    "max": b["max"],
                    "count": int(b["count"]),
                    "mean_pred": round(b["sum_pred"] / b["count"], 4),
                    "mean_hit": round(b["sum_hit"] / b["count"], 4),
                }
            )

        profiles.setdefault(sport, {})[market] = {
            "sample_size": len(rows),
            "brier_score": round(_brier_score(probs, outcomes), 6),
            "log_loss": round(_log_loss(probs, outcomes), 6),
            "bins": out_bins,
        }

    return profiles


def get_files_for_date(predictions_dir: Path, historical_dir: Path, date_str: str):
    live_file = predictions_dir / f"{date_str}.json"
    hist_file = historical_dir / f"{date_str}.json"
    return live_file, hist_file


def get_today_file(predictions_dir: Path, historical_dir: Path):
    today_str = date.today().strftime("%Y-%m-%d")
    live_file, hist_file = get_files_for_date(predictions_dir, historical_dir, today_str)

    if live_file.exists():
        return live_file
    if hist_file.exists():
        return hist_file

    if not predictions_dir.exists():
        raise HTTPException(status_code=404, detail="No existe la carpeta de predicciones live")

    files = sorted(predictions_dir.glob("*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No hay predicciones guardadas")

    return files[-1]


def resolve_prediction_file(predictions_dir: Path, historical_dir: Path, date_str: str):
    selected_date = parse_date_str(date_str)
    today = date.today()
    live_file, hist_file = get_files_for_date(predictions_dir, historical_dir, date_str)

    # Render uses UTC; around midnight, local "today" can look like "yesterday" on server.
    # For near-today dates, prefer live board first to avoid serving stale historical snapshots.
    if selected_date >= (today - timedelta(days=1)):
        if live_file.exists():
            return live_file
        if hist_file.exists():
            return hist_file

    # Pasado => prioriza histórico.
    if selected_date < today:
        if hist_file.exists():
            return hist_file
        if live_file.exists():
            return live_file

    # Hoy o futuro => prioriza predicciones live.
    else:
        if live_file.exists():
            return live_file
        if hist_file.exists():
            return hist_file

    raise HTTPException(
        status_code=404,
        detail=f"No hay predicciones para la fecha {date_str}"
    )


def merge_result_hints_from_historical(events: list, historical_file: Path):
    if not historical_file.exists():
        return events

    historical_events = _normalize_events_payload(read_json_file(historical_file))
    if not historical_events:
        return events

    by_game_id = {
        str(item.get("game_id", "")).strip(): item
        for item in historical_events
        if str(item.get("game_id", "")).strip()
    }

    if not by_game_id:
        return events

    result_keys = [
        "correct_full_game",
        "correct_full_game_adjusted",
        "correct_full_game_base",
        "full_game_hit",
        "correct_spread",
        "correct_total",
        "correct_total_adjusted",
        "q1_hit",
        "actual_result",
        "final_score_text",
    ]

    merged = []
    for event in events:
        item = dict(event)
        game_id = str(item.get("game_id", "")).strip()
        hist = by_game_id.get(game_id)
        if hist:
            for key in result_keys:
                if item.get(key) is None and hist.get(key) is not None:
                    item[key] = hist.get(key)
        merged.append(item)

    return merged


def _best_picks_normalize_ranking_mode(ranking_mode: str | None):
    mode = str(ranking_mode or "balanced").strip().lower()
    if mode not in {"balanced", "best_hit_rate", "best_ev_real_only", "meta"}:
        return "balanced"
    return mode


def _best_picks_snapshot_file(date_str: str, ranking_mode: str = "balanced"):
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    if mode == "balanced":
        return BEST_PICKS_SNAPSHOTS_DIR / f"{date_str}.json"
    return BEST_PICKS_SNAPSHOTS_DIR / f"{date_str}__{mode}.json"


def _best_picks_sports():
    return [
        "nba",
        "mlb",
        "kbo",
        "nhl",
        "liga_mx",
        "laliga",
        "euroleague",
        "ncaa_baseball",
    ]


def _best_picks_load_events_raw_by_date(sport: str, date_str: str):
    config = ensure_sport_exists(sport)
    source_date = _kbo_source_date_from_local(date_str) if sport == "kbo" else date_str
    selected_date = parse_date_str(date_str)
    today = date.today()

    predictions_dir = config["predictions_dir"]
    historical_dir = config["historical_dir"]
    live_file = predictions_dir / f"{source_date}.json"
    hist_file = historical_dir / f"{source_date}.json"

    # Best Picks should prioritize upcoming/live boards. Only past dates can use historical fallback.
    if selected_date >= today:
        if not live_file.exists():
            return []
        file_path = live_file
    else:
        file_path = hist_file if hist_file.exists() else live_file
        if not file_path.exists():
            return []

    events = read_json_file(file_path)
    events = _translate_event_dates_for_sport(sport, events)
    events = [e for e in events if str(e.get("date", ""))[:10] == date_str]
    events = apply_overrides_to_events(sport, date_str, events)
    return events


def _best_picks_events_for_date(date_str: str):
    events_by_sport = {}
    for sport in _best_picks_sports():
        if sport in BEST_PICKS_EXCLUDED_SPORTS:
            continue
        try:
            events_by_sport[sport] = _best_picks_load_events_raw_by_date(sport, date_str)
        except Exception:
            events_by_sport[sport] = []
    return events_by_sport


def _best_picks_filter_excluded_sports(payload: dict):
    out = dict(payload or {})
    picks = [
        p for p in (out.get("picks") or [])
        if str(p.get("sport") or "") not in BEST_PICKS_EXCLUDED_SPORTS
    ]
    out["picks"] = picks

    sports_summary = [
        s for s in (out.get("sports_summary") or [])
        if str(s.get("sport") or "") not in BEST_PICKS_EXCLUDED_SPORTS
    ]
    if sports_summary:
        out["sports_summary"] = sports_summary

    return out


def _best_picks_summarize_sports(picks: list[dict]):
    by_sport = {}
    for p in picks:
        sport = str(p.get("sport") or "")
        if not sport:
            continue
        stats = by_sport.setdefault(
            sport,
            {
                "sport": sport,
                "label": p.get("sport_label") or SPORTS_CONFIG.get(sport, {}).get("label") or sport.upper(),
                "count": 0,
                "avg_score": 0.0,
                "avg_final_rank_score": 0.0,
                "avg_expected_value_per_unit": 0.0,
                "max_score": 0.0,
                "max_final_rank_score": 0.0,
                "max_expected_value_per_unit": -999.0,
            },
        )
        score = float(p.get("score") or 0.0)
        final_rank = float(p.get("final_rank_score") or 0.0)
        ev = float(p.get("expected_value_per_unit") or 0.0)
        stats["count"] += 1
        stats["avg_score"] += score
        stats["avg_final_rank_score"] += final_rank
        stats["avg_expected_value_per_unit"] += ev
        stats["max_score"] = max(stats["max_score"], score)
        stats["max_final_rank_score"] = max(stats["max_final_rank_score"], final_rank)
        stats["max_expected_value_per_unit"] = max(stats["max_expected_value_per_unit"], ev)

    out = []
    for stats in by_sport.values():
        count = max(1, int(stats["count"]))
        stats["avg_score"] = round(stats["avg_score"] / count, 2)
        stats["avg_final_rank_score"] = round(stats["avg_final_rank_score"] / count, 3)
        stats["avg_expected_value_per_unit"] = round(stats["avg_expected_value_per_unit"] / count, 4)
        stats["max_expected_value_per_unit"] = round(stats["max_expected_value_per_unit"], 4)
        out.append(stats)

    return sorted(
        out,
        key=lambda x: (x["max_final_rank_score"], x["avg_final_rank_score"], x["max_score"]),
        reverse=True,
    )


def _best_picks_trim_payload(payload: dict, top_n: int):
    filtered = _best_picks_filter_excluded_sports(payload)
    picks = list(filtered.get("picks") or [])[:max(1, int(top_n))]
    out = dict(filtered)
    out["top_n"] = len(picks)
    out["picks"] = picks
    out["sports_summary"] = _best_picks_summarize_sports(picks)
    return out


def _best_picks_generate_snapshot(date_str: str, generation_top_n: int, ranking_mode: str = "balanced"):
    selected_date = parse_date_str(date_str)
    events_by_sport = _best_picks_events_for_date(date_str)
    calibration_profiles = build_probability_calibration_profiles()
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    # Best Picks is intended for actionable/upcoming markets only.
    include_completed = False
    payload = build_daily_best_picks(
        events_by_sport,
        top_n=max(1, int(generation_top_n)),
        calibration_profiles=calibration_profiles,
        include_completed=include_completed,
        ranking_mode=mode,
    )
    payload["snapshot_date"] = date_str
    payload["snapshot_generated_at"] = datetime.now().isoformat()
    payload["snapshot_source"] = "frozen_daily_snapshot"
    payload["snapshot_ranking_mode"] = mode
    return payload


def _best_picks_save_snapshot(date_str: str, payload: dict, ranking_mode: str = "balanced"):
    path = _best_picks_snapshot_file(date_str, ranking_mode=ranking_mode)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _best_picks_load_snapshot(date_str: str, ranking_mode: str = "balanced"):
    parse_date_str(date_str)
    path = _best_picks_snapshot_file(date_str, ranking_mode=ranking_mode)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _best_picks_event_lookup_for_payload(payload: dict):
    picks = payload.get("picks") or []
    key_pairs = {
        (str(p.get("sport") or ""), str(p.get("date") or ""))
        for p in picks
        if str(p.get("sport") or "") and str(p.get("date") or "")
    }

    lookup = {}
    results_lookup_by_sport = {}
    for sport, date_str in key_pairs:
        try:
            events = _best_picks_load_events_raw_by_date(sport, date_str)
        except Exception:
            continue

        if sport not in results_lookup_by_sport:
            results_lookup_by_sport[sport] = build_results_lookup_for_sport(sport)

        try:
            events = enrich_predictions_with_results(
                sport,
                events,
                lookup=results_lookup_by_sport.get(sport),
                allow_live=False,
            )
        except Exception:
            pass

        for event in events:
            game_id = str(event.get("game_id") or "")
            if not game_id:
                continue
            lookup[(sport, date_str, game_id)] = event
    return lookup


def _best_picks_with_results(payload: dict):
    def _event_is_completed(event: dict | None) -> bool:
        if not isinstance(event, dict):
            return False

        completed = event.get("status_completed")
        if completed is not None:
            try:
                if int(float(completed)) == 1:
                    return True
            except Exception:
                pass

        state = str(event.get("status_state") or "").strip().lower()
        if state in {"post", "final", "completed"}:
            return True

        return bool(event.get("result_available"))

    if not isinstance(payload, dict):
        return payload

    event_lookup = _best_picks_event_lookup_for_payload(payload)
    out = dict(payload)
    out_picks = []

    result_keys = [
        "correct_full_game",
        "correct_full_game_adjusted",
        "correct_full_game_base",
        "full_game_hit",
        "correct_spread",
        "correct_total",
        "correct_total_adjusted",
        "q1_hit",
        "actual_result",
        "final_score_text",
        "status_state",
        "status_description",
        "status_detail",
        "status_completed",
        "result_available",
        "home_score",
        "away_score",
        "home_q1_score",
        "away_q1_score",
        "full_game_result_winner",
        "q1_result_winner",
        "correct_btts",
        "correct_btts_adjusted",
        "correct_corners_adjusted",
        "correct_f5",
        "correct_home_over",
    ]

    for pick in payload.get("picks") or []:
        row = dict(pick)
        sport = str(row.get("sport") or "")
        date_str = str(row.get("date") or "")
        game_id = str(row.get("game_id") or "")

        event = event_lookup.get((sport, date_str, game_id))
        if isinstance(event, dict):
            for key in result_keys:
                value = event.get(key)
                if value is not None:
                    row[key] = value

            if row.get("actual_result") is None and event.get("full_game_result_winner") is not None:
                row["actual_result"] = event.get("full_game_result_winner")

        is_resolved = _event_is_completed(event)
        if is_resolved and row.get("full_game_hit") is None:
            hit = row.get("correct_full_game_adjusted")
            if hit is None:
                hit = row.get("correct_full_game")
            if hit is None:
                hit = row.get("correct_full_game_base")
            if hit is not None:
                row["full_game_hit"] = hit

        if is_resolved:
            hit = row.get("full_game_hit")
            if hit is True:
                row["result_label"] = "ACIERTO"
            elif hit is False:
                row["result_label"] = "FALLO"
            else:
                row["result_label"] = "RESUELTO"
        else:
            row["result_label"] = "PENDIENTE"

        out_picks.append(row)

    out["picks"] = out_picks
    return out

def _best_picks_get_or_create_snapshot(date_str: str, generation_top_n: int, force_refresh: bool, ranking_mode: str = "balanced"):
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    snapshot = None if force_refresh else _best_picks_load_snapshot(date_str, ranking_mode=mode)
    if snapshot is not None:
        return snapshot

    snapshot = _best_picks_generate_snapshot(date_str, generation_top_n, ranking_mode=mode)
    _best_picks_save_snapshot(date_str, snapshot, ranking_mode=mode)
    return snapshot


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "NBA GOLD API running",
        "sports": list(SPORTS_CONFIG.keys()),
    }


@app.get("/api/sports")
def get_sports():
    return [
        {"key": key, "label": value["label"]}
        for key, value in SPORTS_CONFIG.items()
    ]


@app.get("/api/{sport}/predictions/today")
def get_today_predictions(sport: str):
    config = ensure_sport_exists(sport)
    if sport == "kbo":
        local_today = date.today().strftime("%Y-%m-%d")
        source_today = _kbo_source_date_from_local(local_today)
        try:
            latest = resolve_prediction_file(
                config["predictions_dir"],
                config["historical_dir"],
                source_today,
            )
        except HTTPException:
            latest = get_today_file(config["predictions_dir"], config["historical_dir"])
    else:
        latest = get_today_file(config["predictions_dir"], config["historical_dir"])

    events = _normalize_events_payload(read_json_file(latest))
    source_stem = str(latest.stem)
    _, hist_file = get_files_for_date(config["predictions_dir"], config["historical_dir"], source_stem)
    events = merge_result_hints_from_historical(events, hist_file)
    overrides_date = source_today if sport == "kbo" else str(latest.stem)
    events = apply_overrides_to_events(sport, overrides_date, events)
    try:
        events = enrich_predictions_if_available(sport, events)
    except Exception:
        # Return base events rather than failing the endpoint when enrichment has transient issues.
        pass
    payload = _translate_event_dates_for_sport(sport, _normalize_events_payload(events))
    return _sanitize_json_values(payload)


@app.get("/api/{sport}/predictions/{date_str}")
def get_predictions_by_date(sport: str, date_str: str):
    config = ensure_sport_exists(sport)
    source_date = _kbo_source_date_from_local(date_str) if sport == "kbo" else date_str
    file_path = resolve_prediction_file(
        config["predictions_dir"],
        config["historical_dir"],
        source_date,
    )
    events = _normalize_events_payload(read_json_file(file_path))
    _, hist_file = get_files_for_date(config["predictions_dir"], config["historical_dir"], source_date)
    events = merge_result_hints_from_historical(events, hist_file)
    overrides_date = source_date if sport == "kbo" else date_str
    events = apply_overrides_to_events(sport, overrides_date, events)
    try:
        events = enrich_predictions_if_available(sport, events)
    except Exception:
        # Return base events rather than failing the endpoint when enrichment has transient issues.
        pass
    payload = _translate_event_dates_for_sport(sport, _normalize_events_payload(events))
    return _sanitize_json_values(payload)


@app.get("/api/{sport}/available-dates")
def get_available_dates(sport: str):
    config = ensure_sport_exists(sport)
    dates = set()

    predictions_dir = config["predictions_dir"]
    historical_dir = config["historical_dir"]

    if predictions_dir.exists():
        for f in predictions_dir.glob("*.json"):
            dates.add(f.stem)

    if historical_dir.exists():
        for f in historical_dir.glob("*.json"):
            dates.add(f.stem)

    out_dates = sorted(dates)
    if sport == "kbo":
        out_dates = sorted({_kbo_local_date_from_source(d) for d in out_dates})
    return out_dates


@app.get("/api/{sport}/prediction-detail/{date_str}/{game_id}")
def get_prediction_detail(sport: str, date_str: str, game_id: str):
    config = ensure_sport_exists(sport)
    source_date = _kbo_source_date_from_local(date_str) if sport == "kbo" else date_str
    file_path = resolve_prediction_file(
        config["predictions_dir"],
        config["historical_dir"],
        source_date,
    )

    data = _normalize_events_payload(read_json_file(file_path))
    _, hist_file = get_files_for_date(config["predictions_dir"], config["historical_dir"], source_date)
    data = merge_result_hints_from_historical(data, hist_file)
    overrides_date = source_date if sport == "kbo" else date_str
    data = apply_overrides_to_events(sport, overrides_date, data)
    try:
        data = enrich_predictions_if_available(sport, data)
    except Exception:
        pass
    data = _translate_event_dates_for_sport(sport, _normalize_events_payload(data))

    for event in data:
        if str(event.get("game_id")) == str(game_id):
            return _sanitize_json_values(event)

    raise HTTPException(status_code=404, detail="Juego no encontrado")


@app.get("/api/insights/summary")
def get_insights_summary():
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "euroleague"]
    summaries = [build_sport_insights_summary(s) for s in sports]

    return {
        "generated_at": datetime.now().isoformat(),
        "sports": summaries,
    }


@app.get("/api/insights/weekday-scoring")
def get_weekday_scoring_insights():
    configs = [
        SportScoringConfig(
            key="nba",
            label="NBA",
            raw_file=NBA_RAW_HISTORY,
            home_col="home_pts_total",
            away_col="away_pts_total",
            metric_label="Puntos Totales",
        ),
        SportScoringConfig(
            key="mlb",
            label="MLB",
            raw_file=MLB_RAW_HISTORY,
            home_col="home_runs_total",
            away_col="away_runs_total",
            metric_label="Runs Totales",
        ),
        SportScoringConfig(
            key="kbo",
            label="KBO",
            raw_file=KBO_RAW_HISTORY,
            home_col="home_runs_total",
            away_col="away_runs_total",
            metric_label="Runs Totales",
            date_shift_days=-1,
        ),
        SportScoringConfig(
            key="ncaa_baseball",
            label="NCAA Baseball",
            raw_file=NCAA_BASEBALL_RAW_HISTORY,
            home_col="home_runs_total",
            away_col="away_runs_total",
            metric_label="Carreras Totales",
        ),
        SportScoringConfig(
            key="nhl",
            label="NHL",
            raw_file=NHL_RAW_HISTORY,
            home_col="home_score",
            away_col="away_score",
            metric_label="Goals Totales",
        ),
        SportScoringConfig(
            key="liga_mx",
            label="Liga MX",
            raw_file=LIGA_MX_RAW_HISTORY,
            home_col="home_score",
            away_col="away_score",
            metric_label="Goles Totales",
        ),
        SportScoringConfig(
            key="laliga",
            label="LaLiga EA Sports",
            raw_file=LALIGA_RAW_HISTORY,
            home_col="home_score",
            away_col="away_score",
            metric_label="Goles Totales",
        ),
        SportScoringConfig(
            key="euroleague",
            label="EuroLeague",
            raw_file=EUROLEAGUE_RAW_HISTORY,
            home_col="home_pts_total",
            away_col="away_pts_total",
            metric_label="Puntos Totales",
        ),
    ]
    return build_weekday_scoring_summary(configs)


@app.get("/api/insights/best-picks/available-dates")
def get_best_picks_available_dates():
    if not BEST_PICKS_SNAPSHOTS_DIR.exists():
        return []

    out = []
    for f in BEST_PICKS_SNAPSHOTS_DIR.glob("*.json"):
        stem = str(f.stem)
        date_part = stem.split("__", 1)[0]
        try:
            parse_date_str(date_part)
        except HTTPException:
            continue
        out.append(date_part)
    return sorted(set(out))


@app.get("/api/insights/best-picks/today")
def get_best_picks_today(top_n: int = 25, force_refresh: bool = False, ranking_mode: str = "best_hit_rate"):
    today_local = date.today().strftime("%Y-%m-%d")
    generation_top_n = max(int(top_n), 50)
    effective_force_refresh = bool(force_refresh)
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    payload = _best_picks_get_or_create_snapshot(
        date_str=today_local,
        generation_top_n=generation_top_n,
        force_refresh=effective_force_refresh,
        ranking_mode=mode,
    )
    payload = _best_picks_trim_payload(payload, top_n=top_n)
    return _best_picks_with_results(payload)


@app.get("/api/insights/best-picks/{date_str}")
def get_best_picks_by_date(date_str: str, top_n: int = 25, force_refresh: bool = False, ranking_mode: str = "best_hit_rate"):
    parse_date_str(date_str)
    generation_top_n = max(int(top_n), 50)
    effective_force_refresh = bool(force_refresh)
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    payload = _best_picks_get_or_create_snapshot(
        date_str=date_str,
        generation_top_n=generation_top_n,
        force_refresh=effective_force_refresh,
        ranking_mode=mode,
    )
    payload = _best_picks_trim_payload(payload, top_n=top_n)
    return _best_picks_with_results(payload)


@app.get("/api/insights/tier-performance")
def get_tier_performance_insights():
    return build_tier_performance_summary()
