from pathlib import Path
import json
import math
import os
import subprocess
import sys
import threading
from datetime import date, datetime, timedelta
import re
from typing import Optional

import pandas as pd
import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Body, Header
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
try:
    from .auth_storage import (
        create_session,
        create_purchase_order,
        delete_session,
        ensure_admin_user,
        find_user_by_email,
        find_user_by_id,
        find_open_purchase_order_by_email_plan,
        find_purchase_order_by_id,
        delete_user_account,
        get_session,
        get_app_setting,
        hash_password,
        init_auth_db,
        is_access_expired,
        iso_utc,
        list_purchase_orders,
        list_non_pending_users,
        list_pending_users,
        parse_utc,
        register_user,
        reset_user_password,
        row_to_user_payload,
        set_app_setting,
        update_purchase_order_status,
        set_user_approval,
        utc_now,
    )
except ImportError:
    from auth_storage import (
        create_session,
        create_purchase_order,
        delete_session,
        ensure_admin_user,
        find_user_by_email,
        find_user_by_id,
        find_open_purchase_order_by_email_plan,
        find_purchase_order_by_id,
        delete_user_account,
        get_session,
        get_app_setting,
        hash_password,
        init_auth_db,
        is_access_expired,
        iso_utc,
        list_purchase_orders,
        list_non_pending_users,
        list_pending_users,
        parse_utc,
        register_user,
        reset_user_password,
        row_to_user_payload,
        set_app_setting,
        update_purchase_order_status,
        set_user_approval,
        utc_now,
    )

BASE_DIR = Path(__file__).resolve().parent
NBA_RAW_HISTORY = BASE_DIR / "data" / "raw" / "nba_advanced_history.csv"
MLB_RAW_HISTORY = BASE_DIR / "data" / "mlb" / "raw" / "mlb_advanced_history.csv"
NHL_RAW_HISTORY = BASE_DIR / "data" / "nhl" / "raw" / "nhl_advanced_history.csv"
LIGA_MX_RAW_HISTORY = BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_advanced_history.csv"
LALIGA_RAW_HISTORY = BASE_DIR / "data" / "laliga" / "raw" / "laliga_advanced_history.csv"
BUNDESLIGA_RAW_HISTORY = BASE_DIR / "data" / "bundesliga" / "raw" / "bundesliga_advanced_history.csv"
LIGUE1_RAW_HISTORY = BASE_DIR / "data" / "ligue1" / "raw" / "ligue1_advanced_history.csv"
KBO_RAW_HISTORY = BASE_DIR / "data" / "kbo" / "raw" / "kbo_advanced_history.csv"
EUROLEAGUE_RAW_HISTORY = BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_advanced_history.csv"
NCAA_BASEBALL_RAW_HISTORY = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_advanced_history.csv"
NCAA_BASEBALL_RAW_UPCOMING = BASE_DIR / "data" / "ncaa_baseball" / "raw" / "ncaa_baseball_upcoming_schedule.csv"
TENNIS_RAW_HISTORY = BASE_DIR / "data" / "tennis" / "raw" / "tennis_advanced_history.csv"
TRIPLE_A_RAW_HISTORY = BASE_DIR / "data" / "triple_a" / "raw" / "triple_a_advanced_history.csv"
TRIPLE_A_RAW_UPCOMING = BASE_DIR / "data" / "triple_a" / "raw" / "triple_a_upcoming_schedule.csv"
SPORT_RAW_FILES = {
    "nba": {
        "raw_history": NBA_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "raw" / "nba_upcoming_schedule.csv",
    },
    "mlb": {
        "raw_history": MLB_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "mlb" / "raw" / "mlb_upcoming_schedule.csv",
    },
    "kbo": {
        "raw_history": KBO_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "kbo" / "raw" / "kbo_upcoming_schedule.csv",
    },
    "nhl": {
        "raw_history": NHL_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "nhl" / "raw" / "nhl_upcoming_schedule.csv",
    },
    "liga_mx": {
        "raw_history": LIGA_MX_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "liga_mx" / "raw" / "liga_mx_upcoming_schedule.csv",
    },
    "laliga": {
        "raw_history": LALIGA_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "laliga" / "raw" / "laliga_upcoming_schedule.csv",
    },
    "bundesliga": {
        "raw_history": BUNDESLIGA_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "bundesliga" / "raw" / "bundesliga_upcoming_schedule.csv",
    },
    "ligue1": {
        "raw_history": LIGUE1_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "ligue1" / "raw" / "ligue1_upcoming_schedule.csv",
    },
    "euroleague": {
        "raw_history": EUROLEAGUE_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "euroleague" / "raw" / "euroleague_upcoming_schedule.csv",
    },
    "ncaa_baseball": {
        "raw_history": NCAA_BASEBALL_RAW_HISTORY,
        "upcoming_schedule": NCAA_BASEBALL_RAW_UPCOMING,
    },
    "tennis": {
        "raw_history": TENNIS_RAW_HISTORY,
        "upcoming_schedule": BASE_DIR / "data" / "tennis" / "raw" / "tennis_upcoming_schedule.csv",
    },
    "triple_a": {
        "raw_history": TRIPLE_A_RAW_HISTORY,
        "upcoming_schedule": TRIPLE_A_RAW_UPCOMING,
    },
}
BEST_PICKS_SNAPSHOTS_DIR = BASE_DIR / "data" / "insights" / "best_picks"
BEST_PICKS_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
BEST_PICKS_EXCLUDED_SPORTS = {"ncaa_baseball"}
TEAM_FORM_SNAPSHOTS_DIR = BASE_DIR / "data" / "insights" / "team_form_snapshots"
TEAM_FORM_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

ESPN_SCOREBOARD_URLS = {
    "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard",
    "mlb": "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard",
    "kbo": "https://site.api.espn.com/apis/site/v2/sports/baseball/kbo/scoreboard",
    "nhl": "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard",
    "liga_mx": "https://site.api.espn.com/apis/site/v2/sports/soccer/mex.1/scoreboard",
    "laliga": "https://site.api.espn.com/apis/site/v2/sports/soccer/esp.1/scoreboard",
    "bundesliga": "https://site.api.espn.com/apis/site/v2/sports/soccer/ger.1/scoreboard",
    "ligue1": "https://site.api.espn.com/apis/site/v2/sports/soccer/fra.1/scoreboard",
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
    "bundesliga": {
        "predictions_dir": BASE_DIR / "data" / "bundesliga" / "predictions",
        "historical_dir": BASE_DIR / "data" / "bundesliga" / "historical_predictions",
        "label": "Bundesliga",
    },
    "ligue1": {
        "predictions_dir": BASE_DIR / "data" / "ligue1" / "predictions",
        "historical_dir": BASE_DIR / "data" / "ligue1" / "historical_predictions",
        "label": "Ligue 1",
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
    "tennis": {
        "predictions_dir": BASE_DIR / "data" / "tennis" / "predictions",
        "historical_dir": BASE_DIR / "data" / "tennis" / "historical_predictions",
        "label": "Tennis",
    },
    "triple_a": {
        "predictions_dir": BASE_DIR / "data" / "triple_a" / "predictions",
        "historical_dir": BASE_DIR / "data" / "triple_a" / "historical_predictions",
        "label": "Triple-A",
    },
}

TEAM_FORM_CACHE_LOCK = threading.Lock()
TEAM_FORM_CACHE: dict[tuple, dict] = {}
TEAM_FORM_CACHE_TTL_SECONDS = 300
TEAM_FORM_AUTO_SNAPSHOT_LOCK = threading.Lock()
TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE: Optional[str] = None

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

DEFAULT_THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY", "2c7887cad91583f20215b7d590c617e4").strip()

SPORT_UPDATE_PIPELINES = {
    "nba": {
        "label": "NBA",
        "steps": [
            {"key": "ingest", "label": "Ingesta NBA", "script": BASE_DIR / "sports" / "nba" / "data_ingest.py"},
            {"key": "features", "label": "Features NBA", "script": BASE_DIR / "sports" / "nba" / "feature_engineering.py"},
            {"key": "train", "label": "Entrenamiento NBA", "script": BASE_DIR / "sports" / "nba" / "train_models.py"},
            {"key": "historical", "label": "Hist?ricas NBA", "script": BASE_DIR / "sports" / "nba" / "historical_predictions.py"},
            {"key": "today", "label": "Predicciones de hoy NBA", "script": BASE_DIR / "sports" / "nba" / "predict_today.py"},
        ],
        "env": {},
    },
    "mlb": {
        "label": "MLB",
        "steps": [
            {"key": "ingest", "label": "Ingesta MLB", "script": BASE_DIR / "sports" / "mlb" / "data_ingest_mlb.py"},
            {"key": "lineup_strength", "label": "Lineup strength MLB", "script": BASE_DIR / "sports" / "mlb" / "ingest_lineup_strength.py"},
            {"key": "line_movement", "label": "Line movement MLB", "script": BASE_DIR / "sports" / "mlb" / "ingest_line_movement.py"},
            {"key": "umpire_stats", "label": "Umpire stats MLB", "script": BASE_DIR / "sports" / "mlb" / "ingest_umpire_stats.py"},
            {"key": "features", "label": "Features MLB", "script": BASE_DIR / "sports" / "mlb" / "feature_engineering_mlb_core.py"},
            {"key": "train", "label": "Entrenamiento MLB", "script": BASE_DIR / "sports" / "mlb" / "train_models_mlb.py"},
            {"key": "historical", "label": "Walk-forward MLB", "script": BASE_DIR / "sports" / "mlb" / "historical_predictions_mlb_walkforward.py"},
            {"key": "today", "label": "Predicciones de hoy MLB", "script": BASE_DIR / "sports" / "mlb" / "predict_today_mlb.py"},
        ],
        "env": {"THE_ODDS_API_KEY": DEFAULT_THE_ODDS_API_KEY},
    },
    "kbo": {
        "label": "KBO",
        "steps": [
            {"key": "ingest", "label": "Ingesta KBO", "script": BASE_DIR / "sports" / "kbo" / "data_ingest_kbo.py"},
            {"key": "features", "label": "Features KBO", "script": BASE_DIR / "sports" / "kbo" / "feature_engineering_kbo.py"},
            {"key": "train", "label": "Entrenamiento KBO", "script": BASE_DIR / "sports" / "kbo" / "train_models_kbo.py"},
            {"key": "historical", "label": "Hist?ricas KBO", "script": BASE_DIR / "sports" / "kbo" / "historical_predictions_kbo.py"},
            {"key": "today", "label": "Predicciones de hoy KBO", "script": BASE_DIR / "sports" / "kbo" / "predict_today_kbo.py"},
        ],
        "env": {},
    },
    "nhl": {
        "label": "NHL",
        "steps": [
            {"key": "ingest", "label": "Ingesta NHL", "script": BASE_DIR / "sports" / "nhl" / "data_ingest_nhl.py"},
            {"key": "features_core", "label": "Features NHL", "script": BASE_DIR / "sports" / "nhl" / "feature_engineering_nhl.py"},
            {"key": "train", "label": "Entrenamiento NHL", "script": BASE_DIR / "sports" / "nhl" / "train_models_nhl.py"},
            {"key": "historical", "label": "Hist?ricas NHL", "script": BASE_DIR / "sports" / "nhl" / "historical_predictions_nhl.py"},
            {"key": "today", "label": "Predicciones de hoy NHL", "script": BASE_DIR / "sports" / "nhl" / "predict_today_nhl.py"},
        ],
        "env": {},
    },
    "liga_mx": {
        "label": "Liga MX",
        "steps": [
            {"key": "ingest", "label": "Ingesta Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "data_ingest_liga_mx.py"},
            {"key": "features", "label": "Features Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "feature_engineering_liga_mx_v3.py"},
            {"key": "train", "label": "Entrenamiento Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "train_models_liga_mx.py"},
            {"key": "historical", "label": "Hist?ricas Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "historical_predictions_liga_mx.py"},
            {"key": "evaluate", "label": "Accuracy baseline Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "evaluate_baseline.py"},
            {"key": "today", "label": "Predicciones de hoy Liga MX", "script": BASE_DIR / "sports" / "ligamx" / "predict_today_liga_mx.py"},
        ],
        "env": {},
    },
    "laliga": {
        "label": "LaLiga",
        "steps": [
            {"key": "ingest", "label": "Ingesta LaLiga", "script": BASE_DIR / "sports" / "laliga" / "data_ingest_laliga.py"},
            {"key": "adjustments", "label": "Event adjustments LaLiga", "script": BASE_DIR / "sports" / "laliga" / "event_adjustments_laliga.py"},
            {"key": "features", "label": "Features LaLiga", "script": BASE_DIR / "sports" / "laliga" / "feature_engineering_laliga.py"},
            {"key": "train", "label": "Entrenamiento LaLiga", "script": BASE_DIR / "sports" / "laliga" / "train_models_laliga.py"},
            {"key": "historical", "label": "Hist?ricas LaLiga", "script": BASE_DIR / "sports" / "laliga" / "historical_predictions_laliga.py"},
            {"key": "today", "label": "Predicciones de hoy LaLiga", "script": BASE_DIR / "sports" / "laliga" / "predict_today_laliga.py"},
        ],
        "env": {},
    },
    "bundesliga": {
        "label": "Bundesliga",
        "steps": [
            {"key": "ingest", "label": "Ingesta Bundesliga", "script": BASE_DIR / "sports" / "bundesliga" / "data_ingest_bundesliga.py"},
            {"key": "features", "label": "Features Bundesliga", "script": BASE_DIR / "sports" / "bundesliga" / "feature_engineering_bundesliga.py"},
            {"key": "train", "label": "Entrenamiento Bundesliga", "script": BASE_DIR / "sports" / "bundesliga" / "train_models_bundesliga.py"},
            {"key": "historical", "label": "Historicas Bundesliga", "script": BASE_DIR / "sports" / "bundesliga" / "historical_predictions_bundesliga.py"},
            {"key": "today", "label": "Predicciones de hoy Bundesliga", "script": BASE_DIR / "sports" / "bundesliga" / "predict_today_bundesliga.py"},
        ],
        "env": {},
    },
    "ligue1": {
        "label": "Ligue 1",
        "steps": [
            {"key": "ingest", "label": "Ingesta Ligue 1", "script": BASE_DIR / "sports" / "ligue1" / "data_ingest_ligue1.py"},
            {"key": "features", "label": "Features Ligue 1", "script": BASE_DIR / "sports" / "ligue1" / "feature_engineering_ligue1.py"},
            {"key": "train", "label": "Entrenamiento Ligue 1", "script": BASE_DIR / "sports" / "ligue1" / "train_models_ligue1.py"},
            {"key": "historical", "label": "Historicas Ligue 1", "script": BASE_DIR / "sports" / "ligue1" / "historical_predictions_ligue1.py"},
            {"key": "today", "label": "Predicciones de hoy Ligue 1", "script": BASE_DIR / "sports" / "ligue1" / "predict_today_ligue1.py"},
        ],
        "env": {},
    },
    "euroleague": {
        "label": "EuroLeague",
        "steps": [
            {"key": "ingest", "label": "Ingesta EuroLeague", "script": BASE_DIR / "sports" / "euroleague" / "data_ingest_euroleague.py"},
            {"key": "features", "label": "Features EuroLeague", "script": BASE_DIR / "sports" / "euroleague" / "feature_engineering_euroleague.py"},
            {"key": "train", "label": "Entrenamiento EuroLeague", "script": BASE_DIR / "sports" / "euroleague" / "train_models_euroleague.py"},
            {"key": "historical", "label": "Hist?ricas EuroLeague", "script": BASE_DIR / "sports" / "euroleague" / "historical_predictions_euroleague.py"},
            {"key": "today", "label": "Predicciones de hoy EuroLeague", "script": BASE_DIR / "sports" / "euroleague" / "predict_today_euroleague.py"},
        ],
        "env": {},
    },
    "ncaa_baseball": {
        "label": "NCAA Baseball",
        "steps": [
            {"key": "ingest", "label": "Ingesta NCAA Baseball", "script": BASE_DIR / "sports" / "ncaa baseball" / "data_ingest_ncaa_baseball.py"},
            {"key": "features", "label": "Features NCAA Baseball", "script": BASE_DIR / "sports" / "ncaa baseball" / "feature_engineering_ncaa_baseball.py"},
            {"key": "train", "label": "Entrenamiento NCAA Baseball", "script": BASE_DIR / "sports" / "ncaa baseball" / "train_models_ncaa_baseball.py"},
            {"key": "historical", "label": "Hist?ricas NCAA Baseball", "script": BASE_DIR / "sports" / "ncaa baseball" / "historical_predictions_ncaa_baseball.py"},
            {"key": "today", "label": "Predicciones de hoy NCAA Baseball", "script": BASE_DIR / "sports" / "ncaa baseball" / "predict_today_ncaa_baseball.py"},
        ],
        "env": {},
    },
    "tennis": {
        "label": "Tennis",
        "steps": [
            {"key": "ingest", "label": "Ingesta Tennis", "script": BASE_DIR / "sports" / "tennis" / "data_ingest_tennis.py"},
            {"key": "features", "label": "Features Tennis", "script": BASE_DIR / "sports" / "tennis" / "feature_engineering_tennis.py"},
            {"key": "train", "label": "Entrenamiento Tennis", "script": BASE_DIR / "sports" / "tennis" / "train_models_tennis.py"},
            {"key": "historical", "label": "Historicas Tennis", "script": BASE_DIR / "sports" / "tennis" / "historical_predictions_tennis.py"},
            {"key": "today", "label": "Predicciones de hoy Tennis", "script": BASE_DIR / "sports" / "tennis" / "predict_today_tennis.py"},
        ],
        "env": {},
    },
    "triple_a": {
        "label": "Triple-A",
        "steps": [
            {"key": "ingest", "label": "Ingesta Triple-A", "script": BASE_DIR / "sports" / "triple_a" / "data_ingest_triple_a.py"},
            {"key": "features", "label": "Features Triple-A", "script": BASE_DIR / "sports" / "triple_a" / "feature_engineering_triple_a.py"},
            {"key": "train", "label": "Entrenamiento Triple-A", "script": BASE_DIR / "sports" / "triple_a" / "train_models_triple_a.py"},
            {"key": "historical", "label": "Historicas Triple-A", "script": BASE_DIR / "sports" / "triple_a" / "historical_predictions_triple_a.py"},
            {"key": "today", "label": "Predicciones de hoy Triple-A", "script": BASE_DIR / "sports" / "triple_a" / "predict_today_triple_a.py"},
        ],
        "env": {},
    },
}


def _initial_update_state(sport: str):
    pipeline = SPORT_UPDATE_PIPELINES[sport]
    return {
        "status": "idle",
        "percent": 0,
        "message": f"Listo para actualizar {pipeline['label']}.",
        "current_step": None,
        "current_step_label": None,
        "completed_steps": 0,
        "total_steps": len(pipeline["steps"]),
        "started_at": None,
        "finished_at": None,
        "error": None,
        "logs": [],
    }


SPORT_UPDATE_STATES = {sport: _initial_update_state(sport) for sport in SPORT_UPDATE_PIPELINES}
SPORT_UPDATE_LOCKS = {sport: threading.Lock() for sport in SPORT_UPDATE_PIPELINES}
ALL_SPORTS_UPDATE_STATE = {
    "status": "idle",
    "percent": 0,
    "message": "Listo para actualizar todos los deportes.",
    "current_sport": None,
    "current_sport_label": None,
    "completed_sports": 0,
    "total_sports": len(SPORT_UPDATE_PIPELINES),
    "started_at": None,
    "finished_at": None,
    "error": None,
    "logs": [],
}
ALL_SPORTS_UPDATE_LOCK = threading.Lock()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "nba-gold-api",
    }


ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "emilio.andra.na@gmail.com").strip().lower()
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")
init_auth_db()
ensure_admin_user(ADMIN_EMAIL, ADMIN_PASSWORD)

APPROVABLE_ROLES = {"member", "vip", "capper", "admin"}
PURCHASE_ORDER_STATUSES = {"pending_contact", "contacted", "paid", "approved", "cancelled"}
PURCHASE_PLAN_MAP = {
    "starter": {"role": "member", "price_mxn": 99, "label": "Starter"},
    "pro": {"role": "vip", "price_mxn": 249, "label": "Pro"},
    "vip": {"role": "capper", "price_mxn": 499, "label": "VIP"},
}
TELEGRAM_URL = os.getenv("TELEGRAM_URL", "https://t.me/PickGoldApp").strip()


def _build_purchase_order_code() -> str:
    now = datetime.now()
    return f"PG-{now.strftime('%Y%m%d')}-{np.random.randint(1000, 9999)}"


def _build_telegram_order_text(order: dict) -> str:
    plan_info = PURCHASE_PLAN_MAP.get(str(order.get("plan_key") or "").lower(), {})
    lines = [
        "Hola PickGold, quiero activar mi acceso.",
        f"Orden: {order.get('order_code')}",
        f"Plan: {plan_info.get('label') or order.get('plan_key')}",
        f"Email: {order.get('email')}",
        f"Nombre: {order.get('name')}",
        "Comparto comprobante para validacion.",
    ]
    username = str(order.get("telegram_username") or "").strip()
    if username:
        lines.append(f"Telegram user: @{username.lstrip('@')}")
    return "\n".join(lines)


def _build_telegram_deep_link(message: str) -> str:
    base = str(TELEGRAM_URL or "").strip()
    if not base:
        return ""
    encoded = requests.utils.quote(str(message or ""))
    joiner = "&" if "?" in base else "?"
    if "/joinchat/" in base or base.endswith("+") or "/+" in base:
        return base
    return f"{base}{joiner}text={encoded}"


def _extract_bearer_token(authorization: Optional[str]) -> str:
    raw = str(authorization or "").strip()
    if not raw:
        return ""
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return raw


def _build_auth_session_payload(session_data: dict) -> dict:
    user = dict(session_data.get("user") or {})
    return {
        "ok": True,
        "pending": False,
        "token": session_data["token"],
        "session_expires_at": session_data["session_expires_at"],
        "user": user,
    }


def _safe_iso_mtime(file_path: Path) -> Optional[str]:
    try:
        if file_path and file_path.exists():
            return datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(timespec="seconds")
    except Exception:
        return None
    return None


def _latest_json_file_info(directory: Path) -> dict:
    if not directory.exists():
        return {"count": 0, "latest_file": None, "latest_updated_at": None}

    files = sorted(directory.glob("*.json"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
    if not files:
        return {"count": 0, "latest_file": None, "latest_updated_at": None}

    latest = files[0]
    return {
        "count": len(files),
        "latest_file": latest.name,
        "latest_updated_at": _safe_iso_mtime(latest),
    }


def _build_admin_sport_pipeline_snapshot(sport: str) -> dict:
    config = SPORTS_CONFIG[sport]
    pipeline = SPORT_UPDATE_PIPELINES.get(sport, {"label": config["label"], "steps": []})
    raw_files = SPORT_RAW_FILES.get(sport, {})
    predictions_info = _latest_json_file_info(config["predictions_dir"])
    historical_info = _latest_json_file_info(config["historical_dir"])

    raw_history = raw_files.get("raw_history")
    upcoming_schedule = raw_files.get("upcoming_schedule")

    raw_history_updated_at = _safe_iso_mtime(raw_history) if raw_history else None
    upcoming_schedule_updated_at = _safe_iso_mtime(upcoming_schedule) if upcoming_schedule else None

    snapshot = {
        "sport": sport,
        "label": pipeline.get("label") or config["label"],
        "board_route": f"/{sport.replace('_', '-')}",
        "predictions_dir": str(config["predictions_dir"]),
        "historical_dir": str(config["historical_dir"]),
        "prediction_files_count": predictions_info["count"],
        "historical_files_count": historical_info["count"],
        "latest_prediction_file": predictions_info["latest_file"],
        "latest_prediction_updated_at": predictions_info["latest_updated_at"],
        "latest_historical_file": historical_info["latest_file"],
        "latest_historical_updated_at": historical_info["latest_updated_at"],
        "raw_history_file": str(raw_history) if raw_history else None,
        "raw_history_updated_at": raw_history_updated_at,
        "upcoming_schedule_file": str(upcoming_schedule) if upcoming_schedule else None,
        "upcoming_schedule_updated_at": upcoming_schedule_updated_at,
        "steps": [
            {
                "key": step.get("key"),
                "label": step.get("label"),
                "script": str(step.get("script")) if step.get("script") else None,
            }
            for step in pipeline.get("steps", [])
        ],
        "update_status": _copy_update_state(sport),
    }
    snapshot["board_status"] = _build_public_board_status(sport, snapshot=snapshot)
    return snapshot


def _admin_market_label(market_key: str) -> str:
    labels = {
        "full_game": "Full Game",
        "over_25": "Over 2.5",
        "btts": "BTTS",
        "corners_over_95": "Corners O/U 9.5",
        "ht_result": "Half Time Result",
    }
    return labels.get(str(market_key or "").lower(), str(market_key or "").replace("_", " ").title())


def _safe_read_json_object(file_path: Path):
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def _coerce_hit_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(int(value))
        return None
    if isinstance(value, str):
        txt = value.strip().lower()
        if txt in {"1", "true", "yes", "si", "acierto", "win", "won"}:
            return True
        if txt in {"0", "false", "no", "fallo", "lose", "lost"}:
            return False
    return None


def _extract_prediction_rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        games = payload.get("games")
        if isinstance(games, list):
            return games
    return []


def _historical_market_accuracy_fallback(sport: str, max_files: int = 180) -> dict:
    config = SPORTS_CONFIG.get(sport) or {}
    historical_dir = config.get("historical_dir")
    if not isinstance(historical_dir, Path) or not historical_dir.exists():
        return {}

    json_files = sorted(historical_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if max_files > 0:
        json_files = json_files[:max_files]
    if not json_files:
        return {}

    # Candidate hit flags by market across legacy/current schemas.
    market_hit_candidates = {
        "full_game": ["correct_full_game_adjusted", "correct_full_game", "full_game_hit", "moneyline_hit", "pick_hit"],
        "over_25": ["correct_total_adjusted", "correct_total", "totals_hit", "over_under_hit"],
        "btts": ["correct_btts_adjusted", "correct_btts", "btts_hit"],
        "corners_over_95": ["correct_corners_adjusted", "correct_corners", "corners_hit"],
        "ht_result": ["correct_ht_adjusted", "correct_ht", "first_half_hit", "ht_result_hit"],
    }
    counters = {
        market: {"hits": 0, "total": 0}
        for market in market_hit_candidates.keys()
    }

    for file_path in json_files:
        payload = _safe_read_json_object(file_path)
        for row in _extract_prediction_rows(payload):
            if not isinstance(row, dict):
                continue
            for market, keys in market_hit_candidates.items():
                hit = None
                for key in keys:
                    hit = _coerce_hit_bool(row.get(key))
                    if hit is not None:
                        break
                if hit is None:
                    continue
                counters[market]["total"] += 1
                if hit:
                    counters[market]["hits"] += 1

    result = {}
    for market, data in counters.items():
        total = int(data["total"])
        if total <= 0:
            continue
        result[market] = {
            "accuracy": float(data["hits"] / total),
            "samples": total,
        }
    return result


def _walkforward_market_accuracy_fallback(sport: str) -> dict:
    candidates = [
        BASE_DIR / "data" / sport / "walkforward" / f"walkforward_summary_{sport}.json",
        BASE_DIR / "data" / sport / "walkforward" / "walkforward_summary_mlb.json",
        BASE_DIR / "data" / sport / "reports" / f"walkforward_summary_{sport}.json",
    ]
    payload = None
    for path in candidates:
        if path.exists():
            payload = _safe_read_json_object(path)
            if isinstance(payload, dict):
                break
    if not isinstance(payload, dict):
        return {}

    # Tennis summary format (single-market at root).
    if "accuracy" in payload and "rows" in payload and any(k in payload for k in ("status", "tiers", "resolved_rows")):
        acc = _safe_float(payload.get("accuracy"))
        if acc is not None:
            return {"full_game": {"accuracy": acc, "samples": _safe_int(payload.get("resolved_rows") or payload.get("rows"))}}

    result = {}
    for market_key, row in payload.items():
        if not isinstance(row, dict):
            continue
        acc = _safe_float(row.get("published_accuracy"))
        if acc is None:
            acc = _safe_float(row.get("accuracy"))
        if acc is None:
            continue
        samples = _safe_int(row.get("published_rows"))
        if samples is None or samples <= 0:
            samples = _safe_int(row.get("rows"))
        result[str(market_key)] = {"accuracy": acc, "samples": samples}
    return result


def _load_admin_market_accuracy_report_for_sport(sport: str) -> dict:
    reports_dir = BASE_DIR / "data" / sport / "reports"
    if not reports_dir.exists():
        return {"sport": sport, "label": SPORTS_CONFIG[sport]["label"], "markets": [], "has_data": False}

    summary_candidates = sorted(reports_dir.glob("training_summary*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    comparison_candidates = sorted(
        reports_dir.glob("historical_adjustment_comparison*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    summary_payload = _safe_read_json_object(summary_candidates[0]) if summary_candidates else None
    comparison_payload = _safe_read_json_object(comparison_candidates[0]) if comparison_candidates else None

    markets: dict[str, dict] = {}

    if isinstance(summary_payload, dict):
        # Generic single-market summary (e.g. tennis).
        if "accuracy" in summary_payload and isinstance(summary_payload.get("accuracy"), (int, float, str)):
            markets["full_game"] = {
                "market_key": "full_game",
                "market_label": _admin_market_label("full_game"),
                "validation_accuracy": _safe_float(summary_payload.get("accuracy")),
                "validation_log_loss": _safe_float(summary_payload.get("logloss")),
                "historical_base_accuracy": None,
                "historical_adjusted_accuracy": None,
                "historical_base_log_loss": None,
                "historical_adjusted_log_loss": None,
            }

        for market_key, market_data in summary_payload.items():
            if not isinstance(market_data, dict):
                continue
            ensemble_metrics = market_data.get("ensemble_metrics") or {}
            if not isinstance(ensemble_metrics, dict):
                ensemble_metrics = {}

            validation_accuracy = _safe_float(ensemble_metrics.get("accuracy"))
            validation_log_loss = _safe_float(ensemble_metrics.get("logloss"))

            # Some sports persist validation stats directly.
            if validation_accuracy is None:
                validation_accuracy = _safe_float(market_data.get("valid_accuracy"))
            if validation_log_loss is None:
                validation_log_loss = _safe_float(market_data.get("valid_logloss"))

            markets[str(market_key)] = {
                "market_key": str(market_key),
                "market_label": _admin_market_label(str(market_key)),
                "validation_accuracy": validation_accuracy,
                "validation_log_loss": validation_log_loss,
                "historical_base_accuracy": None,
                "historical_adjusted_accuracy": None,
                "historical_base_log_loss": None,
                "historical_adjusted_log_loss": None,
            }

    if isinstance(comparison_payload, dict):
        metrics_block = comparison_payload.get("metrics")
        if isinstance(metrics_block, dict):
            for market_key, metric_data in metrics_block.items():
                if not isinstance(metric_data, dict):
                    continue
                market_key = str(market_key)
                if market_key not in markets:
                    markets[market_key] = {
                        "market_key": market_key,
                        "market_label": _admin_market_label(market_key),
                        "validation_accuracy": None,
                        "validation_log_loss": None,
                        "historical_base_accuracy": None,
                        "historical_adjusted_accuracy": None,
                        "historical_base_log_loss": None,
                        "historical_adjusted_log_loss": None,
                    }
                markets[market_key]["historical_base_accuracy"] = _safe_float(metric_data.get("base_accuracy"))
                markets[market_key]["historical_adjusted_accuracy"] = _safe_float(metric_data.get("adjusted_accuracy"))
                markets[market_key]["historical_base_log_loss"] = _safe_float(metric_data.get("base_log_loss"))
                markets[market_key]["historical_adjusted_log_loss"] = _safe_float(metric_data.get("adjusted_log_loss"))

    fallback_metrics = _historical_market_accuracy_fallback(sport)
    if not fallback_metrics:
        fallback_metrics = _walkforward_market_accuracy_fallback(sport)
    for market_key, metric_data in fallback_metrics.items():
        if market_key not in markets:
            markets[market_key] = {
                "market_key": market_key,
                "market_label": _admin_market_label(market_key),
                "validation_accuracy": None,
                "validation_log_loss": None,
                "historical_base_accuracy": None,
                "historical_adjusted_accuracy": None,
                "historical_base_log_loss": None,
                "historical_adjusted_log_loss": None,
            }

        if markets[market_key].get("historical_base_accuracy") is None:
            markets[market_key]["historical_base_accuracy"] = _safe_float(metric_data.get("accuracy"))
        if markets[market_key].get("historical_adjusted_accuracy") is None:
            markets[market_key]["historical_adjusted_accuracy"] = _safe_float(metric_data.get("accuracy"))

    ordered = sorted(
        markets.values(),
        key=lambda item: (
            0 if item["market_key"] == "full_game" else 1,
            item["market_key"],
        ),
    )

    return {
        "sport": sport,
        "label": SPORTS_CONFIG[sport]["label"],
        "reports_dir": str(reports_dir),
        "training_summary_file": str(summary_candidates[0].name) if summary_candidates else None,
        "historical_comparison_file": str(comparison_candidates[0].name) if comparison_candidates else None,
        "markets": ordered,
        "has_data": len(ordered) > 0,
    }


def _build_public_board_status(sport: str, target_date: Optional[str] = None, snapshot: Optional[dict] = None) -> dict:
    snapshot = snapshot or _build_admin_sport_pipeline_snapshot(sport)
    config = SPORTS_CONFIG[sport]
    source_date = _kbo_source_date_from_local(target_date) if (sport == "kbo" and target_date) else target_date
    prediction_file = None
    historical_file = None
    if source_date:
        prediction_file = config["predictions_dir"] / f"{source_date}.json"
        historical_file = config["historical_dir"] / f"{source_date}.json"

    has_target_snapshot = False
    if prediction_file and prediction_file.exists():
        has_target_snapshot = True
    if historical_file and historical_file.exists():
        has_target_snapshot = True
    if sport == "ncaa_baseball" and target_date:
        has_target_snapshot = has_target_snapshot or (target_date in _get_ncaa_baseball_available_dates())

    latest_prediction_date = None
    latest_prediction_file = snapshot.get("latest_prediction_file")
    if latest_prediction_file:
        latest_prediction_date = Path(latest_prediction_file).stem
        if sport == "kbo":
            latest_prediction_date = _kbo_local_date_from_source(latest_prediction_date)

    def _iso_to_date(value: Optional[str]):
        if not value:
            return None
        try:
            return datetime.fromisoformat(value).date().isoformat()
        except Exception:
            return None

    raw_history_date = _iso_to_date(snapshot.get("raw_history_updated_at"))
    upcoming_schedule_date = _iso_to_date(snapshot.get("upcoming_schedule_updated_at"))
    latest_prediction_update_date = _iso_to_date(snapshot.get("latest_prediction_updated_at"))

    today_local = datetime.now().date().isoformat()
    selected_date = target_date or today_local
    update_state = snapshot.get("update_status") or {}

    freshness = "ok"
    title = "Datos al dia"
    message = "El board esta usando la version mas reciente disponible."

    if update_state.get("status") == "running":
        freshness = "running"
        title = "Actualizacion en curso"
        message = update_state.get("current_step_label") or update_state.get("message") or "Estamos refrescando este deporte ahora mismo."
    elif target_date and not has_target_snapshot:
        freshness = "stale"
        title = "Board pendiente de actualizar"
        if latest_prediction_date:
            message = f"La pestana no tiene snapshot para {selected_date}. El ultimo board disponible es {latest_prediction_date}."
        else:
            message = f"La pestana todavia no tiene un snapshot disponible para {selected_date}."
    elif not raw_history_date or not upcoming_schedule_date:
        freshness = "stale"
        title = "Fuentes incompletas"
        message = "Este deporte todavia no tiene sus archivos base completos. Conviene correr la actualizacion completa antes de confiar en el board."
    elif raw_history_date < today_local or upcoming_schedule_date < today_local:
        freshness = "stale"
        title = "Fuentes desactualizadas"
        message = f"Los datos base de este deporte no se actualizan desde {min(raw_history_date, upcoming_schedule_date)}. Conviene refrescar el pipeline completo."
    elif latest_prediction_date and latest_prediction_date < today_local:
        freshness = "stale"
        title = "Datos desactualizados"
        message = f"Este deporte no se actualiza desde {latest_prediction_date}. Conviene correr la actualizacion completa para llevar el board al dia."
    elif selected_date >= today_local and latest_prediction_date and latest_prediction_date < selected_date:
        freshness = "stale"
        title = "Datos potencialmente desactualizados"
        message = f"El board visible llega hasta {latest_prediction_date}. Conviene correr una actualizacion para cubrir {selected_date}."
    elif not latest_prediction_date:
        freshness = "warning"
        title = "Sin snapshot detectado"
        message = "Todavia no encontramos archivos de predicciones para este deporte."

    return {
        "sport": sport,
        "label": snapshot.get("label") or config["label"],
        "target_date": selected_date,
        "latest_prediction_date": latest_prediction_date,
        "has_target_snapshot": has_target_snapshot,
        "freshness": freshness,
        "title": title,
        "message": message,
        "raw_history_updated_at": snapshot.get("raw_history_updated_at"),
        "upcoming_schedule_updated_at": snapshot.get("upcoming_schedule_updated_at"),
        "latest_prediction_updated_at": snapshot.get("latest_prediction_updated_at"),
        "update_status": update_state,
    }


def _require_admin_session(authorization: Optional[str]) -> dict:
    token = _extract_bearer_token(authorization)
    session_data = get_session(token)
    if not session_data:
        raise HTTPException(status_code=401, detail="Sesion invalida o expirada.")

    user = session_data.get("user") or {}
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="No autorizado.")

    if user.get("status") != "approved":
        raise HTTPException(status_code=403, detail="Tu cuenta admin no esta aprobada.")

    if is_access_expired(user):
        delete_session(token)
        raise HTTPException(status_code=403, detail="El acceso del administrador ya expiro.")

    return session_data


def _safe_float(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    num = _safe_float(value)
    if num is None:
        return None
    try:
        return int(num)
    except Exception:
        return None


def _format_final_score_text(home_score, away_score):
    if home_score is None or away_score is None:
        return ""
    try:
        return f"Final: {int(home_score)} - {int(away_score)}"
    except Exception:
        return ""


def _load_ncaa_baseball_schedule_events(date_str: str) -> list[dict]:
    if not NCAA_BASEBALL_RAW_UPCOMING.exists():
        return []

    try:
        df = pd.read_csv(NCAA_BASEBALL_RAW_UPCOMING, dtype={"game_id": str})
    except Exception:
        return []

    if df.empty or "date" not in df.columns:
        return []

    rows = df[df["date"].astype(str) == str(date_str)].copy()
    if rows.empty:
        return []

    rows["time"] = rows.get("time", "").fillna("").astype(str)
    rows = rows.sort_values(["time", "game_id"], kind="stable")

    events = []
    for _, row in rows.iterrows():
        home_team = str(row.get("home_team") or "HOME").strip() or "HOME"
        away_team = str(row.get("away_team") or "AWAY").strip() or "AWAY"
        home_score = _safe_int(row.get("home_runs_total"))
        away_score = _safe_int(row.get("away_runs_total"))
        odds_details = str(row.get("odds_details") or "").strip()
        total_line = _safe_float(row.get("odds_over_under"))
        total_line_value = None
        if total_line is not None and total_line > 0:
            total_line_value = int(total_line) if float(total_line).is_integer() else round(float(total_line), 1)

        status_state = str(row.get("status_state") or "").strip().lower() or "pre"
        status_completed = _safe_int(row.get("status_completed")) or 0
        result_available = status_completed == 1 or status_state in {"post", "final", "completed"}

        event = {
            "game_id": str(row.get("game_id") or ""),
            "date": str(row.get("date") or date_str),
            "time": str(row.get("time") or ""),
            "season": row.get("season"),
            "home_team": home_team,
            "away_team": away_team,
            "game_name": f"{away_team} @ {home_team}",
            "status_completed": 1 if result_available else 0,
            "status_state": status_state,
            "status_description": str(row.get("status_description") or ("Final" if result_available else "Scheduled")),
            "status_detail": str(row.get("status_detail") or ("FINAL" if result_available else "Scheduled")),
            "home_score": home_score,
            "away_score": away_score,
            "home_runs_total": home_score,
            "away_runs_total": away_score,
            "home_r1": _safe_int(row.get("home_r1")),
            "away_r1": _safe_int(row.get("away_r1")),
            "home_runs_f5": _safe_int(row.get("home_runs_f5")),
            "away_runs_f5": _safe_int(row.get("away_runs_f5")),
            "odds_details": odds_details or "No Line",
            "odds_over_under": total_line if total_line is not None else 0,
            "closing_total_line": total_line_value,
            "full_game_pick": "Sin linea disponible",
            "full_game_confidence": None,
            "recommended_tier": "Sin linea",
            "result_available": result_available,
            "final_score_text": _format_final_score_text(home_score, away_score) if result_available else "",
            "market_missing": 1,
            "board_source": "raw_schedule",
        }
        events.append(event)

    try:
        return enrich_predictions_with_results("ncaa_baseball", events, allow_live=True)
    except Exception:
        return events


def _get_ncaa_baseball_available_dates() -> list[str]:
    if not NCAA_BASEBALL_RAW_UPCOMING.exists():
        return []
    try:
        df = pd.read_csv(NCAA_BASEBALL_RAW_UPCOMING, usecols=["date"])
    except Exception:
        return []
    if df.empty or "date" not in df.columns:
        return []
    return sorted({str(value) for value in df["date"].dropna().astype(str).tolist() if str(value).strip()})


def _merge_ncaa_baseball_board(date_str: str) -> list[dict]:
    base_events = _load_ncaa_baseball_schedule_events(date_str)
    config = SPORTS_CONFIG["ncaa_baseball"]

    predicted_events = []
    hist_file = None
    try:
        file_path = resolve_prediction_file(
            config["predictions_dir"],
            config["historical_dir"],
            date_str,
        )
        predicted_events = _normalize_events_payload(read_json_file(file_path))
        _, hist_file = get_files_for_date(config["predictions_dir"], config["historical_dir"], date_str)
        predicted_events = merge_result_hints_from_historical(predicted_events, hist_file)
        predicted_events = apply_overrides_to_events("ncaa_baseball", date_str, predicted_events)
        try:
            predicted_events = enrich_predictions_if_available("ncaa_baseball", predicted_events)
        except Exception:
            pass
    except HTTPException:
        predicted_events = []

    if not base_events:
        return predicted_events
    if not predicted_events:
        return base_events

    predicted_by_game = {str(item.get("game_id") or ""): item for item in predicted_events}
    merged = []
    seen_ids = set()

    for item in base_events:
        game_id = str(item.get("game_id") or "")
        predicted = predicted_by_game.get(game_id)
        if predicted:
            merged_item = dict(item)
            merged_item.update({k: v for k, v in predicted.items() if v is not None and v != ""})
            merged_item.setdefault("board_source", "merged_schedule_predictions")
            merged.append(merged_item)
            seen_ids.add(game_id)
        else:
            merged.append(item)
            seen_ids.add(game_id)

    for item in predicted_events:
        game_id = str(item.get("game_id") or "")
        if game_id not in seen_ids:
            merged.append(item)

    return merged


@app.post("/api/register")
def register(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    if not name or not email or not password:
        return {"ok": False, "error": "Todos los campos son requeridos."}

    existing = find_user_by_email(email)
    if existing:
        return {"ok": False, "error": "El email ya esta registrado."}

    user = register_user(name, email, password)

    return {
        "ok": True,
        "user": user,
        "message": "Cuenta creada. Queda pendiente de aprobacion por un administrador.",
    }


@app.post("/api/purchase-orders")
def create_manual_purchase_order(payload: dict = Body(...)):
    name = str(payload.get("name", "")).strip()
    email = str(payload.get("email", "")).strip().lower()
    plan_key = str(payload.get("plan_key", "starter")).strip().lower()
    telegram_username = str(payload.get("telegram_username", "")).strip() or None
    notes = str(payload.get("notes", "")).strip() or None

    if not name or not email:
        return {"ok": False, "error": "Nombre y email son requeridos para generar la orden."}

    plan_info = PURCHASE_PLAN_MAP.get(plan_key)
    if not plan_info:
        return {"ok": False, "error": "Plan no soportado."}

    existing = find_open_purchase_order_by_email_plan(email, plan_key)
    if existing:
        message = _build_telegram_order_text(existing)
        return {
            "ok": True,
            "created": False,
            "order": existing,
            "telegram_message": message,
            "telegram_deep_link": _build_telegram_deep_link(message),
            "message": f"Ya existe una orden abierta ({existing.get('order_code')}).",
        }

    linked_user = find_user_by_email(email)
    order = create_purchase_order(
        order_code=_build_purchase_order_code(),
        name=name,
        email=email,
        user_id=(linked_user or {}).get("id"),
        plan_key=plan_key,
        plan_role=str(plan_info.get("role")),
        plan_price_mxn=int(plan_info.get("price_mxn")),
        telegram_username=telegram_username,
        notes=notes,
    )
    text = _build_telegram_order_text(order)
    return {
        "ok": True,
        "created": True,
        "order": order,
        "telegram_message": text,
        "telegram_deep_link": _build_telegram_deep_link(text),
        "message": f"Orden creada: {order.get('order_code')}.",
    }


@app.post("/api/login")
def login(payload: dict = Body(...)):
    email = str(payload.get("email", "")).strip().lower()
    password = str(payload.get("password", ""))

    user = find_user_by_email(email)
    if not user or user["password_hash"] != hash_password(password):
        return {"ok": False, "error": "Credenciales incorrectas."}

    if user["status"] != "approved":
        return {
            "ok": True,
            "pending": True,
            "user": row_to_user_payload(user),
            "message": "Tu cuenta esta pendiente de aprobacion.",
        }

    if is_access_expired(user):
        return {
            "ok": False,
            "error": "Tu acceso ya expiro. Pidele al administrador que renueve tu membresia.",
        }

    session_data = create_session(user["id"])
    _schedule_daily_team_form_snapshot_if_needed()
    return _build_auth_session_payload(session_data)


@app.get("/api/session")
def get_active_session(authorization: Optional[str] = Header(default=None)):
    token = _extract_bearer_token(authorization)
    session_data = get_session(token)
    if not session_data:
        return {"ok": False, "error": "Sesion invalida o expirada."}

    user = session_data.get("user") or {}
    if user.get("status") != "approved":
        delete_session(token)
        return {"ok": False, "error": "Tu cuenta no esta aprobada."}
    if is_access_expired(user):
        delete_session(token)
        return {"ok": False, "error": "Tu acceso ya expiro."}

    _schedule_daily_team_form_snapshot_if_needed()
    return _build_auth_session_payload(session_data)


@app.post("/api/logout")
def logout(authorization: Optional[str] = Header(default=None), payload: dict = Body(default={})):
    token = _extract_bearer_token(authorization) or str(payload.get("token", "")).strip()
    if token:
        delete_session(token)
    return {"ok": True}


@app.get("/api/admin/pending-users")
def pending_users(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    pending = list_pending_users()
    return {"ok": True, "pending": pending}


@app.get("/api/admin/users")
def admin_users(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    users = list_non_pending_users()
    active_users = [user for user in users if user.get("is_active")]
    return {
        "ok": True,
        "users": users,
        "active_count": len(active_users),
        "approved_count": sum(1 for user in users if user.get("status") == "approved"),
    }


@app.get("/api/admin/purchase-orders")
def admin_purchase_orders(
    authorization: Optional[str] = Header(default=None),
    status: Optional[str] = None,
    limit: int = 200,
):
    _require_admin_session(authorization)
    requested_status = str(status or "").strip().lower() or None
    if requested_status and requested_status not in PURCHASE_ORDER_STATUSES:
        return {"ok": False, "error": "Estatus de orden no soportado."}
    orders = list_purchase_orders(status=requested_status, limit=limit)
    return {
        "ok": True,
        "orders": orders,
        "count": len(orders),
    }


@app.post("/api/admin/purchase-orders/status")
def admin_update_purchase_order_status(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    admin_session = _require_admin_session(authorization)
    order_id = str(payload.get("order_id", "")).strip()
    status = str(payload.get("status", "")).strip().lower()
    admin_notes = str(payload.get("admin_notes", "")).strip() or None
    payment_reference = str(payload.get("payment_reference", "")).strip() or None
    access_days = payload.get("access_days")
    role_override = str(payload.get("role", "")).strip().lower() or None

    if not order_id or not status:
        return {"ok": False, "error": "order_id y status son requeridos."}
    if status not in PURCHASE_ORDER_STATUSES:
        return {"ok": False, "error": "Estatus de orden no soportado."}

    current = find_purchase_order_by_id(order_id)
    if not current:
        return {"ok": False, "error": "Orden no encontrada."}

    updated = update_purchase_order_status(
        order_id=order_id,
        status=status,
        admin_notes=admin_notes,
        payment_reference=payment_reference,
        approved_by=(admin_session.get("user") or {}).get("email", ADMIN_EMAIL),
    )
    if not updated:
        return {"ok": False, "error": "No se pudo actualizar la orden."}

    activated_user = None
    if status == "approved":
        plan_key = str(updated.get("plan_key") or "").lower()
        plan_info = PURCHASE_PLAN_MAP.get(plan_key) or {}
        role = role_override or str(plan_info.get("role") or "member").lower()
        if role not in APPROVABLE_ROLES:
            role = "member"

        linked_user = find_user_by_id(str(updated.get("user_id") or "").strip()) if updated.get("user_id") else None
        if not linked_user:
            linked_user = find_user_by_email(str(updated.get("email") or ""))

        if linked_user:
            expires_at = None
            try:
                days = int(access_days if access_days is not None else 30)
            except Exception:
                days = 30
            days = max(1, days)
            expires_at = iso_utc(utc_now() + timedelta(days=days))
            activated_user = set_user_approval(
                str(linked_user.get("id") or ""),
                role=role,
                access_expires_at=expires_at,
                approved_by=(admin_session.get("user") or {}).get("email", ADMIN_EMAIL),
            )

    return {
        "ok": True,
        "order": updated,
        "activated_user": activated_user,
    }


@app.post("/api/admin/approve-user")
def approve_user(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    admin_session = _require_admin_session(authorization)
    user_id = str(payload.get("user_id", "")).strip()
    role = str(payload.get("role", "member")).strip().lower() or "member"
    access_days = payload.get("access_days")
    access_expires_at = str(payload.get("access_expires_at", "")).strip() or None

    if role not in APPROVABLE_ROLES:
        return {"ok": False, "error": "Rol no soportado."}

    user = find_user_by_id(user_id)
    if not user:
        return {"ok": False, "error": "Usuario no encontrado."}

    if access_expires_at:
        if parse_utc(access_expires_at) is None:
            return {"ok": False, "error": "Fecha de expiracion invalida."}
    else:
        try:
            days = int(access_days if access_days is not None else 30)
        except Exception:
            return {"ok": False, "error": "Dias de acceso invalidos."}
        if days <= 0:
            return {"ok": False, "error": "Los dias de acceso deben ser mayores a 0."}
        access_expires_at = iso_utc(utc_now() + timedelta(days=days))

    updated_user = set_user_approval(
        user_id,
        role=role,
        access_expires_at=access_expires_at,
        approved_by=(admin_session.get("user") or {}).get("email", ADMIN_EMAIL),
    )

    return {
        "ok": True,
        "user": updated_user,
    }


@app.post("/api/admin/reset-user-password")
def admin_reset_user_password(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    admin_session = _require_admin_session(authorization)
    user_id = str(payload.get("user_id", "")).strip()
    new_password = str(payload.get("new_password", "")).strip()

    if not user_id or not new_password:
        return {"ok": False, "error": "Usuario y nueva contrasena son requeridos."}
    if len(new_password) < 6:
        return {"ok": False, "error": "La nueva contrasena debe tener al menos 6 caracteres."}

    user = find_user_by_id(user_id)
    if not user:
        return {"ok": False, "error": "Usuario no encontrado."}
    if (user.get("role") if isinstance(user, dict) else user["role"]) == "admin":
        return {"ok": False, "error": "No se puede resetear la contrasena del admin principal desde este panel."}

    updated_user = reset_user_password(user_id, new_password)
    return {
        "ok": True,
        "user": updated_user,
        "message": f"Contrasena reiniciada por {(admin_session.get('user') or {}).get('email', ADMIN_EMAIL)}.",
    }


@app.post("/api/admin/delete-user")
def admin_delete_user(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    admin_session = _require_admin_session(authorization)
    user_id = str(payload.get("user_id", "")).strip()
    if not user_id:
        return {"ok": False, "error": "Usuario requerido."}

    target_user = find_user_by_id(user_id)
    if not target_user:
        return {"ok": False, "error": "Usuario no encontrado."}

    admin_user = admin_session.get("user") or {}
    if user_id == str(admin_user.get("id") or ""):
        return {"ok": False, "error": "No puedes eliminar tu propia cuenta desde este panel."}
    if (target_user.get("role") if isinstance(target_user, dict) else target_user["role"]) == "admin":
        return {"ok": False, "error": "No se puede eliminar el admin principal."}

    delete_user_account(user_id)
    return {"ok": True, "message": "Usuario eliminado correctamente."}


def _app_settings_payload() -> dict:
    social_mode_raw = str(get_app_setting("social_mode", "0") or "0").strip().lower()
    social_mode = social_mode_raw in {"1", "true", "yes", "si", "on"}
    return {
        "social_mode": social_mode,
    }


@app.get("/api/public/app-settings")
def public_app_settings():
    return {
        "ok": True,
        **_app_settings_payload(),
    }


@app.get("/api/admin/app-settings")
def admin_app_settings(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    return {
        "ok": True,
        **_app_settings_payload(),
    }


@app.post("/api/admin/app-settings")
def update_admin_app_settings(payload: dict = Body(...), authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    social_mode = bool(payload.get("social_mode"))
    set_app_setting("social_mode", "1" if social_mode else "0")
    return {
        "ok": True,
        **_app_settings_payload(),
    }


def ensure_sport_exists(sport: str):
    if sport not in SPORTS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Deporte no soportado: {sport}")
    return SPORTS_CONFIG[sport]


def _enrich_progress_state(state: dict) -> dict:
    payload = dict(state)
    started_at_raw = payload.get("started_at")
    finished_at_raw = payload.get("finished_at")
    percent = float(payload.get("percent") or 0)
    now = datetime.now()

    try:
        started_at = datetime.fromisoformat(started_at_raw) if started_at_raw else None
    except Exception:
        started_at = None
    try:
        finished_at = datetime.fromisoformat(finished_at_raw) if finished_at_raw else None
    except Exception:
        finished_at = None

    if started_at:
        elapsed_seconds = max(0, int((now - started_at).total_seconds()))
        payload["elapsed_seconds"] = elapsed_seconds
    else:
        elapsed_seconds = None
        payload["elapsed_seconds"] = None

    payload["eta_seconds"] = None
    payload["estimated_finish_at"] = None
    if payload.get("status") == "running" and started_at and percent > 0:
        remaining_seconds = max(0, int(elapsed_seconds * ((100 - percent) / percent)))
        payload["eta_seconds"] = remaining_seconds
        payload["estimated_finish_at"] = (now + timedelta(seconds=remaining_seconds)).isoformat(timespec="seconds")
    elif payload.get("status") == "completed" and started_at and finished_at:
        payload["elapsed_seconds"] = max(0, int((finished_at - started_at).total_seconds()))

    return payload


def _copy_all_sports_update_state():
    with ALL_SPORTS_UPDATE_LOCK:
        return _enrich_progress_state(dict(ALL_SPORTS_UPDATE_STATE))


def _set_all_sports_update_state(**updates):
    with ALL_SPORTS_UPDATE_LOCK:
        ALL_SPORTS_UPDATE_STATE.update(updates)


def _append_all_sports_log(line: str):
    if not line:
        return
    with ALL_SPORTS_UPDATE_LOCK:
        logs = list(ALL_SPORTS_UPDATE_STATE.get("logs", []))
        logs.append(str(line))
        ALL_SPORTS_UPDATE_STATE["logs"] = logs[-12:]


def _copy_update_state(sport: str):
    state = SPORT_UPDATE_STATES[sport]
    lock = SPORT_UPDATE_LOCKS[sport]
    with lock:
        return _enrich_progress_state(dict(state))


def _append_update_log(sport: str, line: str):
    if not line:
        return
    state = SPORT_UPDATE_STATES[sport]
    lock = SPORT_UPDATE_LOCKS[sport]
    with lock:
        logs = list(state.get("logs", []))
        logs.append(str(line))
        state["logs"] = logs[-8:]


def _set_update_state(sport: str, **updates):
    state = SPORT_UPDATE_STATES[sport]
    lock = SPORT_UPDATE_LOCKS[sport]
    with lock:
        state.update(updates)


def _run_all_sports_update_pipeline():
    sports = list(SPORT_UPDATE_PIPELINES.keys())
    total_sports = len(sports)
    _set_all_sports_update_state(
        status="running",
        percent=0,
        message="Preparando actualizacion global...",
        current_sport=None,
        current_sport_label=None,
        completed_sports=0,
        total_sports=total_sports,
        started_at=datetime.now().isoformat(timespec="seconds"),
        finished_at=None,
        error=None,
        logs=[],
    )

    try:
        for idx, sport in enumerate(sports, start=1):
            pipeline = SPORT_UPDATE_PIPELINES[sport]
            _set_all_sports_update_state(
                current_sport=sport,
                current_sport_label=pipeline["label"],
                message=f"Actualizando {pipeline['label']}...",
                percent=int(round(((idx - 1) / total_sports) * 100)),
            )
            _append_all_sports_log(f"[RUN] {pipeline['label']}")

            worker = threading.Thread(target=_run_update_pipeline, args=(sport,), daemon=True)
            worker.start()

            while worker.is_alive():
                sport_state = _copy_update_state(sport)
                sport_percent = max(0, min(100, float(sport_state.get("percent") or 0)))
                overall_percent = int(round((((idx - 1) + (sport_percent / 100.0)) / total_sports) * 100))
                _set_all_sports_update_state(
                    current_sport=sport,
                    current_sport_label=pipeline["label"],
                    completed_sports=idx - 1,
                    percent=overall_percent,
                    message=sport_state.get("current_step_label") or sport_state.get("message") or f"Actualizando {pipeline['label']}...",
                )
                threading.Event().wait(1.0)

            sport_state = _copy_update_state(sport)
            if sport_state.get("status") != "completed":
                _set_all_sports_update_state(
                    status="failed",
                    message=f"Fallo en {pipeline['label']}.",
                    error=sport_state.get("error") or f"La actualizacion de {pipeline['label']} fallo.",
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    percent=int(round((((idx - 1) + (max(0, min(100, float(sport_state.get('percent') or 0))) / 100.0)) / total_sports) * 100)),
                    completed_sports=idx - 1,
                )
                return

            _set_all_sports_update_state(
                completed_sports=idx,
                percent=int(round((idx / total_sports) * 100)),
                message=f"{pipeline['label']} completado.",
            )
            _append_all_sports_log(f"[OK] {pipeline['label']}")

        _set_all_sports_update_state(
            status="completed",
            percent=100,
            message="Actualizacion completa de todos los deportes terminada.",
            current_sport=None,
            current_sport_label=None,
            completed_sports=total_sports,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=None,
        )
    except Exception as exc:
        _set_all_sports_update_state(
            status="failed",
            message="La actualizacion global se detuvo por un error inesperado.",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=str(exc),
        )


def _run_update_pipeline(sport: str):
    pipeline = SPORT_UPDATE_PIPELINES[sport]
    total_steps = len(pipeline["steps"])
    sport_label = pipeline["label"]

    _set_update_state(
        sport,
        status="running",
        percent=0,
        message=f"Preparando actualizacion {sport_label}...",
        current_step=None,
        current_step_label=None,
        completed_steps=0,
        total_steps=total_steps,
        started_at=datetime.now().isoformat(timespec="seconds"),
        finished_at=None,
        error=None,
        logs=[],
    )

    try:
        process_env = os.environ.copy()
        process_env["PYTHONIOENCODING"] = "utf-8"
        process_env["PYTHONUTF8"] = "1"
        for env_key, env_value in pipeline.get("env", {}).items():
            if env_value:
                process_env[env_key] = str(env_value)

        for idx, step in enumerate(pipeline["steps"], start=1):
            start_percent = int(round(((idx - 1) / total_steps) * 100))
            end_percent = int(round((idx / total_steps) * 100))
            _set_update_state(
                sport,
                current_step=step["key"],
                current_step_label=step["label"],
                percent=start_percent,
                message=f"Ejecutando {step['label']}...",
            )
            _append_update_log(sport, f"[RUN] {step['label']}")

            result = subprocess.run(
                [sys.executable, "-X", "utf8", str(step["script"])],
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=process_env,
            )

            stdout_lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
            stderr_lines = [line.strip() for line in (result.stderr or "").splitlines() if line.strip()]
            for line in stdout_lines[-3:]:
                _append_update_log(sport, line)
            for line in stderr_lines[-2:]:
                _append_update_log(sport, f"[stderr] {line}")

            if result.returncode != 0:
                error_line = stderr_lines[-1] if stderr_lines else (stdout_lines[-1] if stdout_lines else "Error desconocido.")
                _set_update_state(
                    sport,
                    status="failed",
                    percent=start_percent,
                    message=f"Fallo en {step['label']}.",
                    current_step=step["key"],
                    current_step_label=step["label"],
                    completed_steps=idx - 1,
                    finished_at=datetime.now().isoformat(timespec="seconds"),
                    error=error_line,
                )
                return

            _set_update_state(
                sport,
                percent=end_percent,
                completed_steps=idx,
                message=f"{step['label']} completado.",
            )

        _set_update_state(
            sport,
            status="completed",
            percent=100,
            message=f"Actualizacion {sport_label} completada.",
            current_step=None,
            current_step_label=None,
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=None,
        )
    except Exception as exc:
        _set_update_state(
            sport,
            status="failed",
            message=f"La actualizacion {sport_label} se detuvo por un error inesperado.",
            finished_at=datetime.now().isoformat(timespec="seconds"),
            error=str(exc),
        )


def parse_date_str(date_str: str) -> date:
    try:
        return datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail="Formato de fecha inválido. Usa YYYY-MM-DD."
        ) from exc


def _kbo_source_date_from_local(date_str: str) -> str:
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
            "home_pts_total", "away_pts_total", "home_q1", "away_q1", "home_q2", "away_q2", "home_q3", "away_q3", "home_q4", "away_q4",
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
    elif sport == "triple_a":
        file_path = TRIPLE_A_RAW_HISTORY
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
    elif sport == "bundesliga":
        file_path = BUNDESLIGA_RAW_HISTORY
        use_cols = [
            "game_id", "date", "home_team", "away_team", "home_score", "away_score", "home_corners", "away_corners", "total_corners",
        ]
    elif sport == "ligue1":
        file_path = LIGUE1_RAW_HISTORY
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
                home_q2_score = int(row["home_q2"]) if "home_q2" in row and not pd.isna(row["home_q2"]) else None
                away_q2_score = int(row["away_q2"]) if "away_q2" in row and not pd.isna(row["away_q2"]) else None
                home_q3_score = int(row["home_q3"]) if "home_q3" in row and not pd.isna(row["home_q3"]) else None
                away_q3_score = int(row["away_q3"]) if "away_q3" in row and not pd.isna(row["away_q3"]) else None
                home_q4_score = int(row["home_q4"]) if "home_q4" in row and not pd.isna(row["home_q4"]) else None
                away_q4_score = int(row["away_q4"]) if "away_q4" in row and not pd.isna(row["away_q4"]) else None
                home_f5_score = None
                away_f5_score = None
            elif sport in {"mlb", "kbo", "ncaa_baseball", "triple_a"}:
                home_score = int(row["home_runs_total"])
                away_score = int(row["away_runs_total"])
                home_q1_score = int(row["home_r1"])
                away_q1_score = int(row["away_r1"])
                home_q2_score = None
                away_q2_score = None
                home_q3_score = None
                away_q3_score = None
                home_q4_score = None
                away_q4_score = None
                home_f5_score = int(row["home_runs_f5"])
                away_f5_score = int(row["away_runs_f5"])
            else:
                home_score = int(row["home_score"])
                away_score = int(row["away_score"])
                home_q1_score = None
                away_q1_score = None
                home_q2_score = None
                away_q2_score = None
                home_q3_score = None
                away_q3_score = None
                home_q4_score = None
                away_q4_score = None
                home_f5_score = None
                away_f5_score = None

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

            payload = {
                "date": str(row["date"]),
                "home_team": home_team,
                "away_team": away_team,
                "home_score": home_score,
                "away_score": away_score,
                "home_q1_score": home_q1_score,
                "away_q1_score": away_q1_score,
                "home_q2_score": home_q2_score,
                "away_q2_score": away_q2_score,
                "home_q3_score": home_q3_score,
                "away_q3_score": away_q3_score,
                "home_q4_score": home_q4_score,
                "away_q4_score": away_q4_score,
                "home_f5_score": home_f5_score,
                "away_f5_score": away_f5_score,
                "home_corners": home_corners,
                "away_corners": away_corners,
                "total_corners": total_corners,
                "full_game_winner": full_game_winner,
                "q1_winner": q1_winner,
            }
            lookup[game_id] = payload
            lookup[(str(row["date"]), away_team, home_team)] = payload
        except Exception:
            continue

    return lookup


def _market_lookup_columns_for_sport(sport: str):
    common = [
        "odds_details",
        "odds_over_under",
        "odds_data_quality",
        "home_is_favorite",
        "home_spread",
        "spread_abs",
        "closing_spread_line",
        "closing_total_line",
        "home_moneyline_odds",
        "away_moneyline_odds",
        "closing_moneyline_odds",
        "closing_spread_odds",
        "closing_total_odds",
    ]
    if sport == "tennis":
        return common + ["player_a_odds", "player_b_odds"]
    if sport == "euroleague":
        return common + ["odds_spread"]
    return common


def build_market_lookup_for_sport(sport: str):
    raw_files = SPORT_RAW_FILES.get(sport) or {}
    source_files = [raw_files.get("raw_history"), raw_files.get("upcoming_schedule")]
    use_cols = ["game_id", "date", "home_team", "away_team", *_market_lookup_columns_for_sport(sport)]

    lookup = {}
    for file_path in source_files:
        if not file_path or not Path(file_path).exists():
            continue

        try:
            header_cols = list(pd.read_csv(file_path, nrows=0).columns)
        except Exception:
            continue

        selected_cols = [c for c in use_cols if c in header_cols]
        if len(selected_cols) < 4:
            continue

        try:
            df = pd.read_csv(file_path, usecols=selected_cols, dtype={"game_id": str})
        except Exception:
            continue

        for _, row in df.iterrows():
            game_id = str(row.get("game_id") or "").strip()
            date_str = str(row.get("date") or "").strip()
            home_team = str(row.get("home_team") or "").strip()
            away_team = str(row.get("away_team") or "").strip()
            payload = {
                col: row.get(col)
                for col in selected_cols
                if col not in {"game_id", "date", "home_team", "away_team"}
            }
            if game_id:
                lookup[("game_id", game_id)] = payload
            if date_str and home_team and away_team:
                lookup[("matchup", date_str, away_team, home_team)] = payload
    return lookup


def apply_market_data_if_available(sport: str, events: list[dict]):
    def _is_missing(value):
        if value in (None, "", 0, 0.0, "N/A", "No Line"):
            return True
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    rows = _normalize_events_payload(events)
    if not rows:
        return rows

    lookup = build_market_lookup_for_sport(sport)
    if not lookup:
        return rows

    enriched = []
    for row in rows:
        game_id = str(row.get("game_id") or "").strip()
        date_str = str(row.get("date") or "").strip()
        home_team = str(row.get("home_team") or "").strip()
        away_team = str(row.get("away_team") or "").strip()
        market = lookup.get(("game_id", game_id)) or lookup.get(("matchup", date_str, away_team, home_team)) or {}

        for key, value in market.items():
            if _is_missing(row.get(key)) and not _is_missing(value):
                row[key] = value

        if _is_missing(row.get("closing_total_line")):
            total_line = row.get("odds_over_under")
            if not _is_missing(total_line):
                row["closing_total_line"] = total_line

        if _is_missing(row.get("moneyline_odds")):
            pick = str(row.get("full_game_pick") or "").strip()
            pick_upper = pick.upper()
            if pick and (pick == home_team or pick_upper == "HOME WIN") and not _is_missing(row.get("home_moneyline_odds")):
                row["moneyline_odds"] = row.get("home_moneyline_odds")
                row["pick_ml_odds"] = row.get("home_moneyline_odds")
            elif pick and (pick == away_team or pick_upper == "AWAY WIN") and not _is_missing(row.get("away_moneyline_odds")):
                row["moneyline_odds"] = row.get("away_moneyline_odds")
                row["pick_ml_odds"] = row.get("away_moneyline_odds")

        enriched.append(row)
    return enriched


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

    def _to_int_or_none(value):
        try:
            return int(value)
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
                    "home_q1_score": _to_int_or_none((home.get("linescores") or [{}])[0].get("value"))
                    if isinstance(home.get("linescores"), list) and len(home.get("linescores") or []) > 0 else None,
                    "away_q1_score": _to_int_or_none((away.get("linescores") or [{}])[0].get("value"))
                    if isinstance(away.get("linescores"), list) and len(away.get("linescores") or []) > 0 else None,
                    "home_q2_score": _to_int_or_none((home.get("linescores") or [{}, {}])[1].get("value"))
                    if isinstance(home.get("linescores"), list) and len(home.get("linescores") or []) > 1 else None,
                    "away_q2_score": _to_int_or_none((away.get("linescores") or [{}, {}])[1].get("value"))
                    if isinstance(away.get("linescores"), list) and len(away.get("linescores") or []) > 1 else None,
                    "home_q3_score": _to_int_or_none((home.get("linescores") or [{}, {}, {}])[2].get("value"))
                    if isinstance(home.get("linescores"), list) and len(home.get("linescores") or []) > 2 else None,
                    "away_q3_score": _to_int_or_none((away.get("linescores") or [{}, {}, {}])[2].get("value"))
                    if isinstance(away.get("linescores"), list) and len(away.get("linescores") or []) > 2 else None,
                    "home_q4_score": _to_int_or_none((home.get("linescores") or [{}, {}, {}, {}])[3].get("value"))
                    if isinstance(home.get("linescores"), list) and len(home.get("linescores") or []) > 3 else None,
                    "away_q4_score": _to_int_or_none((away.get("linescores") or [{}, {}, {}, {}])[3].get("value"))
                    if isinstance(away.get("linescores"), list) and len(away.get("linescores") or []) > 3 else None,
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
                header_url = f"https://live.euroleague.net/api/Header?gamecode={gamecode}&seasoncode={season_code}"
                header = requests.get(header_url, timeout=20).json() or {}
            except Exception:
                continue

            try:
                home_score = int(float(header.get("ScoreA", 0) or 0))
                away_score = int(float(header.get("ScoreB", 0) or 0))
                game_date = str(header.get("Date") or "")
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

    def _fetch_live_lookup_triple_a(target_date: str):
        try:
            params = {
                "sportId": 11,
                "startDate": target_date,
                "endDate": target_date,
                "hydrate": "linescore,team,statusFlags",
            }
            payload = requests.get(
                "https://statsapi.mlb.com/api/v1/schedule",
                params=params,
                timeout=20,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PickGOLD/1.0)",
                    "Accept": "application/json",
                },
            ).json() or {}
        except Exception:
            return {}

        out = {}
        for block in payload.get("dates") or []:
            for game in block.get("games") or []:
                try:
                    game_id = str(game.get("gamePk") or "").strip()
                    if not game_id:
                        continue
                    status = game.get("status") or {}
                    abstract = str(status.get("abstractGameState") or "").strip().lower()
                    detailed = str(status.get("detailedState") or "").strip() or ("Final" if abstract == "final" else "Scheduled")
                    home = ((game.get("teams") or {}).get("home") or {})
                    away = ((game.get("teams") or {}).get("away") or {})
                    state = "post" if abstract == "final" else ("in" if abstract == "live" else "pre")
                    out[game_id] = {
                        "status_state": state,
                        "status_description": detailed,
                        "status_detail": detailed,
                        "status_completed": 1 if state == "post" else 0,
                        "home_score": int(float(home.get("score") or 0)),
                        "away_score": int(float(away.get("score") or 0)),
                    }
                except Exception:
                    continue
        return out

    def _fetch_live_lookup(target_sport: str, items: list[dict], target_date: str):
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
        if target_sport == "triple_a":
            return _fetch_live_lookup_triple_a(target_date)
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

        item_date = str(item.get("date", "") or "")[:10]
        if item_date:
            try:
                return datetime.strptime(item_date, "%Y-%m-%d").date() < date.today()
            except Exception:
                return False

        return False

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

    def _safe_float_or_none(value):
        try:
            numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric):
            return None
        return numeric

    def _extract_signed_line(value):
        match = re.search(r"([+-]?\d+(?:\.\d+)?)", str(value or "").strip())
        if not match:
            return None
        try:
            return float(match.group(1))
        except Exception:
            return None

    def _format_signed_line(value):
        if value is None:
            return None
        try:
            numeric = float(value)
        except Exception:
            return None
        if not np.isfinite(numeric) or abs(numeric) < 1e-9:
            return None
        return f"{numeric:+.1f}"

    def _resolve_spread_side_team(item: dict):
        home_team = str(item.get("home_team") or "").strip()
        away_team = str(item.get("away_team") or "").strip()
        side_team = str(item.get("spread_side_team") or "").strip()
        if side_team in {home_team, away_team}:
            return side_team

        spread_pick = str(item.get("spread_pick") or "").strip()
        spread_pick_upper = spread_pick.upper()
        if home_team and home_team.upper() in spread_pick_upper:
            return home_team
        if away_team and away_team.upper() in spread_pick_upper:
            return away_team
        if "HOME" in spread_pick_upper or "LOCAL" in spread_pick_upper:
            return home_team or None
        if "AWAY" in spread_pick_upper or "VISITOR" in spread_pick_upper or "VISITANTE" in spread_pick_upper:
            return away_team or None

        full_game_pick = str(item.get("full_game_pick") or "").strip()
        if full_game_pick in {home_team, away_team} and _resolve_spread_context(item):
            return full_game_pick
        return None

    def _resolve_signed_spread_line(item: dict, side_team: str | None):
        if not side_team:
            return None

        home_team = str(item.get("home_team") or "").strip()
        away_team = str(item.get("away_team") or "").strip()
        line_abs = _safe_float_or_none(item.get("spread_abs"))
        if line_abs is None or abs(line_abs) < 1e-9:
            for candidate in (
                item.get("home_spread"),
                item.get("closing_spread_line"),
                _extract_signed_line(item.get("spread_pick")),
            ):
                numeric = _safe_float_or_none(candidate)
                if numeric is not None and abs(numeric) > 0:
                    line_abs = abs(numeric)
                    break

        home_spread = _safe_float_or_none(item.get("home_spread"))
        if home_spread is not None and abs(home_spread) > 0:
            if side_team == home_team:
                return home_spread
            if side_team == away_team:
                return -home_spread

        predicted_home_margin = _safe_float_or_none(item.get("predicted_home_margin"))
        if line_abs is not None and line_abs > 0 and predicted_home_margin is not None:
            side_margin = predicted_home_margin if side_team == home_team else -predicted_home_margin
            return -line_abs if side_margin >= line_abs else line_abs

        explicit_line = _extract_signed_line(item.get("spread_pick"))
        if explicit_line is not None:
            return explicit_line

        closing_line = _safe_float_or_none(item.get("closing_spread_line"))
        if closing_line is not None and abs(closing_line) > 0:
            if side_team == home_team:
                return closing_line
            if side_team == away_team:
                return -closing_line
        return None

    def _normalize_spread_fields(item: dict):
        side_team = _resolve_spread_side_team(item)
        signed_line = _resolve_signed_spread_line(item, side_team)
        if side_team:
            item["spread_side_team"] = side_team
        if signed_line is not None:
            item["spread_line_signed"] = round(float(signed_line), 2)
            item["spread_line_display"] = _format_signed_line(signed_line)
            if side_team:
                item["spread_pick_display"] = f"{side_team} {item['spread_line_display']}"
        elif side_team:
            item["spread_pick_display"] = side_team

    def _evaluate_signed_spread(item: dict, home_score: int, away_score: int):
        side_team = str(item.get("spread_side_team") or "").strip()
        signed_line = _safe_float_or_none(item.get("spread_line_signed"))
        home_team = str(item.get("home_team") or "").strip()
        away_team = str(item.get("away_team") or "").strip()
        if not side_team or signed_line is None or abs(signed_line) < 1e-9:
            return None

        if side_team == home_team:
            adjusted_score = float(home_score) + float(signed_line)
            opponent_score = float(away_score)
        elif side_team == away_team:
            adjusted_score = float(away_score) + float(signed_line)
            opponent_score = float(home_score)
        else:
            return None

        if adjusted_score > opponent_score:
            return True
        if adjusted_score < opponent_score:
            return False
        return None

    def _synthesize_missing_market_picks(item: dict):
        spread_pick = item.get("spread_pick")
        full_game_pick = str(item.get("full_game_pick") or "").strip()
        if _is_pending_pick(spread_pick) and full_game_pick and _resolve_spread_context(item):
            item["spread_pick"] = full_game_pick
        _normalize_spread_fields(item)

    def _normalize_full_game_schema(item: dict):
        def _first_valid_pick(values):
            for value in values:
                if not _is_pending_pick(value):
                    return str(value).strip()
            return ""

        home_team = str(item.get("home_team") or "").strip().upper()
        away_team = str(item.get("away_team") or "").strip().upper()
        market = str(item.get("market") or "").strip().upper()
        generic_pick = str(item.get("pick") or "").strip()
        generic_pick_u = generic_pick.upper()
        is_generic_team_pick = bool(generic_pick) and generic_pick_u in {home_team, away_team}

        candidate_picks = [
            item.get("full_game_pick"),
            item.get("moneyline_pick"),
            item.get("pick_team"),
        ]
        if is_generic_team_pick or market in {"FULL_GAME", "MONEYLINE", "ML"}:
            candidate_picks.append(generic_pick)
        resolved_pick = _first_valid_pick(candidate_picks)
        if resolved_pick:
            item["full_game_pick"] = resolved_pick

        if item.get("full_game_confidence") in (None, "", "nan"):
            for key in ("moneyline_confidence", "confidence"):
                val = item.get(key)
                if val not in (None, "", "nan"):
                    item["full_game_confidence"] = val
                    break

        if item.get("full_game_recommended_score") in (None, "", "nan"):
            for key in ("moneyline_recommended_score", "recommended_score"):
                val = item.get(key)
                if val not in (None, "", "nan"):
                    item["full_game_recommended_score"] = val
                    break

        if item.get("home_score") in (None, "", "nan"):
            for key in ("actual_home_score", "home_final_score"):
                val = item.get(key)
                if val not in (None, "", "nan"):
                    item["home_score"] = val
                    break

        if item.get("away_score") in (None, "", "nan"):
            for key in ("actual_away_score", "away_final_score"):
                val = item.get(key)
                if val not in (None, "", "nan"):
                    item["away_score"] = val
                    break

        if item.get("full_game_result_winner") in (None, "", "nan"):
            for key in ("moneyline_actual", "actual_winner"):
                val = item.get(key)
                if val not in (None, "", "nan"):
                    item["full_game_result_winner"] = val
                    break

        home_score = _to_int_or_none(item.get("home_score"))
        away_score = _to_int_or_none(item.get("away_score"))
        if item.get("full_game_result_winner") in (None, "", "nan") and home_score is not None and away_score is not None:
            item["full_game_result_winner"] = _winner_from_score(
                str(item.get("home_team") or ""),
                str(item.get("away_team") or ""),
                home_score,
                away_score,
            )

        if home_score is not None and away_score is not None:
            item["home_score"] = home_score
            item["away_score"] = away_score
            if item.get("final_score_text") in (None, "", "N/A"):
                item["final_score_text"] = f"{item.get('away_team')} {away_score} - {item.get('home_team')} {home_score}"
            item["result_available"] = True
            item["status_state"] = "post"
            item["status_description"] = item.get("status_description") or "Final"
            item["status_completed"] = 1

        if _to_bool_or_none(item.get("full_game_hit")) is None:
            for key in ("moneyline_correct", "correct_full_game_adjusted", "correct_full_game", "correct"):
                val = _to_bool_or_none(item.get(key))
                if val is not None:
                    item["full_game_hit"] = val
                    break

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
        h1_hit = _to_bool_or_none(item.get("h1_hit"))

        has_any_hit = any(v is not None for v in (full_game_hit, spread_hit, total_hit, q1_hit, h1_hit))
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
        home_q2_score=None,
        away_q2_score=None,
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
            if sport in {"mlb", "kbo", "ncaa_baseball", "triple_a"} and home_q1_score is not None and away_q1_score is not None:
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
                if home_q2_score is not None and away_q2_score is not None:
                    h1_home = int(home_q1_score) + int(home_q2_score)
                    h1_away = int(away_q1_score) + int(away_q2_score)
                    h1_winner = _winner_from_score(home_team, away_team, h1_home, h1_away)
                    item["h1_result_winner"] = h1_winner
                    if _to_bool_or_none(item.get("h1_hit")) is None:
                        item["h1_hit"] = evaluate_team_pick(
                            pick=str(item.get("h1_pick", "")),
                            home_team=home_team,
                            away_team=away_team,
                            winner=h1_winner,
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
            normalized_spread_hit = _evaluate_signed_spread(item, home_score, away_score)
            if normalized_spread_hit is not None:
                item["correct_spread"] = normalized_spread_hit
            elif _to_bool_or_none(item.get("correct_spread")) is None:
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
        _normalize_full_game_schema(item)
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
            if live.get("home_q1_score") is not None:
                item["home_q1_score"] = live.get("home_q1_score")
            if live.get("away_q1_score") is not None:
                item["away_q1_score"] = live.get("away_q1_score")
            if live.get("home_q2_score") is not None:
                item["home_q2_score"] = live.get("home_q2_score")
            if live.get("away_q2_score") is not None:
                item["away_q2_score"] = live.get("away_q2_score")
            if live.get("home_q3_score") is not None:
                item["home_q3_score"] = live.get("home_q3_score")
            if live.get("away_q3_score") is not None:
                item["away_q3_score"] = live.get("away_q3_score")
            if live.get("home_q4_score") is not None:
                item["home_q4_score"] = live.get("home_q4_score")
            if live.get("away_q4_score") is not None:
                item["away_q4_score"] = live.get("away_q4_score")

        result = lookup.get(game_id)
        if not result:
            matchup_key = (
                str(item.get("date") or "")[:10],
                str(item.get("away_team") or "").strip(),
                str(item.get("home_team") or "").strip(),
            )
            result = lookup.get(matchup_key)

        # If ESPN says the game is still pre/live, do not overwrite it with raw-history
        # rows that may already contain partial scores for the same date.
        if live:
            live_state = str(live.get("status_state") or "").strip().lower()
            live_completed = int(live.get("status_completed", 0) or 0)
            if live_completed != 1 and live_state in {"pre", "in"}:
                result = None

        if result and not _same_event_date(item.get("date"), result.get("date")):
            result = None

        if not result:
            if _event_has_completed_result(item):
                home_team = str(item.get("home_team", "") or "")
                away_team = str(item.get("away_team", "") or "")
                home_score = _to_int_or_none(item.get("home_score"))
                away_score = _to_int_or_none(item.get("away_score"))
                home_q1_score = _to_int_or_none(item.get("home_q1_score"))
                away_q1_score = _to_int_or_none(item.get("away_q1_score"))
                home_q2_score = _to_int_or_none(item.get("home_q2_score"))
                away_q2_score = _to_int_or_none(item.get("away_q2_score"))
                home_q3_score = _to_int_or_none(item.get("home_q3_score"))
                away_q3_score = _to_int_or_none(item.get("away_q3_score"))
                home_q4_score = _to_int_or_none(item.get("home_q4_score"))
                away_q4_score = _to_int_or_none(item.get("away_q4_score"))

                if home_score is not None and away_score is not None and home_team and away_team:
                    item["result_available"] = True
                    item["home_score"] = home_score
                    item["away_score"] = away_score

                    if home_q1_score is not None:
                        item["home_q1_score"] = home_q1_score
                    if away_q1_score is not None:
                        item["away_q1_score"] = away_q1_score
                    if home_q2_score is not None:
                        item["home_q2_score"] = home_q2_score
                    if away_q2_score is not None:
                        item["away_q2_score"] = away_q2_score
                    if home_q3_score is not None:
                        item["home_q3_score"] = home_q3_score
                    if away_q3_score is not None:
                        item["away_q3_score"] = away_q3_score
                    if home_q4_score is not None:
                        item["home_q4_score"] = home_q4_score
                    if away_q4_score is not None:
                        item["away_q4_score"] = away_q4_score

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
                        home_q2_score=home_q2_score,
                        away_q2_score=away_q2_score,
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
        if result.get("home_q2_score") is not None:
            item["home_q2_score"] = result["home_q2_score"]
        if result.get("away_q2_score") is not None:
            item["away_q2_score"] = result["away_q2_score"]
        if result.get("home_q3_score") is not None:
            item["home_q3_score"] = result["home_q3_score"]
        if result.get("away_q3_score") is not None:
            item["away_q3_score"] = result["away_q3_score"]
        if result.get("home_q4_score") is not None:
            item["home_q4_score"] = result["home_q4_score"]
        if result.get("away_q4_score") is not None:
            item["away_q4_score"] = result["away_q4_score"]

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
            home_q2_score=result.get("home_q2_score"),
            away_q2_score=result.get("away_q2_score"),
            home_f5_score=result.get("home_f5_score"),
            away_f5_score=result.get("away_f5_score"),
            total_corners=result.get("total_corners"),
        )

        enriched.append(item)

    return enriched


def enrich_predictions_if_available(sport: str, events: list, lookup: dict | None = None):
    events = _normalize_events_payload(events)
    if sport in {"nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "ncaa_baseball", "triple_a"}:
        return enrich_predictions_with_results(sport, events, lookup=lookup)
    return events


def _event_market_hits(event: dict) -> dict:
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

    h1_hit = _to_bool_or_none(event.get("h1_hit"))
    if h1_hit is not None:
        markets["first_half"] = h1_hit

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


def _to_float_or_none(value):
    try:
        v = float(value)
    except Exception:
        return None
    if pd.isna(v):
        return None
    return v


def _team_form_classify(score_pct: float):
    if score_pct < 30.0:
        return "Mala racha"
    if score_pct < 45.0:
        return "Racha media"
    if score_pct < 60.0:
        return "Racha normal"
    if score_pct < 75.0:
        return "Buena racha"
    return "Muy buena racha"


def _team_form_streak_label(outcomes: list[str]):
    if not outcomes:
        return "Sin datos"
    first = outcomes[0]
    count = 0
    for value in outcomes:
        if value == first:
            count += 1
        else:
            break
    return f"{first}{count}"


def _team_form_points_for_outcome(outcome: str):
    if outcome == "W":
        return 3
    if outcome == "D":
        return 1
    return 0


def _team_form_score_from_window(outcomes_window: list[str]):
    if not outcomes_window:
        return 0.0
    pts = sum(_team_form_points_for_outcome(o) for o in outcomes_window)
    max_pts = len(outcomes_window) * 3
    if max_pts <= 0:
        return 0.0
    return (pts / max_pts) * 100.0


def _team_form_estimate_outcome_probs(records: list[dict], lookback: int = 20):
    recent = records[-max(3, int(lookback)):]
    if not recent:
        return {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}

    # Recency-weighted frequencies + Laplace smoothing.
    weights = np.linspace(0.5, 1.0, num=len(recent))
    weighted = {"W": 1.0, "D": 1.0, "L": 1.0}
    for rec, w in zip(recent, weights):
        outcome = str(rec.get("outcome") or "")
        if outcome in weighted:
            weighted[outcome] += float(w)

    total = float(sum(weighted.values()))
    if total <= 0:
        return {"W": 1 / 3, "D": 1 / 3, "L": 1 / 3}
    return {k: float(v / total) for k, v in weighted.items()}


def _team_form_monte_carlo_forecast(
    records: list[dict],
    window_games: int,
    horizon_games: int = 5,
    simulations: int = 4000,
):
    if not records:
        return None

    horizon = max(1, min(int(horizon_games), 12))
    sims = max(500, min(int(simulations), 20000))
    probs = _team_form_estimate_outcome_probs(records, lookback=20)
    p_w = float(probs.get("W", 1 / 3))
    p_d = float(probs.get("D", 1 / 3))
    p_l = float(probs.get("L", 1 / 3))
    # Ensure numeric stability.
    total = p_w + p_d + p_l
    if total <= 0:
        p_w = p_d = p_l = 1 / 3
    else:
        p_w, p_d, p_l = (p_w / total), (p_d / total), (p_l / total)

    base_window = [str(r.get("outcome") or "L") for r in records[-max(3, int(window_games)):]]
    current_score = _team_form_score_from_window(base_window)

    choices = np.array(["W", "D", "L"], dtype=object)
    probs_arr = np.array([p_w, p_d, p_l], dtype=float)
    sampled = np.random.choice(choices, size=(sims, horizon), p=probs_arr)

    final_scores = np.zeros(sims, dtype=float)
    deltas = np.zeros(sims, dtype=float)
    for i in range(sims):
        window = list(base_window)
        for step in sampled[i]:
            window.append(str(step))
            if len(window) > window_games:
                window.pop(0)
        score = _team_form_score_from_window(window)
        final_scores[i] = score
        deltas[i] = score - current_score

    improve_prob = float(np.mean(deltas >= 3.0))
    decline_prob = float(np.mean(deltas <= -3.0))
    stable_prob = float(max(0.0, 1.0 - improve_prob - decline_prob))

    exp_score = float(np.mean(final_scores))
    exp_delta = float(np.mean(deltas))
    p25 = float(np.percentile(final_scores, 25))
    p75 = float(np.percentile(final_scores, 75))

    trend_label = "estable"
    if improve_prob > 0.5:
        trend_label = "mejora probable"
    elif decline_prob > 0.5:
        trend_label = "baja probable"

    return {
        "horizon_games": horizon,
        "simulations": sims,
        "outcome_probs": {
            "win": round(p_w, 4),
            "draw": round(p_d, 4),
            "loss": round(p_l, 4),
        },
        "improve_probability": round(improve_prob, 4),
        "stable_probability": round(stable_prob, 4),
        "decline_probability": round(decline_prob, 4),
        "expected_score_pct": round(exp_score, 2),
        "expected_change_pct": round(exp_delta, 2),
        "score_p25": round(p25, 2),
        "score_p75": round(p75, 2),
        "trend_label": trend_label,
    }


def _team_form_raw_file_for_sport(sport: str) -> Optional[Path]:
    raw_info = SPORT_RAW_FILES.get(sport) or {}
    raw_history = raw_info.get("raw_history")
    if isinstance(raw_history, Path):
        return raw_history
    return None


def _team_form_sources_fingerprint(sports: list[str]):
    parts = []
    for sport in sports:
        raw_path = _team_form_raw_file_for_sport(sport)
        if not raw_path or not raw_path.exists():
            parts.append((sport, None, None))
            continue
        try:
            stat = raw_path.stat()
            parts.append((sport, int(stat.st_mtime), int(stat.st_size)))
        except Exception:
            parts.append((sport, None, None))
    return tuple(parts)


def _team_form_resolved_score_columns(columns: list[str]):
    home_candidates = ["home_score", "home_pts_total", "home_runs_total"]
    away_candidates = ["away_score", "away_pts_total", "away_runs_total"]
    home_col = next((c for c in home_candidates if c in columns), None)
    away_col = next((c for c in away_candidates if c in columns), None)
    return home_col, away_col


def _team_form_records_from_sport_raw(sport: str):
    raw_path = _team_form_raw_file_for_sport(sport)
    if not raw_path or not raw_path.exists():
        return []

    try:
        df = pd.read_csv(raw_path, low_memory=False)
    except Exception:
        return []

    if df.empty or "date" not in df.columns:
        return []

    records = []

    # Tennis schema (player_a/player_b/winner)
    if {"player_a", "player_b", "winner"}.issubset(set(df.columns)):
        for row in df.itertuples(index=False):
            row_dict = row._asdict()
            date_raw = str(row_dict.get("date") or "").strip()
            if not date_raw:
                continue

            player_a = str(row_dict.get("player_a") or "").strip()
            player_b = str(row_dict.get("player_b") or "").strip()
            winner = str(row_dict.get("winner") or "").strip()
            if not player_a or not player_b or not winner:
                continue

            ts = pd.to_datetime(date_raw, errors="coerce")
            if pd.isna(ts):
                continue

            a_sets = _to_float_or_none(row_dict.get("player_a_sets"))
            b_sets = _to_float_or_none(row_dict.get("player_b_sets"))
            if a_sets is None or b_sets is None:
                a_sets = _to_float_or_none(row_dict.get("player_a_games")) or 0.0
                b_sets = _to_float_or_none(row_dict.get("player_b_games")) or 0.0

            if winner == player_a:
                outcome_a, outcome_b = "W", "L"
            elif winner == player_b:
                outcome_a, outcome_b = "L", "W"
            else:
                outcome_a, outcome_b = "D", "D"

            records.append(
                {
                    "sport": sport,
                    "team": player_a,
                    "date": str(ts.date()),
                    "ts": float(ts.timestamp()),
                    "outcome": outcome_a,
                    "gf": float(a_sets),
                    "ga": float(b_sets),
                }
            )
            records.append(
                {
                    "sport": sport,
                    "team": player_b,
                    "date": str(ts.date()),
                    "ts": float(ts.timestamp()),
                    "outcome": outcome_b,
                    "gf": float(b_sets),
                    "ga": float(a_sets),
                }
            )
        return records

    required_team_cols = {"home_team", "away_team"}
    if not required_team_cols.issubset(set(df.columns)):
        return []

    home_col, away_col = _team_form_resolved_score_columns(list(df.columns))
    if not home_col or not away_col:
        return []

    time_col = "time" if "time" in df.columns else None
    for row in df.itertuples(index=False):
        row_dict = row._asdict()
        date_raw = str(row_dict.get("date") or "").strip()
        if not date_raw:
            continue

        home_team = str(row_dict.get("home_team") or "").strip()
        away_team = str(row_dict.get("away_team") or "").strip()
        if not home_team or not away_team:
            continue

        home_score = _to_float_or_none(row_dict.get(home_col))
        away_score = _to_float_or_none(row_dict.get(away_col))
        if home_score is None or away_score is None:
            continue

        time_raw = str(row_dict.get(time_col) or "00:00").strip() if time_col else "00:00"
        if not re.match(r"^\d{1,2}:\d{2}$", time_raw):
            time_raw = "00:00"

        ts = pd.to_datetime(f"{date_raw} {time_raw}", errors="coerce")
        if pd.isna(ts):
            ts = pd.to_datetime(date_raw, errors="coerce")
        if pd.isna(ts):
            continue

        if home_score > away_score:
            home_outcome, away_outcome = "W", "L"
        elif away_score > home_score:
            home_outcome, away_outcome = "L", "W"
        else:
            home_outcome, away_outcome = "D", "D"

        records.append(
            {
                "sport": sport,
                "team": home_team,
                "date": str(ts.date()),
                "ts": float(ts.timestamp()),
                "outcome": home_outcome,
                "gf": float(home_score),
                "ga": float(away_score),
            }
        )
        records.append(
            {
                "sport": sport,
                "team": away_team,
                "date": str(ts.date()),
                "ts": float(ts.timestamp()),
                "outcome": away_outcome,
                "gf": float(away_score),
                "ga": float(home_score),
            }
        )

    return records


def build_team_form_summary(
    sports: list[str],
    window_games: int = 8,
    min_games: int = 3,
    as_of_date: Optional[str] = None,
):
    cutoff_date = None
    if as_of_date:
        cutoff_date = parse_date_str(as_of_date)

    all_records = []
    for sport in sports:
        sport_records = _team_form_records_from_sport_raw(sport)
        if cutoff_date is not None:
            filtered = []
            for rec in sport_records:
                try:
                    rec_date = parse_date_str(str(rec.get("date") or ""))
                except HTTPException:
                    continue
                if rec_date <= cutoff_date:
                    filtered.append(rec)
            sport_records = filtered
        all_records.extend(sport_records)

    by_team = {}
    for rec in all_records:
        key = (str(rec["sport"]), str(rec["team"]))
        by_team.setdefault(key, []).append(rec)

    rows = []
    for (sport, team), items in by_team.items():
        sorted_items = sorted(items, key=lambda x: x["ts"], reverse=True)
        recent = sorted_items[: max(1, int(window_games))]
        if len(recent) < max(1, int(min_games)):
            continue

        wins = sum(1 for x in recent if x["outcome"] == "W")
        losses = sum(1 for x in recent if x["outcome"] == "L")
        draws = sum(1 for x in recent if x["outcome"] == "D")
        gf = float(sum(float(x["gf"]) for x in recent))
        ga = float(sum(float(x["ga"]) for x in recent))
        points = float((wins * 3) + draws)
        max_points = float(len(recent) * 3)
        score_pct = float((points / max_points) * 100.0) if max_points > 0 else 0.0

        rows.append(
            {
                "sportKey": sport,
                "sportLabel": SPORTS_CONFIG.get(sport, {}).get("label", sport.upper()),
                "teamCode": team,
                "teamName": team,
                "gamesSample": len(recent),
                "formScorePct": round(score_pct, 2),
                "wins": int(wins),
                "losses": int(losses),
                "draws": int(draws),
                "gf": round(gf, 2),
                "ga": round(ga, 2),
                "diff": round(gf - ga, 2),
                "streak": _team_form_streak_label([str(x["outcome"]) for x in recent]),
                "formLabel": _team_form_classify(score_pct),
                "lastMatchDate": str(recent[0]["date"]),
            }
        )

    rows = sorted(rows, key=lambda x: (float(x["formScorePct"]), int(x["gamesSample"])), reverse=True)
    return {
        "generated_at": datetime.now().isoformat(),
        "window_games": int(window_games),
        "min_games": int(min_games),
        "rows": rows,
    }


def _team_form_snapshot_file(date_str: str) -> Path:
    return TEAM_FORM_SNAPSHOTS_DIR / f"{date_str}.json"


def _parse_month_str(month_str: str) -> str:
    m = str(month_str or "").strip()
    if not re.match(r"^\d{4}-\d{2}$", m):
        raise HTTPException(status_code=400, detail="Formato de mes invalido. Usa YYYY-MM.")
    return m


def _snapshot_has_pending(snapshot: Optional[dict]) -> bool:
    if not isinstance(snapshot, dict):
        return True
    totals = snapshot.get("totals") or {}
    pending = totals.get("pending")
    try:
        return int(pending) > 0
    except Exception:
        pass
    return True


def _admin_capture_team_form_snapshot(date_str: str, window_games: int = 8, force: bool = False):
    target = parse_date_str(date_str).strftime("%Y-%m-%d")
    existing = _admin_load_team_form_snapshot(target)
    if (not force) and existing is not None and (not _snapshot_has_pending(existing)):
        return existing

    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "ncaa_baseball", "tennis", "triple_a"]

    form_payload = build_team_form_summary(
        sports=sports,
        window_games=window_games,
        min_games=3,
        as_of_date=target,
    )
    form_rows = list(form_payload.get("rows") or [])
    form_map = {(str(r.get("sportKey")), str(r.get("teamCode"))): r for r in form_rows}

    team_stats = {}
    sport_stats = {}

    for sport in sports:
        try:
            events = _best_picks_load_events_raw_by_date(sport, target)
            lookup = build_results_lookup_for_sport(sport)
            events = enrich_predictions_if_available(sport, events, lookup=lookup)
        except Exception:
            events = []

        sport_total = 0
        sport_hits = 0
        sport_pending = 0

        for event in events:
            picked_team = _resolve_picked_team(event)
            if not picked_team:
                continue

            full_hit = _event_market_hits(event).get("full_game")
            key = (sport, str(picked_team))
            stat = team_stats.setdefault(
                key,
                {
                    "sport": sport,
                    "sport_label": SPORTS_CONFIG.get(sport, {}).get("label", sport.upper()),
                    "team": str(picked_team),
                    "picks": 0,
                    "hits": 0,
                    "pending": 0,
                    "avg_form_score_acc": 0.0,
                    "avg_form_score_n": 0,
                    "form_label": None,
                },
            )
            stat["picks"] += 1
            sport_total += 1

            if full_hit is True:
                stat["hits"] += 1
                sport_hits += 1
            elif full_hit is None:
                stat["pending"] += 1
                sport_pending += 1

            mapped = form_map.get((sport, str(picked_team)))
            if mapped:
                stat["avg_form_score_acc"] += float(mapped.get("formScorePct") or 0.0)
                stat["avg_form_score_n"] += 1
                if not stat["form_label"]:
                    stat["form_label"] = mapped.get("formLabel")

        if sport_total > 0:
            sport_stats[sport] = {
                "sport": sport,
                "sport_label": SPORTS_CONFIG.get(sport, {}).get("label", sport.upper()),
                "picks": sport_total,
                "hits": sport_hits,
                "pending": sport_pending,
                "hit_rate": round((sport_hits / sport_total) if sport_total > 0 else 0.0, 4),
            }

    teams_out = []
    for stat in team_stats.values():
        picks = int(stat["picks"])
        hits = int(stat["hits"])
        pending = int(stat["pending"])
        settled = max(0, picks - pending)
        hit_rate = (hits / settled) if settled > 0 else 0.0
        avg_form_score = (
            stat["avg_form_score_acc"] / stat["avg_form_score_n"]
            if int(stat["avg_form_score_n"]) > 0
            else None
        )
        teams_out.append(
            {
                "sport": stat["sport"],
                "sport_label": stat["sport_label"],
                "team": stat["team"],
                "picks": picks,
                "hits": hits,
                "pending": pending,
                "settled": settled,
                "hit_rate": round(hit_rate, 4),
                "avg_form_score_pct": round(avg_form_score, 2) if avg_form_score is not None else None,
                "form_label": stat["form_label"],
            }
        )

    teams_out.sort(key=lambda x: (x["hit_rate"], x["hits"], x["picks"]), reverse=True)
    sports_out = sorted(list(sport_stats.values()), key=lambda x: x["sport"])

    snapshot = {
        "snapshot_date": target,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "is_partial": bool(any(int(t.get("pending") or 0) > 0 for t in teams_out)),
        "window_games": int(window_games),
        "sports": sports_out,
        "teams": teams_out,
        "totals": {
            "sports_count": len(sports_out),
            "teams_count": len(teams_out),
            "picks": int(sum(x["picks"] for x in teams_out)),
            "hits": int(sum(x["hits"] for x in teams_out)),
            "pending": int(sum(x["pending"] for x in teams_out)),
        },
    }

    file_path = _team_form_snapshot_file(target)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return snapshot


def _admin_load_team_form_snapshot(date_str: str):
    target = parse_date_str(date_str).strftime("%Y-%m-%d")
    path = _team_form_snapshot_file(target)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _admin_team_form_snapshot_monthly_summary(month_str: str):
    month_key = _parse_month_str(month_str)
    files = sorted(TEAM_FORM_SNAPSHOTS_DIR.glob(f"{month_key}-*.json"))

    snapshots = []
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                snapshots.append(payload)
        except Exception:
            continue

    by_team = {}
    by_sport = {}
    for snap in snapshots:
        date_str = str(snap.get("snapshot_date") or "")
        for team in snap.get("teams") or []:
            key = (str(team.get("sport")), str(team.get("team")))
            agg = by_team.setdefault(
                key,
                {
                    "sport": str(team.get("sport")),
                    "sport_label": str(team.get("sport_label") or ""),
                    "team": str(team.get("team")),
                    "days": 0,
                    "picks": 0,
                    "hits": 0,
                    "pending": 0,
                    "form_acc": 0.0,
                    "form_n": 0,
                    "last_form_label": None,
                    "last_snapshot_date": None,
                },
            )
            agg["days"] += 1
            agg["picks"] += int(team.get("picks") or 0)
            agg["hits"] += int(team.get("hits") or 0)
            agg["pending"] += int(team.get("pending") or 0)
            avg_form = team.get("avg_form_score_pct")
            if avg_form is not None:
                agg["form_acc"] += float(avg_form)
                agg["form_n"] += 1
            if date_str and (not agg["last_snapshot_date"] or date_str >= agg["last_snapshot_date"]):
                agg["last_snapshot_date"] = date_str
                agg["last_form_label"] = team.get("form_label")

        for sport_row in snap.get("sports") or []:
            sport = str(sport_row.get("sport") or "")
            agg_sport = by_sport.setdefault(
                sport,
                {
                    "sport": sport,
                    "sport_label": str(sport_row.get("sport_label") or sport.upper()),
                    "days": 0,
                    "picks": 0,
                    "hits": 0,
                    "pending": 0,
                },
            )
            agg_sport["days"] += 1
            agg_sport["picks"] += int(sport_row.get("picks") or 0)
            agg_sport["hits"] += int(sport_row.get("hits") or 0)
            agg_sport["pending"] += int(sport_row.get("pending") or 0)

    teams_out = []
    for agg in by_team.values():
        settled = max(0, int(agg["picks"]) - int(agg["pending"]))
        hit_rate = (float(agg["hits"]) / settled) if settled > 0 else 0.0
        avg_form = (agg["form_acc"] / agg["form_n"]) if int(agg["form_n"]) > 0 else None
        teams_out.append(
            {
                "sport": agg["sport"],
                "sport_label": agg["sport_label"],
                "team": agg["team"],
                "days": int(agg["days"]),
                "picks": int(agg["picks"]),
                "hits": int(agg["hits"]),
                "pending": int(agg["pending"]),
                "settled": settled,
                "hit_rate": round(hit_rate, 4),
                "avg_form_score_pct": round(avg_form, 2) if avg_form is not None else None,
                "last_form_label": agg["last_form_label"],
                "last_snapshot_date": agg["last_snapshot_date"],
            }
        )
    teams_out.sort(key=lambda x: (x["hit_rate"], x["hits"], x["picks"]), reverse=True)

    sports_out = []
    for agg in by_sport.values():
        picks = int(agg["picks"])
        hits = int(agg["hits"])
        pending = int(agg["pending"])
        settled = max(0, picks - pending)
        sports_out.append(
            {
                "sport": agg["sport"],
                "sport_label": agg["sport_label"],
                "days": int(agg["days"]),
                "picks": picks,
                "hits": hits,
                "pending": pending,
                "settled": settled,
                "hit_rate": round((hits / settled) if settled > 0 else 0.0, 4),
            }
        )
    sports_out.sort(key=lambda x: x["sport"])

    return {
        "month": month_key,
        "snapshots_count": len(snapshots),
        "snapshots_dates": sorted([str(s.get("snapshot_date") or "") for s in snapshots if str(s.get("snapshot_date") or "")]),
        "sports": sports_out,
        "teams": teams_out,
        "top_teams": teams_out[:25],
        "generated_at": datetime.now().isoformat(),
    }


def _run_auto_team_form_snapshot_for_date(date_str: str):
    global TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE
    try:
        _admin_capture_team_form_snapshot(date_str=date_str, window_games=8)
    except Exception:
        # Non-blocking background task; failures are intentionally silent.
        pass
    finally:
        with TEAM_FORM_AUTO_SNAPSHOT_LOCK:
            if TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE == date_str:
                TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE = None


def _schedule_daily_team_form_snapshot_if_needed():
    global TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE
    today_str = date.today().strftime("%Y-%m-%d")
    existing = _admin_load_team_form_snapshot(today_str)
    if existing is not None and (not _snapshot_has_pending(existing)):
        return

    with TEAM_FORM_AUTO_SNAPSHOT_LOCK:
        if TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE == today_str:
            return
        TEAM_FORM_AUTO_SNAPSHOT_IN_PROGRESS_DATE = today_str

    worker = threading.Thread(
        target=_run_auto_team_form_snapshot_for_date,
        args=(today_str,),
        daemon=True,
    )
    worker.start()


def _admin_team_form_snapshots_compare(base_date: str, target_date: str):
    d1 = parse_date_str(base_date).strftime("%Y-%m-%d")
    d2 = parse_date_str(target_date).strftime("%Y-%m-%d")
    snap_a = _admin_load_team_form_snapshot(d1)
    snap_b = _admin_load_team_form_snapshot(d2)
    if snap_a is None:
        raise HTTPException(status_code=404, detail=f"No existe snapshot para {d1}.")
    if snap_b is None:
        raise HTTPException(status_code=404, detail=f"No existe snapshot para {d2}.")

    by_team_a = {
        (str(t.get("sport")), str(t.get("team"))): t
        for t in (snap_a.get("teams") or [])
    }
    by_team_b = {
        (str(t.get("sport")), str(t.get("team"))): t
        for t in (snap_b.get("teams") or [])
    }
    all_keys = sorted(set(by_team_a.keys()) | set(by_team_b.keys()))

    rows = []
    for key in all_keys:
        a = by_team_a.get(key, {})
        b = by_team_b.get(key, {})
        settled_a = int(a.get("settled") or max(0, int(a.get("picks") or 0) - int(a.get("pending") or 0)))
        settled_b = int(b.get("settled") or max(0, int(b.get("picks") or 0) - int(b.get("pending") or 0)))
        hit_a = float(a.get("hit_rate") or 0.0) if settled_a > 0 else None
        hit_b = float(b.get("hit_rate") or 0.0) if settled_b > 0 else None
        form_a = a.get("avg_form_score_pct")
        form_b = b.get("avg_form_score_pct")
        delta_hit = (hit_b - hit_a) if (hit_a is not None and hit_b is not None) else None
        delta_form = (
            float(form_b) - float(form_a)
            if (form_a is not None and form_b is not None)
            else None
        )

        effective_delta = None
        effective_metric = None
        if delta_hit is not None:
            effective_delta = float(delta_hit)
            effective_metric = "hit_rate"
        elif delta_form is not None:
            effective_delta = float(delta_form) / 100.0
            effective_metric = "form_score"

        rows.append(
            {
                "sport": key[0],
                "sport_label": str((b or a).get("sport_label") or key[0].upper()),
                "team": key[1],
                "base": {
                    "date": d1,
                    "picks": int(a.get("picks") or 0),
                    "hits": int(a.get("hits") or 0),
                    "pending": int(a.get("pending") or 0),
                    "settled": settled_a,
                    "hit_rate": round(hit_a, 4) if hit_a is not None else None,
                    "avg_form_score_pct": form_a,
                },
                "target": {
                    "date": d2,
                    "picks": int(b.get("picks") or 0),
                    "hits": int(b.get("hits") or 0),
                    "pending": int(b.get("pending") or 0),
                    "settled": settled_b,
                    "hit_rate": round(hit_b, 4) if hit_b is not None else None,
                    "avg_form_score_pct": form_b,
                },
                "delta_hit_rate": round(delta_hit, 4) if delta_hit is not None else None,
                "delta_form_score_pct": round(delta_form, 2) if delta_form is not None else None,
                "effective_delta": round(effective_delta, 4) if effective_delta is not None else None,
                "effective_metric": effective_metric,
            }
        )

    comparable = [r for r in rows if r.get("effective_delta") is not None]
    improved = [r for r in comparable if float(r["effective_delta"]) > 0]
    declined = [r for r in comparable if float(r["effective_delta"]) < 0]
    stable = [r for r in comparable if float(r["effective_delta"]) == 0]

    improved.sort(key=lambda x: (x["effective_delta"], x.get("delta_form_score_pct") or 0), reverse=True)
    declined.sort(key=lambda x: (x["effective_delta"], x.get("delta_form_score_pct") or 0))

    return {
        "base_date": d1,
        "target_date": d2,
        "teams_compared": len(rows),
        "teams_comparable": len(comparable),
        "improved_count": len(improved),
        "declined_count": len(declined),
        "stable_count": len(stable),
        "top_improved": improved[:20],
        "top_declined": declined[:20],
        "rows": rows,
        "generated_at": datetime.now().isoformat(),
    }


def build_team_form_team_detail(
    sport: str,
    team: str,
    window_games: int = 8,
    max_points: int = 30,
    horizon_games: int = 5,
    simulations: int = 4000,
):
    sport_key = str(sport or "").strip().lower()
    team_key = str(team or "").strip()
    if not sport_key or not team_key:
        raise HTTPException(status_code=400, detail="Parámetros sport/team inválidos.")
    if sport_key not in SPORTS_CONFIG:
        raise HTTPException(status_code=404, detail=f"Deporte no soportado: {sport_key}")

    records = [r for r in _team_form_records_from_sport_raw(sport_key) if str(r.get("team") or "") == team_key]
    if not records:
        raise HTTPException(status_code=404, detail="No hay historial suficiente para ese equipo.")

    records = sorted(records, key=lambda x: x["ts"])
    window = max(3, min(int(window_games), 20))
    limit_points = max(10, min(int(max_points), 80))

    timeline = []
    for idx, rec in enumerate(records):
        start = max(0, idx - window + 1)
        sample = records[start: idx + 1]
        if len(sample) < 3:
            continue

        wins = sum(1 for x in sample if x["outcome"] == "W")
        draws = sum(1 for x in sample if x["outcome"] == "D")
        losses = sum(1 for x in sample if x["outcome"] == "L")
        points = (wins * 3) + draws
        max_pts = len(sample) * 3
        score_pct = (points / max_pts) * 100.0 if max_pts > 0 else 0.0

        timeline.append(
            {
                "date": str(rec.get("date")),
                "outcome": str(rec.get("outcome")),
                "gf": float(rec.get("gf") or 0.0),
                "ga": float(rec.get("ga") or 0.0),
                "window_games": len(sample),
                "score_pct": round(float(score_pct), 2),
                "form_label": _team_form_classify(float(score_pct)),
                "streak": _team_form_streak_label([str(x.get("outcome")) for x in reversed(sample)]),
                "wins": int(wins),
                "draws": int(draws),
                "losses": int(losses),
            }
        )

    if not timeline:
        raise HTTPException(status_code=404, detail="No hay suficiente muestra para curva de racha.")

    timeline = timeline[-limit_points:]
    latest = timeline[-1]
    first = timeline[0]
    forecast = _team_form_monte_carlo_forecast(
        records=records,
        window_games=window,
        horizon_games=horizon_games,
        simulations=simulations,
    )

    return {
        "generated_at": datetime.now().isoformat(),
        "sport": sport_key,
        "sport_label": SPORTS_CONFIG.get(sport_key, {}).get("label", sport_key.upper()),
        "team": team_key,
        "window_games": window,
        "timeline": timeline,
        "current": latest,
        "change_vs_start_pct": round(float(latest["score_pct"]) - float(first["score_pct"]), 2),
        "forecast": forecast,
    }


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
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "ncaa_baseball", "triple_a"]
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
        "first_half": ["h1_hit"],
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

    status_completed = event.get("status_completed")
    try:
        if status_completed is not None and int(float(status_completed)) != 1:
            return None
    except Exception:
        pass

    status_state = str(event.get("status_state") or "").strip().lower()
    if status_state and status_state not in {"post", "final", "completed"}:
        return None

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
            try:
                line = float(m.group(1)) if m else float(event.get("odds_over_under") or 0.0)
            except Exception:
                line = 0.0
            if line <= 0:
                return None
            return total_points > line

        if "UNDER" in pick_upper:
            m = re.search(r"(\d+(?:\.\d+)?)", pick_text)
            try:
                line = float(m.group(1)) if m else float(event.get("odds_over_under") or 0.0)
            except Exception:
                line = 0.0
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
        return None

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
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "triple_a"]
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

    if selected_date >= (today - timedelta(days=1)):
        if live_file.exists():
            return live_file
        if hist_file.exists():
            return hist_file
    elif selected_date < today:
        if hist_file.exists():
            return hist_file
        if live_file.exists():
            return live_file
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
        "h1_hit",
        "actual_result",
        "final_score_text",
    ]

    merged = []
    for event in events:
        item = dict(event)
        game_id = str(item.get("game_id", "")).strip()
        hist = by_game_id.get(game_id)
        state = str(item.get("status_state") or "").strip().lower()
        completed = int(item.get("status_completed", 0) or 0)
        status_description = str(item.get("status_description") or "").strip().lower()
        status_detail = str(item.get("status_detail") or "").strip().lower()
        status_text = f"{status_description} {status_detail}".strip()
        non_final_states = {
            "pre",
            "in",
            "live",
            "in_progress",
            "inprogress",
            "status_in_progress",
            "halftime",
            "mid",
            "scheduled",
            "not_started",
            "ns",
        }
        has_live_hint = any(
            token in status_text
            for token in (
                "in progress",
                "en vivo",
                "live",
                "q1",
                "q2",
                "q3",
                "q4",
                "quarter",
                "halftime",
                "top",
                "bot",
            )
        )
        should_skip_hints = completed != 1 and (state in non_final_states or has_live_hint)

        if hist and not should_skip_hints:
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
        "bundesliga",
        "ligue1",
        "euroleague",
        "ncaa_baseball",
        "tennis",
        "triple_a",
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


def _best_picks_group_by_sport_payload(picks: list[dict], per_sport_limit: int = 4):
    grouped = {}
    for row in picks or []:
        sport = str(row.get("sport") or "")
        if not sport:
            continue
        grouped.setdefault(sport, []).append(dict(row))

    sections = []
    for sport, sport_rows in grouped.items():
        sport_rows.sort(
            key=lambda x: (
                float(x.get("final_rank_score", 0.0) or 0.0),
                float(x.get("score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        sample = sport_rows[: max(1, int(per_sport_limit))]
        first = sample[0]
        sections.append(
            {
                "sport": sport,
                "sport_label": first.get("sport_label") or sport.upper(),
                "count": len(sport_rows),
                "top_score": round(float(first.get("score", 0.0) or 0.0), 2),
                "top_final_rank_score": round(float(first.get("final_rank_score", 0.0) or 0.0), 3),
                "picks": sample,
            }
        )

    sections.sort(
        key=lambda x: (
            float(x.get("top_final_rank_score", 0.0) or 0.0),
            int(x.get("count", 0) or 0),
        ),
        reverse=True,
    )
    return sections


def _best_picks_trim_payload(payload: dict, top_n: int):
    filtered = _best_picks_filter_excluded_sports(payload)
    picks = list(filtered.get("picks") or [])[:max(1, int(top_n))]
    out = dict(filtered)
    out["top_n"] = len(picks)
    out["picks"] = picks
    out["sports_summary"] = _best_picks_summarize_sports(picks)

    raw_top_by_sport = filtered.get("top_by_sport") or []
    if isinstance(raw_top_by_sport, list) and raw_top_by_sport:
        normalized_sections = []
        for section in raw_top_by_sport:
            if not isinstance(section, dict):
                continue
            section_picks = list(section.get("picks") or [])[:4]
            if not section_picks:
                continue
            normalized = dict(section)
            normalized["count"] = int(section.get("count", len(section_picks)) or len(section_picks))
            normalized["picks"] = section_picks
            normalized_sections.append(normalized)
        out["top_by_sport"] = normalized_sections
    else:
        out["top_by_sport"] = _best_picks_group_by_sport_payload(picks, per_sport_limit=4)
    return out


def _best_picks_generate_snapshot(date_str: str, generation_top_n: int, ranking_mode: str = "balanced"):
    parse_date_str(date_str)
    events_by_sport = _best_picks_events_for_date(date_str)
    calibration_profiles = build_probability_calibration_profiles()
    mode = _best_picks_normalize_ranking_mode(ranking_mode)
    payload = build_daily_best_picks(
        events_by_sport,
        top_n=max(1, int(generation_top_n)),
        calibration_profiles=calibration_profiles,
        include_completed=False,
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
                allow_live=True,
            )
        except Exception:
            pass

        for event in events:
            game_id = str(event.get("game_id") or "")
            if not game_id:
                continue
            lookup[(sport, date_str, game_id)] = event
    return lookup


def _evaluate_best_pick_team_side(pick_text, home_team, away_team, winner):
    pick = str(pick_text or "").strip().upper()
    home = str(home_team or "").strip().upper()
    away = str(away_team or "").strip().upper()
    win = str(winner or "").strip().upper()
    if not pick or not home or not away or not win:
        return None
    if pick in {"HOME", "HOME WIN", "LOCAL"}:
        return win == home
    if pick in {"AWAY", "AWAY WIN", "VISITOR", "VISITANTE"}:
        return win == away
    if home in pick:
        return win == home
    if away in pick:
        return win == away
    return None


def _extract_numeric_line(value):
    import re
    m = re.search(r"(\d+(?:\.\d+)?)", str(value or ""))
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _evaluate_best_pick_total(pick_text, observed_total, fallback_line=None):
    pick = str(pick_text or "").strip().upper()
    if not pick or observed_total is None:
        return None
    line = _extract_numeric_line(pick)
    if line is None:
        try:
            line = float(fallback_line)
        except Exception:
            line = None
    if line is None:
        return None
    if "OVER" in pick:
        return float(observed_total) > float(line)
    if "UNDER" in pick:
        return float(observed_total) < float(line)
    return None


def _coerce_result_bool(value):
    if isinstance(value, bool):
        return value
    try:
        if int(value) in (0, 1):
            return bool(int(value))
    except Exception:
        pass
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes", "si", "acierto", "win", "won"}:
        return True
    if text in {"false", "0", "no", "fallo", "lose", "lost"}:
        return False
    return None


def _best_pick_market_hit(row: dict):
    market = str(row.get("market") or "").strip().lower()

    preferred_keys = {
        "full_game": ["full_game_hit", "correct_full_game_adjusted", "correct_full_game", "correct_full_game_base"],
        "spread": ["correct_spread"],
        "total": ["correct_total_adjusted", "correct_total"],
        "q1_yrfi": ["q1_hit"],
        "q1": ["q1_hit"],
        "h1": ["h1_hit"],
        "btts": ["correct_btts_adjusted", "correct_btts"],
        "f5": ["correct_f5"],
        "home_over": ["correct_home_over"],
        "corners": ["correct_corners_adjusted"],
    }

    for key in preferred_keys.get(market, []):
        value = row.get(key)
        if value is not None:
            return value

    fallback_keys = [
        "full_game_hit",
        "correct_full_game_adjusted",
        "correct_full_game",
        "correct_full_game_base",
        "correct_spread",
        "correct_total_adjusted",
        "correct_total",
        "q1_hit",
        "h1_hit",
        "correct_btts_adjusted",
        "correct_btts",
        "correct_f5",
        "correct_home_over",
        "correct_corners_adjusted",
    ]
    for key in fallback_keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


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

    def _enrich_pick_row(base_row: dict, event_lookup: dict, result_keys: list[str]):
        row = dict(base_row)
        sport = str(row.get("sport") or "")
        date_str = str(row.get("date") or "")
        game_id = str(row.get("game_id") or "")

        event = event_lookup.get((sport, date_str, game_id))
        away_team = str(row.get("away_team") or "").strip().upper()
        home_team = str(row.get("home_team") or "").strip().upper()
        if not isinstance(event, dict) and away_team and home_team:
            event = event_lookup_by_teams.get((sport, date_str, away_team, home_team))
        historical_result = None
        if away_team and home_team:
            historical_result = historical_lookup_by_teams.get((sport, date_str, away_team, home_team))
        if isinstance(event, dict):
            for key in result_keys:
                value = event.get(key)
                if value is not None:
                    row[key] = value
            if row.get("actual_result") is None and event.get("full_game_result_winner") is not None:
                row["actual_result"] = event.get("full_game_result_winner")
        if isinstance(historical_result, dict):
            row["home_score"] = historical_result.get("home_score")
            row["away_score"] = historical_result.get("away_score")
            row["home_q1_score"] = historical_result.get("home_q1_score")
            row["away_q1_score"] = historical_result.get("away_q1_score")
            row["home_q2_score"] = historical_result.get("home_q2_score")
            row["away_q2_score"] = historical_result.get("away_q2_score")
            row["home_q3_score"] = historical_result.get("home_q3_score")
            row["away_q3_score"] = historical_result.get("away_q3_score")
            row["home_q4_score"] = historical_result.get("home_q4_score")
            row["away_q4_score"] = historical_result.get("away_q4_score")
            row["home_f5_score"] = historical_result.get("home_f5_score")
            row["away_f5_score"] = historical_result.get("away_f5_score")
            row["status_state"] = 'post'
            row["status_description"] = 'Final'
            row["status_completed"] = 1
            row["result_available"] = True
            row["final_score_text"] = f"{historical_result.get('away_team')} {historical_result.get('away_score')} - {historical_result.get('home_team')} {historical_result.get('home_score')}"
            row["actual_result"] = historical_result.get("full_game_winner")
            row["full_game_result_winner"] = historical_result.get("full_game_winner")
            if historical_result.get("q1_winner") is not None:
                row["q1_result_winner"] = historical_result.get("q1_winner")

            if row.get("full_game_hit") is None:
                row["full_game_hit"] = _evaluate_best_pick_team_side(
                    row.get("pick"),
                    historical_result.get("home_team"),
                    historical_result.get("away_team"),
                    historical_result.get("full_game_winner"),
                )
            if row.get("q1_hit") is None and historical_result.get("q1_winner") is not None:
                row["q1_hit"] = _evaluate_best_pick_team_side(
                    row.get("pick"),
                    historical_result.get("home_team"),
                    historical_result.get("away_team"),
                    historical_result.get("q1_winner"),
                )
            if row.get("correct_f5") is None and historical_result.get("home_f5_score") is not None and historical_result.get("away_f5_score") is not None:
                if historical_result.get("home_f5_score") > historical_result.get("away_f5_score"):
                    f5_winner = historical_result.get("home_team")
                elif historical_result.get("away_f5_score") > historical_result.get("home_f5_score"):
                    f5_winner = historical_result.get("away_team")
                else:
                    f5_winner = 'TIE'
                row["correct_f5"] = _evaluate_best_pick_team_side(
                    row.get("pick"),
                    historical_result.get("home_team"),
                    historical_result.get("away_team"),
                    f5_winner,
                )
            total_runs = None
            try:
                total_runs = float(historical_result.get("home_score") or 0) + float(historical_result.get("away_score") or 0)
            except Exception:
                total_runs = None
            if row.get("correct_total") is None and total_runs is not None:
                row["correct_total"] = _evaluate_best_pick_total(row.get("pick"), total_runs, row.get("total_line"))
            if row.get("correct_home_over") is None and historical_result.get("home_score") is not None:
                row["correct_home_over"] = _evaluate_best_pick_total(row.get("pick"), historical_result.get("home_score"), None)

        resolved_source = row if isinstance(row, dict) else event
        is_resolved = _event_is_completed(resolved_source) or bool(row.get("result_available"))
        raw_hit = _best_pick_market_hit(row) if is_resolved else None
        hit = _coerce_result_bool(raw_hit)
        row["result_hit"] = hit

        if is_resolved:
            if hit is True:
                row["result_label"] = "ACIERTO"
            elif hit is False:
                row["result_label"] = "FALLO"
            else:
                row["result_label"] = "RESUELTO"
        else:
            row["result_label"] = "PENDIENTE"

        return row

    if not isinstance(payload, dict):
        return payload

    event_lookup = _best_picks_event_lookup_for_payload(payload)
    event_lookup_by_teams = {}
    historical_lookup_by_teams = {}
    sports_in_payload = sorted({str(p.get("sport") or "") for p in (payload.get("picks") or []) if str(p.get("sport") or "")})

    for sport_name in sports_in_payload:
        try:
            sport_lookup = build_results_lookup_for_sport(sport_name)
        except Exception:
            sport_lookup = {}
        for result in (sport_lookup or {}).values():
            if not isinstance(result, dict):
                continue
            away_team = str(result.get("away_team") or "").strip().upper()
            home_team = str(result.get("home_team") or "").strip().upper()
            date_key = str(result.get("date") or "")[:10]
            if away_team and home_team and date_key:
                historical_lookup_by_teams[(sport_name, date_key, away_team, home_team)] = result

    for (sport_key, date_key, _game_id), event in event_lookup.items():
        if not isinstance(event, dict):
            continue
        away_team = str(event.get("away_team") or "").strip().upper()
        home_team = str(event.get("home_team") or "").strip().upper()
        if away_team and home_team:
            event_lookup_by_teams[(str(sport_key), str(date_key), away_team, home_team)] = event
    out = dict(payload)

    result_keys = [
        "correct_full_game",
        "correct_full_game_adjusted",
        "correct_full_game_base",
        "full_game_hit",
        "correct_spread",
        "correct_total",
        "correct_total_adjusted",
        "q1_hit",
        "h1_hit",
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

    out_picks = [
        _enrich_pick_row(pick, event_lookup, result_keys)
        for pick in (payload.get("picks") or [])
    ]
    out["picks"] = out_picks

    top_by_sport = []
    for section in payload.get("top_by_sport") or []:
        if not isinstance(section, dict):
            continue
        section_rows = [
            _enrich_pick_row(pick, event_lookup, result_keys)
            for pick in (section.get("picks") or [])
        ]
        section_rows = [row for row in section_rows if isinstance(row, dict)]
        if section_rows:
            normalized = dict(section)
            normalized["picks"] = section_rows
            top_by_sport.append(normalized)
    out["top_by_sport"] = top_by_sport or _best_picks_group_by_sport_payload(out_picks, per_sport_limit=4)
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
    local_today = date.today().strftime("%Y-%m-%d")
    source_today = _kbo_source_date_from_local(local_today) if sport == "kbo" else local_today

    if sport == "ncaa_baseball":
        candidate_dates = _get_ncaa_baseball_available_dates()
        future_or_today = [d for d in candidate_dates if d >= local_today]
        past = [d for d in candidate_dates if d < local_today]
        scan_dates = future_or_today + list(reversed(past))
        for candidate in scan_dates:
            events = _merge_ncaa_baseball_board(candidate)
            if events:
                return _sanitize_json_values(events)
        return []
    elif sport == "kbo":
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
        pass
    events = apply_market_data_if_available(sport, events)
    payload = _translate_event_dates_for_sport(sport, _normalize_events_payload(events))
    return _sanitize_json_values(payload)


@app.get("/api/{sport}/predictions/{date_str}")
def get_predictions_by_date(sport: str, date_str: str):
    config = ensure_sport_exists(sport)
    source_date = _kbo_source_date_from_local(date_str) if sport == "kbo" else date_str
    if sport == "ncaa_baseball":
        events = _merge_ncaa_baseball_board(date_str)
        return _sanitize_json_values(events)

    try:
        file_path = resolve_prediction_file(
            config["predictions_dir"],
            config["historical_dir"],
            source_date,
        )
    except HTTPException:
        raise
    events = _normalize_events_payload(read_json_file(file_path))
    _, hist_file = get_files_for_date(config["predictions_dir"], config["historical_dir"], source_date)
    events = merge_result_hints_from_historical(events, hist_file)
    overrides_date = source_date if sport == "kbo" else date_str
    events = apply_overrides_to_events(sport, overrides_date, events)
    try:
        events = enrich_predictions_if_available(sport, events)
    except Exception:
        pass
    events = apply_market_data_if_available(sport, events)
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

    if sport == "ncaa_baseball":
        dates.update(_get_ncaa_baseball_available_dates())

    out_dates = sorted(dates)
    if sport == "kbo":
        out_dates = sorted({_kbo_local_date_from_source(d) for d in out_dates})
    return out_dates


@app.get("/api/admin/sport-updates")
def admin_sport_updates(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    sports = [
        _build_admin_sport_pipeline_snapshot(sport)
        for sport in ["nba", "mlb", "tennis", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "ncaa_baseball", "triple_a"]
    ]
    return {"ok": True, "sports": sports}


@app.get("/api/admin/all-sports-update-status")
def admin_all_sports_update_status(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    return {"ok": True, **_copy_all_sports_update_state()}


@app.get("/api/admin/market-accuracy-report")
def admin_market_accuracy_report(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    sports = []
    for sport in [
        "nba",
        "mlb",
        "tennis",
        "kbo",
        "nhl",
        "liga_mx",
        "laliga",
        "bundesliga",
        "ligue1",
        "euroleague",
        "ncaa_baseball",
        "triple_a",
    ]:
        if sport not in SPORTS_CONFIG:
            continue
        sports.append(_load_admin_market_accuracy_report_for_sport(sport))
    return {"ok": True, "sports": sports}


@app.get("/api/admin/team-form-snapshots/status")
def admin_team_form_snapshots_status(
    month: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    _require_admin_session(authorization)
    month_key = _parse_month_str(month) if month else date.today().strftime("%Y-%m")
    summary = _admin_team_form_snapshot_monthly_summary(month_key)
    latest = summary["snapshots_dates"][-1] if summary["snapshots_dates"] else None
    return {
        "ok": True,
        "month": month_key,
        "snapshots_count": summary["snapshots_count"],
        "latest_snapshot_date": latest,
        "snapshots_dates": summary["snapshots_dates"],
    }


@app.get("/api/admin/team-form-snapshots/monthly")
def admin_team_form_snapshots_monthly(
    month: Optional[str] = None,
    authorization: Optional[str] = Header(default=None),
):
    _require_admin_session(authorization)
    month_key = _parse_month_str(month) if month else date.today().strftime("%Y-%m")
    summary = _admin_team_form_snapshot_monthly_summary(month_key)
    return {"ok": True, **summary}


@app.post("/api/admin/team-form-snapshots/capture")
def admin_team_form_snapshots_capture(
    payload: dict = Body(default={}),
    authorization: Optional[str] = Header(default=None),
):
    _require_admin_session(authorization)
    date_str = str((payload or {}).get("date") or date.today().strftime("%Y-%m-%d"))
    window_games = int((payload or {}).get("window_games") or 8)
    force = bool((payload or {}).get("force_refresh", False))
    target = parse_date_str(date_str).strftime("%Y-%m-%d")
    existing = _admin_load_team_form_snapshot(target)
    should_refresh = force or existing is None or _snapshot_has_pending(existing)

    snapshot = _admin_capture_team_form_snapshot(date_str=target, window_games=window_games, force=force)
    monthly = _admin_team_form_snapshot_monthly_summary(date_str[:7])
    return {
        "ok": True,
        "message": f"Snapshot {'actualizado' if should_refresh else 'sin cambios'} para {target}.",
        "updated": bool(should_refresh),
        "snapshot": snapshot,
        "monthly": monthly,
    }


@app.get("/api/admin/team-form-snapshots/compare")
def admin_team_form_snapshots_compare(
    base_date: str,
    target_date: str,
    authorization: Optional[str] = Header(default=None),
):
    _require_admin_session(authorization)
    payload = _admin_team_form_snapshots_compare(base_date=base_date, target_date=target_date)
    return {"ok": True, **payload}


@app.post("/api/admin/update-all-sports")
def admin_update_all_sports(authorization: Optional[str] = Header(default=None)):
    _require_admin_session(authorization)
    state = _copy_all_sports_update_state()
    if state.get("status") == "running":
        return {"ok": True, **state}

    worker = threading.Thread(target=_run_all_sports_update_pipeline, daemon=True)
    worker.start()
    return {
        "ok": True,
        "status": "running",
        "percent": 0,
        "message": "Actualizacion global iniciada.",
        "current_sport": None,
        "current_sport_label": None,
        "completed_sports": 0,
        "total_sports": len(SPORT_UPDATE_PIPELINES),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error": None,
        "logs": [],
    }


@app.get("/api/{sport}/board-status")
def get_sport_board_status(sport: str, date: Optional[str] = None):
    ensure_sport_exists(sport)
    return _build_public_board_status(sport, date)


@app.get("/api/{sport}/update-status")
def get_sport_update_status(sport: str):
    if sport not in SPORT_UPDATE_PIPELINES:
        raise HTTPException(status_code=404, detail=f"Actualizacion no soportada para: {sport}")
    return _copy_update_state(sport)


@app.post("/api/{sport}/update-all")
def start_sport_update_all(sport: str):
    if sport not in SPORT_UPDATE_PIPELINES:
        raise HTTPException(status_code=404, detail=f"Actualizacion no soportada para: {sport}")

    pipeline = SPORT_UPDATE_PIPELINES[sport]
    sport_label = pipeline["label"]
    state = _copy_update_state(sport)
    if state.get("status") == "running":
        return state

    worker = threading.Thread(target=_run_update_pipeline, args=(sport,), daemon=True)
    worker.start()
    return {
        "status": "running",
        "percent": 0,
        "message": f"Actualizacion {sport_label} iniciada.",
        "current_step": None,
        "current_step_label": None,
        "completed_steps": 0,
        "total_steps": len(pipeline["steps"]),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "error": None,
        "logs": [],
    }


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
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "triple_a"]
    summaries = [build_sport_insights_summary(s) for s in sports]

    return {
        "generated_at": datetime.now().isoformat(),
        "sports": summaries,
    }


@app.get("/api/insights/team-form")
def get_team_form_insights(
    window_games: int = 8,
    min_games: int = 3,
    force_refresh: bool = False,
):
    sports = ["nba", "mlb", "kbo", "nhl", "liga_mx", "laliga", "bundesliga", "ligue1", "euroleague", "ncaa_baseball", "tennis", "triple_a"]
    window = max(3, min(int(window_games), 20))
    min_sample = max(2, min(int(min_games), window))
    fingerprint = _team_form_sources_fingerprint(sports)
    cache_key = (window, min_sample, fingerprint)

    now_ts = datetime.now().timestamp()
    if not force_refresh:
        with TEAM_FORM_CACHE_LOCK:
            cache_hit = TEAM_FORM_CACHE.get(cache_key)
            if cache_hit and (now_ts - float(cache_hit.get("created_ts", 0))) <= TEAM_FORM_CACHE_TTL_SECONDS:
                return cache_hit["payload"]

    payload = build_team_form_summary(
        sports=sports,
        window_games=window,
        min_games=min_sample,
    )

    with TEAM_FORM_CACHE_LOCK:
        TEAM_FORM_CACHE[cache_key] = {
            "created_ts": now_ts,
            "payload": payload,
        }

    return payload


@app.get("/api/insights/team-form/team")
def get_team_form_team_insights(
    sport: str,
    team: str,
    window_games: int = 8,
    max_points: int = 30,
    horizon_games: int = 5,
    simulations: int = 4000,
    force_refresh: bool = False,
):
    sport_key = str(sport or "").strip().lower()
    team_key = str(team or "").strip()
    window = max(3, min(int(window_games), 20))
    points = max(10, min(int(max_points), 80))
    horizon = max(1, min(int(horizon_games), 12))
    sims = max(500, min(int(simulations), 20000))
    fingerprint = _team_form_sources_fingerprint([sport_key]) if sport_key else tuple()
    cache_key = ("team_detail", sport_key, team_key, window, points, horizon, sims, fingerprint)

    now_ts = datetime.now().timestamp()
    if not force_refresh:
        with TEAM_FORM_CACHE_LOCK:
            cache_hit = TEAM_FORM_CACHE.get(cache_key)
            if cache_hit and (now_ts - float(cache_hit.get("created_ts", 0))) <= TEAM_FORM_CACHE_TTL_SECONDS:
                return cache_hit["payload"]

    payload = build_team_form_team_detail(
        sport=sport_key,
        team=team_key,
        window_games=window,
        max_points=points,
        horizon_games=horizon,
        simulations=sims,
    )

    with TEAM_FORM_CACHE_LOCK:
        TEAM_FORM_CACHE[cache_key] = {
            "created_ts": now_ts,
            "payload": payload,
        }

    return payload


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
            key="triple_a",
            label="Triple-A",
            raw_file=TRIPLE_A_RAW_HISTORY,
            home_col="home_runs_total",
            away_col="away_runs_total",
            metric_label="Runs Totales",
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
            key="bundesliga",
            label="Bundesliga",
            raw_file=BUNDESLIGA_RAW_HISTORY,
            home_col="home_score",
            away_col="away_score",
            metric_label="Goles Totales",
        ),
        SportScoringConfig(
            key="ligue1",
            label="Ligue 1",
            raw_file=LIGUE1_RAW_HISTORY,
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



