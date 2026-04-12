import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
INPUT_FILE = BASE_DIR / "data" / "nhl" / "processed" / "model_ready_features_nhl.csv"

MODELS_DIR = BASE_DIR / "data" / "nhl" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DIR = BASE_DIR / "data" / "nhl" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_CONFIG = {
    "full_game": {
        "target_col": "TARGET_full_game",
        "description": "Home Win (1) vs Away Win (0)",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "handicap_1_5": {
        "target_col": "TARGET_spread_1_5",
        "description": "Home -1.5 covers (1) vs Away +1.5 covers (0)",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "spread_2_5": {
        "target_col": "TARGET_over_55",
        "description": "Over/Under 5.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "q1_over_15": {
        "target_col": "TARGET_p1_over_15",
        "description": "1P Goals Over/Under 1.5",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
    "home_over_2_5": {
        "target_col": "TARGET_home_over_25",
        "description": "Home Team Over/Under 2.5 goals",
        "problem_type": "binary",
        "num_classes": 2,
        "threshold": 0.50,
    },
}

NON_FEATURE_COLUMNS = {
    "game_id", "date", "date_dt", "time", "season", "home_team", "away_team",
    "home_score", "away_score", "is_draw", "total_goals",
    "home_p1_goals", "away_p1_goals", "total_p1_goals",
    "completed", "venue_name", "odds_details", "odds_over_under",
    "TARGET_full_game", "TARGET_over_55", "TARGET_home_over_25",
    "TARGET_spread_1_5", "TARGET_p1_over_15",
}


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, dtype={"game_id": str})
    df["date"] = df["date"].astype(str)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_dt"]).sort_values(["date_dt", "game_id"]).reset_index(drop=True)
    return df


def get_all_numeric_feature_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS and c != "date_dt"]
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    banned_keywords = ["correct", "actual", "winner", "result", "target"]
    return [c for c in cols if not any(k in c.lower() for k in banned_keywords)]


def get_nhl_market_feature_columns(df: pd.DataFrame, market: str) -> List[str]:
    all_numeric = get_all_numeric_feature_columns(df)

    if market == "full_game":
        keep = []
        keep_prefixes = ("fg_", "ml_")
        keep_exact = {
            "elo_diff",
            "home_elo_pre", "away_elo_pre",
            "h2h_home_win_rate",
            "home_goals_scored_l5", "away_goals_scored_l5",
            "home_goals_allowed_l5", "away_goals_allowed_l5",
            "home_goals_scored_l10", "away_goals_scored_l10",
            "home_goals_allowed_l10", "away_goals_allowed_l10",
            "home_win_rate_l5", "away_win_rate_l5",
            "home_win_rate_l10", "away_win_rate_l10",
            "home_goal_diff_l5", "away_goal_diff_l5",
            "home_goal_diff_l10", "away_goal_diff_l10",
            "home_rest_days", "away_rest_days",
            "home_is_b2b", "away_is_b2b",
            "home_games_last_3", "away_games_last_3",
            "home_games_last_5", "away_games_last_5",
            "home_goalie_found", "away_goalie_found",
            "home_goalie_confirmed", "away_goalie_confirmed",
            "home_goalie_games_started_before", "away_goalie_games_started_before",
            "both_goalies_found", "both_goalies_confirmed",
            "rest_days_diff",
            "win_rate_l5_diff", "win_rate_l10_diff",
            "goals_scored_l5_diff", "goals_allowed_l5_diff",
            "goals_scored_l10_diff", "goals_allowed_l10_diff",
            "goal_diff_l5_diff", "goal_diff_l10_diff", "goal_diff_std_l5_diff",
            "goalie_goals_allowed_l3_diff", "goalie_goals_allowed_l5_diff", "goalie_goals_allowed_l10_diff",
            "goalie_win_rate_l3_diff", "goalie_win_rate_l5_diff", "goalie_win_rate_l10_diff",
            "goalie_rest_days_diff", "goalie_experience_diff",
            "fg_form_diff", "fg_form_long_diff", "fg_scoring_diff", "fg_scoring_long_diff",
            "fg_rest_diff", "fg_volatility_diff", "fg_goalie_diff", "fg_strength_diff",
            "fg_strength_x_goalie_diff", "fg_market_edge", "fg_market_confidence", "fg_market_vs_model_gap",
            "ml_implied_home_prob_no_vig", "ml_implied_away_prob_no_vig",
            "ml_prob_gap_no_vig", "ml_home_is_favorite_market", "ml_abs_price_gap", "ml_odds_available",
        }

        drop_keywords = [
            "over_55", "home_over_25", "total", "under", "over",
            "team_total", "q1_", "1p_", "p1_", "spread_pick", "correct"
        ]

        for c in all_numeric:
            c_low = c.lower()
            if any(k in c_low for k in drop_keywords):
                continue
            if (
                c in keep_exact
                or c.startswith(keep_prefixes)
                or c.endswith("_diff")
            ):
                keep.append(c)

        return sorted(set(keep))

    return sorted(all_numeric)


def time_based_split(df: pd.DataFrame, valid_fraction: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_idx = int(len(df) * (1 - valid_fraction))
    split_idx = max(50, min(split_idx, len(df) - 20))
    train_df = df.iloc[:split_idx].copy()
    valid_df = df.iloc[split_idx:].copy()
    return train_df, valid_df


def build_xgb_binary() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=42,
        n_jobs=-1,
    )


def build_lgbm_binary() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=15,
        subsample=0.80,
        colsample_bytree=0.80,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_lgbm_secondary_binary() -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        n_estimators=300,
        learning_rate=0.06,
        num_leaves=25,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=43,
        n_jobs=-1,
        verbose=-1,
    )


def build_catboost_binary() -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5.0,
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_state=44,
        verbose=0,
        allow_writing_files=False,
    )


def calculate_optimal_threshold(y_valid: np.ndarray, probs: np.ndarray, market: str) -> float:
    if market == "full_game":
        search_space = np.arange(0.45, 0.61, 0.01)
    else:
        search_space = np.arange(0.35, 0.66, 0.01)

    best_threshold = 0.5
    best_accuracy = -1

    for threshold in search_space:
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_valid, preds)
        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = float(threshold)

    return best_threshold


def _ensemble_binary_probs(models: Dict, X: pd.DataFrame) -> np.ndarray:
    xgb_probs = models["xgb"].predict_proba(X)[:, 1]
    lgbm_probs = models["lgbm"].predict_proba(X)[:, 1]
    lgbm_sec_probs = models["lgbm_sec"].predict_proba(X)[:, 1]
    catboost_probs = models["catboost"].predict_proba(X)[:, 1]
    return np.clip((xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0, 1e-6, 1 - 1e-6)


def _load_market_models(market: str) -> Optional[Dict]:
    market_dir = MODELS_DIR / market
    metadata_path = market_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return {
            "xgb": joblib.load(market_dir / "xgboost_model.pkl"),
            "lgbm": joblib.load(market_dir / "lgbm_model.pkl"),
            "lgbm_sec": joblib.load(market_dir / "lgbm_secondary_model.pkl"),
            "catboost": joblib.load(market_dir / "catboost_model.pkl"),
            "metadata": metadata,
        }
    except Exception as e:
        print(f"   [WARN] Could not load {market} models for V2 meta: {e}")
        return None


def _safe_prob_series(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    s = pd.to_numeric(df[col], errors="coerce")
    if default >= 0.0:
        return s.fillna(default)
    return s


def _build_v2_meta_matrix(
    full_home_prob: np.ndarray,
    spread_home_cover_prob: np.ndarray,
    market_home_prob: np.ndarray,
    market_gap: np.ndarray,
) -> np.ndarray:
    base_p = np.clip(np.asarray(full_home_prob, dtype=float), 1e-6, 1 - 1e-6)
    spread_p = np.asarray(spread_home_cover_prob, dtype=float)
    market_p = np.asarray(market_home_prob, dtype=float)
    gap = np.asarray(market_gap, dtype=float)

    spread_p = np.where(np.isnan(spread_p), 0.5, spread_p)
    market_p = np.where(np.isnan(market_p), 0.5, market_p)
    gap = np.where(np.isnan(gap), 0.0, gap)
    gap = np.clip(np.abs(gap), 0.0, 1.0)

    spread_proxy = np.clip(0.5 + (spread_p - 0.5) * 0.60, 1e-6, 1 - 1e-6)
    return np.column_stack(
        [
            base_p,
            market_p,
            spread_p,
            spread_proxy,
            gap,
            base_p - market_p,
            base_p - spread_proxy,
            market_p - spread_proxy,
            base_p * gap,
            market_p * gap,
        ]
    )


def train_full_game_v2_meta_model(df: pd.DataFrame) -> Optional[Dict]:
    print("\n🧠 Training FULL_GAME V2 meta model (base + handicap + market)")

    full_assets = _load_market_models("full_game")
    spread_assets = _load_market_models("handicap_1_5")
    if full_assets is None:
        print("   [WARN] full_game assets missing; skipping V2 meta model")
        return None

    full_features = full_assets["metadata"].get("feature_columns", [])
    spread_features = spread_assets["metadata"].get("feature_columns", []) if spread_assets else []

    target = pd.to_numeric(df.get("TARGET_full_game"), errors="coerce")
    valid_idx = target.isin([0, 1])
    if valid_idx.sum() < 200:
        print("   [WARN] Not enough rows for V2 meta model")
        return None

    df_work = df.loc[valid_idx].copy()
    y = target.loc[valid_idx].astype(int)

    split_idx = int(len(df_work) * 0.8)
    split_idx = max(80, min(split_idx, len(df_work) - 40))

    df_train = df_work.iloc[:split_idx].copy()
    df_valid = df_work.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_valid = y.iloc[split_idx:].copy()

    X_full_train = df_train[full_features].fillna(0)
    X_full_valid = df_valid[full_features].fillna(0)

    full_prob_train = _ensemble_binary_probs(full_assets, X_full_train)
    full_prob_valid = _ensemble_binary_probs(full_assets, X_full_valid)

    if spread_assets and spread_features:
        X_spread_train = df_train[spread_features].fillna(0)
        X_spread_valid = df_valid[spread_features].fillna(0)
        spread_prob_train = _ensemble_binary_probs(spread_assets, X_spread_train)
        spread_prob_valid = _ensemble_binary_probs(spread_assets, X_spread_valid)
    else:
        spread_prob_train = np.full(len(df_train), 0.5, dtype=float)
        spread_prob_valid = np.full(len(df_valid), 0.5, dtype=float)

    market_home_train = _safe_prob_series(df_train, "ml_implied_home_prob_no_vig", 0.5).to_numpy(dtype=float)
    market_home_valid = _safe_prob_series(df_valid, "ml_implied_home_prob_no_vig", 0.5).to_numpy(dtype=float)
    market_gap_train = _safe_prob_series(df_train, "ml_prob_gap_no_vig", 0.0).to_numpy(dtype=float)
    market_gap_valid = _safe_prob_series(df_valid, "ml_prob_gap_no_vig", 0.0).to_numpy(dtype=float)

    X_meta_train = _build_v2_meta_matrix(
        full_home_prob=full_prob_train,
        spread_home_cover_prob=spread_prob_train,
        market_home_prob=market_home_train,
        market_gap=market_gap_train,
    )
    X_meta_valid = _build_v2_meta_matrix(
        full_home_prob=full_prob_valid,
        spread_home_cover_prob=spread_prob_valid,
        market_home_prob=market_home_valid,
        market_gap=market_gap_valid,
    )

    if len(np.unique(y_train)) < 2:
        print("   [WARN] Meta train split has one class; skipping V2 meta model")
        return None

    meta_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    meta_model.fit(X_meta_train, y_train)

    train_probs = meta_model.predict_proba(X_meta_train)[:, 1]
    valid_probs = meta_model.predict_proba(X_meta_valid)[:, 1]
    meta_threshold = calculate_optimal_threshold(y_train.values, train_probs, "full_game")

    valid_preds = (valid_probs >= meta_threshold).astype(int)
    base_valid_preds = (full_prob_valid >= float(full_assets["metadata"].get("threshold", 0.5))).astype(int)

    valid_acc_meta = float(accuracy_score(y_valid, valid_preds))
    valid_acc_base = float(accuracy_score(y_valid, base_valid_preds))

    market_dir = MODELS_DIR / "full_game"
    model_path = market_dir / "meta_model_full_game_v2.pkl"
    metadata_path = market_dir / "meta_model_full_game_v2_metadata.json"
    joblib.dump(meta_model, model_path)

    meta_metadata = {
        "model": "logistic_regression",
        "features": [
            "base_home_prob",
            "market_home_prob",
            "spread_home_cover_prob",
            "spread_home_win_proxy",
            "market_gap_abs",
            "base_minus_market",
            "base_minus_spread_proxy",
            "market_minus_spread_proxy",
            "base_x_gap",
            "market_x_gap",
        ],
        "threshold": float(meta_threshold),
        "valid_rows": int(len(y_valid)),
        "valid_accuracy_base": float(valid_acc_base),
        "valid_accuracy_v2": float(valid_acc_meta),
        "valid_delta_v2_vs_base_pp": float((valid_acc_meta - valid_acc_base) * 100.0),
        "base_full_game_threshold": float(full_assets["metadata"].get("threshold", 0.5)),
        "spread_model_available": bool(spread_assets is not None),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(meta_metadata, f, indent=2, ensure_ascii=False)

    print(f"   FULL_GAME base valid acc: {valid_acc_base:.4f}")
    print(f"   FULL_GAME V2   valid acc: {valid_acc_meta:.4f}")
    print(f"   Delta V2-Base         : {(valid_acc_meta - valid_acc_base) * 100.0:+.2f} pp")
    print(f"   Saved meta model      : {model_path}")

    return {
        "market": "full_game_v2_meta",
        **meta_metadata,
    }


def train_single_market(df: pd.DataFrame, market: str, config: Dict):
    target_col = config["target_col"]
    problem_type = config["problem_type"]

    print(f"\n🎯 Entrenando mercado: {market} ({problem_type})")
    print(f"   Descripción: {config['description']}")

    feature_cols = get_nhl_market_feature_columns(df, market)
    X = df[feature_cols].fillna(0)
    y = pd.to_numeric(df[target_col], errors="coerce")

    valid_idx = ~y.isna()
    X = X.loc[valid_idx].copy()
    y = y.loc[valid_idx].copy()

    if market == "full_game":
        # Ya viene binario desde feature engineering: 1=home, 0=away
        y = y.astype(int)

    print(f"   Juegos: {len(X)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Classes: {sorted(y.unique().tolist())}")

    if len(X) < 80 or len(np.unique(y)) < 2:
        print(f"   [WARN] No hay suficiente data útil para {market}")
        return None

    split_idx = int(len(X) * 0.8)
    split_idx = max(50, min(split_idx, len(X) - 20))
    X_train, X_valid = X.iloc[:split_idx].copy(), X.iloc[split_idx:].copy()
    y_train, y_valid = y.iloc[:split_idx].copy(), y.iloc[split_idx:].copy()

    print("   Entrenando 4 modelos...")

    xgb = build_xgb_binary()
    lgbm = build_lgbm_binary()
    lgbm_sec = build_lgbm_secondary_binary()
    catboost = build_catboost_binary()

    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)
    lgbm_sec.fit(X_train, y_train)
    catboost.fit(X_train, y_train)

    xgb_probs = xgb.predict_proba(X_valid)[:, 1]
    lgbm_probs = lgbm.predict_proba(X_valid)[:, 1]
    lgbm_sec_probs = lgbm_sec.predict_proba(X_valid)[:, 1]
    catboost_probs = catboost.predict_proba(X_valid)[:, 1]

    ensemble_probs = (xgb_probs + lgbm_probs + lgbm_sec_probs + catboost_probs) / 4.0
    threshold = calculate_optimal_threshold(y_valid.values, ensemble_probs, market)
    ensemble_preds = (ensemble_probs >= threshold).astype(int)

    accuracy = accuracy_score(y_valid, ensemble_preds)
    try:
        logloss = log_loss(y_valid, np.clip(ensemble_probs, 1e-6, 1 - 1e-6))
    except Exception:
        logloss = None

    try:
        auc = roc_auc_score(y_valid, ensemble_probs)
    except Exception:
        auc = None

    print(f"   Accuracy valid: {accuracy:.4f}")
    if logloss is not None:
        print(f"   LogLoss valid: {logloss:.4f}")
    if auc is not None:
        print(f"   ROC AUC valid: {auc:.4f}")
    print(f"   Best threshold: {threshold:.2f}")

    market_dir = MODELS_DIR / market
    market_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(xgb, market_dir / "xgboost_model.pkl")
    joblib.dump(lgbm, market_dir / "lgbm_model.pkl")
    joblib.dump(lgbm_sec, market_dir / "lgbm_secondary_model.pkl")
    joblib.dump(catboost, market_dir / "catboost_model.pkl")

    metadata = {
        "market": market,
        "target_col": target_col,
        "problem_type": problem_type,
        "threshold": float(threshold),
        "feature_columns": feature_cols,
        "n_rows": int(len(X)),
        "n_features": int(len(feature_cols)),
        "valid_accuracy": float(accuracy),
        "valid_logloss": None if logloss is None else float(logloss),
        "valid_auc": None if auc is None else float(auc),
        "classes": [int(x) for x in sorted(np.unique(y).tolist())],
    }

    with open(market_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def train_nhl_models():
    print("🏒 NHL MODEL TRAINING")
    print("=" * 60)

    df = load_dataset()
    print(f"📂 Dataset cargado: {len(df)} juegos")

    results = {}
    for market, config in TARGET_CONFIG.items():
        result = train_single_market(df, market, config)
        if result is not None:
            results[market] = result

    v2_result = train_full_game_v2_meta_model(df)
    if v2_result is not None:
        results["full_game_v2_meta"] = v2_result

    summary_file = REPORTS_DIR / "training_summary_nhl.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Summary guardado en: {summary_file}")


if __name__ == "__main__":
    train_nhl_models()
