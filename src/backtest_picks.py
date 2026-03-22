from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTIONS = REPORTS_DIR / "backtest_predictions.csv"
OUTPUT_THRESHOLDS = REPORTS_DIR / "backtest_thresholds_summary.csv"
OUTPUT_BUCKETS = REPORTS_DIR / "backtest_buckets_summary.csv"
OUTPUT_TIERS = REPORTS_DIR / "backtest_tiers_summary.csv"


def load_pick_params():
    default_params = {
        "game": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
        "q1": {"xgb_weight": 0.5, "lgb_weight": 0.5, "threshold": 0.5},
    }

    params_file = MODELS_DIR / "pick_params.pkl"
    if not params_file.exists():
        return default_params

    try:
        loaded = joblib.load(params_file)
        if not isinstance(loaded, dict):
            return default_params

        for key in ["game", "q1"]:
            if key not in loaded:
                loaded[key] = default_params[key]
            for p in ["xgb_weight", "lgb_weight", "threshold"]:
                if p not in loaded[key]:
                    loaded[key][p] = default_params[key][p]
        return loaded
    except Exception:
        return default_params


def load_models():
    required_files = [
        MODELS_DIR / "xgb_game.pkl",
        MODELS_DIR / "lgb_game.pkl",
        MODELS_DIR / "xgb_q1.pkl",
        MODELS_DIR / "lgb_q1.pkl",
        MODELS_DIR / "feature_names.pkl",
    ]

    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        raise FileNotFoundError(
            "Faltan archivos de modelo:\n- " + "\n- ".join(missing)
        )

    xgb_game = joblib.load(MODELS_DIR / "xgb_game.pkl")
    lgb_game = joblib.load(MODELS_DIR / "lgb_game.pkl")
    xgb_q1 = joblib.load(MODELS_DIR / "xgb_q1.pkl")
    lgb_q1 = joblib.load(MODELS_DIR / "lgb_q1.pkl")
    feature_names = joblib.load(MODELS_DIR / "feature_names.pkl")

    return xgb_game, lgb_game, xgb_q1, lgb_q1, feature_names


def load_holdout():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"No existe el archivo: {PROCESSED_DATA}")

    df = pd.read_csv(PROCESSED_DATA).sort_values("date").reset_index(drop=True)

    cols_to_drop = [
        "game_id",
        "date",
        "season",
        "home_team",
        "away_team",
        "TARGET_home_win",
        "TARGET_home_win_q1",
    ]

    missing_cols = [c for c in cols_to_drop if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en features: {missing_cols}")

    X = df.drop(columns=cols_to_drop)
    y_game = df["TARGET_home_win"].astype(int)
    y_q1 = df["TARGET_home_win_q1"].astype(int)

    split_idx = int(len(df) * 0.80)

    df_meta = df.iloc[split_idx:].copy().reset_index(drop=True)
    X_test = X.iloc[split_idx:].copy().reset_index(drop=True)
    y_game_test = y_game.iloc[split_idx:].copy().reset_index(drop=True)
    y_q1_test = y_q1.iloc[split_idx:].copy().reset_index(drop=True)

    return df_meta, X_test, y_game_test, y_q1_test


def assign_pick_tier(confidence: float) -> str:
    if confidence >= 70:
        return "ELITE"
    if confidence >= 65:
        return "PREMIUM"
    if confidence >= 60:
        return "STRONG"
    if confidence >= 57:
        return "NORMAL"
    return "PASS"


def build_predictions(df_meta, X_test, y_game_test, y_q1_test, feature_names):
    xgb_game, lgb_game, xgb_q1, lgb_q1, _ = load_models()
    pick_params = load_pick_params()

    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    game_wx = float(pick_params["game"]["xgb_weight"])
    game_wl = float(pick_params["game"]["lgb_weight"])
    game_th = float(pick_params["game"]["threshold"])

    q1_wx = float(pick_params["q1"]["xgb_weight"])
    q1_wl = float(pick_params["q1"]["lgb_weight"])
    q1_th = float(pick_params["q1"]["threshold"])

    # ---------- FULL GAME ----------
    prob_xgb_game = xgb_game.predict_proba(X_test)[:, 1]
    prob_lgb_game = lgb_game.predict_proba(X_test)[:, 1]
    prob_game_home = game_wx * prob_xgb_game + game_wl * prob_lgb_game
    pred_game_home = (prob_game_home >= game_th).astype(int)

    # ---------- Q1 ----------
    prob_xgb_q1 = xgb_q1.predict_proba(X_test)[:, 1]
    prob_lgb_q1 = lgb_q1.predict_proba(X_test)[:, 1]
    prob_q1_home = q1_wx * prob_xgb_q1 + q1_wl * prob_lgb_q1
    pred_q1_home = (prob_q1_home >= q1_th).astype(int)

    out = df_meta.copy()

    # FULL GAME
    out["actual_game_home_win"] = y_game_test
    out["pred_game_home_win"] = pred_game_home
    out["prob_game_home_win"] = prob_game_home
    out["prob_game_away_win"] = 1 - prob_game_home
    out["game_confidence"] = np.maximum(out["prob_game_home_win"], out["prob_game_away_win"]) * 100
    out["game_pick_side"] = np.where(out["pred_game_home_win"] == 1, "HOME", "AWAY")
    out["game_pick_team"] = np.where(out["pred_game_home_win"] == 1, out["home_team"], out["away_team"])
    out["game_correct"] = (out["pred_game_home_win"] == out["actual_game_home_win"]).astype(int)
    out["game_tier"] = out["game_confidence"].apply(assign_pick_tier)

    # Q1
    out["actual_q1_home_win"] = y_q1_test
    out["pred_q1_home_win"] = pred_q1_home
    out["prob_q1_home_win"] = prob_q1_home
    out["prob_q1_away_win"] = 1 - prob_q1_home
    out["q1_confidence"] = np.maximum(out["prob_q1_home_win"], out["prob_q1_away_win"]) * 100
    out["q1_pick_side"] = np.where(out["pred_q1_home_win"] == 1, "HOME", "AWAY")
    out["q1_pick_team"] = np.where(out["pred_q1_home_win"] == 1, out["home_team"], out["away_team"])
    out["q1_correct"] = (out["pred_q1_home_win"] == out["actual_q1_home_win"]).astype(int)
    out["q1_tier"] = out["q1_confidence"].apply(assign_pick_tier)

    # Buckets
    bins = [50, 55, 60, 65, 70, 75, 80, 101]
    labels = ["50-54.9", "55-59.9", "60-64.9", "65-69.9", "70-74.9", "75-79.9", "80+"]

    out["game_conf_bucket"] = pd.cut(
        out["game_confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    out["q1_conf_bucket"] = pd.cut(
        out["q1_confidence"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    return out


def print_overall_metrics(df):
    print("\n======================================================")
    print("📈 RESUMEN GENERAL DEL BACKTEST")
    print("======================================================")

    # Full game
    game_acc = accuracy_score(df["actual_game_home_win"], df["pred_game_home_win"])
    game_auc = roc_auc_score(df["actual_game_home_win"], df["prob_game_home_win"])
    game_brier = brier_score_loss(df["actual_game_home_win"], df["prob_game_home_win"])
    game_logloss = log_loss(df["actual_game_home_win"], df["prob_game_home_win"], labels=[0, 1])

    # Q1
    q1_acc = accuracy_score(df["actual_q1_home_win"], df["pred_q1_home_win"])
    q1_auc = roc_auc_score(df["actual_q1_home_win"], df["prob_q1_home_win"])
    q1_brier = brier_score_loss(df["actual_q1_home_win"], df["prob_q1_home_win"])
    q1_logloss = log_loss(df["actual_q1_home_win"], df["prob_q1_home_win"], labels=[0, 1])

    print("🏆 FULL GAME")
    print(f"   Accuracy : {game_acc:.4f}")
    print(f"   AUC      : {game_auc:.4f}")
    print(f"   Brier    : {game_brier:.4f}")
    print(f"   LogLoss  : {game_logloss:.4f}")

    print("\n⏱️ PRIMER CUARTO")
    print(f"   Accuracy : {q1_acc:.4f}")
    print(f"   AUC      : {q1_auc:.4f}")
    print(f"   Brier    : {q1_brier:.4f}")
    print(f"   LogLoss  : {q1_logloss:.4f}")

    print(f"\nTotal juegos evaluados: {len(df)}")


def summarize_thresholds(df, conf_col, correct_col, title, thresholds=None):
    if thresholds is None:
        thresholds = [50, 52, 55, 57, 60, 62, 65, 70]

    print(f"\n📌 {title} - Accuracy por umbral")
    print("-" * 52)

    rows = []
    for th in thresholds:
        subset = df[df[conf_col] >= th].copy()
        n = len(subset)
        acc = subset[correct_col].mean() if n > 0 else np.nan

        rows.append({
            "market": title,
            "threshold": th,
            "n_picks": n,
            "accuracy": acc,
        })

    result = pd.DataFrame(rows)
    print(result[["threshold", "n_picks", "accuracy"]].to_string(index=False))
    return result


def summarize_buckets(df, bucket_col, correct_col, title):
    print(f"\n📊 {title} - Accuracy por bucket")
    print("-" * 52)

    result = (
        df.groupby(bucket_col, dropna=False, observed=False)
        .agg(
            picks=(correct_col, "size"),
            accuracy=(correct_col, "mean"),
        )
        .reset_index()
    )

    result.insert(0, "market", title)
    print(result[[bucket_col, "picks", "accuracy"]].to_string(index=False))
    return result


def summarize_tiers(df, tier_col, correct_col, title):
    print(f"\n🔥 {title} - Accuracy por tier")
    print("-" * 52)

    tier_order = ["PASS", "NORMAL", "STRONG", "PREMIUM", "ELITE"]
    temp = df.copy()
    temp[tier_col] = pd.Categorical(temp[tier_col], categories=tier_order, ordered=True)

    result = (
        temp.groupby(tier_col, observed=False)
        .agg(
            picks=(correct_col, "size"),
            accuracy=(correct_col, "mean"),
        )
        .reset_index()
    )

    result.insert(0, "market", title)
    print(result[[tier_col, "picks", "accuracy"]].to_string(index=False))
    return result


def save_outputs(df, threshold_game, threshold_q1, bucket_game, bucket_q1, tier_game, tier_q1):
    df.to_csv(OUTPUT_PREDICTIONS, index=False)

    thresholds_all = pd.concat([threshold_game, threshold_q1], ignore_index=True)
    thresholds_all.to_csv(OUTPUT_THRESHOLDS, index=False)

    buckets_all = pd.concat([bucket_game, bucket_q1], ignore_index=True)
    buckets_all.to_csv(OUTPUT_BUCKETS, index=False)

    tiers_all = pd.concat([tier_game, tier_q1], ignore_index=True)
    tiers_all.to_csv(OUTPUT_TIERS, index=False)

    print(f"\n💾 Predicciones guardadas en: {OUTPUT_PREDICTIONS}")
    print(f"💾 Resumen umbrales guardado en: {OUTPUT_THRESHOLDS}")
    print(f"💾 Resumen buckets guardado en: {OUTPUT_BUCKETS}")
    print(f"💾 Resumen tiers guardado en: {OUTPUT_TIERS}")


def main():
    print("🚀 Iniciando backtest de picks históricos...")

    _, _, _, _, feature_names = load_models()
    df_meta, X_test, y_game_test, y_q1_test = load_holdout()

    df_backtest = build_predictions(
        df_meta=df_meta,
        X_test=X_test,
        y_game_test=y_game_test,
        y_q1_test=y_q1_test,
        feature_names=feature_names,
    )

    print_overall_metrics(df_backtest)

    threshold_game = summarize_thresholds(
        df_backtest,
        conf_col="game_confidence",
        correct_col="game_correct",
        title="FULL GAME",
    )

    bucket_game = summarize_buckets(
        df_backtest,
        bucket_col="game_conf_bucket",
        correct_col="game_correct",
        title="FULL GAME",
    )

    tier_game = summarize_tiers(
        df_backtest,
        tier_col="game_tier",
        correct_col="game_correct",
        title="FULL GAME",
    )

    threshold_q1 = summarize_thresholds(
        df_backtest,
        conf_col="q1_confidence",
        correct_col="q1_correct",
        title="PRIMER CUARTO",
    )

    bucket_q1 = summarize_buckets(
        df_backtest,
        bucket_col="q1_conf_bucket",
        correct_col="q1_correct",
        title="PRIMER CUARTO",
    )

    tier_q1 = summarize_tiers(
        df_backtest,
        tier_col="q1_tier",
        correct_col="q1_correct",
        title="PRIMER CUARTO",
    )

    save_outputs(
        df=df_backtest,
        threshold_game=threshold_game,
        threshold_q1=threshold_q1,
        bucket_game=bucket_game,
        bucket_q1=bucket_q1,
        tier_game=tier_game,
        tier_q1=tier_q1,
    )

    print("\n✅ Backtest terminado.")


if __name__ == "__main__":
    main()