from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

# --- RUTAS ---
BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_ensemble_cv(X: pd.DataFrame, y: pd.Series, label: str):
    print(f"\n📊 Validación temporal CV para {label}...")
    tscv = TimeSeriesSplit(n_splits=5)

    acc_scores = []
    auc_scores = []
    brier_scores = []
    logloss_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=42,
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
        )

        xgb_model.fit(X_train, y_train)
        lgb_model.fit(X_train, y_train)

        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        prob_ens = (prob_xgb + prob_lgb) / 2
        pred_ens = (prob_ens >= 0.5).astype(int)

        acc = accuracy_score(y_test, pred_ens)
        try:
            auc = roc_auc_score(y_test, prob_ens)
        except Exception:
            auc = np.nan

        brier = brier_score_loss(y_test, prob_ens)
        ll = log_loss(y_test, prob_ens, labels=[0, 1])

        acc_scores.append(acc)
        auc_scores.append(auc)
        brier_scores.append(brier)
        logloss_scores.append(ll)

        print(
            f"   Fold {fold}: "
            f"ACC={acc:.4f} | "
            f"AUC={auc:.4f} | "
            f"Brier={brier:.4f} | "
            f"LogLoss={ll:.4f}"
        )

    print(f"\n✅ CV Media {label}")
    print(f"   ACC    : {np.nanmean(acc_scores):.4f}")
    print(f"   AUC    : {np.nanmean(auc_scores):.4f}")
    print(f"   Brier  : {np.nanmean(brier_scores):.4f}")
    print(f"   LogLoss: {np.nanmean(logloss_scores):.4f}")


def build_time_splits(n_rows: int):
    """
    Split temporal:
    - 70% train base
    - 10% calibración
    - 20% holdout final
    """
    train_end = int(n_rows * 0.70)
    calib_end = int(n_rows * 0.80)

    if train_end <= 0 or calib_end <= train_end or calib_end >= n_rows:
        raise ValueError("No hay suficientes filas para hacer split temporal 70/10/20.")

    return train_end, calib_end


def fit_model(base_model, X_train, y_train):
    """
    Entrena modelo base sin calibración adicional para priorizar precisión de pick.
    """
    base_model.fit(X_train, y_train)
    return base_model


def optimize_ensemble_params(prob_xgb, prob_lgb, y_true, label: str):
    """
    Busca el mejor peso de ensamble sobre el bloque de calibración,
    usando umbral fijo 0.5 para evitar sobreajuste del threshold.
    Criterio principal: accuracy. Desempate: menor Brier.
    """
    weights = np.arange(0.20, 0.81, 0.05)
    threshold = 0.5

    best = {
        "xgb_weight": 0.5,
        "lgb_weight": 0.5,
        "threshold": threshold,
        "accuracy": -1.0,
        "brier": np.inf,
    }

    for w in weights:
        probs = w * prob_xgb + (1 - w) * prob_lgb
        brier = brier_score_loss(y_true, probs)
        preds = (probs >= threshold).astype(int)
        acc = accuracy_score(y_true, preds)

        if (acc > best["accuracy"]) or (np.isclose(acc, best["accuracy"]) and brier < best["brier"]):
            best = {
                "xgb_weight": float(w),
                "lgb_weight": float(1 - w),
                "threshold": float(threshold),
                "accuracy": float(acc),
                "brier": float(brier),
            }

    print(
        f"\n🎯 Parámetros óptimos ({label}) en calibración: "
        f"w_xgb={best['xgb_weight']:.2f}, "
        f"w_lgb={best['lgb_weight']:.2f}, "
        f"threshold={best['threshold']:.3f}, "
        f"ACC={best['accuracy']:.4f}, "
        f"Brier={best['brier']:.4f}"
    )

    return best


def print_holdout_metrics(y_true, probs, label: str):
    preds = (probs >= 0.5).astype(int)

    print(f"\n📊 HOLDOUT FINAL - {label}")
    print(f"   Accuracy : {accuracy_score(y_true, preds):.4f}")
    print(f"   AUC      : {roc_auc_score(y_true, probs):.4f}")
    print(f"   Brier    : {brier_score_loss(y_true, probs):.4f}")
    print(f"   LogLoss  : {log_loss(y_true, probs, labels=[0, 1]):.4f}")


def train_all_models():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {PROCESSED_DATA}")

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
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing_cols}")

    X = df.drop(columns=cols_to_drop)
    y_game = df["TARGET_home_win"].astype(int)
    y_q1 = df["TARGET_home_win_q1"].astype(int)

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    print(f"🧠 Entrenando IAs con {len(feature_names)} features...")

    # -------------------------------------------------
    # 1) VALIDACIÓN TEMPORAL CV
    # -------------------------------------------------
    evaluate_ensemble_cv(X, y_game, "PARTIDO COMPLETO")
    evaluate_ensemble_cv(X, y_q1, "PRIMER CUARTO")

    # -------------------------------------------------
    # 2) SPLIT TEMPORAL FINAL: 70 / 10 / 20
    # -------------------------------------------------
    train_end, calib_end = build_time_splits(len(df))

    X_train = X.iloc[:train_end]
    X_calib = X.iloc[train_end:calib_end]
    X_test = X.iloc[calib_end:]

    yg_train = y_game.iloc[:train_end]
    yg_calib = y_game.iloc[train_end:calib_end]
    yg_test = y_game.iloc[calib_end:]

    yq_train = y_q1.iloc[:train_end]
    yq_calib = y_q1.iloc[train_end:calib_end]
    yq_test = y_q1.iloc[calib_end:]

    print("\n🧩 Split temporal final:")
    print(f"   Train base : {len(X_train)}")
    print(f"   Calibración: {len(X_calib)}")
    print(f"   Holdout    : {len(X_test)}")

    # -------------------------------------------------
    # 3) MODELOS FINALES FULL GAME
    # -------------------------------------------------
    print("\n⏳ Entrenando modelos finales para PARTIDO COMPLETO (sin calibración)...")

    xgb_game_base = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )

    lgb_game_base = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )

    xgb_game = fit_model(
        base_model=xgb_game_base,
        X_train=X_train,
        y_train=yg_train,
    )

    lgb_game = fit_model(
        base_model=lgb_game_base,
        X_train=X_train,
        y_train=yg_train,
    )

    prob_xgb_g_cal = xgb_game.predict_proba(X_calib)[:, 1]
    prob_lgb_g_cal = lgb_game.predict_proba(X_calib)[:, 1]
    game_params = optimize_ensemble_params(prob_xgb_g_cal, prob_lgb_g_cal, yg_calib, "PARTIDO COMPLETO")

    prob_xgb_g = xgb_game.predict_proba(X_test)[:, 1]
    prob_lgb_g = lgb_game.predict_proba(X_test)[:, 1]
    prob_ens_g = game_params["xgb_weight"] * prob_xgb_g + game_params["lgb_weight"] * prob_lgb_g

    print_holdout_metrics(yg_test, prob_ens_g, "PARTIDO COMPLETO")
    print(f"   Threshold óptimo aplicado: {game_params['threshold']:.3f}")
    print(f"   Accuracy@threshold: {accuracy_score(yg_test, (prob_ens_g >= game_params['threshold']).astype(int)):.4f}")

    joblib.dump(xgb_game, MODELS_DIR / "xgb_game.pkl")
    joblib.dump(lgb_game, MODELS_DIR / "lgb_game.pkl")

    # -------------------------------------------------
    # 4) MODELOS FINALES Q1
    # -------------------------------------------------
    print("\n⏳ Entrenando modelos finales para PRIMER CUARTO (sin calibración)...")

    xgb_q1_base = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )

    lgb_q1_base = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )

    xgb_q1 = fit_model(
        base_model=xgb_q1_base,
        X_train=X_train,
        y_train=yq_train,
    )

    lgb_q1 = fit_model(
        base_model=lgb_q1_base,
        X_train=X_train,
        y_train=yq_train,
    )

    prob_xgb_q1_cal = xgb_q1.predict_proba(X_calib)[:, 1]
    prob_lgb_q1_cal = lgb_q1.predict_proba(X_calib)[:, 1]
    q1_params = optimize_ensemble_params(prob_xgb_q1_cal, prob_lgb_q1_cal, yq_calib, "PRIMER CUARTO")

    prob_xgb_q1 = xgb_q1.predict_proba(X_test)[:, 1]
    prob_lgb_q1 = lgb_q1.predict_proba(X_test)[:, 1]
    prob_ens_q1 = q1_params["xgb_weight"] * prob_xgb_q1 + q1_params["lgb_weight"] * prob_lgb_q1

    print_holdout_metrics(yq_test, prob_ens_q1, "PRIMER CUARTO")
    print(f"   Threshold óptimo aplicado: {q1_params['threshold']:.3f}")
    print(f"   Accuracy@threshold: {accuracy_score(yq_test, (prob_ens_q1 >= q1_params['threshold']).astype(int)):.4f}")

    joblib.dump(xgb_q1, MODELS_DIR / "xgb_q1.pkl")
    joblib.dump(lgb_q1, MODELS_DIR / "lgb_q1.pkl")

    pick_params = {
        "game": {
            "xgb_weight": game_params["xgb_weight"],
            "lgb_weight": game_params["lgb_weight"],
            "threshold": game_params["threshold"],
        },
        "q1": {
            "xgb_weight": q1_params["xgb_weight"],
            "lgb_weight": q1_params["lgb_weight"],
            "threshold": q1_params["threshold"],
        },
    }
    joblib.dump(pick_params, MODELS_DIR / "pick_params.pkl")

    print("\n✅ ¡Todos los modelos fueron entrenados y guardados!")
    print("🎛️ Parámetros de pick guardados en: pick_params.pkl")
    print(f"💾 Modelos guardados en: {MODELS_DIR}")


if __name__ == "__main__":
    train_all_models()