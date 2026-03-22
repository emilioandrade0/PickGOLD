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
        pred_ens = (prob_ens > 0.5).astype(int)

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
    print(f"   ACC   : {np.nanmean(acc_scores):.4f}")
    print(f"   AUC   : {np.nanmean(auc_scores):.4f}")
    print(f"   Brier : {np.nanmean(brier_scores):.4f}")
    print(f"   LogLoss: {np.nanmean(logloss_scores):.4f}")


def train_all_models():
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

    X = df.drop(columns=cols_to_drop)
    y_game = df["TARGET_home_win"]
    y_q1 = df["TARGET_home_win_q1"]

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    print(f"🧠 Entrenando IAs con {len(feature_names)} features...")

    # --- Validación temporal CV ---
    evaluate_ensemble_cv(X, y_game, "PARTIDO COMPLETO")
    evaluate_ensemble_cv(X, y_q1, "PRIMER CUARTO")

    # --- Split final holdout más reciente ---
    split_idx = int(len(df) * 0.80)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    yg_train = y_game.iloc[:split_idx]
    yg_test = y_game.iloc[split_idx:]

    yq_train = y_q1.iloc[:split_idx]
    yq_test = y_q1.iloc[split_idx:]

    print("\n⏳ Entrenando modelos finales para PARTIDO COMPLETO...")
    xgb_game = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    lgb_game = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )

    xgb_game.fit(X_train, yg_train)
    lgb_game.fit(X_train, yg_train)

    prob_xgb_g = xgb_game.predict_proba(X_test)[:, 1]
    prob_lgb_g = lgb_game.predict_proba(X_test)[:, 1]
    prob_ens_g = (prob_xgb_g + prob_lgb_g) / 2
    pred_ens_g = (prob_ens_g > 0.5).astype(int)

    print("\n📊 HOLDOUT FINAL - PARTIDO COMPLETO")
    print(f"   Accuracy : {accuracy_score(yg_test, pred_ens_g):.4f}")
    print(f"   AUC      : {roc_auc_score(yg_test, prob_ens_g):.4f}")
    print(f"   Brier    : {brier_score_loss(yg_test, prob_ens_g):.4f}")
    print(f"   LogLoss  : {log_loss(yg_test, prob_ens_g, labels=[0, 1]):.4f}")

    joblib.dump(xgb_game, MODELS_DIR / "xgb_game.pkl")
    joblib.dump(lgb_game, MODELS_DIR / "lgb_game.pkl")

    print("\n⏳ Entrenando modelos finales para PRIMER CUARTO...")
    xgb_q1 = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42,
    )
    lgb_q1 = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )

    xgb_q1.fit(X_train, yq_train)
    lgb_q1.fit(X_train, yq_train)

    prob_xgb_q1 = xgb_q1.predict_proba(X_test)[:, 1]
    prob_lgb_q1 = lgb_q1.predict_proba(X_test)[:, 1]
    prob_ens_q1 = (prob_xgb_q1 + prob_lgb_q1) / 2
    pred_ens_q1 = (prob_ens_q1 > 0.5).astype(int)

    print("\n📊 HOLDOUT FINAL - PRIMER CUARTO")
    print(f"   Accuracy : {accuracy_score(yq_test, pred_ens_q1):.4f}")
    print(f"   AUC      : {roc_auc_score(yq_test, prob_ens_q1):.4f}")
    print(f"   Brier    : {brier_score_loss(yq_test, prob_ens_q1):.4f}")
    print(f"   LogLoss  : {log_loss(yq_test, prob_ens_q1, labels=[0, 1]):.4f}")

    joblib.dump(xgb_q1, MODELS_DIR / "xgb_q1.pkl")
    joblib.dump(lgb_q1, MODELS_DIR / "lgb_q1.pkl")

    print("\n✅ ¡Todos los modelos fueron entrenados y guardados!")


if __name__ == "__main__":
    train_all_models()