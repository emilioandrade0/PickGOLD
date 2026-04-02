from pathlib import Path
import sys

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, classification_report, log_loss, roc_auc_score

# --- RUTAS ---
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
PROCESSED_DATA = BASE_DIR / "data" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "lgb_nba_model.pkl"
FEATURES_PATH = MODELS_DIR / "lgb_feature_names.pkl"


def train_lightgbm():
    print(f"📥 Cargando features desde: {PROCESSED_DATA}")
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
    y = df["TARGET_home_win"]

    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"🧠 Entrenando LightGBM con {len(X.columns)} features...")

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        verbosity=-1,
    )

    print("⏳ Entrenando el algoritmo LightGBM...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n📊 EVALUACIÓN DEL MODELO LightGBM:")
    print(f"🎯 Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"🎯 AUC      : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"🎯 Brier    : {brier_score_loss(y_test, y_prob):.4f}")
    print(f"🎯 LogLoss  : {log_loss(y_test, y_prob, labels=[0, 1]):.4f}")

    print("\nReporte Detallado:")
    print(classification_report(y_test, y_pred, target_names=["Gana Visitante", "Gana Local"]))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(X.columns.tolist(), FEATURES_PATH)

    print(f"💾 Modelo guardado en: {MODEL_PATH}")
    print(f"💾 Features guardadas en: {FEATURES_PATH}")

    importances = pd.Series(model.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    print("\n🌟 TOP 10 VARIABLES MÁS IMPORTANTES:")
    for feat, val in top_features.items():
        print(f"   - {feat}: {val}")


if __name__ == "__main__":
    train_lightgbm()
