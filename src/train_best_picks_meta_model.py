from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

BASE_DIR = Path(__file__).resolve().parent
SNAPSHOTS_DIR = BASE_DIR / "data" / "insights" / "best_picks"
META_DIR = BASE_DIR / "data" / "insights" / "meta_model"
META_DIR.mkdir(parents=True, exist_ok=True)
META_MODEL_FILE = META_DIR / "best_picks_meta.pkl"


def _load_snapshot(path: Path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if not isinstance(payload.get("picks"), list):
        return None
    return payload


def _feature_row(row: dict):
    return {
        "sport": str(row.get("sport") or ""),
        "market": str(row.get("market") or ""),
        "tier": str(row.get("tier") or ""),
        "score": float(row.get("score") or 0.0),
        "final_rank_score": float(row.get("final_rank_score") or 0.0),
        "model_probability": float(row.get("model_probability") or 0.0),
        "implied_probability_estimate": float(row.get("implied_probability_estimate") or 0.0),
        "edge_proxy": float(row.get("edge_proxy") or 0.0),
        "stability_factor": float(row.get("stability_factor") or 0.0),
        "reliability_factor": float(row.get("reliability_factor") or 0.0),
        "correlation_penalty": float(row.get("correlation_penalty") or 1.0),
        "decimal_odds_used": float(row.get("decimal_odds_used") or 0.0),
        "odds_is_fallback": int(bool(row.get("odds_is_fallback"))),
    }


def _collect_training_rows():
    try:
        from src.api import _best_picks_with_results
    except Exception:
        from api import _best_picks_with_results

    if not SNAPSHOTS_DIR.exists():
        return [], []

    files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    X_rows = []
    y = []

    for path in files:
        payload = _load_snapshot(path)
        if not payload:
            continue

        # Enrich with resolved result labels/hits using current API logic.
        try:
            enriched = _best_picks_with_results(payload)
        except Exception:
            continue

        for row in enriched.get("picks") or []:
            hit = row.get("result_hit")
            if hit is None:
                continue
            X_rows.append(_feature_row(row))
            y.append(1 if bool(hit) else 0)

    return X_rows, y


def train_meta_model():
    X_rows, y = _collect_training_rows()
    n = len(y)

    if n < 50:
        print(f"[WARN] Not enough resolved rows for meta-model: {n}")
        return

    y_arr = np.array(y, dtype=int)

    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(X_rows)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X, y_arr)

    probs = model.predict_proba(X)[:, 1]

    try:
        auc = roc_auc_score(y_arr, probs)
    except Exception:
        auc = None

    try:
        ll = log_loss(y_arr, probs, labels=[0, 1])
    except Exception:
        ll = None

    payload = {
        "model": model,
        "vectorizer": vectorizer,
        "meta": {
            "sample_size": int(n),
            "positive_rate": float(y_arr.mean()),
            "auc_in_sample": float(auc) if auc is not None else None,
            "logloss_in_sample": float(ll) if ll is not None else None,
        },
    }

    joblib.dump(payload, META_MODEL_FILE)

    print(f"[OK] Meta-model saved: {META_MODEL_FILE}")
    print(f"[INFO] Samples: {n} | Pos rate: {y_arr.mean():.4f}")
    if auc is not None:
        print(f"[INFO] In-sample AUC: {auc:.4f}")
    if ll is not None:
        print(f"[INFO] In-sample logloss: {ll:.4f}")


if __name__ == "__main__":
    train_meta_model()
