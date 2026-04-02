from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
PROCESSED = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
REPORTS_DIR = BASE_DIR / "data" / "mlb" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def try_import_catboost():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None


def dynamic_feature_selection(df, date_col="date", label_col="TARGET_home_win", early_frac=0.6, drop_ratio=0.5):
    """Select features by comparing correlation in early vs recent periods.

    Keeps features whose absolute correlation with label has not dropped by more than `drop_ratio`.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=[label_col])
    dates = sorted(df[date_col].dt.date.unique())
    if len(dates) < 2:
        return []
    split_idx = max(1, int(len(dates) * early_frac))
    early_dates = set(dates[:split_idx])
    recent_dates = set(dates[split_idx:])

    numeric = df.select_dtypes(include=[np.number])
    if label_col in numeric.columns:
        numeric = numeric.drop(columns=[label_col])
    features = [c for c in numeric.columns if not c.startswith("game_id")]

    keep = []
    for f in features:
        try:
            ce = float(df[df[date_col].dt.date.isin(early_dates)][f].corr(df[df[date_col].dt.date.isin(early_dates)][label_col]))
            cr = float(df[df[date_col].dt.date.isin(recent_dates)][f].corr(df[df[date_col].dt.date.isin(recent_dates)][label_col]))
        except Exception:
            continue
        if np.isnan(ce) or np.isnan(cr):
            continue
        if abs(cr) >= abs(ce) * (1 - drop_ratio):
            keep.append(f)
    return keep


def build_ensemble(allow_cat=True):
    estimators = []
    # sklearn GBC
    estimators.append(("gb", GradientBoostingClassifier(n_estimators=200, random_state=42)))
    # MLP
    estimators.append(("mlp", MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)))
    # CatBoost if available
    Cat = try_import_catboost() if allow_cat else None
    if Cat is not None:
        estimators.append(("cat", Cat(iterations=200, verbose=0, random_state=42)))

    # Stacking meta-learner
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3, n_jobs=1, passthrough=False)
    return stack


def calibrate_model(model, Xtr, ytr, method="isotonic"):
    try:
        calib = CalibratedClassifierCV(model, method=method, cv=3)
        calib.fit(Xtr, ytr)
        return calib
    except Exception:
        # fallback to sigmoid
        try:
            calib = CalibratedClassifierCV(model, method="sigmoid", cv=3)
            calib.fit(Xtr, ytr)
            return calib
        except Exception:
            return None


def walkforward_compare(df, features_all, label_col="TARGET_home_win", date_col="date", min_train_days=90, test_window_days=30):
    df = df.copy()
    df = df.dropna(subset=[label_col])
    df[date_col] = pd.to_datetime(df[date_col])
    dates = sorted(df[date_col].dt.date.unique())
    results = []
    for i in range(min_train_days, len(dates) - test_window_days + 1, test_window_days):
        train_dates = dates[:i]
        test_dates = dates[i:i + test_window_days]
        tr = df[df[date_col].dt.date.isin(train_dates)]
        te = df[df[date_col].dt.date.isin(test_dates)]
        if tr.empty or te.empty:
            continue

        # dynamic selection on training set
        keep = dynamic_feature_selection(tr, date_col=date_col, label_col=label_col)
        # fallback: if keep empty use top numeric features
        if not keep:
            keep = [c for c in tr.select_dtypes(include=[np.number]).columns if c != label_col][:30]

        # train baseline (single GBC on full features_all)
        Xtr_b = tr[features_all].select_dtypes(include=[np.number]).fillna(0).values
        Xte_b = te[features_all].select_dtypes(include=[np.number]).fillna(0).values
        ytr = tr[label_col].astype(int).values
        yte = te[label_col].astype(int).values

        base = GradientBoostingClassifier(n_estimators=200, random_state=42)
        base.fit(Xtr_b, ytr)
        pb = base.predict_proba(Xte_b)[:, 1]
        acc_b = accuracy_score(yte, (pb >= 0.5).astype(int))
        auc_b = roc_auc_score(yte, pb)

        # new ensemble on selected features
        Xtr_e = tr[keep].select_dtypes(include=[np.number]).fillna(0).values
        Xte_e = te[keep].select_dtypes(include=[np.number]).fillna(0).values

        ensemble = build_ensemble()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ensemble.fit(Xtr_e, ytr)

        # calibrate
        calib = calibrate_model(ensemble, Xtr_e, ytr, method="isotonic")
        if calib is not None:
            pe = calib.predict_proba(Xte_e)[:, 1]
        else:
            pe = ensemble.predict_proba(Xte_e)[:, 1]

        acc_e = accuracy_score(yte, (pe >= 0.5).astype(int))
        auc_e = roc_auc_score(yte, pe)

        results.append({
            "train_start": str(train_dates[0]),
            "train_end": str(train_dates[-1]),
            "test_start": str(test_dates[0]),
            "test_end": str(test_dates[-1]),
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "acc_baseline": float(acc_b),
            "auc_baseline": float(auc_b),
            "acc_ensemble": float(acc_e),
            "auc_ensemble": float(auc_e),
            "n_selected_features": int(len(keep)),
        })

    if not results:
        return {}
    dfres = pd.DataFrame(results)
    summary = {
        "n_folds": int(len(dfres)),
        "baseline_acc_mean": float(dfres["acc_baseline"].mean()),
        "ensemble_acc_mean": float(dfres["acc_ensemble"].mean()),
        "baseline_auc_mean": float(dfres["auc_baseline"].mean()),
        "ensemble_auc_mean": float(dfres["auc_ensemble"].mean()),
        "selected_features_median": int(dfres["n_selected_features"].median()),
    }
    return {"summary": summary, "folds": results}


def main():
    p = PROCESSED
    if not p.exists():
        print("Processed features file not found:", p)
        return
    df = pd.read_csv(p)
    label = "TARGET_home_win"
    # prepare features_all (numeric, excluding targets)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features_all = [c for c in numeric if not c.upper().startswith("TARGET_") and c not in ("game_id",)]

    print("Running walkforward comparison (baseline vs ensemble + dynamic selection)")
    out = walkforward_compare(df, features_all, label_col=label)
    out_path = REPORTS_DIR / "ensemble_selection_compare.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print("Saved results to:", out_path)


if __name__ == "__main__":
    main()
