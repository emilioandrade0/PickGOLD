from pathlib import Path
import sys

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = SRC_ROOT
PROCESSED = BASE_DIR / "data" / "mlb" / "processed" / "model_ready_features_mlb.csv"
REPORTS_DIR = BASE_DIR / "data" / "mlb" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def main(label_col="TARGET_home_win"):
    p = PROCESSED
    if not p.exists():
        print("Processed features file not found:", p)
        return
    df = pd.read_csv(p)
    if label_col not in df.columns:
        print(f"Label '{label_col}' not found in processed features")
        return

    df = df.dropna(subset=[label_col])
    # select numeric features only
    numeric = df.select_dtypes(include=[np.number]).copy()
    # drop label and obvious ids
    drop_cols = ["game_id", "season"]
    for c in drop_cols:
        if c in numeric.columns:
            numeric = numeric.drop(columns=[c])
    if label_col in numeric.columns:
        numeric = numeric.drop(columns=[label_col])

    features = [c for c in numeric.columns if not c.upper().startswith("TARGET_")]
    if not features:
        print("No numeric features found")
        return

    X = numeric[features].fillna(0).values
    y = df[label_col].astype(int).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, random_state=42)
    model.fit(Xtr, ytr)

    importances = model.feature_importances_
    fi = sorted(list(zip(features, importances)), key=lambda x: x[1], reverse=True)
    top_fi = fi[:50]

    # correlations
    corr = {}
    for f in features:
        try:
            corr[f] = float(pd.Series(df[f]).corr(pd.Series(y)))
        except Exception:
            corr[f] = None

    top_corr = sorted([(k, v) for k, v in corr.items() if v is not None], key=lambda x: abs(x[1]), reverse=True)[:50]

    out = {
        "top_feature_importances": [{"feature": f, "importance": imp} for f, imp in top_fi],
        "top_correlations": [{"feature": f, "corr": float(v)} for f, v in top_corr],
        "n_features": len(features),
    }

    out_path = REPORTS_DIR / "feature_analysis.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    # save CSVs
    pd.DataFrame(out["top_feature_importances"]).to_csv(REPORTS_DIR / "feature_importances.csv", index=False)
    pd.DataFrame(out["top_correlations"]).to_csv(REPORTS_DIR / "feature_correlations.csv", index=False)

    print("Feature analysis saved to:", out_path)


if __name__ == "__main__":
    main()
