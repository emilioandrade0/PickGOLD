import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = SRC_ROOT / "data" / "liga_mx" / "processed"
REPORTS_DIR = SRC_ROOT / "data" / "liga_mx" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_FILE = PROCESSED_DIR / "model_ready_features_liga_mx.csv"
V3_FILE = PROCESSED_DIR / "model_ready_features_liga_mx_v3.csv"

SUMMARY_JSON = REPORTS_DIR / "liga_mx_baseline_vs_v3_temporal_summary.json"
SUMMARY_CSV = REPORTS_DIR / "liga_mx_baseline_vs_v3_temporal_summary.csv"
FOLDS_CSV = REPORTS_DIR / "liga_mx_baseline_vs_v3_temporal_folds.csv"
REACTIVATION_CSV = REPORTS_DIR / "liga_mx_market_reactivation_recommendations.csv"
REACTIVATION_MD = REPORTS_DIR / "liga_mx_market_reactivation_recommendations.md"
SELECTIVE_PLAN_CSV = REPORTS_DIR / "liga_mx_selective_upgrade_plan.csv"
SELECTIVE_PLAN_JSON = REPORTS_DIR / "liga_mx_selective_upgrade_plan.json"

TARGETS = {
    "full_game": {"target": "TARGET_full_game", "problem": "multiclass"},
    "over_25": {"target": "TARGET_over_25", "problem": "binary"},
    "btts": {"target": "TARGET_btts", "problem": "binary"},
}

NON_FEATURE_COLS = {
    "game_id",
    "date",
    "season",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "is_draw",
    "total_goals",
    "TARGET_full_game",
    "TARGET_over_25",
    "TARGET_btts",
    "TARGET_corners_over_95",
}


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path, dtype={"game_id": str})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "game_id"]).reset_index(drop=True)
    return df


def build_time_folds(n_rows: int, min_train_rows: int = 260, n_folds: int = 5) -> List[Tuple[int, int]]:
    if n_rows < (min_train_rows + 60):
        split = max(int(n_rows * 0.75), min_train_rows)
        split = min(split, n_rows - 20)
        return [(split, n_rows)]

    start = max(min_train_rows, int(n_rows * 0.55))
    remaining = n_rows - start
    fold_size = max(35, remaining // n_folds)

    folds = []
    train_end = start
    while train_end + 20 <= n_rows and len(folds) < n_folds:
        valid_end = min(train_end + fold_size, n_rows)
        if valid_end - train_end < 20:
            break
        folds.append((train_end, valid_end))
        train_end = valid_end

    if not folds:
        folds.append((start, n_rows))
    return folds


def build_model(problem: str, random_state: int) -> LGBMClassifier:
    if problem == "multiclass":
        return LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=380,
            learning_rate=0.045,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=18,
            reg_alpha=0.10,
            reg_lambda=1.10,
            random_state=random_state,
            verbose=-1,
        )

    return LGBMClassifier(
        objective="binary",
        n_estimators=340,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=16,
        reg_alpha=0.08,
        reg_lambda=1.00,
        random_state=random_state,
        verbose=-1,
    )


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int = 3) -> float:
    one_hot = np.zeros((len(y_true), n_classes), dtype=float)
    one_hot[np.arange(len(y_true)), y_true.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def evaluate_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    problem: str,
    train_end: int,
    valid_end: int,
    random_state: int,
) -> Dict[str, float]:
    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()

    train_df = train_df.dropna(subset=[target_col])
    valid_df = valid_df.dropna(subset=[target_col])

    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y_train = train_df[target_col].astype(int).to_numpy()
    y_valid = valid_df[target_col].astype(int).to_numpy()

    if len(X_train) < 50 or len(X_valid) < 20:
        return {
            "rows_train": len(X_train),
            "rows_valid": len(X_valid),
            "accuracy": np.nan,
            "logloss": np.nan,
            "brier": np.nan,
        }

    if np.unique(y_train).size < 2:
        return {
            "rows_train": len(X_train),
            "rows_valid": len(X_valid),
            "accuracy": np.nan,
            "logloss": np.nan,
            "brier": np.nan,
        }

    model = build_model(problem=problem, random_state=random_state)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_valid)

    if problem == "multiclass":
        probs = np.asarray(probs, dtype=float)
        probs = np.clip(probs, 1e-9, None)
        probs = probs / probs.sum(axis=1, keepdims=True)
        preds = np.argmax(probs, axis=1)
        return {
            "rows_train": len(X_train),
            "rows_valid": len(X_valid),
            "accuracy": float(accuracy_score(y_valid, preds)),
            "logloss": float(log_loss(y_valid, probs, labels=[0, 1, 2])),
            "brier": multiclass_brier(y_valid, probs, n_classes=3),
        }

    p1 = np.asarray(probs)[:, 1]
    preds = (p1 >= 0.5).astype(int)
    return {
        "rows_train": len(X_train),
        "rows_valid": len(X_valid),
        "accuracy": float(accuracy_score(y_valid, preds)),
        "logloss": float(log_loss(y_valid, p1, labels=[0, 1])),
        "brier": float(brier_score_loss(y_valid, p1)),
    }


def evaluate_dataset(df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    folds = build_time_folds(len(df), min_train_rows=260, n_folds=5)
    fold_rows: List[Dict] = []

    for market, config in TARGETS.items():
        target_col = config["target"]
        problem = config["problem"]

        feature_cols = [
            c for c in df.columns if c not in NON_FEATURE_COLS and c != "date_dt"
        ]

        if target_col not in df.columns:
            continue

        for fold_idx, (train_end, valid_end) in enumerate(folds, start=1):
            metrics = evaluate_fold(
                df=df,
                feature_cols=feature_cols,
                target_col=target_col,
                problem=problem,
                train_end=train_end,
                valid_end=valid_end,
                random_state=42 + fold_idx,
            )
            fold_rows.append(
                {
                    "dataset": dataset_name,
                    "market": market,
                    "fold": fold_idx,
                    "train_end_idx": train_end,
                    "valid_end_idx": valid_end,
                    **metrics,
                }
            )

    folds_df = pd.DataFrame(fold_rows)
    summary_df = (
        folds_df.groupby(["dataset", "market"], as_index=False)
        .agg(
            folds=("fold", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            logloss_mean=("logloss", "mean"),
            logloss_std=("logloss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            valid_rows_mean=("rows_valid", "mean"),
        )
        .sort_values(["market", "dataset"])
        .reset_index(drop=True)
    )

    return folds_df, summary_df


def build_reactivation_recommendations(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df[summary_df["dataset"] == "baseline"].set_index("market")
    v3 = summary_df[summary_df["dataset"] == "v3"].set_index("market")

    rows = []
    for market in TARGETS.keys():
        if market not in baseline.index or market not in v3.index:
            continue

        base_acc = float(baseline.loc[market, "accuracy_mean"])
        v3_acc = float(v3.loc[market, "accuracy_mean"])
        base_ll = float(baseline.loc[market, "logloss_mean"])
        v3_ll = float(v3.loc[market, "logloss_mean"])
        base_br = float(baseline.loc[market, "brier_mean"])
        v3_br = float(v3.loc[market, "brier_mean"])

        delta_acc = v3_acc - base_acc
        delta_ll = base_ll - v3_ll
        delta_br = base_br - v3_br

        if market == "full_game":
            if v3_acc >= 0.50 and v3_ll <= 1.03 and delta_acc >= 0.005 and delta_ll >= 0.003:
                decision = "REACTIVATE"
            elif v3_acc >= 0.48 and delta_acc >= 0.0:
                decision = "WATCHLIST"
            else:
                decision = "KEEP_FROZEN"
        else:
            if v3_acc >= 0.56 and v3_ll <= 0.69 and v3_br <= 0.245 and delta_acc >= 0.005:
                decision = "REACTIVATE"
            elif v3_acc >= 0.54 and delta_acc >= 0.0:
                decision = "WATCHLIST"
            else:
                decision = "KEEP_FROZEN"

        rows.append(
            {
                "market": market,
                "baseline_accuracy": round(base_acc, 4),
                "v3_accuracy": round(v3_acc, 4),
                "delta_accuracy": round(delta_acc, 4),
                "baseline_logloss": round(base_ll, 4),
                "v3_logloss": round(v3_ll, 4),
                "delta_logloss_improvement": round(delta_ll, 4),
                "baseline_brier": round(base_br, 4),
                "v3_brier": round(v3_br, 4),
                "delta_brier_improvement": round(delta_br, 4),
                "decision": decision,
            }
        )

    return pd.DataFrame(rows)


def build_selective_upgrade_plan(summary_df: pd.DataFrame) -> pd.DataFrame:
    baseline = summary_df[summary_df["dataset"] == "baseline"].set_index("market")
    v3 = summary_df[summary_df["dataset"] == "v3"].set_index("market")

    rows = []
    for market in TARGETS.keys():
        if market not in baseline.index or market not in v3.index:
            continue

        base_acc = float(baseline.loc[market, "accuracy_mean"])
        v3_acc = float(v3.loc[market, "accuracy_mean"])
        base_ll = float(baseline.loc[market, "logloss_mean"])
        v3_ll = float(v3.loc[market, "logloss_mean"])
        base_br = float(baseline.loc[market, "brier_mean"])
        v3_br = float(v3.loc[market, "brier_mean"])

        delta_acc = v3_acc - base_acc
        delta_ll = base_ll - v3_ll
        delta_br = base_br - v3_br

        # Composite score: prioritize accuracy but penalize calibration regressions.
        composite_score = delta_acc + (0.25 * delta_ll) + (0.25 * delta_br)

        # Apply only if there is net improvement and regressions stay within tolerance.
        apply_change = (
            (delta_acc > 0.0)
            and (delta_ll >= -0.0400)
            and (delta_br >= -0.0200)
            and (composite_score > 0.0)
        )

        rows.append(
            {
                "market": market,
                "baseline_accuracy": round(base_acc, 4),
                "v3_accuracy": round(v3_acc, 4),
                "delta_accuracy": round(delta_acc, 4),
                "baseline_logloss": round(base_ll, 4),
                "v3_logloss": round(v3_ll, 4),
                "delta_logloss_improvement": round(delta_ll, 4),
                "baseline_brier": round(base_br, 4),
                "v3_brier": round(v3_br, 4),
                "delta_brier_improvement": round(delta_br, 4),
                "composite_score": round(composite_score, 4),
                "apply_change": bool(apply_change),
                "selected_feature_source": "v3" if apply_change else "baseline",
            }
        )

    return pd.DataFrame(rows)


def write_reactivation_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# Liga MX Market Reactivation Recommendations")
    lines.append("")
    lines.append("Decision rule: prioritize temporal accuracy and calibration improvements from baseline to v3.")
    lines.append("")
    lines.append("| Market | Baseline Acc | V3 Acc | Delta Acc | Baseline LogLoss | V3 LogLoss | Decision |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    for _, row in df.iterrows():
        lines.append(
            f"| {row['market']} | {row['baseline_accuracy']:.4f} | {row['v3_accuracy']:.4f} | "
            f"{row['delta_accuracy']:+.4f} | {row['baseline_logloss']:.4f} | {row['v3_logloss']:.4f} | {row['decision']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_selective_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append("# Liga MX Selective Upgrade Plan")
    lines.append("")
    lines.append("Rule: apply modifications only where net improvement exists.")
    lines.append("")
    lines.append("| Market | Delta Acc | Delta LogLoss (improvement) | Delta Brier (improvement) | Composite | Apply | Source |")
    lines.append("|---|---:|---:|---:|---:|---|---|")

    for _, row in df.iterrows():
        lines.append(
            f"| {row['market']} | {row['delta_accuracy']:+.4f} | {row['delta_logloss_improvement']:+.4f} | "
            f"{row['delta_brier_improvement']:+.4f} | {row['composite_score']:+.4f} | "
            f"{str(bool(row['apply_change']))} | {row['selected_feature_source']} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("LIGA MX BASELINE vs V3 TEMPORAL AUDIT")
    print("=" * 72)

    baseline_df = load_dataset(BASELINE_FILE)
    v3_df = load_dataset(V3_FILE)

    baseline_folds, baseline_summary = evaluate_dataset(baseline_df, dataset_name="baseline")
    v3_folds, v3_summary = evaluate_dataset(v3_df, dataset_name="v3")

    folds_df = pd.concat([baseline_folds, v3_folds], ignore_index=True)
    summary_df = pd.concat([baseline_summary, v3_summary], ignore_index=True)

    reactivation_df = build_reactivation_recommendations(summary_df)
    selective_df = build_selective_upgrade_plan(summary_df)

    FOLDS_CSV.write_text(folds_df.to_csv(index=False), encoding="utf-8")
    SUMMARY_CSV.write_text(summary_df.to_csv(index=False), encoding="utf-8")
    SUMMARY_JSON.write_text(summary_df.to_json(orient="records", indent=2), encoding="utf-8")
    REACTIVATION_CSV.write_text(reactivation_df.to_csv(index=False), encoding="utf-8")
    write_reactivation_markdown(reactivation_df, REACTIVATION_MD)
    SELECTIVE_PLAN_CSV.write_text(selective_df.to_csv(index=False), encoding="utf-8")
    SELECTIVE_PLAN_JSON.write_text(selective_df.to_json(orient="records", indent=2), encoding="utf-8")
    write_selective_markdown(selective_df, REPORTS_DIR / "liga_mx_selective_upgrade_plan.md")

    print(f"[OK] Fold metrics : {FOLDS_CSV}")
    print(f"[OK] Summary      : {SUMMARY_CSV}")
    print(f"[OK] Reactivation : {REACTIVATION_CSV}")
    print(f"[OK] Markdown     : {REACTIVATION_MD}")
    print(f"[OK] Selective CSV: {SELECTIVE_PLAN_CSV}")
    print(f"[OK] Selective JS : {SELECTIVE_PLAN_JSON}")

    print("\nTopline decisions:")
    for _, row in reactivation_df.iterrows():
        print(
            f"  - {row['market']}: {row['decision']} "
            f"(acc {row['baseline_accuracy']:.3f} -> {row['v3_accuracy']:.3f})"
        )

    print("\nSelective apply plan (only improvements):")
    for _, row in selective_df.iterrows():
        action = "APPLY" if bool(row["apply_change"]) else "KEEP_BASELINE"
        print(
            f"  - {row['market']}: {action} "
            f"(delta_acc={row['delta_accuracy']:+.4f}, composite={row['composite_score']:+.4f})"
        )


if __name__ == "__main__":
    main()
