"""Runner for YRFI experiments: walk-forward CV, stacking+calibration, and feature pruning.

Outputs metrics to stdout.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, accuracy_score

from src.sports.mlb import train_models_mlb as t

SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT


def walk_forward_experiment(df, market_key="yrfi", n_splits=5, min_train=200, val_size=200, step=200):
    feature_cols = t.get_market_feature_columns(df, market_key)
    target_col = t.TARGET_CONFIG[market_key]["target_col"]

    # keep only rows with target
    df = df.dropna(subset=[target_col]).copy()
    df = df.reset_index(drop=True)

    results = []

    start = min_train
    while start + val_size <= len(df):
        train_df = df.iloc[:start].copy()
        valid_df = df.iloc[start:start+val_size].copy()

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[target_col].astype(int)
        X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_valid = valid_df[target_col].astype(int)

        scale = t.get_scale_pos_weight(y_train)
        xgb = t.build_xgb(scale, market_key)
        lgbm = t.build_lgbm(scale, market_key)

        if y_train.nunique() < 2 or len(y_valid) == 0:
            start += step
            continue

        # Fit base models
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        xgb_valid = xgb.predict_proba(X_valid)[:, 1]
        lgbm_valid = lgbm.predict_proba(X_valid)[:, 1]

        # Ensemble via weight search on valid
        ens_params = t.choose_best_ensemble_params(y_valid, xgb_valid, lgbm_valid, market_key)
        ens_prob = ens_params["xgb_weight"] * xgb_valid + ens_params["lgbm_weight"] * lgbm_valid

        # Stacking: create manual OOF probs on train using TimeSeriesSplit
        tss = TimeSeriesSplit(n_splits=min(5, max(2, len(X_train)//50)))
        try:
            xgb_oof = np.zeros(len(X_train))
            lgbm_oof = np.zeros(len(X_train))
            for train_idx, val_idx in tss.split(X_train):
                xgb_fold = clone(xgb)
                lgbm_fold = clone(lgbm)
                xgb_fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                lgbm_fold.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                xgb_oof[val_idx] = xgb_fold.predict_proba(X_train.iloc[val_idx])[:, 1]
                lgbm_oof[val_idx] = lgbm_fold.predict_proba(X_train.iloc[val_idx])[:, 1]

            stack_train = np.column_stack([xgb_oof, lgbm_oof])
            stack_clf = LogisticRegression(max_iter=1000)
            stack_clf.fit(stack_train, y_train)

            # stack valid
            xgb_valid_full = xgb.predict_proba(X_valid)[:, 1]
            lgbm_valid_full = lgbm.predict_proba(X_valid)[:, 1]
            stack_valid = np.column_stack([xgb_valid_full, lgbm_valid_full])
            stack_prob = stack_clf.predict_proba(stack_valid)[:, 1]

            # Calibrate stack with isotonic on train OOF
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(x=stack_clf.predict_proba(stack_train)[:, 1], y=y_train)
            stack_calib = iso.transform(stack_prob)
        except Exception:
            stack_prob = np.zeros(len(y_valid))
            stack_calib = np.zeros(len(y_valid))

        # metrics
        def metrics(y, probs, thr=0.5):
            preds = (probs >= thr).astype(int)
            return {"auc": float(roc_auc_score(y, probs)), "acc": float(accuracy_score(y, preds))}

        res = {
            "start": start,
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "xgb": metrics(y_valid, xgb_valid),
            "lgbm": metrics(y_valid, lgbm_valid),
            "ensemble": metrics(y_valid, ens_prob, thr=ens_params["threshold"]),
            "stack": metrics(y_valid, stack_prob) if stack_prob.sum() > 0 else None,
            "stack_calib": metrics(y_valid, stack_calib) if stack_calib.sum() > 0 else None,
            "ens_params": ens_params,
        }
        results.append(res)

        start += step

    return results


def prune_and_rerun(df, market_key="yrfi", top_k=12):
    # train single global model to get importances
    feature_cols = t.get_market_feature_columns(df, market_key)
    target_col = t.TARGET_CONFIG[market_key]["target_col"]
    df = df.dropna(subset=[target_col]).copy().reset_index(drop=True)

    # use nearly all data minus last 300 for training importances
    if len(df) < 500:
        return None
    train_df = df.iloc[:-300]
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train = train_df[target_col].astype(int)
    scale = t.get_scale_pos_weight(y_train)
    xgb = t.build_xgb(scale, market_key)
    lgbm = t.build_lgbm(scale, market_key)
    xgb.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    xgb_imp = getattr(xgb, "feature_importances_", np.zeros(len(feature_cols)))
    lgbm_imp = getattr(lgbm, "feature_importances_", np.zeros(len(feature_cols)))
    combined = [(c, float((float(x)+float(l))/2.0)) for c, x, l in zip(feature_cols, xgb_imp, lgbm_imp)]
    combined_sorted = sorted(combined, key=lambda x: x[1], reverse=True)
    top_features = [c for c, _ in combined_sorted[:top_k]]

    # rerun walk-forward using only top_features
    print(f"Pruned top_{top_k}: {top_features}")
    # temporarily replace market priority with pruned list
    old = t.MARKET_FEATURE_PRIORITY.get(market_key)
    t.MARKET_FEATURE_PRIORITY[market_key] = top_features
    res = walk_forward_experiment(df, market_key=market_key, n_splits=5, min_train=200, val_size=200, step=200)
    # restore
    t.MARKET_FEATURE_PRIORITY[market_key] = old
    return res


def evaluate_catboost_available():
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except Exception:
        return None
def stacked_meta_experiment(df, market_key="yrfi", n_oof_splits=5, min_train=200, val_size=200, step=200):
    """Train a meta-model on OOF predictions (TimeSeriesSplit over historical data),
    then evaluate that meta-model across the same walk-forward windows.
    """
    feature_cols = t.get_market_feature_columns(df, market_key)
    target_col = t.TARGET_CONFIG[market_key]["target_col"]

    df = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    if len(df) < (min_train + val_size):
        return []

    # Produce OOF predictions for base models across the historical series
    X_all = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    y_all = df[target_col].astype(int)

    tss = TimeSeriesSplit(n_splits=n_oof_splits)
    xgb_oof = np.zeros(len(X_all))
    lgbm_oof = np.zeros(len(X_all))

    for train_idx, val_idx in tss.split(X_all):
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr = y_all.iloc[train_idx]
        scale = t.get_scale_pos_weight(y_tr)
        xgb = t.build_xgb(scale, market_key)
        lgbm = t.build_lgbm(scale, market_key)
        xgb.fit(X_tr, y_tr)
        lgbm.fit(X_tr, y_tr)
        xgb_oof[val_idx] = xgb.predict_proba(X_val)[:, 1]
        lgbm_oof[val_idx] = lgbm.predict_proba(X_val)[:, 1]

    # train meta-model on rows that received OOF predictions (non-zero length)
    oof_mask = (xgb_oof != 0) | (lgbm_oof != 0)
    if oof_mask.sum() < 50:
        return []

    stack_train = np.column_stack([xgb_oof[oof_mask], lgbm_oof[oof_mask]])
    y_stack = y_all.iloc[oof_mask].values
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(stack_train, y_stack)

    # calibrate meta on its training set
    try:
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(x=meta_clf.predict_proba(stack_train)[:, 1], y=y_stack)
    except Exception:
        iso = None

    # Now run walk-forward windows and evaluate using meta_clf
    results = []
    start = min_train
    while start + val_size <= len(df):
        train_df = df.iloc[:start].copy()
        valid_df = df.iloc[start:start + val_size].copy()

        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[target_col].astype(int)
        X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_valid = valid_df[target_col].astype(int)

        if y_train.nunique() < 2 or len(y_valid) == 0:
            start += step
            continue

        scale = t.get_scale_pos_weight(y_train)
        xgb = t.build_xgb(scale, market_key)
        lgbm = t.build_lgbm(scale, market_key)
        xgb.fit(X_train, y_train)
        lgbm.fit(X_train, y_train)

        xgb_valid = xgb.predict_proba(X_valid)[:, 1]
        lgbm_valid = lgbm.predict_proba(X_valid)[:, 1]

        # meta predict
        stack_valid = np.column_stack([xgb_valid, lgbm_valid])
        meta_prob = meta_clf.predict_proba(stack_valid)[:, 1]
        if iso is not None:
            meta_calib = iso.transform(meta_prob)
        else:
            meta_calib = meta_prob

        # ensemble baseline params for reference
        ens_params = t.choose_best_ensemble_params(y_valid, xgb_valid, lgbm_valid, market_key)

        def metrics(y, probs, thr=0.5):
            preds = (probs >= thr).astype(int)
            return {"auc": float(roc_auc_score(y, probs)), "acc": float(accuracy_score(y, preds))}

        res = {
            "start": start,
            "train_rows": len(X_train),
            "valid_rows": len(X_valid),
            "xgb": metrics(y_valid, xgb_valid),
            "lgbm": metrics(y_valid, lgbm_valid),
            "ensemble": metrics(y_valid, ens_params["xgb_weight"] * xgb_valid + ens_params["lgbm_weight"] * lgbm_valid, thr=ens_params["threshold"]),
            "meta_stack": metrics(y_valid, meta_prob),
            "meta_stack_calib": metrics(y_valid, meta_calib),
            "ens_params": ens_params,
        }
        results.append(res)
        start += step

    return results

def save_results(results, name="yrfi_experiments.json"):
    try:
        path = t.REPORTS_DIR / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {path}")
    except Exception as e:
        print("Failed to save results:", e)


if __name__ == "__main__":
    df = t.load_dataset()
    df = df.tail(3000).copy()
    print("Running walk-forward baseline for yrfi...")
    wf = walk_forward_experiment(df, market_key="yrfi", n_splits=5, min_train=200, val_size=200, step=200)
    print(json.dumps(wf, indent=2))
    save_results({"walk_forward": wf}, name="yrfi_experiments_walkforward.json")

    print("\nRunning prune+rerun (top12)...")
    pr = prune_and_rerun(df, market_key="yrfi", top_k=12)
    print(json.dumps(pr, indent=2))
    save_results({"prune_rerun": pr}, name="yrfi_experiments_prune_top12.json")

    # Try CatBoost quick check
    CatBoost = evaluate_catboost_available()
    if CatBoost is not None:
        print('\nCatBoost detected — running quick baseline compare...')
        # quick fit on last 2000 rows train / 200 valid
        df_small = df.tail(2200).copy()
        feature_cols = t.get_market_feature_columns(df_small, 'yrfi')
        target_col = t.TARGET_CONFIG['yrfi']['target_col']
        df_small = df_small.dropna(subset=[target_col]).reset_index(drop=True)
        train_df = df_small.iloc[:-200]
        valid_df = df_small.iloc[-200:]
        X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_train = train_df[target_col].astype(int)
        X_valid = valid_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        y_valid = valid_df[target_col].astype(int)

        cb = CatBoost(iterations=200, depth=4, learning_rate=0.03, verbose=0, random_seed=42)
        try:
            cb.fit(X_train, y_train)
            cb_probs = cb.predict_proba(X_valid)[:, 1]
            cb_metrics = {"auc": float(roc_auc_score(y_valid, cb_probs)), "acc": float(accuracy_score((cb_probs>=0.5).astype(int), y_valid))}
            print('CatBoost metrics:', cb_metrics)
            save_results({"catboost": cb_metrics}, name="yrfi_experiments_catboost.json")
        except Exception as e:
            print('CatBoost run failed:', e)
    else:
        print('\nCatBoost not installed — skipping CatBoost experiment')
