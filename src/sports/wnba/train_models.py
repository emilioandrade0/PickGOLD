from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from typing import Tuple
try:
    from calibration import calibrate_probability, load_calibration_config
except Exception:
    def calibrate_probability(p, sport=None, market=None, calibration_config=None):
        try:
            return float(p)
        except Exception:
            return p

    def load_calibration_config(path):
        return None

# --- RUTAS ---
import sys
SRC_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

BASE_DIR = SRC_ROOT
PROCESSED_DATA = BASE_DIR / "data" / "wnba" / "processed" / "model_ready_features.csv"
MODELS_DIR = BASE_DIR / "data" / "wnba" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def is_gpu_available() -> bool:
    """Detecta si hay GPU NVIDIA disponible en el sistema (nvidia-smi)."""
    try:
        import shutil, subprocess
        if shutil.which("nvidia-smi"):
            res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=3)
            return res.returncode == 0 and len(res.stdout.strip()) > 0
    except Exception:
        return False
    return False


# GPU disabled by request: force CPU-only runs to avoid CUDA/XGBoost GPU constructor issues
USE_GPU = False
print("ℹ️ GPU usage disabled — forcing CPU-only training")


class ConstantBinaryModel:
    """Modelo binario constante para escenarios single-class."""

    def __init__(self, p_one: float):
        self.p_one = float(np.clip(p_one, 0.0, 1.0))

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        n = len(X)
        p1 = np.full(n, self.p_one, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def _binary_class_stats(y: pd.Series) -> tuple[int, int, float]:
    counts = y.value_counts(dropna=True)
    n_classes = int(counts.shape[0])
    if n_classes == 0:
        return 0, 0, 0.0
    if n_classes == 1:
        return 1, int(counts.iloc[0]), 0.0
    minority = int(counts.min())
    ratio = float(minority / max(int(counts.sum()), 1))
    return 2, minority, ratio


def evaluate_ensemble_cv(X: pd.DataFrame, y: pd.Series, label: str):
    """
    Validación Cruzada Temporal para los 3 modelos base promediados simple,
    solo para darte una idea rápida del rendimiento antes del stacking.
    """
    print(f"\n📊 Validación temporal CV para {label}...")
    tscv = TimeSeriesSplit(n_splits=5)

    acc_scores = []
    auc_scores = []
    brier_scores = []
    logloss_scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Entrenar modelos base (intenta GPU si está disponible)
        xgb_model, lgb_model, cat_model = fit_base_models(X_train, y_train, use_gpu=USE_GPU)

        prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
        prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
        prob_cat = cat_model.predict_proba(X_test)[:, 1]
        
        # Promedio simple solo para el reporte de CV
        prob_ens = (prob_xgb + prob_lgb + prob_cat) / 3
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
            f"    Fold {fold}: "
            f"ACC={acc:.4f} | "
            f"AUC={auc:.4f} | "
            f"Brier={brier:.4f} | "
            f"LogLoss={ll:.4f}"
        )

    print(f"\n✅ CV Media {label} (Promedio Simple)")
    print(f"    ACC    : {np.nanmean(acc_scores):.4f}")
    print(f"    AUC    : {np.nanmean(auc_scores):.4f}")
    print(f"    Brier  : {np.nanmean(brier_scores):.4f}")
    print(f"    LogLoss: {np.nanmean(logloss_scores):.4f}")


def build_time_splits(n_rows: int):
    train_end = int(n_rows * 0.70)
    calib_end = int(n_rows * 0.80)

    if train_end <= 0 or calib_end <= train_end or calib_end >= n_rows:
        raise ValueError("No hay suficientes filas para hacer split temporal 70/10/20.")

    return train_end, calib_end


def fit_base_models(X_train, y_train, use_gpu: bool = False):
    """Construye y entrena los tres modelos base; intenta usar GPU si `use_gpu=True`.
    Hace fallback a constructores CPU si la construcción/fit con GPU falla.
    """
    y_train = pd.Series(y_train).astype(int)
    if y_train.nunique() < 2:
        const_p = float(y_train.mean()) if len(y_train) else 0.5
        print(f"⚠️ Fold con una sola clase ({int(round(const_p))}). Usando modelos constantes.")
        const_model = ConstantBinaryModel(const_p)
        return const_model, const_model, const_model

    # XGBoost
    try:
        if use_gpu:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=4,
                subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42,
                base_score=0.5,
                tree_method="gpu_hist"
            )
        else:
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=4,
                subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, base_score=0.5
            )
        xgb_model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ XGBoost GPU constructor/fit falló, usando CPU fallback: {e}")
        xgb_model = xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, subsample=0.85, colsample_bytree=0.85, eval_metric="logloss", random_state=42, base_score=0.5)
        xgb_model.fit(X_train, y_train)

    # LightGBM
    try:
        if use_gpu:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1,
                device='gpu'
            )
        else:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1
            )
        lgb_model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ LightGBM GPU constructor/fit falló, usando CPU fallback: {e}")
        lgb_model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, num_leaves=31, subsample=0.85, colsample_bytree=0.85, random_state=42, verbosity=-1)
        lgb_model.fit(X_train, y_train)

    # CatBoost
    try:
        if use_gpu:
            # For GPU CatBoost, use bootstrap_type compatible with subsample
            cat_model = CatBoostClassifier(
                iterations=300, learning_rate=0.03, depth=4,
                subsample=0.85, random_state=42, verbose=0,
                task_type='GPU', devices='0', bootstrap_type='Bernoulli'
            )
        else:
            cat_model = CatBoostClassifier(
                iterations=300, learning_rate=0.03, depth=4,
                subsample=0.85, random_state=42, verbose=0
            )
        cat_model.fit(X_train, y_train)
    except Exception as e:
        print(f"⚠️ CatBoost GPU constructor/fit falló, usando CPU fallback: {e}")
        cat_model = CatBoostClassifier(iterations=300, learning_rate=0.03, depth=4, subsample=0.85, random_state=42, verbose=0)
        cat_model.fit(X_train, y_train)

    return xgb_model, lgb_model, cat_model


def train_meta_learner_calibrated(base_models, X_calib, y_calib):
    xgb_m, lgb_m, cat_m = base_models
    
    p_xgb = xgb_m.predict_proba(X_calib)[:, 1]
    p_lgb = lgb_m.predict_proba(X_calib)[:, 1]
    p_cat = cat_m.predict_proba(X_calib)[:, 1]
    
    X_meta = np.column_stack([p_xgb, p_lgb, p_cat])
    
    # Cambiamos 'isotonic' por 'sigmoid' para mayor estabilidad y evitar el NameError
    meta_model = LogisticRegression()
    calibrated_meta = CalibratedClassifierCV(estimator=meta_model, method='sigmoid', cv=5)
    calibrated_meta.fit(X_meta, y_calib)
    
    return calibrated_meta


def generate_oof_preds(X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Tuple[np.ndarray, Tuple, dict]:
    """
    Genera predicciones out-of-fold para los 3 modelos base en X (temporal).
    Devuelve un array (n_rows, 3) con las probabilidades OOF y los modelos finales
    entrenados sobre todo X.
    """
    n = len(X)
    oof = np.zeros((n, 3), dtype=float)
    oof[:] = np.nan

    tscv = TimeSeriesSplit(n_splits=n_splits)

    imp_xgb_list = []
    imp_lgb_list = []
    imp_cat_list = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr = y.iloc[train_idx]

        xgb_m, lgb_m, cat_m = fit_base_models(X_tr, y_tr, use_gpu=USE_GPU)

        oof[test_idx, 0] = xgb_m.predict_proba(X_te)[:, 1]
        oof[test_idx, 1] = lgb_m.predict_proba(X_te)[:, 1]
        oof[test_idx, 2] = cat_m.predict_proba(X_te)[:, 1]
        # recolectar importancias por fold (si están disponibles)
        try:
            imp_xgb_list.append(getattr(xgb_m, "feature_importances_") / (getattr(xgb_m, "feature_importances_").sum() + 1e-12))
        except Exception:
            imp_xgb_list.append(np.zeros(X.shape[1], dtype=float))
        try:
            imp_lgb_list.append(getattr(lgb_m, "feature_importances_") / (getattr(lgb_m, "feature_importances_").sum() + 1e-12))
        except Exception:
            imp_lgb_list.append(np.zeros(X.shape[1], dtype=float))
        try:
            # CatBoost tiene get_feature_importance
            imp_cat = np.array(cat_m.get_feature_importance())
            imp_cat_list.append(imp_cat / (imp_cat.sum() + 1e-12))
        except Exception:
            imp_cat_list.append(np.zeros(X.shape[1], dtype=float))

    # Entrenar modelos finales sobre todo X (para usar en producción/meta-test)
    final_xgb, final_lgb, final_cat = fit_base_models(X, y, use_gpu=USE_GPU)

    # Promediar importancias por columna
    imp_dict = {}
    if len(imp_xgb_list):
        imp_dict["xgb"] = np.nanmean(np.vstack(imp_xgb_list), axis=0)
    else:
        imp_dict["xgb"] = np.zeros(X.shape[1], dtype=float)
    if len(imp_lgb_list):
        imp_dict["lgb"] = np.nanmean(np.vstack(imp_lgb_list), axis=0)
    else:
        imp_dict["lgb"] = np.zeros(X.shape[1], dtype=float)
    if len(imp_cat_list):
        imp_dict["cat"] = np.nanmean(np.vstack(imp_cat_list), axis=0)
    else:
        imp_dict["cat"] = np.zeros(X.shape[1], dtype=float)

    return oof, (final_xgb, final_lgb, final_cat), imp_dict


def train_meta_from_oof(X_meta: np.ndarray, y_meta: pd.Series):
    y_meta = pd.Series(y_meta).astype(int)
    if len(y_meta) == 0 or y_meta.nunique() < 2:
        const_p = float(y_meta.mean()) if len(y_meta) else 0.5
        print("⚠️ Meta-learner con una sola clase. Usando meta constante.")
        return ConstantBinaryModel(const_p)

    min_class = int(y_meta.value_counts().min())
    cv_folds = min(5, min_class)
    if cv_folds < 2:
        const_p = float(y_meta.mean())
        print("⚠️ Meta-learner sin suficientes ejemplos por clase. Usando meta constante.")
        return ConstantBinaryModel(const_p)

    meta_model = LogisticRegression()
    calibrated_meta = CalibratedClassifierCV(estimator=meta_model, method='sigmoid', cv=cv_folds)
    calibrated_meta.fit(X_meta, y_meta)
    return calibrated_meta


def print_holdout_metrics(base_models, meta_model, X_test, y_test, label: str):
    xgb_m, lgb_m, cat_m = base_models
    
    p_xgb = xgb_m.predict_proba(X_test)[:, 1]
    p_lgb = lgb_m.predict_proba(X_test)[:, 1]
    p_cat = cat_m.predict_proba(X_test)[:, 1]
    
    X_meta_test = np.column_stack([p_xgb, p_lgb, p_cat])
    final_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    
    preds = (final_probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    brier = brier_score_loss(y_test, final_probs)
    try:
        auc = roc_auc_score(y_test, final_probs)
    except Exception:
        auc = np.nan
    try:
        ll = log_loss(y_test, final_probs, labels=[0, 1])
    except Exception:
        ll = np.nan

    print(f"\n📊 HOLDOUT FINAL - {label} (Stacking + Sigmoid)")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    AUC      : {auc:.4f}" if np.isfinite(auc) else "    AUC      : NA (single-class holdout)")
    print(f"    Brier    : {brier:.4f}")
    print(f"    LogLoss  : {ll:.4f}" if np.isfinite(ll) else "    LogLoss  : NA")


def find_best_threshold(probs: np.ndarray, y_true: pd.Series, low: float = 0.47, high: float = 0.53, step: float = 0.005):
    thresholds = np.arange(low, high + 1e-9, step)
    best_thr = 0.5
    best_acc = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = accuracy_score(y_true, preds)
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return best_thr, best_acc


def search_best_weights(p_x: np.ndarray, p_l: np.ndarray, p_c: np.ndarray, y_true: pd.Series, calib_cfg=None, sport="wnba", market="full_game", step: float = 0.1):
    """Busca combinación de pesos (w_x, w_l, w_c) en una malla que maximice accuracy en holdout.
    Devuelve (best_weights, best_acc, best_probs)
    """
    best_acc = -1.0
    best_w = (1 / 3, 1 / 3, 1 / 3)
    best_probs = None

    grid = np.arange(0.0, 1.0 + 1e-9, step)
    for wx in grid:
        for wl in grid:
            wc = 1.0 - wx - wl
            if wc < -1e-9:
                continue
            wc = max(0.0, wc)
            probs = wx * p_x + wl * p_l + wc * p_c
            if calib_cfg is not None:
                probs = np.array([calibrate_probability(float(v), sport=sport, market=market, calibration_config=calib_cfg) for v in probs])
            preds = (probs >= 0.5).astype(int)
            acc = accuracy_score(y_true, preds)
            if acc > best_acc:
                best_acc = acc
                best_w = (float(wx), float(wl), float(wc))
                best_probs = probs.copy()

    return best_w, best_acc, best_probs


def train_all_models():
    if not PROCESSED_DATA.exists():
        raise FileNotFoundError(f"No existe el archivo de features: {PROCESSED_DATA}")

    df = pd.read_csv(PROCESSED_DATA).sort_values("date").reset_index(drop=True)

    # LISTA MAESTRA DE COLUMNAS PROHIBIDAS PARA EL ENTRENAMIENTO
    cols_to_drop = [
    "game_id", "date", "season", "home_team", "away_team",
    "TARGET_home_win", "TARGET_home_win_q1", "TARGET_home_win_h1", "TARGET_home_cover_spread", "TARGET_over_total",
    "home_pts_total", "away_pts_total", "home_q1", "away_q1" # <--- ESTO ES LO QUE EVITA EL 1.0
]

    # Borrar solo las que existan en el DataFrame actual
    existing_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=existing_drop)

    missing_cols = [c for c in cols_to_drop if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas esperadas en el dataset: {missing_cols}")

    X = df.drop(columns=cols_to_drop)
    
    # --- FASE 2: FEATURE SELECTION DINÁMICA (solo numéricas) ---
    # Eliminamos columnas numéricas con varianza casi cero; evitamos .var() en datetime/objetos
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        num_selector = X[num_cols].var() > 1e-6
        keep_numeric = num_selector[num_selector].index.tolist()
    else:
        keep_numeric = []
    X = X.loc[:, keep_numeric]
    # selector: columnas finales que usaremos como features (útil para splits posteriores)
    selector = X.columns.tolist()
    # ------------------------------------------

    y_game = df["TARGET_home_win"].astype(int)
    y_q1 = df["TARGET_home_win_q1"].astype(int)
    y_h1 = pd.to_numeric(df.get("TARGET_home_win_h1"), errors="coerce")
    y_spread = pd.to_numeric(df.get("TARGET_home_cover_spread"), errors="coerce")
    y_total = pd.to_numeric(df.get("TARGET_over_total"), errors="coerce")

    feature_names = X.columns.tolist()
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")

    print(f"🧠 Entrenando IAs con {len(feature_names)} features (después de selección dinámica)...")

    # 1) VALIDACIÓN TEMPORAL CV
    evaluate_ensemble_cv(X, y_game, "PARTIDO COMPLETO")
    evaluate_ensemble_cv(X, y_q1, "PRIMER CUARTO")
    evaluate_ensemble_cv(X.loc[y_h1.notna()], y_h1.loc[y_h1.notna()].astype(int), "PRIMERA MITAD")

    # 2) SPLIT TEMPORAL FINAL: 70 / 10 / 20
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

    yh_train = y_h1.iloc[:train_end]
    yh_calib = y_h1.iloc[train_end:calib_end]
    yh_test = y_h1.iloc[calib_end:]

    print("\n🧩 Split temporal final:")
    print(f"    Train base : {len(X_train)}")
    print(f"    Calibración: {len(X_calib)}")
    print(f"    Holdout    : {len(X_test)}")

    # 3) MODELOS FINALES FULL GAME
    # Helper: train pipeline for a given dataset (can be reused for market splits)
    def train_for_dataframe(X_df: pd.DataFrame, y_game_ser: pd.Series, y_q1_ser: pd.Series, y_h1_ser: pd.Series, suffix: str = "") -> dict:
        """Entrena base/meta para game, q1 y h1 sobre X_df. Guarda modelos con sufijo y retorna pick_params parciales."""
        result = {}
        print(f"\n⏳ Generando OOF para PARTIDO COMPLETO {suffix} (5 folds temporales)...")
        X_oof_game = X_df.iloc[:calib_end]
        y_oof_game = y_game_ser.iloc[:calib_end]

        oof_game, final_base_game, imp_game = generate_oof_preds(X_oof_game, y_oof_game, n_splits=5)
        valid_idx = ~np.isnan(oof_game).any(axis=1)
        X_meta_game = oof_game[valid_idx]
        y_meta_game = y_oof_game.iloc[valid_idx]

        print("🧠 Entrenando Meta-Learner sobre OOF (Partido Completo)...")
        meta_g = train_meta_from_oof(X_meta_game, y_meta_game)

        print_holdout_metrics(final_base_game, meta_g, X_test, yg_test, f"PARTIDO COMPLETO{suffix}")

        try:
            calib_cfg = load_calibration_config(MODELS_DIR / "calibration_params.json")
        except Exception:
            calib_cfg = None

        p_x_test = final_base_game[0].predict_proba(X_test)[:, 1]
        p_l_test = final_base_game[1].predict_proba(X_test)[:, 1]
        p_c_test = final_base_game[2].predict_proba(X_test)[:, 1]
        X_meta_test = np.column_stack([p_x_test, p_l_test, p_c_test])
        final_probs = meta_g.predict_proba(X_meta_test)[:, 1]
        if calib_cfg is not None:
            calibrated_probs = np.array([calibrate_probability(float(v), sport="wnba", market="full_game", calibration_config=calib_cfg) for v in final_probs])
        else:
            calibrated_probs = final_probs.copy()

        best_thr_game, best_acc_game = find_best_threshold(calibrated_probs, yg_test, low=0.47, high=0.53, step=0.005)
        print(f"📌 Best threshold (GAME){suffix} on holdout: {best_thr_game:.3f} (acc={best_acc_game:.4f})")

        # buscar mejores pesos para promedio ponderado y compararlo con meta
        best_w, best_w_acc, best_w_probs = search_best_weights(p_x_test, p_l_test, p_c_test, yg_test, calib_cfg, sport="wnba", market="full_game", step=0.1)
        meta_preds = (calibrated_probs >= best_thr_game).astype(int)
        meta_acc = accuracy_score(yg_test, meta_preds)
        use_meta = True
        chosen_weights = {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}
        if best_w_acc > meta_acc:
            use_meta = False
            chosen_weights = {"xgb": best_w[0], "lgb": best_w[1], "cat": best_w[2]}

        # Guardar modelos finales (con sufijo)
        suffix_label = f"{suffix}" if suffix else ""
        joblib.dump(final_base_game[0], MODELS_DIR / f"xgb_game{suffix_label}.pkl")
        joblib.dump(final_base_game[1], MODELS_DIR / f"lgb_game{suffix_label}.pkl")
        joblib.dump(final_base_game[2], MODELS_DIR / f"cat_game{suffix_label}.pkl")
        joblib.dump(meta_g, MODELS_DIR / f"meta_game{suffix_label}.pkl")

        # Calcular features estables (promedio de importancias entre modelos)
        feat_names = X_df.columns.tolist()
        mean_imp = (imp_game.get("xgb", np.zeros(len(feat_names))) + imp_game.get("lgb", np.zeros(len(feat_names))) + imp_game.get("cat", np.zeros(len(feat_names)))) / 3.0
        # elegir top-N estables, guardar lista para inspección
        keep_top_n = min(100, len(feat_names))
        top_idx = np.argsort(-mean_imp)[:keep_top_n]
        stable_features = [feat_names[i] for i in top_idx]
        try:
            joblib.dump(stable_features, MODELS_DIR / f"stable_features{suffix_label}.pkl")
        except Exception:
            pass

        result_game = {
            "threshold": float(best_thr_game),
            "threshold_acc": float(best_acc_game),
            "use_meta": bool(use_meta),
            "weights": chosen_weights,
            "importances": imp_game,
            "stable_features": stable_features,
        }

        # Q1
        print(f"\n⏳ Generando OOF para PRIMER CUARTO {suffix} (5 folds temporales)...")
        X_oof_q1 = X_df.iloc[:calib_end]
        y_oof_q1 = y_q1_ser.iloc[:calib_end]

        oof_q1, final_base_q1, imp_q1 = generate_oof_preds(X_oof_q1, y_oof_q1, n_splits=5)
        valid_idx_q1 = ~np.isnan(oof_q1).any(axis=1)
        X_meta_q1 = oof_q1[valid_idx_q1]
        y_meta_q1 = y_oof_q1.iloc[valid_idx_q1]

        print("🧠 Entrenando Meta-Learner sobre OOF (Primer Cuarto)...")
        meta_q1 = train_meta_from_oof(X_meta_q1, y_meta_q1)

        print_holdout_metrics(final_base_q1, meta_q1, X_test, yq_test, f"PRIMER CUARTO{suffix}")

        p_x_q_test = final_base_q1[0].predict_proba(X_test)[:, 1]
        p_l_q_test = final_base_q1[1].predict_proba(X_test)[:, 1]
        p_c_q_test = final_base_q1[2].predict_proba(X_test)[:, 1]
        X_meta_test_q = np.column_stack([p_x_q_test, p_l_q_test, p_c_q_test])
        final_probs_q = meta_q1.predict_proba(X_meta_test_q)[:, 1]
        if calib_cfg is not None:
            calibrated_probs_q = np.array([calibrate_probability(float(v), sport="wnba", market="q1", calibration_config=calib_cfg) for v in final_probs_q])
        else:
            calibrated_probs_q = final_probs_q.copy()

        best_thr_q1, best_acc_q1 = find_best_threshold(calibrated_probs_q, yq_test, low=0.47, high=0.53, step=0.005)
        print(f"📌 Best threshold (Q1){suffix} on holdout: {best_thr_q1:.3f} (acc={best_acc_q1:.4f})")

        best_w_q, best_w_acc_q, best_w_probs_q = search_best_weights(p_x_q_test, p_l_q_test, p_c_q_test, yq_test, calib_cfg, sport="wnba", market="q1", step=0.1)
        meta_preds_q = (calibrated_probs_q >= best_thr_q1).astype(int)
        meta_acc_q = accuracy_score(yq_test, meta_preds_q)
        use_meta_q = True
        chosen_weights_q = {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}
        if best_w_acc_q > meta_acc_q:
            use_meta_q = False
            chosen_weights_q = {"xgb": best_w_q[0], "lgb": best_w_q[1], "cat": best_w_q[2]}

        joblib.dump(final_base_q1[0], MODELS_DIR / f"xgb_q1{suffix_label}.pkl")
        joblib.dump(final_base_q1[1], MODELS_DIR / f"lgb_q1{suffix_label}.pkl")
        joblib.dump(final_base_q1[2], MODELS_DIR / f"cat_q1{suffix_label}.pkl")
        joblib.dump(meta_q1, MODELS_DIR / f"meta_q1{suffix_label}.pkl")

        # Q1 stable features
        feat_names_q = X_df.columns.tolist()
        mean_imp_q = (imp_q1.get("xgb", np.zeros(len(feat_names_q))) + imp_q1.get("lgb", np.zeros(len(feat_names_q))) + imp_q1.get("cat", np.zeros(len(feat_names_q)))) / 3.0
        keep_top_n_q = min(100, len(feat_names_q))
        top_idx_q = np.argsort(-mean_imp_q)[:keep_top_n_q]
        stable_features_q = [feat_names_q[i] for i in top_idx_q]
        try:
            joblib.dump(stable_features_q, MODELS_DIR / f"stable_features_q1{suffix_label}.pkl")
        except Exception:
            pass

        result_q1 = {
            "threshold": float(best_thr_q1),
            "threshold_acc": float(best_acc_q1),
            "use_meta": bool(use_meta_q),
            "weights": chosen_weights_q,
            "importances": imp_q1,
            "stable_features": stable_features_q,
        }

        # H1
        mask_h1_df = y_h1_ser.notna()
        if mask_h1_df.any():
            X_h1_df = X_df.loc[mask_h1_df]
            y_h1_clean = y_h1_ser.loc[mask_h1_df].astype(int)
            train_end_h1, calib_end_h1 = build_time_splits(len(X_h1_df))

            X_oof_h1 = X_h1_df.iloc[:calib_end_h1]
            y_oof_h1 = y_h1_clean.iloc[:calib_end_h1]
            X_test_h1 = X_h1_df.iloc[calib_end_h1:]
            y_test_h1 = y_h1_clean.iloc[calib_end_h1:]

            print(f"\n⏳ Generando OOF para PRIMERA MITAD {suffix} (5 folds temporales)...")
            oof_h1, final_base_h1, imp_h1 = generate_oof_preds(X_oof_h1, y_oof_h1, n_splits=5)
            valid_idx_h1 = ~np.isnan(oof_h1).any(axis=1)
            X_meta_h1 = oof_h1[valid_idx_h1]
            y_meta_h1 = y_oof_h1.iloc[valid_idx_h1]

            print("🧠 Entrenando Meta-Learner sobre OOF (Primera Mitad)...")
            meta_h1 = train_meta_from_oof(X_meta_h1, y_meta_h1)

            print_holdout_metrics(final_base_h1, meta_h1, X_test_h1, y_test_h1, f"PRIMERA MITAD{suffix}")

            p_x_h1_test = final_base_h1[0].predict_proba(X_test_h1)[:, 1]
            p_l_h1_test = final_base_h1[1].predict_proba(X_test_h1)[:, 1]
            p_c_h1_test = final_base_h1[2].predict_proba(X_test_h1)[:, 1]
            X_meta_test_h1 = np.column_stack([p_x_h1_test, p_l_h1_test, p_c_h1_test])
            final_probs_h1 = meta_h1.predict_proba(X_meta_test_h1)[:, 1]
            if calib_cfg is not None:
                calibrated_probs_h1 = np.array([calibrate_probability(float(v), sport="wnba", market="h1", calibration_config=calib_cfg) for v in final_probs_h1])
            else:
                calibrated_probs_h1 = final_probs_h1.copy()

            best_thr_h1, best_acc_h1 = find_best_threshold(calibrated_probs_h1, y_test_h1, low=0.47, high=0.53, step=0.005)
            print(f"📌 Best threshold (H1){suffix} on holdout: {best_thr_h1:.3f} (acc={best_acc_h1:.4f})")

            best_w_h1, best_w_acc_h1, _ = search_best_weights(p_x_h1_test, p_l_h1_test, p_c_h1_test, y_test_h1, calib_cfg, sport="wnba", market="h1", step=0.1)
            meta_preds_h1 = (calibrated_probs_h1 >= best_thr_h1).astype(int)
            meta_acc_h1 = accuracy_score(y_test_h1, meta_preds_h1)
            use_meta_h1 = True
            chosen_weights_h1 = {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}
            if best_w_acc_h1 > meta_acc_h1:
                use_meta_h1 = False
                chosen_weights_h1 = {"xgb": best_w_h1[0], "lgb": best_w_h1[1], "cat": best_w_h1[2]}

            joblib.dump(final_base_h1[0], MODELS_DIR / f"xgb_h1{suffix_label}.pkl")
            joblib.dump(final_base_h1[1], MODELS_DIR / f"lgb_h1{suffix_label}.pkl")
            joblib.dump(final_base_h1[2], MODELS_DIR / f"cat_h1{suffix_label}.pkl")
            joblib.dump(meta_h1, MODELS_DIR / f"meta_h1{suffix_label}.pkl")

            feat_names_h1 = X_df.columns.tolist()
            mean_imp_h1 = (imp_h1.get("xgb", np.zeros(len(feat_names_h1))) + imp_h1.get("lgb", np.zeros(len(feat_names_h1))) + imp_h1.get("cat", np.zeros(len(feat_names_h1)))) / 3.0
            keep_top_n_h1 = min(100, len(feat_names_h1))
            top_idx_h1 = np.argsort(-mean_imp_h1)[:keep_top_n_h1]
            stable_features_h1 = [feat_names_h1[i] for i in top_idx_h1]
            try:
                joblib.dump(stable_features_h1, MODELS_DIR / f"stable_features_h1{suffix_label}.pkl")
            except Exception:
                pass

            result_h1 = {
                "threshold": float(best_thr_h1),
                "threshold_acc": float(best_acc_h1),
                "use_meta": bool(use_meta_h1),
                "weights": chosen_weights_h1,
                "importances": imp_h1,
                "stable_features": stable_features_h1,
            }
        else:
            result_h1 = {"threshold": 0.5, "threshold_acc": 0.0, "use_meta": True, "weights": {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}}

        return {"game": result_game, "q1": result_q1, "h1": result_h1}

    def train_single_market(
        X_all: pd.DataFrame,
        y_all: pd.Series,
        market_key: str,
        label: str,
        low_thr: float = 0.47,
        high_thr: float = 0.53,
    ) -> dict | None:
        mask = y_all.notna()
        X_m = X_all.loc[mask].copy()
        y_m = y_all.loc[mask].astype(int).copy()
        n_classes, minority, ratio = _binary_class_stats(y_m)
        if len(X_m) < 300 or n_classes < 2:
            print(f"⚠️ Saltando mercado {label}: muestra insuficiente ({len(X_m)}) o target sin clases.")
            return None
        if minority < 25 or ratio < 0.03:
            print(
                f"⚠️ Saltando mercado {label}: clases muy desbalanceadas "
                f"(minority={minority}, ratio={ratio:.3f})."
            )
            return None

        print(f"\n📊 Validación temporal CV para {label}...")
        evaluate_ensemble_cv(X_m, y_m, label)

        train_end_m, calib_end_m = build_time_splits(len(X_m))
        X_oof = X_m.iloc[:calib_end_m]
        y_oof = y_m.iloc[:calib_end_m]
        X_test_m = X_m.iloc[calib_end_m:]
        y_test_m = y_m.iloc[calib_end_m:]

        print(f"\n⏳ Generando OOF para {label} (5 folds temporales)...")
        oof, final_base, _ = generate_oof_preds(X_oof, y_oof, n_splits=5)
        valid_idx = ~np.isnan(oof).any(axis=1)
        X_meta = oof[valid_idx]
        y_meta = y_oof.iloc[valid_idx]

        print(f"🧠 Entrenando Meta-Learner sobre OOF ({label})...")
        meta = train_meta_from_oof(X_meta, y_meta)
        print_holdout_metrics(final_base, meta, X_test_m, y_test_m, label)

        try:
            calib_cfg = load_calibration_config(MODELS_DIR / "calibration_params.json")
        except Exception:
            calib_cfg = None

        p_x = final_base[0].predict_proba(X_test_m)[:, 1]
        p_l = final_base[1].predict_proba(X_test_m)[:, 1]
        p_c = final_base[2].predict_proba(X_test_m)[:, 1]
        X_meta_test = np.column_stack([p_x, p_l, p_c])
        final_probs = meta.predict_proba(X_meta_test)[:, 1]

        if calib_cfg is not None:
            final_probs = np.array(
                [calibrate_probability(float(v), sport="wnba", market=market_key, calibration_config=calib_cfg) for v in final_probs]
            )

        best_thr, best_acc = find_best_threshold(final_probs, y_test_m, low=low_thr, high=high_thr, step=0.005)
        print(f"📌 Best threshold ({label}) on holdout: {best_thr:.3f} (acc={best_acc:.4f})")

        best_w, best_w_acc, _ = search_best_weights(
            p_x, p_l, p_c, y_test_m, calib_cfg, sport="wnba", market=market_key, step=0.1
        )
        meta_preds = (final_probs >= best_thr).astype(int)
        meta_acc = accuracy_score(y_test_m, meta_preds)
        use_meta = True
        chosen_weights = {"xgb": 1 / 3, "lgb": 1 / 3, "cat": 1 / 3}
        if best_w_acc > meta_acc:
            use_meta = False
            chosen_weights = {"xgb": best_w[0], "lgb": best_w[1], "cat": best_w[2]}

        joblib.dump(final_base[0], MODELS_DIR / f"xgb_{market_key}.pkl")
        joblib.dump(final_base[1], MODELS_DIR / f"lgb_{market_key}.pkl")
        joblib.dump(final_base[2], MODELS_DIR / f"cat_{market_key}.pkl")
        joblib.dump(meta, MODELS_DIR / f"meta_{market_key}.pkl")

        return {
            "threshold": float(best_thr),
            "threshold_acc": float(best_acc),
            "use_meta": bool(use_meta),
            "weights": chosen_weights,
            "min_edge": 0.07 if market_key == "spread" else 0.0,
            "alt_min_edge": 0.085 if market_key == "spread" else 0.0,
            "preferred_line_min": 3.5 if market_key == "spread" else 0.0,
            "preferred_line_max": 6.0 if market_key == "spread" else 0.0,
            "long_line_min": 10.5 if market_key == "spread" else 0.0,
        }


    # Entrenar global
    global_params = train_for_dataframe(X, y_game, y_q1, y_h1, suffix="")

    # Si existe columna de mercado, entrenar rutas separadas
    market_col = None
    for c in ["market_missing", "home_spread", "spread"]:
        if c in df.columns:
            market_col = c
            break

    splits = {"global": global_params}

    if market_col is not None:
        print(f"\n🔀 Column detectada para split de mercado: {market_col}. Entrenando rutas con/sin mercado...")
        if market_col == "market_missing":
            has_market_df = df[df[market_col].fillna(1) == 0]
            no_market_df = df[df[market_col].fillna(1) == 1]
        else:
            has_market_df = df[df[market_col].notna()]
            no_market_df = df[~df.index.isin(has_market_df.index)]

        if not has_market_df.empty and len(has_market_df) > 50:
            X_has = has_market_df.drop(columns=cols_to_drop).loc[:, selector]
            yg_has = has_market_df["TARGET_home_win"].astype(int)
            yq_has = has_market_df["TARGET_home_win_q1"].astype(int)
            yh_has = pd.to_numeric(has_market_df["TARGET_home_win_h1"], errors="coerce")
            splits["with_market"] = train_for_dataframe(X_has, yg_has, yq_has, yh_has, suffix="_with_market")

        if not no_market_df.empty and len(no_market_df) > 50:
            X_no = no_market_df.drop(columns=cols_to_drop).loc[:, selector]
            yg_no = no_market_df["TARGET_home_win"].astype(int)
            yq_no = no_market_df["TARGET_home_win_q1"].astype(int)
            yh_no = pd.to_numeric(no_market_df["TARGET_home_win_h1"], errors="coerce")
            splits["no_market"] = train_for_dataframe(X_no, yg_no, yq_no, yh_no, suffix="_no_market")

    # Guardar pick_params para consumo en runtime (mantener compatibilidad con estructura previa)
    final_pick_params = {
        "game": splits["global"]["game"],
        "q1": splits["global"]["q1"],
        "h1": splits["global"]["h1"],
        "splits": splits,
    }

    spread_params = train_single_market(X, y_spread, market_key="spread", label="HANDICAP/SPREAD", low_thr=0.47, high_thr=0.53)
    total_params = train_single_market(X, y_total, market_key="total", label="OVER/UNDER TOTAL", low_thr=0.47, high_thr=0.53)
    if spread_params is not None:
        final_pick_params["spread"] = spread_params
    if total_params is not None:
        final_pick_params["total"] = total_params
    joblib.dump(final_pick_params, MODELS_DIR / "pick_params.pkl")

    print("\n✅ ¡Todos los modelos (global y splits) fueron entrenados y guardados!")


if __name__ == "__main__":
    train_all_models()
