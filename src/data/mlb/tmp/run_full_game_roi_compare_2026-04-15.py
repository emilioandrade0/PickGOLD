import os
import json
import csv
import subprocess
import math
from pathlib import Path

root = Path('.').resolve()
py = str((root / '.venv/Scripts/python.exe').resolve())
wf = str((root / 'src/sports/mlb/historical_predictions_mlb_walkforward.py').resolve())
summary_path = root / 'src/data/mlb/walkforward/walkforward_summary_mlb.json'
out_csv = root / 'src/data/mlb/tmp/full_game_roi_objective_compare_2026-04-15.csv'
log_dir = root / 'src/data/mlb/tmp'

base_env = {
    'NBA_MLB_MARKETS': 'full_game',
    'NBA_MLB_INPUT_FILE': str((root / 'src/data/mlb/processed/snapshots/model_ready_features_mlb_HEAD.csv').resolve()),
    'NBA_MLB_MAX_TRAIN_DATE': '',
    'NBA_MLB_FULL_GAME_XGB_WEIGHT_GRID': '0.00,0.20,0.35,0.50,0.65,0.80,1.00',
    'NBA_MLB_FULL_GAME_BRIER_WEIGHT': '0.08',
    'NBA_MLB_FULL_GAME_PROB_SHIFT_ENABLED': '1',
    'NBA_MLB_FULL_GAME_PROB_SHIFT_MIN': '-0.02',
    'NBA_MLB_FULL_GAME_PROB_SHIFT_MAX': '0.02',
    'NBA_MLB_FULL_GAME_PROB_SHIFT_STEP': '0.01',
    'NBA_MLB_FULL_GAME_CALIBRATOR_MODE': 'global_lr',
    'NBA_MLB_FULL_GAME_THR_MIN': '0.54',
    'NBA_MLB_FULL_GAME_THR_MAX': '0.66',
    'NBA_MLB_FULL_GAME_THR_STEP': '0.01',
    'NBA_MLB_FULL_GAME_EXTRA_FEATURES': '',
    'NBA_MLB_FULL_GAME_DROP_FEATURES': '',
    'NBA_MLB_META_GATE_ENABLED': '0',
}

scenarios = [
    ('accuracy_cov_ref', {
        'NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE': 'accuracy_cov',
    }),
    ('roi_no_edge', {
        'NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE': 'roi',
        'NBA_MLB_FULL_GAME_ROI_MIN_EDGE': '0.00',
        'NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY': '0.45',
        'NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS': '8',
    }),
    ('roi_edge_0015', {
        'NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE': 'roi',
        'NBA_MLB_FULL_GAME_ROI_MIN_EDGE': '0.015',
        'NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY': '0.45',
        'NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS': '8',
    }),
]

clear_keys = set(base_env) | {
    'NBA_MLB_FULL_GAME_THRESHOLD_OBJECTIVE',
    'NBA_MLB_FULL_GAME_ROI_MIN_EDGE',
    'NBA_MLB_FULL_GAME_ROI_MIN_ACCURACY',
    'NBA_MLB_FULL_GAME_ROI_MIN_PRICED_ROWS',
    'NBA_MLB_FULL_GAME_ROI_SCORE_ROI_WEIGHT',
    'NBA_MLB_FULL_GAME_ROI_SCORE_ACC_WEIGHT',
    'NBA_MLB_FULL_GAME_ROI_SCORE_COV_WEIGHT',
}

rows = []
for i, (name, overrides) in enumerate(scenarios, start=1):
    env = os.environ.copy()
    for k in clear_keys:
        env.pop(k, None)
    env.update(base_env)
    env.update(overrides)

    log_path = log_dir / f'full_game_roi_compare_{name}.log'
    print(f'[{i}/{len(scenarios)}] START {name}')
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run([py, wf], cwd=str(root), env=env, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        rows.append({
            'scenario': name,
            'accuracy': math.nan,
            'brier': math.nan,
            'logloss': math.nan,
            'published_accuracy': math.nan,
            'published_coverage': math.nan,
            'published_roi_per_bet': math.nan,
            'published_total_return_units': math.nan,
            'published_priced_picks': -1,
            'published_priced_coverage': math.nan,
            'published_mean_ev_edge': math.nan,
            'threshold_objective_mode': 'run_failed',
            'avg_calib_roi_per_bet': math.nan,
            'avg_calib_priced_coverage': math.nan,
            'avg_calib_mean_ev_edge': math.nan,
            'log_file': str(log_path),
            'exit_code': proc.returncode,
        })
        print(f'[{i}/{len(scenarios)}] FAIL {name} exit={proc.returncode}')
        continue

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fg = summary.get('full_game', {})
    row = {
        'scenario': name,
        'accuracy': float(fg.get('accuracy', float('nan'))),
        'brier': float(fg.get('brier', float('nan'))),
        'logloss': float(fg.get('logloss', float('nan'))),
        'published_accuracy': float(fg.get('published_accuracy', float('nan'))),
        'published_coverage': float(fg.get('published_coverage', float('nan'))),
        'published_roi_per_bet': float(fg.get('published_roi_per_bet', 0.0)),
        'published_total_return_units': float(fg.get('published_total_return_units', 0.0)),
        'published_priced_picks': int(fg.get('published_priced_picks', 0)),
        'published_priced_coverage': float(fg.get('published_priced_coverage', 0.0)),
        'published_mean_ev_edge': float(fg.get('published_mean_ev_edge', 0.0)),
        'threshold_objective_mode': str(fg.get('threshold_objective_mode', 'accuracy_cov')),
        'avg_calib_roi_per_bet': float(fg.get('avg_calib_roi_per_bet', 0.0)),
        'avg_calib_priced_coverage': float(fg.get('avg_calib_priced_coverage', 0.0)),
        'avg_calib_mean_ev_edge': float(fg.get('avg_calib_mean_ev_edge', 0.0)),
        'log_file': str(log_path),
        'exit_code': 0,
    }
    rows.append(row)
    print(
        f"[{i}/{len(scenarios)}] DONE {name} "
        f"acc={row['accuracy']:.12f} pub_acc={row['published_accuracy']:.6f} pub_cov={row['published_coverage']:.6f} "
        f"roi={row['published_roi_per_bet']:+.5f} priced={row['published_priced_picks']}"
    )

rows_sorted = sorted(
    rows,
    key=lambda r: (
        -1e9 if math.isnan(r['published_roi_per_bet']) else -r['published_roi_per_bet'],
        -1e9 if math.isnan(r['accuracy']) else -r['accuracy'],
        -1e9 if math.isnan(r['published_accuracy']) else -r['published_accuracy'],
    ),
)

fields = [
    'scenario',
    'accuracy',
    'brier',
    'logloss',
    'published_accuracy',
    'published_coverage',
    'published_roi_per_bet',
    'published_total_return_units',
    'published_priced_picks',
    'published_priced_coverage',
    'published_mean_ev_edge',
    'threshold_objective_mode',
    'avg_calib_roi_per_bet',
    'avg_calib_priced_coverage',
    'avg_calib_mean_ev_edge',
    'log_file',
    'exit_code',
]

with open(out_csv, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows_sorted)

print('SAVED', out_csv)
for r in rows_sorted:
    print(
        f"{r['scenario']}: acc={r['accuracy']:.12f} pub_acc={r['published_accuracy']:.6f} pub_cov={r['published_coverage']:.6f} "
        f"roi={r['published_roi_per_bet']:+.5f} priced={r['published_priced_picks']} thr_mode={r['threshold_objective_mode']}"
    )
