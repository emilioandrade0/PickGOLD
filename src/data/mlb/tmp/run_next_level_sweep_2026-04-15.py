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
splits_path = root / 'src/data/mlb/walkforward/full_game/walkforward_splits_summary.csv'
out_csv = root / 'src/data/mlb/tmp/full_game_next_level_sweep_2026-04-15.csv'
log_dir = root / 'src/data/mlb/tmp'
log_dir.mkdir(parents=True, exist_ok=True)

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

light_gate = {
    'NBA_MLB_META_GATE_ENABLED': '1',
    'NBA_MLB_META_GATE_MODEL_C': '1.0',
    'NBA_MLB_META_GATE_MIN_CALIB_ROWS': '160',
    'NBA_MLB_META_GATE_MIN_BASE_ROWS': '20',
    'NBA_MLB_META_GATE_THRESHOLD_MIN': '0.48',
    'NBA_MLB_META_GATE_THRESHOLD_MAX': '0.78',
    'NBA_MLB_META_GATE_THRESHOLD_STEP': '0.02',
    'NBA_MLB_META_GATE_MIN_KEEP_ROWS': '10',
    'NBA_MLB_META_GATE_COVERAGE_BONUS': '0.05',
    'NBA_MLB_META_GATE_RETENTION_TARGET': '0.45',
    'NBA_MLB_META_GATE_RETENTION_PENALTY': '0.10',
    'NBA_MLB_META_GATE_MIN_ACC_GAIN': '0.004',
    'NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION': '0.30',
}

balanced_gate = {
    'NBA_MLB_META_GATE_ENABLED': '1',
    'NBA_MLB_META_GATE_MODEL_C': '0.8',
    'NBA_MLB_META_GATE_MIN_CALIB_ROWS': '180',
    'NBA_MLB_META_GATE_MIN_BASE_ROWS': '25',
    'NBA_MLB_META_GATE_THRESHOLD_MIN': '0.50',
    'NBA_MLB_META_GATE_THRESHOLD_MAX': '0.82',
    'NBA_MLB_META_GATE_THRESHOLD_STEP': '0.02',
    'NBA_MLB_META_GATE_MIN_KEEP_ROWS': '12',
    'NBA_MLB_META_GATE_COVERAGE_BONUS': '0.04',
    'NBA_MLB_META_GATE_RETENTION_TARGET': '0.50',
    'NBA_MLB_META_GATE_RETENTION_PENALTY': '0.14',
    'NBA_MLB_META_GATE_MIN_ACC_GAIN': '0.006',
    'NBA_MLB_META_GATE_MIN_COVERAGE_RETENTION': '0.40',
}

scenarios = [
    ('baseline_ref', {}),
    (
        'no_gate_form_power_auto',
        {
            'NBA_MLB_FULL_GAME_CALIBRATOR_MODE': 'auto',
            'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_form_power,away_form_power',
        },
    ),
    ('gate_light', light_gate),
    (
        'gate_light_form_power',
        dict(light_gate, **{'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_form_power,away_form_power'}),
    ),
    (
        'gate_light_vsleague',
        dict(light_gate, **{'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_run_diff_L10_vs_league,away_run_diff_L10_vs_league'}),
    ),
    (
        'gate_light_home_contact',
        dict(light_gate, **{'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_baserunners_allowed_L10,home_hits_allowed_L10'}),
    ),
    (
        'gate_light_form_power_regime',
        dict(
            light_gate,
            **{
                'NBA_MLB_FULL_GAME_CALIBRATOR_MODE': 'regime_aware',
                'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_form_power,away_form_power',
            },
        ),
    ),
    (
        'gate_balanced_form_power_swap',
        dict(
            balanced_gate,
            **{
                'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_form_power,away_form_power',
                'NBA_MLB_FULL_GAME_DROP_FEATURES': 'diff_win_pct_L10_vs_league,diff_run_diff_L10_vs_league',
            },
        ),
    ),
    (
        'gate_balanced_vsleague_swap',
        dict(
            balanced_gate,
            **{
                'NBA_MLB_FULL_GAME_EXTRA_FEATURES': 'home_run_diff_L10_vs_league,away_run_diff_L10_vs_league',
                'NBA_MLB_FULL_GAME_DROP_FEATURES': 'diff_surface_win_pct_L5,diff_surface_run_diff_L5',
            },
        ),
    ),
]

clear_keys = set(base_env) | set(light_gate) | set(balanced_gate) | {
    'NBA_MLB_FULL_GAME_CALIBRATOR_MODE',
    'NBA_MLB_FULL_GAME_EXTRA_FEATURES',
    'NBA_MLB_FULL_GAME_DROP_FEATURES',
}

rows = []
for i, (name, overrides) in enumerate(scenarios, start=1):
    env = os.environ.copy()
    for k in clear_keys:
        env.pop(k, None)
    env.update(base_env)
    env.update(overrides)

    log_path = log_dir / f'full_game_next_level_{name}.log'
    print(f'[{i}/{len(scenarios)}] START {name}')
    with open(log_path, 'w', encoding='utf-8') as lf:
        proc = subprocess.run([py, wf], cwd=str(root), env=env, stdout=lf, stderr=subprocess.STDOUT)

    if proc.returncode != 0:
        rows.append(
            {
                'scenario': name,
                'accuracy': math.nan,
                'brier': math.nan,
                'logloss': math.nan,
                'coverage': math.nan,
                'published_accuracy': math.nan,
                'published_coverage': math.nan,
                'rows': -1,
                'splits': -1,
                'meta_gate_on_splits': -1,
                'calibrator_mode': 'run_failed',
                'extra_features': env.get('NBA_MLB_FULL_GAME_EXTRA_FEATURES', ''),
                'drop_features': env.get('NBA_MLB_FULL_GAME_DROP_FEATURES', ''),
                'log_file': str(log_path),
                'exit_code': proc.returncode,
            }
        )
        print(f'[{i}/{len(scenarios)}] FAIL {name} exit={proc.returncode}')
        continue

    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    fg = summary.get('full_game', {})

    meta_on = 0
    split_rows = 0
    cal_modes = set()
    if splits_path.exists():
        with open(splits_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                split_rows += 1
                try:
                    if int(float(r.get('meta_gate_enabled', 0) or 0)) == 1:
                        meta_on += 1
                except Exception:
                    pass
                cm = (r.get('calibrator_mode') or '').strip()
                if cm:
                    cal_modes.add(cm)

    row = {
        'scenario': name,
        'accuracy': float(fg.get('accuracy', float('nan'))),
        'brier': float(fg.get('brier', float('nan'))),
        'logloss': float(fg.get('logloss', float('nan'))),
        'coverage': float(fg.get('coverage', float('nan'))),
        'published_accuracy': float(fg.get('published_accuracy', float('nan'))),
        'published_coverage': float(fg.get('published_coverage', float('nan'))),
        'rows': int(fg.get('rows', -1)),
        'splits': split_rows,
        'meta_gate_on_splits': meta_on,
        'calibrator_mode': '|'.join(sorted(cal_modes)) if cal_modes else env.get('NBA_MLB_FULL_GAME_CALIBRATOR_MODE', ''),
        'extra_features': env.get('NBA_MLB_FULL_GAME_EXTRA_FEATURES', ''),
        'drop_features': env.get('NBA_MLB_FULL_GAME_DROP_FEATURES', ''),
        'log_file': str(log_path),
        'exit_code': 0,
    }
    rows.append(row)
    print(
        f"[{i}/{len(scenarios)}] DONE {name} "
        f"acc={row['accuracy']:.12f} brier={row['brier']:.12f} "
        f"pub_acc={row['published_accuracy']:.6f} pub_cov={row['published_coverage']:.6f} meta_on={meta_on}"
    )

rows_sorted = sorted(
    rows,
    key=lambda r: (
        -1e9 if math.isnan(r['accuracy']) else -r['accuracy'],
        1e9 if math.isnan(r['brier']) else r['brier'],
        -1e9 if math.isnan(r['published_accuracy']) else -r['published_accuracy'],
    ),
)

fields = [
    'scenario',
    'accuracy',
    'brier',
    'logloss',
    'coverage',
    'published_accuracy',
    'published_coverage',
    'rows',
    'splits',
    'meta_gate_on_splits',
    'calibrator_mode',
    'extra_features',
    'drop_features',
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
        f"{r['scenario']}: acc={r['accuracy']:.12f} brier={r['brier']:.12f} "
        f"pub_acc={r['published_accuracy']:.6f} pub_cov={r['published_coverage']:.6f} "
        f"meta_on={r['meta_gate_on_splits']} cal={r['calibrator_mode']}"
    )
