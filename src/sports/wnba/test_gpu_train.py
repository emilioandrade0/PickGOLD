import time
from pathlib import Path
import pandas as pd
from train_models import PROCESSED_DATA, fit_base_models, USE_GPU

print('USE_GPU=', USE_GPU)
if not Path(PROCESSED_DATA).exists():
    print('Processed CSV not found:', PROCESSED_DATA)
    raise SystemExit(1)

df = pd.read_csv(PROCESSED_DATA).sort_values('date').reset_index(drop=True)
cols_to_drop = ["game_id", "date", "season", "home_team", "away_team", "TARGET_home_win", "TARGET_home_win_q1", "home_pts_total", "away_pts_total", "home_q1", "away_q1"]
existing_drop = [c for c in cols_to_drop if c in df.columns]
X = df.drop(columns=existing_drop)
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
X = X.loc[:, num_cols]
if len(X) < 200:
    n = len(X)
else:
    n = 200
X_sample = X.iloc[:n]
y_sample = df['TARGET_home_win'].astype(int).iloc[:n]

print(f'Training small sample (n={n}) — GPU requested: {USE_GPU}')
start = time.time()
models = fit_base_models(X_sample, y_sample, use_gpu=USE_GPU)
elapsed = time.time() - start
print('Elapsed sec:', round(elapsed,2))
for m in models:
    print(type(m), getattr(m, '__class__', None))
