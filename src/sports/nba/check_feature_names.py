import joblib
from pathlib import Path
p = Path('../../../models/feature_names.pkl')
if not p.exists():
    print('feature_names.pkl not found at', p)
else:
    names = joblib.load(p)
    want = ['intensity_score','travel_penalty','playoff_multiplier','timezone_difference']
    print('Found feature_names.pkl with', len(names), 'features')
    print('Presence:', [(w, w in names) for w in want])
    for w in want:
        print(w, 'index', names.index(w) if w in names else 'MISSING')
