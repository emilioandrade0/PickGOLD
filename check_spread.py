import pandas as pd

df = pd.read_csv('src/data/nhl/processed/model_ready_features_nhl.csv')
print('TARGET_over_55 distribution:')
print(df['TARGET_over_55'].value_counts())
print(f'\nOver rate: {df["TARGET_over_55"].sum() / len(df) * 100:.1f}%')
