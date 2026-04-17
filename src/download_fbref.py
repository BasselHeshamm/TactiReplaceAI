import pandas as pd
df = pd.read_csv('data/fbref_PL_2024-25.csv')
print(df.shape)
print(df.columns.tolist())
print(df[df['Player'] == 'Trent Alexander-Arnold'])