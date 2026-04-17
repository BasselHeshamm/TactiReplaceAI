from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

leagues = {
    'Premier League': 'data/standard.html',
    'La Liga': 'data/laliga.html',
    'Serie A': 'data/seriea.html',
    'Bundesliga': 'data/bundesliga.html',
    'Ligue 1': 'data/ligue1.html',
}

dfs = []

for league, path in leagues.items():
    print(f'Extracting {league}...')
    with open(path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    table = soup.find('table', {'id': 'stats_standard'})
    df = pd.read_html(StringIO(str(table)))[0]

    # Flatten multi-level columns
    df.columns = [col[1] if isinstance(col, tuple) else col for col in df.columns]

    # Remove duplicate columns by keeping only first occurrence
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    df['League'] = league
    print(f'  {len(df)} players, columns: {df.columns.tolist()}')
    dfs.append(df)

# Combine
combined = pd.concat(dfs, ignore_index=True)

# Drop duplicate headers
combined = combined[combined['Player'] != 'Player']

# Keep only columns we need
keep_cols = ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'MP', 
             'Starts', 'Min', '90s', 'Gls', 'Ast', 'CrdY', 'CrdR',
             'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR', 'League']

# Only keep columns that exist
keep_cols = [c for c in keep_cols if c in combined.columns]
combined = combined[keep_cols]

print(f'\nColumns kept: {keep_cols}')

# Convert numeric
numeric_cols = ['MP','Starts','Min','90s','Gls','Ast','CrdY','CrdR',
                'xG','npxG','xAG','PrgC','PrgP','PrgR']
for col in numeric_cols:
    if col in combined.columns:
        combined[col] = pd.to_numeric(combined[col], errors='coerce')

# Filter min 5 appearances
combined = combined[combined['MP'] >= 5]
combined = combined.fillna(0)
combined['Pos'] = combined['Pos'].str.split(',').str[0]
combined = combined[combined['Pos'].isin(['GK','DF','MF','FW'])]

# Deduplicate loans
combined = combined.sort_values('Min', ascending=False)
combined = combined.drop_duplicates(subset='Player', keep='first')
combined = combined.reset_index(drop=True)

combined.to_csv('data/all_leagues_2425.csv', index=False)
print(f'\nFinal dataset: {combined.shape}')
print(combined['League'].value_counts())