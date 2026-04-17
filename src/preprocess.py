import pandas as pd
import numpy as np

# Load the rich dataset
df = pd.read_csv('data/players_data-2024_2025.csv')
print(f'Raw dataset: {df.shape}')

# Rename Comp to League
df = df.rename(columns={'Comp': 'League'})

# Clean league names
df['League'] = df['League'].str.replace('eng Premier League', 'Premier League', regex=False)
df['League'] = df['League'].str.replace('es La Liga', 'La Liga', regex=False)
df['League'] = df['League'].str.replace('it Serie A', 'Serie A', regex=False)
df['League'] = df['League'].str.replace('de Bundesliga', 'Bundesliga', regex=False)
df['League'] = df['League'].str.replace('fr Ligue 1', 'Ligue 1', regex=False)

# Keep only the columns we need
keep_cols = [
    'Player', 'Nation', 'Pos', 'Squad', 'League', 'Age', 'Born',
    'MP', 'Starts', 'Min', '90s',
    # Attacking
    'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR',
    # Shooting
    'Sh', 'SoT', 'SoT%', 'G/Sh',
    # Passing
    'Cmp%', 'KP', 'PPA', 'CrsPA', 'Crs',
    # Defensive
    'Tkl', 'TklW', 'Int', 'Clr', 'Blocks_stats_defense', 'Err',
    # Possession
    'Touches', 'Succ', 'Succ%', 'Mis', 'Dis',
    # Zone touches
    'Def 3rd', 'Mid 3rd', 'Att 3rd', 'Att Pen', 'Def Pen',
    # Aerial
    'Won', 'Won%',
    # Creativity
    'SCA', 'GCA',
    # Discipline
    'CrdY', 'CrdR',
    # GK
    'GA90', 'Saves', 'Save%', 'CS', 'CS%', 'PSxG', 'PSxG+/-',
]

# Only keep columns that exist
keep_cols = [c for c in keep_cols if c in df.columns]
df = df[keep_cols].copy()

# Rename messy column names
df = df.rename(columns={
    'Blocks_stats_defense': 'Blocks',
    'Save%': 'SavePct',
    'CS%': 'CSPct',
    'SoT%': 'SoTPct',
    'Won%': 'AerialWonPct',
    'Succ%': 'DribSuccPct',
    'PSxG+/-': 'PSxGDiff',
    'Cmp%': 'CmpPct',
    'Def 3rd': 'DefThird',
    'Mid 3rd': 'MidThird',
    'Att 3rd': 'AttThird',
    'Att Pen': 'AttPen',
    'Def Pen': 'DefPen',
})

# Convert all numeric columns
skip_cols = ['Player', 'Nation', 'Pos', 'Squad', 'League']
for col in df.columns:
    if col not in skip_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop players with less than 5 appearances
df = df[df['MP'] >= 5]

# Fill NaN with 0
df = df.fillna(0)

# Keep primary position only
df['Pos'] = df['Pos'].str.split(',').str[0]

# Keep valid positions
df = df[df['Pos'].isin(['GK', 'DF', 'MF', 'FW'])]

# ── SUBPOSITION CLASSIFICATION ──
def classify_subpos(row):
    pos = row['Pos']
    if pos == 'GK': return 'GK'
    if pos == 'FW': return 'FW'
    if pos == 'DF':
        if row['PrgC'] >= 30 or row['PrgR'] >= 55: return 'FB'
        return 'CB'
    if pos == 'MF':
        if row['xAG'] >= 2.4 or row['Gls'] >= 3: return 'CAM'
        if row['xAG'] < 1.2 and row['CrdY'] >= 3: return 'CDM'
        return 'CM'
    return pos

df['SubPos'] = df.apply(classify_subpos, axis=1)

print('\nSubPosition breakdown:')
print(df['SubPos'].value_counts())

# ── DEDUPLICATE LOAN PLAYERS ──
before = len(df)
df = df.sort_values('Min', ascending=False)
df = df.drop_duplicates(subset='Player', keep='first')
df = df.reset_index(drop=True)
print(f'Removed {before - len(df)} duplicate loan entries')

# ── MERGE MARKET VALUES ──
import unicodedata
def normalize_name(s):
    if pd.isna(s): return ''
    return unicodedata.normalize('NFD', str(s).lower().strip()).encode('ascii', 'ignore').decode()

tm_players = pd.read_csv('data/players.csv', usecols=['name', 'market_value_in_eur'])
tm_players = tm_players.rename(columns={'name': 'Player', 'market_value_in_eur': 'MarketValue'})
tm_players = tm_players.dropna(subset=['MarketValue'])
tm_players['MarketValue'] = tm_players['MarketValue'].astype(int)
tm_players['PlayerNorm'] = tm_players['Player'].apply(normalize_name)
tm_players = tm_players.sort_values('MarketValue', ascending=False).drop_duplicates(subset='PlayerNorm', keep='first')
df['PlayerNorm'] = df['Player'].apply(normalize_name)
df = df.merge(tm_players[['PlayerNorm', 'MarketValue']], on='PlayerNorm', how='left')
df = df.drop(columns=['PlayerNorm'])


df['MarketValue'] = df['MarketValue'].fillna(0).astype(int)
matched = (df['MarketValue'] > 0).sum()
print(f'Market values matched: {matched}/{len(df)} players')
matched = (df['MarketValue'] > 0).sum()
print(f'Market values matched: {matched}/{len(df)} players')

df = df.sort_values('MarketValue', ascending=False)
df = df.drop_duplicates(subset='Player', keep='first')
df = df.reset_index(drop=True)
print(f'After final dedup: {len(df)} players')

# Verify known high-value players have correct values
for name in ['Vinicius Júnior', 'Julián Álvarez', 'Álex Baena', 'Kylian Mbappé']:
    row = df[df['Player'] == name]
    if not row.empty:
        print(f'{name}: €{int(row.iloc[0]["MarketValue"])//1000000}M')
    else:
        print(f'{name}: NOT FOUND')

# Save
df.to_csv('data/all_leagues_2425.csv', index=False)
print(f'\nFinal dataset: {df.shape}')
print(f'Leagues: {df["League"].value_counts().to_dict()}')
print(f'Columns: {df.columns.tolist()}')