import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/all_leagues_2425.csv')

FEATURES = {
    'GK':  ['MP', 'Starts', 'Min'],
    'CB':  ['PrgC', 'PrgP', 'xAG', 'Ast', 'CrdY', 'Gls'],
    'FB':  ['PrgC', 'PrgP', 'PrgR', 'xAG', 'Ast', 'CrdY'],
    'CDM': ['PrgP', 'PrgC', 'CrdY', 'Ast', 'xAG', 'Gls'],
    'CM':  ['Gls', 'Ast', 'xAG', 'PrgC', 'PrgP', 'PrgR'],
    'CAM': ['Gls', 'Ast', 'xAG', 'xG', 'npxG', 'PrgC'],
    'FW':  ['Gls', 'Ast', 'xG', 'xAG', 'npxG', 'PrgC', 'PrgR']
}

def normalize(s):
    import unicodedata
    return unicodedata.normalize('NFD', s.lower().strip()).encode('ascii', 'ignore').decode()

def find_replacements(player_name, top_n=10):
    player_row = df[df['Player'].apply(normalize) == normalize(player_name)]
    if player_row.empty:
        print(f"NOT FOUND: {player_name}")
        return None

    player = player_row.iloc[0]
    subpos = player['SubPos']
    cluster = player['Cluster']
    features = FEATURES.get(subpos, FEATURES['CM'])

    cluster_df = df[(df['SubPos'] == subpos) & (df['Cluster'] == cluster)].copy().reset_index(drop=True)
    if len(cluster_df) < 15:
        cluster_df = df[df['SubPos'] == subpos].copy().reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[features])
    idx = cluster_df[cluster_df['Player'].apply(normalize) == normalize(player_name)].index[0]
    scores = cosine_similarity([scaled[idx]], scaled)[0]
    cluster_df['similarity'] = scores

    results = cluster_df[cluster_df['Player'].apply(normalize) != normalize(player_name)]
    results = results.sort_values('similarity', ascending=False).head(top_n)
    return results[['Player', 'Squad', 'League', 'Age', 'SubPos', 'similarity'] + features]

# ── CASE STUDIES ──
case_studies = [
    {
        'name': 'Trent Alexander-Arnold',
        'context': 'Liverpool FB — announced departure to Real Madrid summer 2025',
        'actual_replacement': 'Jeremie Frimpong (Bayer Leverkusen)',
        'check_for': 'Frimpong'
    },
    {
        'name': 'Marcus Rashford',
        'context': 'Man Utd FW — fell out of favor, loaned to Aston Villa Jan 2025',
        'actual_replacement': 'Patrick Dorgu (Lecce to Man Utd)',
        'check_for': 'Dorgu'
    },
    {
        'name': 'Romelu Lukaku',
        'context': 'Chelsea FW — sold permanently to Napoli summer 2024',
        'actual_replacement': 'Nicolas Jackson (already at Chelsea)',
        'check_for': 'Jackson'
    },
    {
        'name': 'Antoine Griezmann',
        'context': 'Atletico Madrid FW — future uncertain, key creative forward',
        'actual_replacement': 'No direct replacement signed',
        'check_for': None
    },
    {
        'name': 'Mason Mount',
        'context': 'Man Utd CDM — struggled at United, low minutes',
        'actual_replacement': 'Manuel Ugarte (PSG to Man Utd)',
        'check_for': 'Ugarte'
    },
]

print("=" * 70)
print("TACTIREPLACE EVALUATION — CASE STUDIES")
print("=" * 70)

for cs in case_studies:
    print(f"\n{'=' * 70}")
    print(f"CASE STUDY: {cs['name']}")
    print(f"Context: {cs['context']}")
    print(f"Actual replacement: {cs['actual_replacement']}")
    print(f"{'=' * 70}")

    results = find_replacements(cs['name'], top_n=10)
    if results is None:
        continue

    print(results.to_string(index=False))

    if cs['check_for']:
        found = results[results['Player'].str.contains(cs['check_for'], case=False, na=False)]
        if not found.empty:
            rank = results.index.get_loc(found.index[0]) + 1
            print(f"\n✅ ACTUAL SIGNING '{cs['check_for']}' FOUND at rank #{rank}")
        else:
            print(f"\n❌ Actual signing '{cs['check_for']}' not in top 10")
            not_found = df[df['Player'].str.contains(cs['check_for'], case=False, na=False)]
            if not_found.empty:
                print(f"   (Player not in dataset at all)")
            else:
                print(f"   (Player exists in dataset but ranked outside top 10)")