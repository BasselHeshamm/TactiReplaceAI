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
    return s.lower().strip()

def find_replacements(player_name, top_n=5):
    player_row = df[df['Player'].apply(normalize) == normalize(player_name)]
    if player_row.empty:
        print(f"Player '{player_name}' not found.")
        return

    subpos = player_row['SubPos'].values[0]
    cluster = player_row['Cluster'].values[0]
    features = FEATURES.get(subpos, FEATURES['CM'])

    print(f"\nPlayer: {player_name}")
    print(f"SubPos: {subpos} | Cluster: {cluster} | Club: {player_row['Squad'].values[0]} | League: {player_row['League'].values[0]}")
    print(f"Features: {features}")

    # Filter to same subpos AND same cluster
    cluster_df = df[(df['SubPos'] == subpos) & (df['Cluster'] == cluster)].copy().reset_index(drop=True)
    print(f"Players in same cluster: {len(cluster_df)}")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df[features])

    player_idx = cluster_df[cluster_df['Player'].apply(normalize) == normalize(player_name)].index[0]
    player_vec = scaled[player_idx]

    scores = cosine_similarity([player_vec], scaled)[0]
    cluster_df['similarity'] = scores

    results = cluster_df[cluster_df['Player'].apply(normalize) != normalize(player_name)]
    results = results.sort_values('similarity', ascending=False).head(top_n)

    print(f"\nTop {top_n} replacements:\n")
    print(results[['Player', 'Squad', 'League', 'Age', 'similarity'] + features].to_string(index=False))
    return results

# Test cases
find_replacements('Trent Alexander-Arnold')
find_replacements('Virgil van Dijk')
find_replacements('Mohamed Salah')
find_replacements('Rodri')
find_replacements('Cole Palmer')