import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('data/all_leagues_2425.csv')

FEATURES = {
    'GK':  ['SavePct', 'GA90', 'CS', 'CSPct', 'PSxGDiff', 'Saves', 'DefPen'],
    'CB':  ['Tkl', 'Int', 'Clr', 'AerialWonPct', 'CmpPct', 'PrgP', 'Blocks', 'DefThird', 'DefPen'],
    'FB':  ['PrgC', 'PrgP', 'PrgR', 'Crs', 'xAG', 'Ast', 'Tkl', 'Int', 'AttThird', 'DefThird'],
    'CDM': ['Tkl', 'TklW', 'Int', 'CmpPct', 'PrgP', 'AerialWonPct', 'Blocks', 'DefThird', 'MidThird'],
    'CM':  ['KP', 'SCA', 'PrgC', 'PrgP', 'CmpPct', 'Gls', 'Ast', 'xAG', 'MidThird', 'AttThird'],
    'CAM': ['Gls', 'Ast', 'xAG', 'xG', 'KP', 'PPA', 'SCA', 'GCA', 'PrgC', 'AttThird', 'AttPen'],
    'FW':  ['Gls', 'Ast', 'xG', 'xAG', 'npxG', 'SoTPct', 'DribSuccPct', 'PrgC', 'KP', 'AttPen', 'AttThird']
}

N_CLUSTERS = {
    'GK': 2, 'CB': 5, 'FB': 3,
    'CDM': 2, 'CM': 3, 'CAM': 3, 'FW': 8
}

def assign_clusters(df):
    df = df.copy()
    df['Cluster'] = -1
    df['ClusterLabel'] = ''

    for subpos, features in FEATURES.items():
        pos_df = df[df['SubPos'] == subpos].copy()
        if len(pos_df) < N_CLUSTERS[subpos]:
            continue

        scaler = StandardScaler()
        scaled = scaler.fit_transform(pos_df[features])

        k = N_CLUSTERS[subpos]
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(scaled)

        df.loc[pos_df.index, 'Cluster'] = clusters
        df.loc[pos_df.index, 'ClusterLabel'] = subpos + '_' + clusters.astype(str)

    return df

df = assign_clusters(df)

# Print cluster compositions to understand what each cluster represents
for subpos in FEATURES.keys():
    print(f'\n=== {subpos} Clusters ===')
    pos_df = df[df['SubPos'] == subpos]
    features = FEATURES[subpos]
    for cluster_id in sorted(pos_df['Cluster'].unique()):
        cluster_players = pos_df[pos_df['Cluster'] == cluster_id]
        means = cluster_players[features].mean()
        print(f'\n  Cluster {cluster_id} ({len(cluster_players)} players):')
        for f in features:
            print(f'    {f}: {means[f]:.2f}')
        print(f'  Top players: {", ".join(cluster_players.nlargest(3, features[0])["Player"].tolist())}')

# Save with cluster assignments
df.to_csv('data/all_leagues_2425.csv', index=False)
print('\nClusters saved to dataset.')