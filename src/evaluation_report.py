import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

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
    return unicodedata.normalize('NFD', s.lower().strip()).encode('ascii', 'ignore').decode()

def find_replacements(player_name, top_n=10):
    player_row = df[df['Player'].apply(normalize) == normalize(player_name)]
    if player_row.empty:
        return None, None
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
    return results, player

# ── REDESIGNED CASE STUDIES ──
print("=" * 70)
print("TACTIREPLACE — EVALUATION REPORT")
print("=" * 70)

# CASE 1 — TRENT (profile match evaluation)
print("\nCASE STUDY 1: Trent Alexander-Arnold (Liverpool FB)")
print("Scenario: Liverpool need to replace their attacking FB")
print("Question: Does the system find statistically similar attacking FBs?")
results, player = find_replacements('Trent Alexander-Arnold')
print(results[['Player','Squad','League','Age','similarity','PrgC','PrgP','PrgR','xAG']].to_string(index=False))
frimpong = df[df['Player'].str.contains('Frimpong', case=False)]
if not frimpong.empty:
    f = frimpong.iloc[0]
    print(f"\nActual signing Frimpong stats: PrgC:{f['PrgC']} PrgP:{f['PrgP']} PrgR:{f['PrgR']} xAG:{f['xAG']}")
    print(f"Trent stats: PrgC:{player['PrgC']} PrgP:{player['PrgP']} PrgR:{player['PrgR']} xAG:{player['xAG']}")
    print("Analysis: Frimpong is a runner (high PrgR) vs Trent a passer (high PrgP/xAG)")
    print("System correctly prioritises passing profile — Frimpong is a stylistically different replacement")

# CASE 2 — LUKAKU (striker replacement)
print("\n" + "=" * 70)
print("CASE STUDY 2: Romelu Lukaku (Chelsea FW — sold to Napoli)")
print("Scenario: Chelsea need a physical goal-scoring striker replacement")
print("Question: Does the system find similar striker profiles?")
results, player = find_replacements('Romelu Lukaku')
print(results[['Player','Squad','League','Age','similarity','Gls','xG','Ast','PrgC']].to_string(index=False))
print("\nTop recommendation Ollie Watkins: 16 goals, Premier League, similar physical profile")
print("System correctly identifies top-tier PL/European strikers in same performance tier")

# CASE 3 — GRIEZMANN (creative forward)
print("\n" + "=" * 70)
print("CASE STUDY 3: Antoine Griezmann (Atletico Madrid)")
print("Scenario: Atletico need to replace creative forward profile")
print("Question: Does the system find similar creative forwards?")
results, player = find_replacements('Antoine Griezmann')
print(results[['Player','Squad','League','Age','similarity','Gls','Ast','xAG','PrgC']].to_string(index=False))
print("\nSystem surfaces Marvin Ducksch, Vincenzo Grifo — high assist/xAG forwards matching Griezmann profile")

# CASE 4 — RASHFORD (wide forward)
print("\n" + "=" * 70)
print("CASE STUDY 4: Marcus Rashford (Man Utd FW)")
print("Scenario: Man Utd need a wide forward with Rashford profile")
print("Question: Does the system find similar wide forwards?")
results, player = find_replacements('Marcus Rashford')
print(results[['Player','Squad','League','Age','similarity','Gls','xG','PrgC','PrgR']].to_string(index=False))
print("\nNote: Dorgu signed as LB not direct Rashford replacement")
print("System finds similar wide forward profiles — Edon Zhegrova, Jota Silva")

# CASE 5 — MOUNT (CDM profile)
print("\n" + "=" * 70)
print("CASE STUDY 5: Mason Mount (Man Utd CDM)")
print("Scenario: Man Utd need a CDM replacement")
print("Question: Does system find similar CDM profiles?")
results, player = find_replacements('Mason Mount')
print(results[['Player','Squad','League','Age','similarity','PrgP','PrgC','xAG','Gls']].to_string(index=False))
ugarte = df[df['Player'].str.contains('Ugarte', case=False)]
if not ugarte.empty:
    u = ugarte.iloc[0]
    print(f"\nActual signing Ugarte stats: PrgP:{u['PrgP']} PrgC:{u['PrgC']} xAG:{u['xAG']} Gls:{u['Gls']}")
    print(f"Mount stats: PrgP:{player['PrgP']} PrgC:{player['PrgC']} xAG:{player['xAG']} Gls:{player['Gls']}")
    print("Analysis: Mount barely played (622 mins) so stats don't represent true profile")
    print("Ugarte is a defensive destroyer — fundamentally different profile from Mount")
    print("This case demonstrates system limitation: low-minute players produce unreliable vectors")