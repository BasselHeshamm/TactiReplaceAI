from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel
from typing import Optional, Dict

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from trajectory import get_trajectory
    TRAJECTORY_ENABLED = True
except ImportError:
    TRAJECTORY_ENABLED = False
app = FastAPI(title="TactiReplace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Stats that are already rates/percentages — do NOT divide by 90s
RATE_STATS = {'SavePct', 'CSPct', 'AerialWonPct', 'CmpPct', 'SoTPct', 'DribSuccPct', 'PSxGDiff', 'GA90'}

# All stats to return for display
ALL_COLS = [
    'Gls', 'Ast', 'xG', 'npxG', 'xAG', 'PrgC', 'PrgP', 'PrgR',
    'Sh', 'SoT', 'SoTPct', 'KP', 'PPA', 'Crs',
    'Tkl', 'TklW', 'Int', 'Clr', 'Blocks',
    'Won', 'AerialWonPct', 'CmpPct',
    'SCA', 'GCA', 'DribSuccPct',
    'CrdY', 'CrdR', 'MP', 'Starts', 'Min',
    'Saves', 'SavePct', 'GA90', 'CS', 'CSPct', 'PSxGDiff',
    'DefThird', 'MidThird', 'AttThird', 'AttPen', 'DefPen'
]

def normalize(s):
    import unicodedata
    return unicodedata.normalize('NFD', s.lower().strip()).encode('ascii', 'ignore').decode()

def make_stats(row, cols):
    return {f: float(row[f]) for f in cols if f in row.index}

def make_stats_p90(row, cols):
    try:
        p90 = float(row['90s'])
        if p90 <= 0: p90 = 1
    except:
        p90 = 1
    CONTEXT_STATS = {'MP', 'Min', 'Starts'}
    result = {}
    for f in cols:
        if f not in row.index:
            continue
        val = float(row[f])
        if f in RATE_STATS or f in CONTEXT_STATS:
            result[f] = round(val, 2)
        else:
            result[f] = round(val / p90, 2)
    return result

@app.get("/")
def root():
    return {"message": "TactiReplace API running", "players": len(df)}

@app.get("/players")
def get_players():
    players = df[['Player', 'Squad', 'League', 'Pos', 'SubPos', 'Age', 'MarketValue']].to_dict(orient='records')
    return {"players": players}

@app.get("/player/{player_name}")
def get_player(player_name: str):
    row = df[df['Player'].apply(normalize) == normalize(player_name)]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")
    r = row.iloc[0]
    return {
        "Player": r['Player'],
        "Squad": r['Squad'],
        "League": r['League'],
        "Pos": r['Pos'],
        "SubPos": r['SubPos'],
        "Age": r['Age'],
        "MarketValue": int(r['MarketValue']) if r['MarketValue'] > 0 else None,
        "stats": make_stats(r, ALL_COLS),
        "stats_p90": make_stats_p90(r, ALL_COLS)
    }

@app.get("/replace/{player_name}")
def get_replacements(player_name: str, top_n: int = 5, age_min: int = 16, age_max: int = 40, max_value: int = 0):
    player_row = df[df['Player'].apply(normalize) == normalize(player_name)]
    if player_row.empty:
        raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    player = player_row.iloc[0]
    subpos = player['SubPos']
    cluster = player['Cluster']
    features = FEATURES.get(subpos, FEATURES['CM'])

    pos_df = df[
        (df['SubPos'] == subpos) &
        (df['Age'] >= age_min) &
        (df['Age'] <= age_max)
    ].copy().reset_index(drop=True)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(pos_df[features])

    player_idx = pos_df[pos_df['Player'].apply(normalize) == normalize(player_name)].index
    if len(player_idx) == 0:
        pos_df = df[df['SubPos'] == subpos].copy().reset_index(drop=True)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pos_df[features])
        player_idx = pos_df[pos_df['Player'].apply(normalize) == normalize(player_name)].index
        if len(player_idx) == 0:
            raise HTTPException(status_code=404, detail=f"Player '{player_name}' not found")

    player_vec = scaled[player_idx[0]]
    scores = cosine_similarity([player_vec], scaled)[0]
    pos_df['similarity'] = scores

    results = pos_df[pos_df['Player'].apply(normalize) != normalize(player_name)].copy()

    age_filtered = results[(results['Age'] >= age_min) & (results['Age'] <= age_max)]
    if len(age_filtered) >= top_n:
        results = age_filtered

    if max_value > 0:
        value_filtered = results[results['MarketValue'] <= max_value]
        if len(value_filtered) >= top_n:
            results = value_filtered

    if len(results) == 0:
        raise HTTPException(status_code=404, detail="No replacement candidates found matching the selected filters. Try widening your age range or increasing the budget.")

    results['final_score'] = results.apply(
        lambda row: row['similarity'] * 1.1 if row['Cluster'] == cluster else row['similarity'],
        axis=1
    )
    results = results.sort_values('final_score', ascending=False).head(top_n)

    return {
        "player": {
            "name": player['Player'],
            "squad": player['Squad'],
            "league": player['League'],
            "pos": player['Pos'],
            "subpos": subpos,
            "cluster": int(cluster),
            "age": player['Age'],
            "market_value": int(player['MarketValue']) if player['MarketValue'] > 0 else None,
            "stats": make_stats(player, ALL_COLS),
            "stats_p90": make_stats_p90(player, ALL_COLS)
        },
        "features": features,
        "cluster_size": len(pos_df),
        "replacements": [
            {
                "name": row['Player'],
                "squad": row['Squad'],
                "league": row['League'],
                "pos": row['Pos'],
                "subpos": row['SubPos'],
                "age": row['Age'],
                "similarity": round(float(row['similarity']), 4),
                "market_value": int(row['MarketValue']) if row['MarketValue'] > 0 else None,
                "stats": make_stats(row, ALL_COLS),
                "stats_p90": make_stats_p90(row, ALL_COLS)
            }
            for _, row in results.iterrows()
        ]
    }

class DescribeRequest(BaseModel):
    subpos: Optional[str] = None
    key_stats: Dict[str, str] = {}
    age_min: int = 16
    age_max: int = 40
    max_value: int = 0
    top_n: int = 5

import httpx

class AIRequest(BaseModel):
    text: str

@app.post("/ai/extract")
async def ai_extract(req: AIRequest):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": "YOUR_API_KEY_HERE",
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "system": """You are a football data analyst. Extract structured recruitment criteria from natural language descriptions.
Return ONLY valid JSON with these fields:
{
  "subpos": one of ["GK","CB","FB","CDM","CM","CAM","FW"] or null,
  "age_max": number or null,
  "age_min": number or null,
  "max_value_eur": number or null,
  "key_stats": {stat_name: "high"|"medium"|"low"},
  "description": brief 1-sentence summary of the profile
}
Use only these stat names: Gls,Ast,xG,xAG,npxG,PrgC,PrgP,PrgR,Tkl,TklW,Int,Clr,Blocks,KP,PPA,Crs,SCA,GCA,CmpPct,AerialWonPct,DribSuccPct,SoTPct,SavePct,CS,Saves,GA90,AttPen,AttThird,DefThird,MidThird,DefPen
Return only JSON, no other text.""",
                "messages": [{"role": "user", "content": req.text}]
            },
            timeout=30.0
        )
        return response.json()

@app.post("/describe")
def describe_search(req: DescribeRequest):
    subpos = req.subpos or 'FW'
    features = FEATURES.get(subpos, FEATURES['CM'])

    pos_df = df[
        (df['SubPos'] == subpos) &
        (df['Age'] >= req.age_min) &
        (df['Age'] <= req.age_max)
    ].copy().reset_index(drop=True)

    if req.max_value > 0:
        pos_df = pos_df[pos_df['MarketValue'] <= req.max_value]

    if len(pos_df) == 0:
        raise HTTPException(status_code=404, detail="No players found matching these criteria.")

    # Build synthetic target vector from key_stats
    scaler = StandardScaler()
    scaled = scaler.fit_transform(pos_df[features])

    # Map high/medium/low to percentile targets
    level_map = {'high': 0.85, 'medium': 0.5, 'low': 0.15}
    target = np.zeros(len(features))
    for i, f in enumerate(features):
        if f in req.key_stats:
            pct = level_map.get(req.key_stats[f].lower(), 0.5)
            target[i] = np.percentile(scaled[:, i], pct * 100)

    scores = cosine_similarity([target], scaled)[0]
    pos_df['similarity'] = scores
    pos_df['final_score'] = pos_df['similarity']
    results = pos_df.sort_values('final_score', ascending=False).head(req.top_n)

    return {
        "features": features,
        "replacements": [
            {
                "name": row['Player'],
                "squad": row['Squad'],
                "league": row['League'],
                "pos": row['Pos'],
                "subpos": row['SubPos'],
                "age": row['Age'],
                "similarity": round(float(row['similarity']), 4),
                "market_value": int(row['MarketValue']) if row['MarketValue'] > 0 else None,
                "stats": make_stats(row, ALL_COLS),
                "stats_p90": make_stats_p90(row, ALL_COLS)
            }
            for _, row in results.iterrows()
        ]
    }


@app.get("/trajectory/{player_name}")
def trajectory(player_name: str):
    """
    Returns population age curve + individual trajectory + projections
    for a named player. Requires historical data processed by historical_loader.py.
    """
    if not TRAJECTORY_ENABLED:
        return {"error": "Trajectory module not available. Make sure trajectory.py is in the src/ folder."}
    try:
        result = get_trajectory(player_name)
        return result
    except FileNotFoundError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Trajectory error: {str(e)}"}