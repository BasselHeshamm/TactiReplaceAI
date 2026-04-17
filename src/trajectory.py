"""
src/trajectory.py

Builds population age curves and individual player development trajectories.
"""

import os
import numpy as np
import pandas as pd

# ── trajectory stats per subposition ─────────────────────────────────────────
TRAJECTORY_STATS = {
    "GK":  ["SavePct", "GA90", "CSPct"],
    "CB":  ["Tkl_p90", "Int_p90", "Clr_p90", "AerialWonPct", "CmpPct", "PrgP_p90"],
    "FB":  ["PrgC_p90", "PrgR_p90", "Crs_p90", "xAG_p90", "Tkl_p90"],
    "CDM": ["Tkl_p90", "TklW_p90", "Int_p90", "CmpPct", "PrgP_p90"],
    "CM":  ["KP_p90", "SCA_p90", "PrgC_p90", "PrgP_p90", "CmpPct", "xAG_p90"],
    "CAM": ["Gls_p90", "Ast_p90", "xAG_p90", "KP_p90", "SCA_p90", "GCA_p90"],
    "FW":  ["Gls_p90", "xG_p90", "npxG_p90", "Ast_p90", "xAG_p90", "SoTPct", "PrgC_p90"],
}

AGE_MIN        = 17
AGE_MAX        = 36
PROJ_YEARS     = 3
MIN_MINUTES    = 450
MIN_POP_POINTS = 20
POLY_DEGREE    = 2

IDENTITY_COLS = {"Player", "Nation", "Pos", "Squad", "League", "Season", "SubPos", "NormName"}


def _rate_stats():
    return {"SavePct", "CSPct", "AerialWonPct", "CmpPct",
            "SoTPct", "DribSuccPct", "GA90", "G/Sh"}


def _load_combined(current_csv="data/all_leagues_2425.csv",
                   history_csv="data/history/processed/all_history_combined.csv"):
    frames = []

    if os.path.exists(current_csv):
        cur = pd.read_csv(current_csv, low_memory=False)
        cur["Season"] = "2024-25"
        frames.append(cur)

    if os.path.exists(history_csv):
        hist = pd.read_csv(history_csv, low_memory=False)
        frames.append(hist)

    if not frames:
        raise FileNotFoundError(
            "Neither current nor historical dataset found. "
            "Run preprocess.py and historical_loader.py first."
        )

    df = pd.concat(frames, ignore_index=True)

    # Force ALL non-identity columns to numeric before any arithmetic
    for col in df.columns:
        if col not in IDENTITY_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["90s"] = df["Min"] / 90.0

    # Recompute p90 columns that may be missing
    rate_stats = _rate_stats()
    nineties = df["90s"].replace(0, np.nan)
    for col in list(df.columns):
        p90_col = col + "_p90"
        if p90_col not in df.columns and col not in rate_stats and col not in IDENTITY_COLS:
            df[p90_col] = df[col] / nineties

    return df


def _fit_population_curve(df, subpos, stat):
    sub = df[
        (df["SubPos"] == subpos) &
        (df["Age"] >= AGE_MIN) &
        (df["Age"] <= AGE_MAX) &
        (df["Min"] >= MIN_MINUTES) &
        (df[stat].notna()) &
        (np.isfinite(df[stat]))
    ].copy()

    if len(sub) < MIN_POP_POINTS:
        return {}

    ages   = sub["Age"].values.astype(float)
    values = sub[stat].values.astype(float)
    cap    = np.nanpercentile(values, 99)
    mask   = values <= cap
    ages, values = ages[mask], values[mask]
    weights = np.sqrt(sub["Min"].values.astype(float)[mask])

    try:
        coeffs = np.polyfit(ages, values, POLY_DEGREE, w=weights)
        poly   = np.poly1d(coeffs)
    except Exception:
        return {}

    std_resid = float(np.std(values - poly(ages)))
    curve = {}
    for age in range(AGE_MIN, AGE_MAX + 1):
        m = float(poly(age))
        curve[age] = {
            "mean":  round(max(m, 0), 3),
            "lower": round(max(m - 1.28 * std_resid, 0), 3),
            "upper": round(max(m + 1.28 * std_resid, 0), 3),
        }
    return curve


def _percentile_at_age(df, subpos, stat, age, value):
    sub = df[
        (df["SubPos"] == subpos) &
        (df["Age"].between(age - 1, age + 1)) &
        (df["Min"] >= MIN_MINUTES) &
        (df[stat].notna()) &
        (np.isfinite(df[stat]))
    ][stat].values

    if len(sub) < 5:
        return -1
    return round(float(np.mean(sub <= value)) * 100)


def _player_history(df, norm_name, stat):
    import unicodedata

    def norm(s):
        if not isinstance(s, str):
            return ""
        nfkd = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    if "NormName" not in df.columns:
        df["NormName"] = df["Player"].apply(norm)

    rows = df[
        (df["NormName"] == norm_name) &
        (df["Min"] >= MIN_MINUTES) &
        (df[stat].notna()) &
        (np.isfinite(df[stat]))
    ].sort_values("Age")

    return [
        {
            "season":  str(row.get("Season", "")),
            "age":     int(row["Age"]),
            "value":   round(float(row[stat]), 3),
            "minutes": int(row["Min"]),
        }
        for _, row in rows.iterrows()
    ]


def _project(curve, history, age_now):
    if not curve or not history:
        return []

    deltas = [
        point["value"] - curve[point["age"]]["mean"]
        for point in history
        if point["age"] in curve
    ]
    if not deltas:
        return []

    avg_delta   = float(np.mean(deltas))
    projections = []

    for i in range(1, PROJ_YEARS + 1):
        future_age = age_now + i
        if future_age not in curve:
            continue
        proj_val = max(round(curve[future_age]["mean"] + avg_delta * (0.6 ** i), 3), 0)
        ci_width  = (curve[future_age]["upper"] - curve[future_age]["lower"]) / 2
        ci_extra  = ci_width * 0.3 * i
        projections.append({
            "age":   future_age,
            "value": proj_val,
            "lower": round(max(proj_val - ci_width - ci_extra, 0), 3),
            "upper": round(proj_val + ci_width + ci_extra, 3),
        })

    return projections


# ── public API ────────────────────────────────────────────────────────────────

_combined_df = None


def get_combined_df():
    global _combined_df
    if _combined_df is None:
        _combined_df = _load_combined()
    return _combined_df


def get_trajectory(player_name):
    import unicodedata

    def norm(s):
        if not isinstance(s, str):
            return ""
        nfkd = unicodedata.normalize("NFKD", s)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()

    df = get_combined_df()

    if "NormName" not in df.columns:
        df["NormName"] = df["Player"].apply(norm)

    norm_name = norm(player_name)
    current   = df[(df["NormName"] == norm_name) & (df["Season"] == "2024-25")]

    if current.empty:
        current = df[df["NormName"] == norm_name].sort_values("Season", ascending=False)
    if current.empty:
        return {"error": f"Player '{player_name}' not found in dataset."}

    player_row   = current.iloc[0]
    subpos       = str(player_row.get("SubPos", ""))
    age_now      = int(player_row.get("Age", 0))
    display_name = str(player_row.get("Player", player_name))

    if subpos not in TRAJECTORY_STATS:
        return {"error": f"Sub-position '{subpos}' not supported for trajectory."}

    stats_out = {}

    for stat in TRAJECTORY_STATS[subpos]:
        if stat not in df.columns:
            continue

        pop_curve = _fit_population_curve(df, subpos, stat)
        if not pop_curve:
            continue

        history    = _player_history(df, norm_name, stat)
        projection = _project(pop_curve, history, age_now)

        for proj in projection:
            proj["percentile"] = _percentile_at_age(
                df, subpos, stat, proj["age"], proj["value"]
            )

        current_val = player_row.get(stat)
        current_pct = -1
        try:
            cv = float(current_val)
            if np.isfinite(cv):
                current_pct = _percentile_at_age(df, subpos, stat, age_now, cv)
        except (TypeError, ValueError):
            pass

        stats_out[stat] = {
            "label":              _stat_label(stat),
            "population_curve":   [{"age": a, **v} for a, v in sorted(pop_curve.items())],
            "player_history":     history,
            "projection":         projection,
            "current_value":      round(float(current_val), 3) if current_val is not None else None,
            "current_percentile": current_pct,
        }

    return {
        "player":  display_name,
        "subpos":  subpos,
        "age_now": age_now,
        "squad":   str(player_row.get("Squad", "")),
        "league":  str(player_row.get("League", "")),
        "stats":   stats_out,
    }


def _stat_label(stat):
    labels = {
        "xG_p90":       "xG per 90",
        "Gls_p90":      "Goals per 90",
        "npxG_p90":     "npxG per 90",
        "Ast_p90":      "Assists per 90",
        "xAG_p90":      "xAG per 90",
        "SoTPct":       "Shot accuracy %",
        "PrgC_p90":     "Prog. carries per 90",
        "KP_p90":       "Key passes per 90",
        "SCA_p90":      "SCA per 90",
        "GCA_p90":      "GCA per 90",
        "Tkl_p90":      "Tackles per 90",
        "TklW_p90":     "Tackles won per 90",
        "Int_p90":      "Interceptions per 90",
        "Clr_p90":      "Clearances per 90",
        "AerialWonPct": "Aerial duel win %",
        "CmpPct":       "Pass completion %",
        "PrgP_p90":     "Prog. passes per 90",
        "PrgR_p90":     "Prog. runs per 90",
        "Crs_p90":      "Crosses per 90",
        "SavePct":      "Save %",
        "GA90":         "Goals against per 90",
        "CSPct":        "Clean sheet %",
    }
    return labels.get(stat, stat)


if __name__ == "__main__":
    result = get_trajectory("Mohamed Salah")
    if "error" in result:
        print(result["error"])
    else:
        print(f"\nTrajectory for {result['player']} ({result['subpos']}, age {result['age_now']})")
        for stat, data in result["stats"].items():
            print(f"\n  {data['label']}")
            print(f"    Current: {data['current_value']} (top {100 - data['current_percentile']}%)")
            for proj in data["projection"]:
                pct_str = f"top {100 - proj['percentile']}%" if proj["percentile"] >= 0 else "n/a"
                print(f"    Age {proj['age']}: {proj['lower']}–{proj['upper']}  ({pct_str})")
