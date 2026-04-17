# TactiReplace ⚽

> AI-powered football player recruitment and replacement system across the top 5 European leagues.

---

## Overview

**TactiReplace** is an AI-driven scouting and recruitment tool that helps analysts, coaches, and clubs find the most statistically similar alternatives to any player across the **Premier League, La Liga, Bundesliga, Serie A, and Ligue 1** for the 2024/25 season.

The system combines unsupervised machine learning with similarity search to deliver intelligent, data-backed player recommendations — filtering by position, age, budget, and playing style.

---

## Features

- **Two-Layer Recommendation Engine** — K-Means clustering groups players by playing style, followed by cosine similarity on per-90 normalized stats to rank the closest replacements
- **Goalkeeper Support** — Separate pipeline for goalkeeper stats and recommendations
- **"Describe a Player" Search** — Natural language rule-based search to find players matching a custom profile
- **Age & Budget Filters** — Market valuations sourced from Transfermarkt to filter candidates by transfer budget
- **PDF Scouting Reports** — Exportable reports for shortlisted players
- **Historical Trajectory Analysis** — Player performance trend analysis across past seasons (2017–2024)
- **All Five Major Leagues** — Full coverage of PL, La Liga, Bundesliga, Serie A, and Ligue 1

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn (K-Means, Cosine Similarity) |
| Data Sources | FBref (stats), Transfermarkt (valuations) |
| Reporting | ReportLab / FPDF (PDF export) |
| Frontend | HTML, CSS, JavaScript |

---

## How It Works

1. **Data Collection** — Player stats scraped from FBref across all five leagues, merged with Transfermarkt market valuations
2. **Preprocessing** — Stats normalized per 90 minutes, missing values handled, goalkeepers separated
3. **Clustering** — K-Means groups players into clusters based on playing style and statistical profile
4. **Similarity Search** — Within a player's cluster, cosine similarity ranks the closest alternatives
5. **Filtering** — Results filtered by position, age range, and transfer budget
6. **Output** — Recommendations displayed in the UI with optional PDF scouting report export

---

## Project Structure

```
TactiReplace/
│
├── src/
│   ├── main.py               # Entry point
│   ├── build_dataset.py      # Data merging and preparation
│   ├── preprocess.py         # Normalization and cleaning
│   ├── clustering.py         # K-Means clustering
│   ├── similarity.py         # Cosine similarity engine
│   ├── evaluation.py         # Model evaluation
│   ├── evaluation_report.py  # Report generation
│   ├── find_transfers.py     # Transfer candidate search
│   ├── historical_loader.py  # Historical season data loader
│   └── trajectory.py         # Player trajectory analysis
│
├── data/
│   ├── all_leagues_2425.csv          # Final cleaned dataset (all 5 leagues)
│   ├── players_data-2024_2025.csv    # Raw FBref master dataset
│   ├── goalkeeper_stats_2425.csv     # Goalkeeper stats
│   ├── players.csv                   # Transfermarkt biographical & valuation data
│   ├── player_valuations.csv         # Market valuations
│   ├── top5-players24-25.xlsx        # Top player reference sheet
│   └── history/                      # Historical season CSVs (2017–2024)
│
├── models/                   # Saved model files
├── notebooks/                # Exploratory notebooks
├── index.html                # Frontend interface
└── README.md
```

---

## Data Sources

- **[FBref](https://fbref.com)** — Player statistics (standard, shooting, passing, defense, possession, goalkeeping)
- **[Transfermarkt](https://www.transfermarkt.com)** — Market valuations and biographical data

---

## Academic Context

This project was developed as a graduation thesis at the **German University in Cairo (GUC)**, Faculty of Management Technology, under the supervision of **Dr. Ahmed Okasha**.

---

## Author

**Bassel Hesham**
Business Informatics — German University in Cairo
[GitHub](https://github.com/BasselHeshamm)
