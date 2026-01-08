import numpy as np
import pandas as pd
import sqlite3

nfl_teams_city_name = {
    "Arizona": "Cardinals",
    "Atlanta": "Falcons",
    "Baltimore": "Ravens",
    "Buffalo": "Bills",
    "Carolina": "Panthers",
    "Chicago": "Bears",
    "Cincinnati": "Bengals",
    "Cleveland": "Browns",
    "Dallas": "Cowboys",
    "Denver": "Broncos",
    "Detroit": "Lions",
    "Green Bay": "Packers",
    "Houston": "Texans",
    "Indianapolis": "Colts",
    "Jacksonville": "Jaguars",
    "Kansas City": "Chiefs",
    "Las Vegas": "Raiders",
    "Los Angeles2": "Rams",  # note: Chargers are also LA
    "Los Angeles1": "Chargers",
    "Miami": "Dolphins",
    "Minnesota": "Vikings",
    "New England": "Patriots",
    "New Orleans": "Saints",
    "New York1": "Giants",
    "New York2": "Jets",
    "Philadelphia": "Eagles",
    "Pittsburgh": "Steelers",
    "Seattle": "Seahawks",
    "San Francisco": "49ers",
    "Tampa Bay": "Buccaneers",
    "Tennessee": "Titans",
    "Washington": "Commanders"
}


df = pd.read_csv("2025_schedule.csv")

df_ids = pd.read_csv("all_game_ids.csv")

df['Home'] = df['Home'].str.lstrip('@').str.strip()

df["Visitor"] = df["Visitor"].map(nfl_teams_city_name)
df["Home"] = df["Home"].map(nfl_teams_city_name)

df_ids.columns = ["Index", "Game", "ID"]


df_ids[['Away', 'Home']] = df_ids['Game'].str.split(' @ ', expand=True)

merged = df.merge(df_ids, left_on=["Home", "Visitor"], right_on=["Home", "Away"], how="left")

merged["ID"] = merged["ID"].fillna(0).astype(int)

merged = merged.drop(columns=["Away", "Game", "Index"])

print(merged)

merged.to_csv("merged_df.csv")

