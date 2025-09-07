import numpy as np
import pandas as pd
import sqlite3

city_abbr = {
        "Arizona": "ARI",
        "Atlanta": "ATL",
        "Baltimore": "BAL",
        "Buffalo": "BUF",
        "Carolina": "CAR",
        "Chicago": "CHI",
        "Cincinnati": "CIN",
        "Cleveland": "CLE",
        "Dallas": "DAL",
        "Denver": "DEN",
        "Detroit": "DET",
        "Green Bay": "GB",
        "Houston": "HOU",
        "Indianapolis": "IND",
        "Jacksonville": "JAX",
        "Kansas City": "KC",
        "Las Vegas": "LV",
        "Los Angeles1": "LAC",  
        "Los Angeles2": "LA",  
        "Miami": "MIA",
        "Minnesota": "MIN",
        "New England": "NE",
        "New Orleans": "NO",
        "New York1": "NYG",  
        "New York2": "NYJ",  
        "Philadelphia": "PHI",
        "Pittsburgh": "PIT",
        "San Francisco": "SF",
        "Seattle": "SEA",
        "Tampa Bay": "TB",
        "Tennessee": "TEN",
        "Washington": "WAS"
    }

df = pd.read_csv("2025_schedule.csv")

df['Home'] = df['Home'].str.lstrip('@').str.strip()

df["Visitor"] = df["Visitor"].map(city_abbr)
df["Home"] = df["Home"].map(city_abbr)

print(df)

conn = sqlite3.connect("nfl.db")

# Write DataFrame to SQL table
df.to_sql("nfl2025schedule", conn, if_exists="replace", index=False)

# Verify by reading back
check = pd.read_sql("SELECT * FROM nfl2025schedule", conn)
print(check.head())

conn.commit()
conn.close()