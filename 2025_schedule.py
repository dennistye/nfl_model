import numpy as np
import pandas as pd
import sqlite3

team_dict = {
    "Cardinals": "ARI",
    "Falcons": "ATL",
    "Ravens": "BAL",
    "Bills": "BUF",
    "Panthers": "CAR",
    "Bears": "CHI",
    "Bengals": "CIN",
    "Browns": "CLE",
    "Cowboys": "DAL",
    "Broncos": "DEN",
    "Lions": "DET",
    "Packers": "GB",
    "Texans": "HOU",
    "Colts": "IND",
    "Jaguars": "JAX",
    "Chiefs": "KC",
    "Raiders": "LV",
    "Chargers": "LAC",
    "Rams": "LAR",
    "Dolphins": "MIA",
    "Vikings": "MIN",
    "Patriots": "NE",
    "Saints": "NO",
    "Giants": "NYG",
    "Jets": "NYJ",
    "Eagles": "PHI",
    "Steelers": "PIT",
    "49ers": "SF",
    "Seahawks": "SEA",
    "Buccaneers": "TB",
    "Titans": "TEN",
    "Commanders": "WAS"
}

df = pd.read_csv("merged_df.csv")

df['Home'] = df['Home'].str.lstrip('@').str.strip()

df["Visitor"] = df["Visitor"].map(team_dict)
df["Home"] = df["Home"].map(team_dict)

print(df)

conn = sqlite3.connect("nfl.db")

# Write DataFrame to SQL table
df.to_sql("nfl2025schedule", conn, if_exists="replace", index=False)

# Verify by reading back
check = pd.read_sql("SELECT * FROM nfl2025schedule", conn)
print(check.head())

conn.commit()
conn.close()