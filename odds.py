import requests
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import numpy as np
import sqlite3

team_abbr = {
        "Arizona Cardinals": "ARI",
        "Atlanta Falcons": "ATL",
        "Baltimore Ravens": "BAL",
        "Buffalo Bills": "BUF",
        "Carolina Panthers": "CAR",
        "Chicago Bears": "CHI",
        "Cincinnati Bengals": "CIN",
        "Cleveland Browns": "CLE",
        "Dallas Cowboys": "DAL",
        "Denver Broncos": "DEN",
        "Detroit Lions": "DET",
        "Green Bay Packers": "GB",
        "Houston Texans": "HOU",
        "Indianapolis Colts": "IND",
        "Jacksonville Jaguars": "JAX",
        "Kansas City Chiefs": "KC",
        "Las Vegas Raiders": "LV",
        "Los Angeles Chargers": "LAC",
        "Los Angeles Rams": "LA",
        "Miami Dolphins": "MIA",
        "Minnesota Vikings": "MIN",
        "New England Patriots": "NE",
        "New Orleans Saints": "NO",
        "New York Giants": "NYG",
        "New York Jets": "NYJ",
        "Philadelphia Eagles": "PHI",
        "Pittsburgh Steelers": "PIT",
        "San Francisco 49ers": "SF",
        "Seattle Seahawks": "SEA",
        "Tampa Bay Buccaneers": "TB",
        "Tennessee Titans": "TEN",
        "Washington Commanders": "WAS"
    }


url = f"https://www.espn.com/nfl/odds"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}

res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")
# Find the <script> tag that contains the JSON
# Find the script tag that contains 'odds'
#print(soup)
# Method 1: Convert soup to string and split by lines
# lines = str(soup).splitlines()
# first_few = lines[:78]  # get first 5 lines
# for line in first_few:
#     print(line)

# print(soup)

df = pd.DataFrame()
values = []



nums = []

# Get game details
for game in soup.select('[data-track-extras]'):
    extras = game['data-track-extras']
    game_info = json.loads(extras)
    #print(game_info)
    nums.append(game_info)

df = pd.DataFrame(nums)

#Get each OddsCell
for odds_cell in soup.select('[data-testid="OddsCell"]'):
    moneyline = odds_cell.select_one('.FTMw.FuEs')
    if moneyline:
        #print(moneyline.text)
        values.append(moneyline.text)

# df["moneyline"] = values[:len(df)]  # match length just in case
df = df.drop_duplicates()
df = df.loc[np.repeat(df.index, 8)].reset_index(drop=True)
df["moneyline"] = values[:len(df)]  # match length just in case

# Split on first space only
df[["game_id", "matchup"]] = df["game_detail"].str.split(" ", n=1, expand=True)

# print(df)

df_pivot = (
    df.groupby(["game_id", "matchup"], sort=False)["moneyline"]
    .apply(list)
    .reset_index()
)

df_pivot.to_csv("csv_folder/odds.csv")
# print(df_pivot)

# Expand list of values into separate columns
df_expanded = pd.DataFrame(df_pivot["moneyline"].tolist())
df_final = pd.concat([df_pivot[["game_id", "matchup"]], df_expanded], axis=1)

df_final[["visitor_team", "home_team"]] = df_final["matchup"].str.split(" vs ", n=1, expand=True)
del df_final["matchup"]

df = df_final.rename(columns={
    0: "open_total",
    1: "visitor_spread",
    2: "o_total",
    3: "visitor_ml",
    4: "open_spread",
    5: "home_spread",
    6: "u_total",
    7: "home_ml"
})

df["home_team"] = df["home_team"].replace(team_abbr)
df["visitor_team"] = df["visitor_team"].replace(team_abbr)


# print(df)
# df.to_csv("csv_folder/odds.csv")

conn = sqlite3.connect("nfl.db")

# Write DataFrame to SQL table
df.to_sql("odds", conn, if_exists="replace", index=False)

# Verify by reading back
check = pd.read_sql("SELECT * FROM odds", conn)
print(check.head())