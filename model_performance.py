import sqlite3

import pandas as pd

conn = sqlite3.connect("nfl.db")
cursor = conn.cursor()

box_scores_2025_df = pd.read_csv("csv_folder/2025_box_scores.csv")

cursor.execute("SELECT * FROM predictions;")
columns = cursor.fetchall()

my_list = []
for col in columns:
    # print(col)
    my_list.append(col)


df = pd.DataFrame(my_list)
print(df)
print(box_scores_2025_df)

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

#change the Full team name to just the abbriviation for visitor and home 

box_scores_2025_df['Visitor'] = box_scores_2025_df['Visitor'].map(team_abbr)
box_scores_2025_df['Home'] = box_scores_2025_df['Home'].map(team_abbr)


merged_df = pd.merge(
    df,
    box_scores_2025_df,
    left_on=[2, 1],   # column names in df1
    right_on=["Visitor", "Home"],              # column names in df2
    how="left"
)

print(merged_df)

merged_df["total_actual"] = merged_df["Visitor_score"] + merged_df["Home_score"]
merged_df["spread_actual"] = abs(merged_df["Visitor_score"] - merged_df["Home_score"])
merged_df["Home_win"] = None
merged_df["Visitor_win"] = None
merged_df["Total_win"] = None
merged_df["Spread_win"] = None



def calc_win(row):
    home_win = row["Home_score"] > row["Visitor_score"]
    visitor_win = row["Visitor_score"] > row["Home_score"]
    return pd.Series([home_win, visitor_win])


def calc_total(row):
    s = row[15]
    num = float(s.replace("u", ""))

    if row[17] == "Over" and num < row["total_actual"]:
       total_win = 1
    elif row[17] == "Under" and num > row["total_actual"]:
        total_win = 1
    else:
        total_win = 0
    return pd.Series([total_win])

def calc_spread(row):
    
    home_spread = row[14]
    visitor_spread = row[10]

    if row[18] == "Home" and home_spread < 0 and row["Home_win"] and row["spread_actual"] >= home_spread:
        spread_win = 1
    elif row[18] == "Away" and visitor_spread < 0 and row["Visitor_win"] and row["spread_actual"] >= visitor_spread:
        spread_win = 1
    elif row[18] == "Home" and home_spread > 0 and row["Visitor_win"] and row["spread_actual"] >= home_spread:
        spread_win = 1
    elif row[18] == "Away" and visitor_spread > 0 and row["Home_win"] and row["spread_actual"] >= home_spread:
        spread_win = 1
    else:
        spread_win = 0
    return pd.Series([spread_win])

merged_df[["Home_win", "Visitor_win"]] = merged_df.apply(calc_win, axis=1)

merged_df[["Total_win"]] = merged_df.apply(calc_total, axis=1)

merged_df[["Spread_win"]] = merged_df.apply(calc_spread, axis=1)

overs = (merged_df[17] == "Over").sum()
unders = (merged_df[17] == "Under").sum()

home_spread = (merged_df[18] == "Home").sum()
away_spread = (merged_df[18] == "Away").sum()

total_num_spread_predictions = home_spread + away_spread

spread_wins = (merged_df["Spread_win"] == 1).sum()


total_wins = (merged_df["Total_win"] == 1).sum()

total_num_total_predictions = overs + unders

print(total_num_total_predictions)
print(total_wins)
print(total_num_spread_predictions)
print(spread_wins)

print("Total win rate: " + str(round((total_wins/total_num_total_predictions)*100, 1)) + "%")
print("Spread win rate: " + str(round((spread_wins/total_num_spread_predictions)*100, 1)) + "%")

# merged_df.to_csv("win_rate.csv")