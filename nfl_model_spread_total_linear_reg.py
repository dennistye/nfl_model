import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sqlite3
from datetime import date
import math
from scipy.stats import norm

global_week = 0

def clean_data(box_scores_2022_df, box_scores_2023_df, box_scores_2024_df, box_scores_2025_df, pbp_2022_df, pbp_2023_df, pbp_2024_df, pbp_2025_df):
    
    #change the box scores from NaN to REG for the OTFLag column
    pbp_2022_df['DefenseTeam'] = pbp_2022_df['DefenseTeam'].replace('LAR', 'LA')
    pbp_2022_df['OffenseTeam'] = pbp_2022_df['OffenseTeam'].replace('LAR', 'LA')

    pbp_2023_df['DefenseTeam'] = pbp_2023_df['DefenseTeam'].replace('LAR', 'LA')
    pbp_2023_df['OffenseTeam'] = pbp_2023_df['OffenseTeam'].replace('LAR', 'LA')

    pbp_2024_df['DefenseTeam'] = pbp_2024_df['DefenseTeam'].replace('LAR', 'LA')
    pbp_2024_df['OffenseTeam'] = pbp_2024_df['OffenseTeam'].replace('LAR', 'LA')

    box_scores_2022_df['OTFlag'] = box_scores_2022_df['OTFlag'].fillna('REG')
    box_scores_2023_df['OTFlag'] = box_scores_2023_df['OTFlag'].fillna('REG')
    box_scores_2024_df['OTFlag'] = box_scores_2024_df['OTFlag'].fillna('REG')
    box_scores_2025_df['OTFlag'] = box_scores_2025_df['OTFlag'].fillna('REG')

    #get rid of the box score column because it is redundant
    box_scores_2022_df = box_scores_2022_df.drop(columns=['Box Score'], errors='ignore')
    box_scores_2023_df = box_scores_2023_df.drop(columns=['Box Score'], errors='ignore')
    box_scores_2024_df = box_scores_2024_df.drop(columns=['Box Score'], errors='ignore')
    box_scores_2025_df = box_scores_2025_df.drop(columns=['Box Score'], errors='ignore')

    #drop any columns that are empty
    pbp_2022_df = pbp_2022_df.dropna(axis=1, how='all')
    pbp_2023_df = pbp_2023_df.dropna(axis=1, how='all')
    pbp_2024_df = pbp_2024_df.dropna(axis=1, how='all')
    pbp_2025_df = pbp_2025_df.dropna(axis=1, how='all')

    #fill any values that are empty to Unknown
    pbp_2022_df['Formation'] = pbp_2022_df['Formation'].fillna('UNKNOWN')
    pbp_2022_df['PlayType'] = pbp_2022_df['PlayType'].fillna('UNKNOWN')
    pbp_2022_df['PassType'] = pbp_2022_df['PassType'].fillna('UNKNOWN')
    pbp_2022_df['RushDirection'] = pbp_2022_df['RushDirection'].fillna('UNKNOWN')
    pbp_2022_df['PenaltyTeam'] = pbp_2022_df['PenaltyTeam'].fillna('UNKNOWN')
    pbp_2022_df['PenaltyType'] = pbp_2022_df['PenaltyType'].fillna('UNKNOWN')

    pbp_2023_df['Formation'] = pbp_2023_df['Formation'].fillna('UNKNOWN')
    pbp_2023_df['PlayType'] = pbp_2023_df['PlayType'].fillna('UNKNOWN')
    pbp_2023_df['PassType'] = pbp_2023_df['PassType'].fillna('UNKNOWN')
    pbp_2023_df['RushDirection'] = pbp_2023_df['RushDirection'].fillna('UNKNOWN')
    pbp_2023_df['PenaltyTeam'] = pbp_2023_df['PenaltyTeam'].fillna('UNKNOWN')
    pbp_2023_df['PenaltyType'] = pbp_2023_df['PenaltyType'].fillna('UNKNOWN')

    pbp_2024_df['Formation'] = pbp_2024_df['Formation'].fillna('UNKNOWN')
    pbp_2024_df['PlayType'] = pbp_2024_df['PlayType'].fillna('UNKNOWN')
    pbp_2024_df['PassType'] = pbp_2024_df['PassType'].fillna('UNKNOWN')
    pbp_2024_df['RushDirection'] = pbp_2024_df['RushDirection'].fillna('UNKNOWN')
    pbp_2024_df['PenaltyTeam'] = pbp_2024_df['PenaltyTeam'].fillna('UNKNOWN')
    pbp_2024_df['PenaltyType'] = pbp_2024_df['PenaltyType'].fillna('UNKNOWN')

    # pbp_2025_df['ScoringPlay'] = pbp_2025_df['ScoringPlay'].fillna('UNKNOWN')
    # pbp_2025_df['PlayType'] = pbp_2025_df['PlayType'].astype(str)
    # pbp_2025_df['PlayType'] = pbp_2025_df['PlayType'].replace('None', 'UNKNOWN')
    # pbp_2025_df['PlayType'] = pbp_2025_df['PlayType'].fillna('UNKNOWN')
    pbp_2025_df = pbp_2025_df.fillna(0)        # replace NaN with 0
    # pbp_2025_df = pbp_2025_df.fillna('')       # replace NaN with empty string
    pbp_2025_df = pbp_2025_df.rename(columns={"Down_num": "Down"})
    pbp_2025_df["Quarter"] = pbp_2025_df["Quarter"].astype(int)
    pbp_2025_df["ToGo"] = pbp_2025_df["ToGo"].astype(int)
    pbp_2025_df["Yardline"] = pbp_2025_df["Yardline"].astype(int)
    pbp_2025_df["SeriesFirstDown"] = pbp_2025_df["SeriesFirstDown"].astype(int)
    pbp_2025_df["NextScore"] = pbp_2025_df["NextScore"].astype(int)
    pbp_2025_df["TeamWin"] = pbp_2025_df["TeamWin"].astype(int)
    pbp_2025_df["Yards"] = pbp_2025_df["Yards"].astype(int)
    pbp_2025_df["IsRush"] = pbp_2025_df["IsRush"].astype(int)
    pbp_2025_df["IsPass"] = pbp_2025_df["IsPass"].astype(int)
    pbp_2025_df["IsIncomplete"] = pbp_2025_df["IsIncomplete"].astype(int)
    pbp_2025_df["IsTouchdown"] = pbp_2025_df["IsTouchdown"].astype(int)
    pbp_2025_df["IsSack"] = pbp_2025_df["IsSack"].astype(int)
    pbp_2025_df["IsChallenge"] = pbp_2025_df["IsChallenge"].astype(int)
    pbp_2025_df["IsChallengeReversed"] = pbp_2025_df["IsChallengeReversed"].astype(int)
    pbp_2025_df["Challenger"] = pbp_2025_df["Challenger"].astype(int)
    pbp_2025_df["IsMeasurement"] = pbp_2025_df["IsMeasurement"].astype(int)
    pbp_2025_df["IsInterception"] = pbp_2025_df["IsInterception"].astype(int)
    pbp_2025_df["IsFumble"] = pbp_2025_df["IsFumble"].astype(int)
    pbp_2025_df["IsPenalty"] = pbp_2025_df["IsPenalty"].astype(int)
    pbp_2025_df["IsTwoPointConversion"] = pbp_2025_df["IsTwoPointConversion"].astype(int)
    pbp_2025_df["IsTwoPointConversionSuccessful"] = pbp_2025_df["IsTwoPointConversionSuccessful"].astype(int)
    pbp_2025_df["RushDirection"] = pbp_2025_df["RushDirection"].astype(int)
    pbp_2025_df["YardLineFixed"] = pbp_2025_df["YardLineFixed"].astype(int)
    pbp_2025_df["IsPenaltyAccepted"] = pbp_2025_df["IsPenaltyAccepted"].astype(int)
    pbp_2025_df["PenaltyTeam"] = pbp_2025_df["PenaltyTeam"].astype(int)
    pbp_2025_df["IsNoPlay"] = pbp_2025_df["IsNoPlay"].astype(int)
    pbp_2025_df["PenaltyType"] = pbp_2025_df["PenaltyType"].astype(int)
    pbp_2025_df["PenaltyYards"] = pbp_2025_df["PenaltyYards"].astype(int)
    pbp_2025_df["Formation"] = pbp_2025_df["Formation"].replace(0, "UNKNOWN")
    pbp_2025_df["PassType"] = pbp_2025_df["PassType"].fillna("UNKNOWN")

    #change data from 2023-11-19 → 11/19/2023 format in play by play df and make sure the date columns are in date format
    pbp_2022_df['GameDate'] = pd.to_datetime(pbp_2022_df['GameDate'])
    pbp_2022_df['GameDate'] = pbp_2022_df['GameDate'].dt.strftime('%m/%d/%Y')

    pbp_2023_df['GameDate'] = pd.to_datetime(pbp_2023_df['GameDate'])
    pbp_2023_df['GameDate'] = pbp_2023_df['GameDate'].dt.strftime('%m/%d/%Y')

    pbp_2024_df['GameDate'] = pd.to_datetime(pbp_2024_df['GameDate'])
    pbp_2024_df['GameDate'] = pbp_2024_df['GameDate'].dt.strftime('%m/%d/%Y')

    pbp_2025_df['GameDate'] = pd.to_datetime(pbp_2025_df['GameDate'])
    pbp_2025_df['GameDate'] = pbp_2025_df['GameDate'].dt.strftime('%m/%d/%Y')

    

    box_scores_2022_df['Date'] = pd.to_datetime(box_scores_2022_df['Date'], errors='coerce')
    box_scores_2022_df['Date'] = box_scores_2022_df['Date'].dt.strftime('%m/%d/%Y')

    box_scores_2023_df['Date'] = pd.to_datetime(box_scores_2023_df['Date'], errors='coerce')
    box_scores_2023_df['Date'] = box_scores_2023_df['Date'].dt.strftime('%m/%d/%Y')

    box_scores_2024_df['Date'] = pd.to_datetime(box_scores_2024_df['Date'], errors='coerce')
    box_scores_2024_df['Date'] = box_scores_2024_df['Date'].dt.strftime('%m/%d/%Y')

    box_scores_2025_df['Date'] = pd.to_datetime(box_scores_2025_df['Date'], errors='coerce')
    box_scores_2025_df['Date'] = box_scores_2025_df['Date'].dt.strftime('%m/%d/%Y')

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
    box_scores_2022_df['Visitor'] = box_scores_2022_df['Visitor'].map(team_abbr)
    box_scores_2022_df['Home'] = box_scores_2022_df['Home'].map(team_abbr)

    box_scores_2023_df['Visitor'] = box_scores_2023_df['Visitor'].map(team_abbr)
    box_scores_2023_df['Home'] = box_scores_2023_df['Home'].map(team_abbr)

    box_scores_2024_df['Visitor'] = box_scores_2024_df['Visitor'].map(team_abbr)
    box_scores_2024_df['Home'] = box_scores_2024_df['Home'].map(team_abbr)

    box_scores_2025_df['Visitor'] = box_scores_2025_df['Visitor'].map(team_abbr)
    box_scores_2025_df['Home'] = box_scores_2025_df['Home'].map(team_abbr)



    # Merge play-by-play data for 2022 with scores data for 2022 based on Date, OffenseTeam, and DefenseTeam
    merged_2022_df = pbp_2022_df.merge(box_scores_2022_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Visitor', 'Home'], how='left')
    merged_2022_df = merged_2022_df.merge(box_scores_2022_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Home', 'Visitor'], how='left', suffixes=('', '_reverse'))

    for column in ['Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']:
        merged_2022_df[column] = merged_2022_df[column].combine_first(merged_2022_df[column + '_reverse'])


    columns_to_drop = [col + '_reverse' for col in ['Date','Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']]
    merged_2022_df = merged_2022_df.drop(columns=columns_to_drop)
    merged_2022_df = merged_2022_df.drop(columns='Date')


    # Merge play-by-play data for 2023 with scores data for 2023 based on Date, OffenseTeam, and DefenseTeam
    merged_2023_df = pbp_2023_df.merge(box_scores_2023_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Visitor', 'Home'], how='left')
    merged_2023_df = merged_2023_df.merge(box_scores_2023_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Home', 'Visitor'], how='left', suffixes=('', '_reverse'))

    for column in ['Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']:
        merged_2023_df[column] = merged_2023_df[column].combine_first(merged_2023_df[column + '_reverse'])


    columns_to_drop = [col + '_reverse' for col in ['Date','Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']]
    merged_2023_df = merged_2023_df.drop(columns=columns_to_drop)
    merged_2023_df = merged_2023_df.drop(columns='Date')



    # Merge play-by-play data for 2024 with scores data for 2024 based on Date, OffenseTeam, and DefenseTeam
    merged_2024_df = pbp_2024_df.merge(box_scores_2024_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Visitor', 'Home'], how='left')
    merged_2024_df = merged_2024_df.merge(box_scores_2024_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Home', 'Visitor'], how='left', suffixes=('', '_reverse'))

    for column in ['Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']:
        merged_2024_df[column] = merged_2024_df[column].combine_first(merged_2024_df[column + '_reverse'])


    columns_to_drop = [col + '_reverse' for col in ['Date','Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']]
    merged_2024_df = merged_2024_df.drop(columns=columns_to_drop)
    merged_2024_df = merged_2024_df.drop(columns='Date')



    merged_2025_df = pbp_2025_df.merge(box_scores_2025_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Visitor', 'Home'], how='left')
    merged_2025_df = merged_2025_df.merge(box_scores_2025_df, left_on=['GameDate', 'OffenseTeam', 'DefenseTeam'], right_on=['Date', 'Home', 'Visitor'], how='left', suffixes=('', '_reverse'))

    for column in ['Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']:
        merged_2025_df[column] = merged_2025_df[column].combine_first(merged_2025_df[column + '_reverse'])


    columns_to_drop = [col + '_reverse' for col in ['Date','Visitor', 'Visitor_score', 'Home', 'Home_score', 'OTFlag']]
    merged_2025_df = merged_2025_df.drop(columns=columns_to_drop)
    merged_2025_df = merged_2025_df.drop(columns='Date')



    # Adding "HomeWon" Column which is just a binary 1 or 0 when a home team won or lost
    merged_2022_df['HomeWon'] = merged_2022_df['Home_score'] > merged_2022_df['Visitor_score']
    merged_2023_df['HomeWon'] = merged_2023_df['Home_score'] > merged_2023_df['Visitor_score']
    merged_2024_df['HomeWon'] = merged_2024_df['Home_score'] > merged_2024_df['Visitor_score']
    merged_2025_df['HomeWon'] = merged_2025_df['Home_score'] > merged_2025_df['Visitor_score']

    merged_2022_df['Spread'] = merged_2022_df['Home_score'] - merged_2022_df['Visitor_score']
    merged_2022_df['Total'] = merged_2022_df['Home_score'] + merged_2022_df['Visitor_score']

    merged_2023_df['Spread'] = merged_2023_df['Home_score'] - merged_2023_df['Visitor_score']
    merged_2023_df['Total'] = merged_2023_df['Home_score'] + merged_2023_df['Visitor_score']

    merged_2024_df['Spread'] = merged_2024_df['Home_score'] - merged_2024_df['Visitor_score']
    merged_2024_df['Total'] = merged_2024_df['Home_score'] + merged_2024_df['Visitor_score']

    merged_2025_df['Spread'] = merged_2025_df['Home_score'] - merged_2025_df['Visitor_score']
    merged_2025_df['Total'] = merged_2025_df['Home_score'] + merged_2025_df['Visitor_score']

    merged_2025_df.rename(columns={"Yardline": "YardLine"}, inplace=True)
    merged_2025_df.rename(columns={"IsMeaurement": "IsMeasurement"}, inplace=True)

    merged_2025_df["YardLine"] = merged_2025_df["YardLine"].fillna(0).astype(int)
    # print(merged_2022_df.isna().sum())
    # merged_2025_df.to_csv("merged_2025_df.csv")

    # print(merged_2025_df)
    # print(merged_2023_df)
    

    all_data = pd.concat([merged_2023_df, merged_2024_df, merged_2025_df])
    # all_data = all_data.drop('Unnamed: 0.1', axis=1)  # drop single column

    # all_data["Yardline"] = all_data["Yardline"].fillna(0).astype(int)
    # all_data["IsMeaurement"] = all_data["IsMeaurement"].fillna(0).astype(int)
    all_data["Challenger"] = all_data["Challenger"].fillna(0).astype(int)
    # all_data["Yardline"] = all_data["Yardline"].fillna(0).astype(int)
    print(all_data.isna().sum())

    #all_data['Weight'] = all_data['SeasonYear'].apply(lambda x: 1.5 if x == 2024 else 1)

    if all_data.isna().any().any():
        print("all_data DataFrame contains NaN values.")
    else:
        print("No NaN values in DataFrame.")

    #all_data = merged_2024_df

    return all_data

def calculate_home_field_advantage(all_data):
    """
    Calculates both league-wide and team-specific Home Field Advantage (HFA) per season.

    Returns:
        pd.DataFrame with columns:
        [Season, Team, Simple_Avg, Home_Win_Pct, Away_Win_Pct, Regression]
        where Team = "LEAGUE_AVG" for league-wide values.
    """
    records = []

    for season in sorted(all_data["SeasonYear"].unique()):
        season_data = all_data[all_data["SeasonYear"] == season]

        # --- League-wide HFA ---
        league_hfa_avg = (season_data["Home_score"] - season_data["Visitor_score"]).mean()
        league_home_win_pct = (season_data["Home_score"] > season_data["Visitor_score"]).mean()

        # Regression for league HFA
        home_rows = season_data[["GameId", "Home", "Home_score"]].rename(
            columns={"Home": "Team", "Home_score": "Points"}
        )
        home_rows["is_home"] = 1

        away_rows = season_data[["GameId", "Visitor", "Visitor_score"]].rename(
            columns={"Visitor": "Team", "Visitor_score": "Points"}
        )
        away_rows["is_home"] = 0

        team_level = pd.concat([home_rows, away_rows], ignore_index=True)
        X = team_level[["is_home"]]
        y = team_level["Points"]

        league_model = LinearRegression().fit(X, y)
        league_regression_hfa = league_model.coef_[0]

        records.append({
            "Season": int(season),
            "Team": "LEAGUE_AVG",
            "Simple_Avg": round(league_hfa_avg, 2),
            "Home_Win_Pct": round(league_home_win_pct * 100, 2),
            "Away_Win_Pct": round((1 - league_home_win_pct) * 100, 2),
            "Regression": round(league_regression_hfa, 2)
        })

        # --- Team-specific HFA ---
        for team in season_data["Home"].unique():
            team_data = season_data[season_data["Home"] == team]

            if len(team_data) < 10:
                continue  # skip teams with too few games

            team_hfa_avg = (team_data["Home_score"] - team_data["Visitor_score"]).mean()
            team_home_win_pct = (team_data["Home_score"] > team_data["Visitor_score"]).mean()

            home_rows = team_data[["GameId", "Home", "Home_score"]].rename(
                columns={"Home": "Team", "Home_score": "Points"}
            )
            home_rows["is_home"] = 1

            away_rows = team_data[["GameId", "Visitor", "Visitor_score"]].rename(
                columns={"Visitor": "Team", "Visitor_score": "Points"}
            )
            away_rows["is_home"] = 0

            team_level = pd.concat([home_rows, away_rows], ignore_index=True)
            X = team_level[["is_home"]]
            y = team_level["Points"]

            team_model = LinearRegression().fit(X, y)
            team_regression_hfa = team_model.coef_[0]

            records.append({
                "Season": int(season),
                "Team": team,
                "Simple_Avg": round(team_hfa_avg, 2),
                "Home_Win_Pct": round(team_home_win_pct * 100, 2),
                "Away_Win_Pct": round((1 - team_home_win_pct) * 100, 2),
                "Regression": round(team_regression_hfa, 2)
            })

    return pd.DataFrame(records)

def calculate_total_home_field_advantage(all_data, box_scores_2023_df, box_scores_2024_df, box_scores_2025_df):
    tau = 10.0  # tunable; larger tau -> more shrinkage for small n

    def shrink(team_val, n, league_mean, tau=50.0):
        weight = n / (n + tau)
        return weight * team_val + (1 - weight) * league_mean

    # Clean data
    for df in [box_scores_2023_df, box_scores_2024_df, box_scores_2025_df]:
        df['OTFlag'] = df['OTFlag'].fillna('REG')
        df.drop(columns=['Box Score'], errors='ignore', inplace=True)



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

    # Combine
    df_combined = pd.concat([box_scores_2023_df, box_scores_2024_df, box_scores_2025_df], axis=0, ignore_index=True)

    df_combined['Visitor'] = df_combined['Visitor'].map(team_abbr)
    df_combined['Home'] = df_combined['Home'].map(team_abbr)

    # League-wide HFA
    league_hfa_avg = (df_combined["Home_score"] - df_combined["Visitor_score"]).mean()
    league_home_win_pct = (df_combined["Home_score"] > df_combined["Visitor_score"]).mean()
    print(f"League HFA avg: {league_hfa_avg}, Home Win %: {league_home_win_pct*100:.2f}%")

    results = []
    home_games_count_dict = {}

    teams = pd.concat([df_combined["Home"], df_combined["Visitor"]]).unique()

    for team in teams:
        home_games = df_combined[df_combined["Home"] == team]
        away_games = df_combined[df_combined["Visitor"] == team]

        home_game_num = len(home_games)
        away_game_num = len(away_games)
        home_games_count_dict[team] = home_game_num

        # Regression for home games
        if home_game_num > 0:
            X_home = home_games[["Visitor_score"]]
            y_home = home_games["Home_score"]
            model_home = LinearRegression().fit(X_home, y_home)
            home_coef = model_home.intercept_
        else:
            home_coef = None

        # Regression for away games
        if away_game_num > 0:
            X_away = away_games[["Home_score"]]
            y_away = away_games["Visitor_score"]
            model_away = LinearRegression().fit(X_away, y_away)
            away_coef = model_away.intercept_
        else:
            away_coef = None

        results.append({
            "team": team,
            "regression_home": home_coef,
            "regression_away": away_coef,
            "reg_diff": None if (home_coef is None or away_coef is None) else home_coef - away_coef,
            "home_games": home_game_num,
            "away_games": away_game_num
        })

    team_hfa_df = pd.DataFrame(results)

    # Shrink reg_diff
    team_hfa_df['reg_diff_shrunk'] = team_hfa_df.apply(
        lambda r: shrink(r['reg_diff'], home_games_count_dict.get(r['team'], 0), league_hfa_avg, tau)
        if r['reg_diff'] is not None else None,
        axis=1
    )

    return team_hfa_df

def team_features(all_data):
    # Add QB passing yards and QB rushing yards

    # QB Touchdowns per game

    # Sacks per game

    # Rushing yards allowed per game

    # Passing yards allowed per game

    # Points allowed <----- already have this 

    # Home or away, home team has a 3% higher chance of winning


    # 1. Average Pointes Scored
    # Calculate the average points scored by each team when home and away
    avg_points_scored_home = all_data.groupby('Home')['Home_score'].mean()
    avg_points_scored_visitor = all_data.groupby('Visitor')['Visitor_score'].mean()

    # 2. Average Points Allowed
    # Calculate the average points allowed by each the when home and visitor teams.
    avg_points_allowed_home = all_data.groupby('Home')['Visitor_score'].mean()
    avg_points_allowed_visitor = all_data.groupby('Visitor')['Home_score'].mean()

    # Calculate the overall average points scored by each team
    overall_avg_points_scored = (avg_points_scored_home + avg_points_scored_visitor) / 2

    # 3. Win Rate
    # Calculate the total number of wins for each home and visitor teams
    home_wins = all_data.groupby('Home')['HomeWon'].sum()
    visitor_wins = all_data.groupby('Visitor').apply(lambda x: len(x) - x['HomeWon'].sum())

    # Calculate the total number of games played by each teams as home and visitor.
    total_games_home = all_data['Home'].value_counts()
    total_games_visitor = all_data['Visitor'].value_counts()

    # Calculate the overall number of wins and total games played by each team.
    overall_wins = home_wins + visitor_wins
    total_games = total_games_home + total_games_visitor

    # Calculate the win rate for each team.
    win_rate = overall_wins / total_games

    # Calculate the average outcome of games between each pair of teams (home vs visitor).
    team_features = pd.DataFrame({
        'AvgPointsScored': overall_avg_points_scored,
        'WinRate': win_rate
    })

    # Reset the index of the team_features DataFrame and rename the index column to "Team".
    team_features.reset_index(inplace=True)
    team_features.rename(columns={'Home': 'Team'}, inplace=True)


    # Calculate defensive features for each NFL team.

    # 1. Average points defended:
    overall_avg_points_defended = (avg_points_allowed_home + avg_points_allowed_visitor) / 2

    # 2. Average conceded plays:
    # A play is considered successful for the offense if it results in a touchdown or doesn't result in a turnover.
    # Create a new column 'SuccessfulPlay' in the all_data DataFrame to represent this.
    #all_data['SuccessfulPlay'] = all_data['IsTouchdown'] | (~all_data['IsInterception'] & ~all_data['IsFumble'])
    all_data['SuccessfulPlay'] = (
    (all_data['IsTouchdown']) |
    (
        ~all_data['IsInterception'] & 
        ~all_data['IsFumble'] & 
        (
            ((all_data['PlayType'] == 'Pass') & (all_data['Yards'] >= 10)) |
            ((all_data['PlayType'] == 'Run') & (all_data['Yards'] >= 4))
        )
    )
    )

    # # Calculate the average rate of successful plays conceded when playing at home.
    avg_conceded_plays_home = all_data.groupby('Home')['SuccessfulPlay'].mean()

    # # Calculate the average rate of successful plays conceded when playing as a visitor.
    avg_conceded_plays_visitor = all_data.groupby('Visitor')['SuccessfulPlay'].mean()

    # # Calculate the overall average rate of successful plays conceded for each team.
    # overall_avg_conceded_plays = (avg_conceded_plays_home + avg_conceded_plays_visitor) / 2

    # Count of plays faced at home and away
    num_plays_home = all_data.groupby('Home').size()
    num_plays_visitor = all_data.groupby('Visitor').size()

    # Weighted overall average
    overall_avg_conceded_plays = (
    (avg_conceded_plays_home * num_plays_home + avg_conceded_plays_visitor * num_plays_visitor)
    / (num_plays_home + num_plays_visitor)
    )

    # 3. Average forced turnovers:
    # Create a new column 'Turnover' that indicates if a play resulted in a turnover (either interception or fumble).
    all_data['Turnover'] = all_data['IsInterception'] | all_data['IsFumble']

    # Calculate the average rate of turnovers forced when playing at home.
    avg_forced_turnovers_home = all_data.groupby('Home')['Turnover'].mean()

    # Calculate the average rate of turnovers forced when playing as a visitor.
    avg_forced_turnovers_visitor = all_data.groupby('Visitor')['Turnover'].mean()

    # Calculate the overall average rate of turnovers forced for each team.
    #overall_avg_forced_turnovers = (avg_forced_turnovers_home + avg_forced_turnovers_visitor) / 2

    # Number of plays by each team on defense
    num_def_plays_home = all_data.groupby('Home').size()
    num_def_plays_visitor = all_data.groupby('Visitor').size()

    # Weighted turnover rate
    overall_avg_forced_turnovers = (
    avg_forced_turnovers_home * num_def_plays_home + 
    avg_forced_turnovers_visitor * num_def_plays_visitor
    ) / (num_def_plays_home + num_def_plays_visitor)

    # Create a new DataFrame to store the defensive features for each team.
    team_features_defensive = pd.DataFrame({
        'Team': team_features['Team'].values,
        'AvgPointsDefended': overall_avg_points_defended.values,
        'AvgConcededPlays': overall_avg_conceded_plays.values,
        'AvgForcedTurnovers': overall_avg_forced_turnovers.values
    })

    # Merge the defensive features with the original team features to create a combined DataFrame.
    team_features_combined = team_features.merge(team_features_defensive, on='Team')

    # Calculate additional offensive features

    # 1. Average yards per play
    avg_yards_per_play_home = all_data.groupby('Home')['Yards'].mean()
    avg_yards_per_play_visitor = all_data.groupby('Visitor')['Yards'].mean()
    overall_avg_yards_per_play = (avg_yards_per_play_home + avg_yards_per_play_visitor) / 2

    # 2. Average total yards per game
    total_yards_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
    total_yards_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
    overall_avg_yards_per_game = (total_yards_per_game_home + total_yards_per_game_visitor).groupby(level=1).mean()

    # 3. Average pass completion rate
    avg_pass_completion_rate_home = all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
    avg_pass_completion_rate_visitor = all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
    overall_avg_pass_completion_rate = (avg_pass_completion_rate_home + avg_pass_completion_rate_visitor) / 2

    # 4. Average touchdowns per game
    avg_touchdowns_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
    avg_touchdowns_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
    overall_avg_touchdowns_per_game = (avg_touchdowns_per_game_home + avg_touchdowns_per_game_visitor).groupby(level=1).mean()

    # 5. Average rush success rate
    avg_rush_success_rate_home = all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
    avg_rush_success_rate_visitor = all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
    overall_avg_rush_success_rate = (avg_rush_success_rate_home + avg_rush_success_rate_visitor) / 2

    # Creating a dataframe for the new offensive features
    new_offensive_features = pd.DataFrame({
        'Team': team_features_combined['Team'],
        'AvgYardsPerPlay': overall_avg_yards_per_play.values,
        'AvgYardsPerGame': overall_avg_yards_per_game.values,
        'AvgPassCompletionRate': overall_avg_pass_completion_rate.values,
        'AvgTouchdownsPerGame': overall_avg_touchdowns_per_game.values,
        'AvgRushSuccessRate': overall_avg_rush_success_rate.values
    })

    # Merging with the existing combined features
    team_features_expanded = team_features_combined.merge(new_offensive_features, on='Team')


    # Calculate additional defensive features

    # 1. Average yards allowed per play
    avg_yards_allowed_per_play_home = all_data.groupby('Home')['Yards'].mean()
    avg_yards_allowed_per_play_visitor = all_data.groupby('Visitor')['Yards'].mean()
    overall_avg_yards_allowed_per_play = (avg_yards_allowed_per_play_home + avg_yards_allowed_per_play_visitor) / 2

    # 2. Average total yards allowed per game
    total_yards_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
    total_yards_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['Yards'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
    overall_avg_yards_allowed_per_game = (total_yards_allowed_per_game_home + total_yards_allowed_per_game_visitor).groupby(level=1).mean()

    # 3. Average pass completion allowed rate (only considering pass plays)
    pass_plays = all_data[all_data['IsPass'] == 1]
    avg_pass_completion_allowed_rate_home = pass_plays.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
    avg_pass_completion_allowed_rate_visitor = pass_plays.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
    overall_avg_pass_completion_allowed_rate = (avg_pass_completion_allowed_rate_home + avg_pass_completion_allowed_rate_visitor) / 2

    # 4. Average touchdowns allowed per game
    avg_touchdowns_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
    avg_touchdowns_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
    overall_avg_touchdowns_allowed_per_game = (avg_touchdowns_allowed_per_game_home + avg_touchdowns_allowed_per_game_visitor).groupby(level=1).mean()

    # 5. Average rush success allowed rate (e.g., rushes > 4 yards)
    rush_plays = all_data[all_data['IsRush'] == 1]
    avg_rush_success_allowed_rate_home = rush_plays.groupby('Home').apply(lambda x: (x['Yards'] > 4).mean())
    avg_rush_success_allowed_rate_visitor = rush_plays.groupby('Visitor').apply(lambda x: (x['Yards'] > 4).mean())
    overall_avg_rush_success_allowed_rate = (avg_rush_success_allowed_rate_home + avg_rush_success_allowed_rate_visitor) / 2


    # Creating a dataframe for the new defensive features
    new_defensive_features = pd.DataFrame({
        'Team': team_features_expanded['Team'],
        'AvgYardsAllowedPerPlay': overall_avg_yards_allowed_per_play.values,
        'AvgYardsAllowedPerGame': overall_avg_yards_allowed_per_game.values,
        'AvgPassCompletionAllowedRate': overall_avg_pass_completion_allowed_rate.values,
        'AvgTouchdownsAllowedPerGame': overall_avg_touchdowns_allowed_per_game.values,
        'AvgRushSuccessAllowedRate': overall_avg_rush_success_allowed_rate.values
    })

    # Merging with the existing combined features
    team_features_complete = team_features_expanded.merge(new_defensive_features, on='Team')

    # After creating all_data in clean_data
    if team_features_complete.isna().any().any():
        print("DataFrame contains NaN values.")
    else:
        print("No NaN values in DataFrame.")

    return team_features_complete

def clean_schedule_merge_with_features(team_features_complete):

    # schedule_2025_df = pd.read_csv("csv_folder/2025_schedule.csv")

    # week1_df = schedule_2025_df[schedule_2025_df['Week'] == 1]
    # week1_df = week1_df[['Home', 'Visitor']]
    # week1_df['Home'] = week1_df['Home'].str.lstrip('@').str.strip()

    # city_abbr = {
    #     "Arizona": "ARI",
    #     "Atlanta": "ATL",
    #     "Baltimore": "BAL",
    #     "Buffalo": "BUF",
    #     "Carolina": "CAR",
    #     "Chicago": "CHI",
    #     "Cincinnati": "CIN",
    #     "Cleveland": "CLE",
    #     "Dallas": "DAL",
    #     "Denver": "DEN",
    #     "Detroit": "DET",
    #     "Green Bay": "GB",
    #     "Houston": "HOU",
    #     "Indianapolis": "IND",
    #     "Jacksonville": "JAX",
    #     "Kansas City": "KC",
    #     "Las Vegas": "LV",
    #     "Los Angeles1": "LAC",  
    #     "Los Angeles2": "LA",  
    #     "Miami": "MIA",
    #     "Minnesota": "MIN",
    #     "New England": "NE",
    #     "New Orleans": "NO",
    #     "New York1": "NYG",  
    #     "New York2": "NYJ",  
    #     "Philadelphia": "PHI",
    #     "Pittsburgh": "PIT",
    #     "San Francisco": "SF",
    #     "Seattle": "SEA",
    #     "Tampa Bay": "TB",
    #     "Tennessee": "TEN",
    #     "Washington": "WAS"
    # }

    # week1_df['Home'] = week1_df['Home'].map(city_abbr)
    # week1_df['Visitor'] = week1_df['Visitor'].map(city_abbr)

    conn = sqlite3.connect("nfl.db")
    query = """
        SELECT *
        FROM "nfl2025schedule"
    """

    # Load directly into DataFrame
    df = pd.read_sql_query(query, conn)
    df_first = df.sort_values(["Week", "Date", "Time"]) \
             .groupby("Week", as_index=False) \
             .first()
    
    # Compute absolute difference in days
    df_first["Date"] = pd.to_datetime(df_first["Date"], errors="coerce")

    # Get today's date
    today = pd.to_datetime(date.today())
    df_first["Diff"] = (df_first["Date"] - today).abs()

    # Find the row with the smallest difference
    closest_week = df_first.loc[df_first["Diff"].idxmin(), "Week"]
    
    print(closest_week)
    global global_week
    global_week = closest_week

    conn = sqlite3.connect("nfl.db")
    query = f"""
        SELECT *
        FROM "nfl2025schedule"
        WHERE Week = 10
    """

    

    # Load directly into DataFrame
    week_number_df = pd.read_sql_query(query, conn)
    # print(week_number_df)
    week_number_df = week_number_df[['Home', 'Visitor']]
    # print(week1_df)

    # Close connection
    conn.close()

    week_number_df['Home'] = week_number_df['Home'].replace('LAR', 'LA')
    week_number_df['Visitor'] = week_number_df['Visitor'].replace('LAR', 'LA')



    upcoming_encoded_home = week_number_df.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
    # print(upcoming_encoded_home)
    upcoming_encoded_both = upcoming_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')
    # print(upcoming_encoded_both)

    # Calculate the difference in features as this might be a more predictive representation
    for col in ['AvgPointsScored', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
                'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
                'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
        upcoming_encoded_both[f'Diff_{col}'] = upcoming_encoded_both[f'{col}_Home'] - upcoming_encoded_both[f'{col}_Visitor']

    # Selecting only the difference columns and the teams for clarity
    upcoming_encoded_final = upcoming_encoded_both[['Home', 'Visitor'] + [col for col in upcoming_encoded_both.columns if 'Diff_' in col]]
    # print(upcoming_encoded_final)
    # After creating all_data in clean_data
    if upcoming_encoded_final.isna().any().any():
        print("DataFrame contains NaN values.")
    else:
        print("No NaN values in DataFrame.")
    return upcoming_encoded_final

def prep_and_train(upcoming_encoded_final, team_features_complete, all_data, total_hfa):
    
    # Merge play-by-play data with team features for home teams
    training_encoded_home = all_data.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
    # Merge the result with team features for visitor teams
    training_encoded_both = training_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')


    training_encoded_both = training_encoded_both.merge(
    total_hfa[['team', 'reg_diff_shrunk']], left_on='Home', right_on='team', how='left'
    ).rename(columns={'reg_diff_shrunk': 'HFA_HOME'}).drop(columns=['team'])

    training_encoded_both = training_encoded_both.merge(
        total_hfa[['team', 'reg_diff_shrunk']], left_on='Visitor', right_on='team', how='left'
    ).rename(columns={'reg_diff_shrunk': 'HFA_VISITOR'}).drop(columns=['team'])

    # Create combined HFA difference
    training_encoded_both['Diff_HFA'] = training_encoded_both['HFA_HOME'] - training_encoded_both['HFA_VISITOR']
    # training_encoded_both.drop(columns=['Diff_HFA_Visitor'], inplace=True)

    # upcoming_encoded_final.to_csv("upcoming_encoded_final.csv")
    # team_features_complete.to_csv("team_features_complete.csv")
    # all_data.to_csv("all_data.csv")

    

    # Calculate the difference in features
    for col in ['AvgPointsScored', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
                'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
                'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
        training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Visitor']


    # training_encoded_both.to_csv("training.csv")

    

    # Filtering out the required columns
    # Feature matrix
    X_train = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
    X_train['Diff_HFA'] = training_encoded_both['Diff_HFA']  # Add HFA difference as a feature


    # X_train = X_train.fillna(0)

    # Target vectors
    y_spread = all_data['Spread']
    # print(y_spread)
    y_total = all_data['Total']
    # print(y_total)

    # y_spread = y_spread.fillna(0)
    # y_total = y_total.fillna(0)

    # ---- Apply Weights ----


    qb_history = {
        "ARI":[2024, 2025],
        "ATL":[2025],
        "BAL":[2023, 2024, 2025],
        "BUF":[2023, 2024, 2025],
        "CAR":[2023, 2024, 2025],
        "CHI":[2024, 2025],
        "CIN":[2023, 2025],
        "CLE":[2025],
        "DAL":[2023, 2024, 2025],
        "DEN":[2025],
        "DET":[2023, 2024, 2025],
        "GB":[2023, 2024, 2025],
        "HOU":[2023, 2024, 2025],
        "IND":[2025],
        "JAX":[2023, 2024, 2025],
        "KC":[2023, 2024, 2025],
        "LV":[2025],
        "LAC":[2023, 2024, 2025], 
        "LA":[2023, 2024, 2025],
        "MIA":[2023, 2024, 2025],
        "MIN":[2025],
        "NE":[2025],
        "NO":[2025],
        "NYG":[2025],  
        "NYJ":[2025],  
        "PHI":[2023, 2024, 2025],
        "PIT":[2025], 
        "SF":[2023, 2024, 2025],
        "SEA":[2025],
        "TB":[2023, 2024, 2025],
        "TEN":[2025],
        "WAS":[2024, 2025],
    }

    alpha = 0.7
    current_year = 2025
    weight_map = {year: alpha**(current_year - year) for year in [2023, 2024, 2025]}

    # Map weights to each row based on SeasonYear
    sample_weights = all_data['SeasonYear'].map(weight_map).fillna(1.0)

    # def get_weight(row):
    #     offense_team = row["OffenseTeam"]
    #     year = row["SeasonYear"]

    #     # If team not in QB history, neutral weight
    #     if offense_team not in qb_history:
    #         return 1.0

    #     qb_years = qb_history[offense_team]

    #     # ✅ Full weight only for years QB actually played
    #     if year in qb_years:
    #         return 1.0
    #     else:
    #         # Year QB didn’t play — small weight
    #         return 0.05


    # # Apply to your play-by-play or team-level data
    # sample_weights = all_data.apply(get_weight, axis=1)

    spread_model = LinearRegression()
    total_model = LinearRegression()

    spread_model.fit(X_train, y_spread, sample_weight=sample_weights)
    total_model.fit(X_train, y_total, sample_weight=sample_weights)

    X_upcoming = upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]]
    # X_upcoming.to_csv("x_upcoming.csv")
    # print(X_upcoming.isna().sum())


    upcoming_encoded_final = upcoming_encoded_final.merge(
        total_hfa[['team', 'reg_diff_shrunk']], left_on='Home', right_on='team', how='left'
    ).rename(columns={'reg_diff_shrunk': 'HFA_HOME'}).drop(columns=['team'])

    upcoming_encoded_final = upcoming_encoded_final.merge(
        total_hfa[['team', 'reg_diff_shrunk']], left_on='Visitor', right_on='team', how='left'
    ).rename(columns={'reg_diff_shrunk': 'HFA_VISITOR'}).drop(columns=['team'])

    upcoming_encoded_final['Diff_HFA'] = upcoming_encoded_final['HFA_HOME'] - upcoming_encoded_final['HFA_VISITOR']

    # Add to upcoming X
    X_upcoming['Diff_HFA'] = upcoming_encoded_final['Diff_HFA']

    cols = ['Diff_HFA'] + [col for col in X_upcoming.columns if col != 'Diff_HFA']
    X_upcoming = X_upcoming[cols]

    predicted_spreads = spread_model.predict(X_upcoming)
    predicted_totals = total_model.predict(X_upcoming)

    print("Spread model R²:", spread_model.score(X_train, y_spread, sample_weight=sample_weights))
    print("Total model R²:", total_model.score(X_train, y_total, sample_weight=sample_weights))


    upcoming_encoded_final['PredictedSpread'] = predicted_spreads
    upcoming_encoded_final['PredictedTotal'] = predicted_totals

    final_predictions = upcoming_encoded_final[['Home', 'Visitor', 'PredictedSpread', 'PredictedTotal']]

    return final_predictions

def spread_edge(diff, std_dev=13.5, odds=-110):
    """
    Estimate % edge given a spread difference (model - Vegas).

    Parameters:
        diff (float): Difference between your predicted spread and Vegas spread.
                      Positive means your model favors covering more than Vegas.
        std_dev (float): Standard deviation of NFL point margins. Default 13.5.
        odds (float): Sportsbook odds. Default -110.

    Returns:
        float: Approximate % edge
    """
    # Convert spread difference to probability
    model_prob = 1 - norm.cdf(-diff / std_dev)

    # Break-even probability for given odds
    break_even = abs(odds) / (abs(odds) + 100)

    # Edge in %
    edge_percent = (model_prob - break_even) * 100
    print(edge_percent)
    return edge_percent

def spread_percent_edge(complete_df):

    complete_df["spread_percent_edge"] = None

    for index, row in complete_df.iterrows():
        if row["best_spread"] == "Home":
            complete_df.at[index, "spread_percent_edge"] = spread_edge(row["diff_spread"])
        elif row["best_spread"] == "Away":
            complete_df.at[index, "spread_percent_edge"] = spread_edge(row["diff_spread"])
    
    complete_df['spread_percent_edge'] = complete_df['spread_percent_edge'].fillna(0).astype(int)


    return complete_df

def total_percent_edge(complete_df):

    complete_df["total_percent_edge"] = None

    for index, row in complete_df.iterrows():
        if row["best_total"] == "Over":
            complete_df.at[index, "total_percent_edge"] = spread_edge(row["diff_total"])
        elif row["best_total"] == "Under":
            complete_df.at[index, "total_percent_edge"] = spread_edge(row["diff_total"])

    complete_df['total_percent_edge'] = complete_df['total_percent_edge'].fillna(0).astype(int)


    return complete_df

def test_model(upcoming_encoded_final, team_features_complete, all_data):
     # Merge play-by-play data with team features for home teams
    training_encoded_home = all_data.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
    # Merge the result with team features for visitor teams
    training_encoded_both = training_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')

    # upcoming_encoded_final.to_csv("upcoming_encoded_final.csv")
    # team_features_complete.to_csv("team_features_complete.csv")
    # all_data.to_csv("all_data.csv")

    

    # Calculate the difference in features
    for col in ['AvgPointsScored', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
                'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
                'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
        training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Visitor']


    # training_encoded_both.to_csv("training.csv")

    

    # Filtering out the required columns
    # Feature matrix
    X_train = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]
    # X_train['Diff_HFA'] = training_encoded_both['Diff_HFA']

    # X_train.to_csv("xtrain.csv")
    # print(X_train.isna().sum())


    # X_train = X_train.fillna(0)

    # Target vectors
    y_spread = all_data['Spread']
    # print(y_spread)
    y_total = all_data['Total']
    # print(y_total)

    # y_spread = y_spread.fillna(0)
    # y_total = y_total.fillna(0)

    # ---- Apply Weights ----
    alpha = 0.7
    current_year = 2025
    weight_map = {year: alpha**(current_year - year) for year in [2023, 2024, 2025]}

    # Map weights to each row based on SeasonYear
    sample_weights = all_data['SeasonYear'].map(weight_map).fillna(1.0)

    spread_model = LinearRegression()
    total_model = LinearRegression()

    spread_model.fit(X_train, y_spread, sample_weight=sample_weights)
    total_model.fit(X_train, y_total, sample_weight=sample_weights)

    X_upcoming = upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]]
    # X_upcoming.to_csv("x_upcoming.csv")
    # print(X_upcoming.isna().sum())

    predicted_spreads = spread_model.predict(X_upcoming)
    predicted_totals = total_model.predict(X_upcoming)

    upcoming_encoded_final['PredictedSpread'] = predicted_spreads
    upcoming_encoded_final['PredictedTotal'] = predicted_totals

    final_predictions = upcoming_encoded_final[['Home', 'Visitor', 'PredictedSpread', 'PredictedTotal']]

    return final_predictions
    
def best_bets(complete_df):

    complete_df["o_total"] = complete_df["o_total"].str.replace("o", "", regex=False)
    complete_df["o_total"] = pd.to_numeric(complete_df["o_total"], errors="coerce")
    complete_df["home_spread"] = pd.to_numeric(complete_df["home_spread"], errors="coerce")
    complete_df["best_total"] = None
    complete_df["best_spread"] = None
    complete_df["diff_spread"] = None
    complete_df["diff_total"] = None
    # complete_df["top_three_spread"] = None
    # complete_df["top_three_total"] = None

    for index, row in complete_df.iterrows():
        complete_df.at[index, "diff_total"] = abs(row["PredictedTotal"] - row["o_total"])
        if row["PredictedTotal"] - row["o_total"] >= 5:
            complete_df.at[index, "best_total"] = "Over"
        elif row["PredictedTotal"] - row["o_total"] <= -5:
            complete_df.at[index, "best_total"] = "Under"
        else:
            complete_df.at[index, "best_total"] = "Mininmal Edge"


    for index, row in complete_df.iterrows():
        complete_df.at[index, "diff_spread"] = abs(abs(row["HomeSpread"]) - abs(row["home_spread"]))
        if row["HomeSpread"] - row["home_spread"] >= 5 and row["HomeSpread"] > 0 and row["home_spread"] > 0:
            complete_df.at[index, "best_spread"] = "Away"
        elif row["HomeSpread"] - row["home_spread"] <= -5 and row["HomeSpread"] < 0 and row["home_spread"] < 0:
            complete_df.at[index, "best_spread"] = "Home" 
        elif row["HomeSpread"] - row["home_spread"] >= 5 and row["HomeSpread"] < 0 and row["home_spread"] < 0:
            complete_df.at[index, "best_spread"] = "Away" 
        elif row["HomeSpread"] - row["home_spread"] <= -5 and row["HomeSpread"] > 0 and row["home_spread"] > 0:
            complete_df.at[index, "best_spread"] = "Home" 
        elif row["HomeSpread"] - row["home_spread"] >= 5 and row["HomeSpread"] > 0 and row["home_spread"] < 0:
            complete_df.at[index, "best_spread"] = "Away" 
        elif row["HomeSpread"] - row["home_spread"] <= -5 and row["HomeSpread"] < 0 and row["home_spread"] > 0:
            complete_df.at[index, "best_spread"] = "Home" 
        else:
            complete_df.at[index, "best_spread"] = "Mininmal Edge"
    

    #     complete_df["top_three_spread"] = (
    #     complete_df["diff_spread"]
    #     .where(complete_df["diff_spread"] >= 5)
    #     .rank(method="first", ascending=False)
    #     .where(lambda x: x <= 3)
    # )
        
    # complete_df["top_three_total"] = (
    #     complete_df["diff_total"]
    #     .where(complete_df["diff_total"] >= 5)
    #     .rank(method="first", ascending=False)
    #     .where(lambda x: x <= 3)
    # )

    # complete_df.drop(columns=["diff_spread", "diff_total"], inplace=True)

    # complete_df['top_three_spread'] = complete_df['top_three_spread'].fillna(0).astype(int)
    # complete_df['top_three_total'] = complete_df['top_three_total'].fillna(0).astype(int)

    complete_df["o_total"] = "o" + complete_df["o_total"].astype(str)

    return complete_df

def convert_spread_to_reg(spread):
    if spread > 0:
        spread = spread * -1
    elif spread < 0:
        spread = spread * -1
    return spread

def main():

    # Load data
    box_scores_2022_df = pd.read_csv("csv_folder/2022_box_scores.csv")

    box_scores_2023_df = pd.read_csv("csv_folder/2023_box_scores.csv")

    box_scores_2024_df = pd.read_csv("csv_folder/2024_box_scores.csv")

    box_scores_2025_df = pd.read_csv("csv_folder/2025_box_scores.csv")

    # schedule_2025_df = pd.read_csv("csv_folder/2025_schedule.csv")
    pbp_2022_df = pd.read_csv("csv_folder/pbp-2022.csv")

    pbp_2023_df = pd.read_csv("csv_folder/pbp-2023.csv")

    pbp_2024_df = pd.read_csv("csv_folder/pbp-2024.csv")

    pbp_2025_df = pd.read_csv("csv_folder/pbp-2025.csv")

    pbp_2025_df = pbp_2025_df.drop("Unnamed: 0", axis=1)

    # pbp_2025_df = pbp_2025_df[~pbp_2025_df['Type'].isin(['Timeout', 'Period'])]
    pbp_2025_df['DefenseTeam'] = pbp_2025_df['DefenseTeam'].replace('LAR', 'LA')
    pbp_2025_df['OffenseTeam'] = pbp_2025_df['OffenseTeam'].replace('LAR', 'LA')
    # pbp_2025_df = pbp_2025_df.drop(columns=['Season', 'Week'])

    vegas_odds = pd.read_csv("csv_folder/odds.csv")
#     vegas_odds2 = pd.read_csv("vegas_lines.csv")
#     vegas_odds2 = vegas_odds2[vegas_odds2["G#"] == 1]
#     vegas_odds2['Opp'] = vegas_odds2['Opp'].str.lstrip('@')
#     vegas_odds2.rename(columns={"Opp": "visitor_team"}, inplace=True)
#     vegas_odds2.rename(columns={"Team": "home_team"}, inplace=True)
#     vegas_odds2.rename(columns={"Spread": "home_spread"}, inplace=True)
#     vegas_odds2.rename(columns={"Over/Under": "o_total"}, inplace=True)

#     vegas_odds2['o_total'] = vegas_odds2['o_total'].astype(float)  # to float
#     vegas_odds2['home_spread'] = vegas_odds2['home_spread'].astype(float)  # to float

#     team_abbr = {
#     "CRD": "ARI",
#     "SDG": "LAC",
#     "ARI": "ARI",
#     "GNB": "GB",
#     "HTX": "HOU",
#     "IND": "IND",
#     "JAX": "JAX",
#     "KAN": "KC",
#     "LVR": "LV",
#     "LAR": "LA",
#     "NWE": "NE",
#     "NOR": "NO",
#     "SFO": "SF",
#     "TAM": "TB",
#     "ATL": "ATL",
#     "RAV": "BAL",
#     "BAL": "BAL",
#     "BUF": "BUF",
#     "CAR": "CAR",
#     "CHI": "CHI",
#     "CIN": "CIN",
#     "CLE": "CLE",
#     "DAL": "DAL",
#     "DEN": "DEN",
#     "DET": "DET",
#     "GNB": "GB",
#     "LAC": "LAC",
#     "MIA": "MIA",
#     "MIN": "MIN",
#     "NO": "NO",
#     "NYG": "NYG",
#     "NYJ": "NYJ",
#     "PHI": "PHI",
#     "PIT": "PIT",
#     "SEA": "SEA",
#     "OTI": "TEN",
#     "WAS": "WAS",
#     "CLT": "MIA",
#     "TEN": "TEN",
#     "RAM": "LA",
#     "RAI": "LV",
#     "HOU": "HOU"
# }

#     vegas_odds2['home_team'] = vegas_odds2['home_team'].map(team_abbr)
#     vegas_odds2['visitor_team'] = vegas_odds2['visitor_team'].map(team_abbr)

    # print(vegas_odds2)

    pbp_2022_df = pbp_2022_df[~(pbp_2022_df['OffenseTeam'].isna() & pbp_2022_df['DefenseTeam'].isna())]

    # k_neighbors_classifier = pd.read_csv("supporting_files/k_neighbors_classifier.csv")

    # k_neighbors_classifier[['Visitor', 'Home']] = k_neighbors_classifier['Game'].str.split(' @ ', expand=True)

    #pinnacle_probs_df = pd.read_csv("csv_folder/Pinnacle_odds.csv")

    # Cleaned data
    all_data = clean_data(box_scores_2022_df, box_scores_2023_df, box_scores_2024_df, box_scores_2025_df, pbp_2022_df, pbp_2023_df, pbp_2024_df, pbp_2025_df)

    # home_field_advantage = calculate_home_field_advantage(all_data)

    # print(home_field_advantage)

    total_hfa = calculate_total_home_field_advantage(all_data, box_scores_2023_df, box_scores_2024_df, box_scores_2024_df)
    print(total_hfa)

    # Adding team features
    team_features_complete = team_features(all_data)

    # Cleaning the schedule and merging with features
    upcoming_encoded_final = clean_schedule_merge_with_features(team_features_complete)

    # Prepares data and uses Logistic Regression to train 
    upcoming_predictions = prep_and_train(upcoming_encoded_final, team_features_complete, all_data, total_hfa)

    # test_model(upcoming_encoded_final, team_features_complete, all_data)

    # Output prediciton to a csv file, eventually will be ouputing to a database

    # upcoming_predictions["VegasSpread"] = pinnacle_probs_df['VegasSpread'].values
    # upcoming_predictions["VegasTotal"] = pinnacle_probs_df['VegasTotal'].values

    upcoming_predictions['HomeSpread'] = upcoming_predictions['PredictedSpread'].apply(convert_spread_to_reg)
    # upcoming_predictions['HomeVegasSpread'] = upcoming_predictions['VegasSpread'].apply(convert_spread_to_reg)
    # upcoming_predictions['DiffSpread'] = upcoming_predictions['HomeSpread'] - upcoming_predictions['HomeVegasSpread']

    upcoming_predictions['VisitorSpread'] = upcoming_predictions['HomeSpread'].apply(lambda x: x*-1)
    # upcoming_predictions['VisitorVegasSpread'] = upcoming_predictions['HomeVegasSpread'].apply(lambda x: x*-1)
    # upcoming_predictions['DiffVisitorSpread'] = upcoming_predictions['VisitorSpread'] - upcoming_predictions['VisitorVegasSpread']

    # upcoming_predictions['DiffTotal'] = upcoming_predictions['PredictedTotal'] - upcoming_predictions['VegasTotal']

    # upcoming_predictions['DiffSpread'] = upcoming_predictions['DiffSpread'].apply(lambda x: x*-1 if x < 0 else x)
    # upcoming_predictions['DiffTotal'] = upcoming_predictions['DiffTotal'].apply(lambda x: x*-1 if x < 0 else x)

    upcoming_predictions = upcoming_predictions.drop(columns=['PredictedSpread']) 

    upcoming_predictions['HomeSpread'] = upcoming_predictions['HomeSpread'].round(1)
    upcoming_predictions['VisitorSpread'] = upcoming_predictions['VisitorSpread'].round(1)
    upcoming_predictions['PredictedTotal'] = upcoming_predictions['PredictedTotal'].round(1)
    upcoming_predictions['o_PredictedTotal'] = upcoming_predictions['PredictedTotal'].apply(lambda x: f"o{x}" if pd.notnull(x) else None)
    upcoming_predictions['u_PredictedTotal'] = upcoming_predictions['PredictedTotal'].apply(lambda x: f"u{x}" if pd.notnull(x) else None)
    # upcoming_predictions = upcoming_predictions.drop(columns=['PredictedTotal']) 

    # print(vegas_odds)
    # print(upcoming_predictions)
    # print(k_neighbors_classifier)

    # print(upcoming_predictions)
    # print(vegas_odds)

    complete_df = upcoming_predictions.merge(vegas_odds, left_on=["Home", "Visitor"], right_on=["home_team", "visitor_team"], how="inner")

    # print(complete_df)

    # complete_df2 = upcoming_predictions.merge(vegas_odds2, left_on=["Home", "Visitor"], right_on=["home_team", "visitor_team"], how="inner")

    # complete_df2 = best_bets(complete_df2)

    complete_df = best_bets(complete_df)

    complete_df = complete_df.drop(columns=["Unnamed: 0"])
    complete_df = complete_df.drop(columns=["visitor_team"])
    complete_df = complete_df.drop(columns=["home_team"])

    complete_df = spread_percent_edge(complete_df)
    complete_df = total_percent_edge(complete_df)

    print(complete_df)


    # complete_df = complete_df.merge(k_neighbors_classifier, left_on=["Home", "Visitor"], right_on=["Home", "Visitor"], how="inner")

    # complete_df = complete_df.drop(columns=["Unnamed: 0_y", "Game", "Spread", "Total"])
    # # Sort by Unnamed: 0_x
    # complete_df.rename(columns={"Unnamed: 0_x": "Index"}, inplace=True)
    # complete_df.rename(columns={"KNC(7)": "knc"}, inplace=True)
    # complete_df = complete_df.sort_values("Index").reset_index(drop=True)

    # print(complete_df)




    complete_df.to_csv("csv_folder/complete_weights.csv", index=False)


    # upcoming_predictions['DiffSpread'] = upcoming_predictions['DiffSpread'].round(1)
    # upcoming_predictions['DiffVisitorSpread'] = upcoming_predictions['DiffVisitorSpread'].round(1)
    # upcoming_predictions['DiffTotal'] = upcoming_predictions['DiffTotal'].round(1)

    global global_week

    complete_df["week"] = global_week

    print(global_week)

    conn = sqlite3.connect("nfl.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
    week INTEGER,
    Home TEXT,
    Visitor TEXT,
    PredictedTotal REAL,
    HomeSpread REAL,
    VisitorSpread REAL,
    o_PredictedTotal TEXT,
    u_PredictedTotal TEXT,
    game_id TEXT,
    open_total REAL,
    visitor_spread REAL,
    o_total TEXT,
    visitor_ml REAL,
    open_spread REAL,
    home_spread REAL,
    u_total TEXT,
    home_ml REAL,
    best_total TEXT,
    best_spread TEXT,
    diff_spread REAL,
    diff_total REAL,
    top_three_spread INTEGER,
    top_three_total INTEGER
    )
    """)
    conn.commit()

    # complete_df = complete_df.drop('spread_percent_edge', axis=1)
    # complete_df = complete_df.drop('total_percent_edge', axis=1)


    # complete_df.to_sql("predictions", conn, if_exists="append", index=False)


    upcoming_predictions.to_csv("csv_folder/week1_predictions_linear_reg.csv", index=False)

    return upcoming_predictions, global_week

if __name__ == "__main__":
    main()