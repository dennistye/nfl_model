import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error


box_scores_2023_df = pd.read_csv("csv_folder/2023_box_scores.csv")

box_scores_2024_df = pd.read_csv("csv_folder/2024_box_scores.csv")

schedule_2025_df = pd.read_csv("csv_folder/2025_schedule.csv")

pbp_2023_df = pd.read_csv("csv_folder/pbp-2023.csv")

pbp_2024_df = pd.read_csv("csv_folder/pbp-2024.csv")

pinnacle_odds_df = pd.read_csv("csv_folder/Pinnacle_odds.csv")
#print(pinnacle_odds_df)

#change the box scores from NaN to REG for the OTFLag column

box_scores_2023_df['OTFlag'] = box_scores_2023_df['OTFlag'].fillna('REG')
box_scores_2024_df['OTFlag'] = box_scores_2024_df['OTFlag'].fillna('REG')

#get rid of the box score column because it is redundant

box_scores_2023_df = box_scores_2023_df.drop(columns=['Box Score'], errors='ignore')
box_scores_2024_df = box_scores_2024_df.drop(columns=['Box Score'], errors='ignore')

#drop any columns that are empty

pbp_2023_df = pbp_2023_df.dropna(axis=1, how='all')
pbp_2024_df = pbp_2024_df.dropna(axis=1, how='all')

#fill any values that are empty to Unknown

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


#change data from 2023-11-19 → 11/19/2023 format in play by play df and make sure the date columns are in date format

pbp_2023_df['GameDate'] = pd.to_datetime(pbp_2023_df['GameDate'])
pbp_2023_df['GameDate'] = pbp_2023_df['GameDate'].dt.strftime('%m/%d/%Y')

pbp_2024_df['GameDate'] = pd.to_datetime(pbp_2024_df['GameDate'])
pbp_2024_df['GameDate'] = pbp_2024_df['GameDate'].dt.strftime('%m/%d/%Y')

box_scores_2023_df['Date'] = pd.to_datetime(box_scores_2023_df['Date'], errors='coerce')
box_scores_2023_df['Date'] = box_scores_2023_df['Date'].dt.strftime('%m/%d/%Y')

box_scores_2023_df['Date'] = pd.to_datetime(box_scores_2023_df['Date'], errors='coerce')
box_scores_2023_df['Date'] = box_scores_2023_df['Date'].dt.strftime('%m/%d/%Y')

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

box_scores_2023_df['Visitor'] = box_scores_2023_df['Visitor'].map(team_abbr)
box_scores_2023_df['Home'] = box_scores_2023_df['Home'].map(team_abbr)

box_scores_2024_df['Visitor'] = box_scores_2024_df['Visitor'].map(team_abbr)
box_scores_2024_df['Home'] = box_scores_2024_df['Home'].map(team_abbr)


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


# Adding "HomeWon" Column which is just a binary 1 or 0 when a home team won or lost
merged_2023_df['HomeWon'] = merged_2023_df['Home_score'] > merged_2023_df['Visitor_score']
merged_2024_df['HomeWon'] = merged_2024_df['Home_score'] > merged_2024_df['Visitor_score']

merged_2023_df['Spread'] = merged_2023_df['Home_score'] - merged_2023_df['Visitor_score']
merged_2023_df['Total'] = merged_2023_df['Home_score'] + merged_2023_df['Visitor_score']

merged_2024_df['Spread'] = merged_2024_df['Home_score'] - merged_2024_df['Visitor_score']
merged_2024_df['Total'] = merged_2024_df['Home_score'] + merged_2024_df['Visitor_score']

#all_data = pd.concat([merged_2023_df, merged_2024_df])

all_data = merged_2024_df

# 1. Average Pointes Scored
# Calculate the average points scored by the home and visitor teams.
avg_points_scored_home = all_data.groupby('Home')['Home_score'].mean()
avg_points_scored_visitor = all_data.groupby('Visitor')['Visitor_score'].mean()

# 2. Average Points Allowed
# Calculate the average points allowed by the home and visitor teams.
avg_points_allowed_home = all_data.groupby('Home')['Visitor_score'].mean()
avg_points_allowed_visitor = all_data.groupby('Visitor')['Home_score'].mean()

# Calculate the overall average points scored and allowed by combining the home and visitor averages.
overall_avg_points_scored = (avg_points_scored_home + avg_points_scored_visitor) / 2
overall_avg_points_allowed = (avg_points_allowed_home + avg_points_allowed_visitor) / 2

# 3. Win Rate
# Calculate the total number of wins for home and visitor teams
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
# head_to_head = all_data.groupby(['Home', 'Visitor'])['HomeWon'].mean()

team_features = pd.DataFrame({
    'AvgPointsScored': overall_avg_points_scored,
    'AvgPointsAllowed': overall_avg_points_allowed,
    'WinRate': win_rate
})

# Reset the index of the team_features DataFrame and rename the index column to "Team".
team_features.reset_index(inplace=True)
team_features.rename(columns={'Home': 'Team'}, inplace=True)


# Calculate defensive features for each NFL team.

# 1. Average points defended:
# This metric is the same as AvgPointsAllowed, which is already computed

# 2. Average conceded plays:
# A play is considered successful for the offense if it results in a touchdown or doesn't result in a turnover.
# Create a new column 'SuccessfulPlay' in the all_data DataFrame to represent this.
all_data['SuccessfulPlay'] = all_data['IsTouchdown'] | (~all_data['IsInterception'] & ~all_data['IsFumble'])

# Calculate the average rate of successful plays conceded when playing at home.
avg_conceded_plays_home = all_data.groupby('Home')['SuccessfulPlay'].mean()

# Calculate the average rate of successful plays conceded when playing as a visitor.
avg_conceded_plays_visitor = all_data.groupby('Visitor')['SuccessfulPlay'].mean()

# Calculate the overall average rate of successful plays conceded for each team.
overall_avg_conceded_plays = (avg_conceded_plays_home + avg_conceded_plays_visitor) / 2

# 3. Average forced turnovers:
# Create a new column 'Turnover' that indicates if a play resulted in a turnover (either interception or fumble).
all_data['Turnover'] = all_data['IsInterception'] | all_data['IsFumble']

# Calculate the average rate of turnovers forced when playing at home.
avg_forced_turnovers_home = all_data.groupby('Home')['Turnover'].mean()

# Calculate the average rate of turnovers forced when playing as a visitor.
avg_forced_turnovers_visitor = all_data.groupby('Visitor')['Turnover'].mean()

# Calculate the overall average rate of turnovers forced for each team.
overall_avg_forced_turnovers = (avg_forced_turnovers_home + avg_forced_turnovers_visitor) / 2

# Create a new DataFrame to store the defensive features for each team.
team_features_defensive = pd.DataFrame({
    'Team': team_features['Team'].values,
    'AvgPointsDefended': team_features['AvgPointsAllowed'].values,
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

# 3. Average pass completion allowed rate
avg_pass_completion_allowed_rate_home = all_data.groupby('Home').apply(lambda x: 1 - x['IsIncomplete'].mean())
avg_pass_completion_allowed_rate_visitor = all_data.groupby('Visitor').apply(lambda x: 1 - x['IsIncomplete'].mean())
overall_avg_pass_completion_allowed_rate = (avg_pass_completion_allowed_rate_home + avg_pass_completion_allowed_rate_visitor) / 2

# 4. Average touchdowns allowed per game
avg_touchdowns_allowed_per_game_home = all_data.groupby(['SeasonYear', 'Home'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Home']).size()
avg_touchdowns_allowed_per_game_visitor = all_data.groupby(['SeasonYear', 'Visitor'])['IsTouchdown'].sum() / all_data.groupby(['SeasonYear', 'Visitor']).size()
overall_avg_touchdowns_allowed_per_game = (avg_touchdowns_allowed_per_game_home + avg_touchdowns_allowed_per_game_visitor).groupby(level=1).mean()

# 5. Average rush success allowed rate
avg_rush_success_allowed_rate_home = all_data.groupby('Home').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
avg_rush_success_allowed_rate_visitor = all_data.groupby('Visitor').apply(lambda x: x['Yards'][x['IsRush'] == 1].mean())
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

week1_df = schedule_2025_df[schedule_2025_df['Week'] == 1]
week1_df = week1_df[['Home', 'Visitor']]
week1_df['Home'] = week1_df['Home'].str.lstrip('@').str.strip()

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

week1_df['Home'] = week1_df['Home'].map(city_abbr)
week1_df['Visitor'] = week1_df['Visitor'].map(city_abbr)

upcoming_encoded_home = week1_df.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
upcoming_encoded_both = upcoming_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')


# Calculate the difference in features as this might be a more predictive representation
for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
            'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
            'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
    upcoming_encoded_both[f'Diff_{col}'] = upcoming_encoded_both[f'{col}_Home'] - upcoming_encoded_both[f'{col}_Visitor']

# Selecting only the difference columns and the teams for clarity
upcoming_encoded_final = upcoming_encoded_both[['Home', 'Visitor'] + [col for col in upcoming_encoded_both.columns if 'Diff_' in col]]



# Prepare training data

# Merge play-by-play data with team features for home teams
training_encoded_home = all_data.merge(team_features_complete, left_on='Home', right_on='Team', how='left')
# Merge the result with team features for visitor teams
training_encoded_both = training_encoded_home.merge(team_features_complete, left_on='Visitor', right_on='Team', suffixes=('_Home', '_Visitor'), how='left')

# Calculate the difference in features
for col in ['AvgPointsScored', 'AvgPointsAllowed', 'WinRate', 'AvgPointsDefended', 'AvgConcededPlays', 'AvgForcedTurnovers',
            'AvgYardsPerPlay', 'AvgYardsPerGame', 'AvgPassCompletionRate', 'AvgTouchdownsPerGame', 'AvgRushSuccessRate',
            'AvgYardsAllowedPerPlay', 'AvgYardsAllowedPerGame', 'AvgPassCompletionAllowedRate', 'AvgTouchdownsAllowedPerGame', 'AvgRushSuccessAllowedRate']:
    training_encoded_both[f'Diff_{col}'] = training_encoded_both[f'{col}_Home'] - training_encoded_both[f'{col}_Visitor']

# Filtering out the required columns
# Feature matrix
X_train = training_encoded_both[[col for col in training_encoded_both.columns if 'Diff_' in col]]

# Target vectors
y_spread = all_data['Spread']
y_total = all_data['Total']

spread_model = LinearRegression()
total_model = LinearRegression()

spread_model.fit(X_train, y_spread)
total_model.fit(X_train, y_total)

X_upcoming = upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]]

predicted_spreads = spread_model.predict(X_upcoming)
predicted_totals = total_model.predict(X_upcoming)

upcoming_encoded_final['PredictedSpread'] = predicted_spreads
upcoming_encoded_final['PredictedTotal'] = predicted_totals

final_predictions = upcoming_encoded_final[['Home', 'Visitor', 'PredictedSpread', 'PredictedTotal']]
final_predictions.to_csv("csv_folder/week1_spread_total_predictions.csv", index=False)

# # Initialize the logistic regression model
# logreg = LogisticRegression(max_iter=1000)

# # Evaluate the model's performance using cross-validation
# cross_val_scores = cross_val_score(logreg, training_data, training_labels, cv=5)

# cross_val_scores_mean = cross_val_scores.mean()

# print(cross_val_scores_mean)

# logreg.fit(training_data, training_labels)

# upcoming_game_probabilities = logreg.predict_proba(upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]])

# # Extract the probability that the home team will win (second column of the result)
# upcoming_game_prob_home_win = upcoming_game_probabilities[:, 1]

# # Add the predictions to the upcoming games dataframe
# upcoming_encoded_final['HomeWinProbability'] = upcoming_game_prob_home_win

# # Sort by the probability of the home team winning for better visualization
# upcoming_predictions = upcoming_encoded_final[['Home', 'Visitor', 'HomeWinProbability']].sort_values(by='HomeWinProbability', ascending=False)

# #joblib.dump(logreg, "nfl_model.pkl")

# upcoming_predictions.to_csv("csv_folder/week1_predictions.csv", index=False)

# Merge with model predictions and actual results
# --- PREDICT WEEK 1 SPREAD/TOTAL ---

X_upcoming = upcoming_encoded_final[[col for col in upcoming_encoded_final.columns if 'Diff_' in col]]

predicted_spreads = spread_model.predict(X_upcoming)
predicted_totals = total_model.predict(X_upcoming)

# Fix SettingWithCopyWarning
upcoming_encoded_final = upcoming_encoded_final.copy()
upcoming_encoded_final.loc[:, 'PredictedSpread'] = predicted_spreads
upcoming_encoded_final.loc[:, 'PredictedTotal'] = predicted_totals

final_predictions = upcoming_encoded_final[['Home', 'Visitor', 'PredictedSpread', 'PredictedTotal']]
final_predictions.to_csv("csv_folder/week1_spread_total_predictions.csv", index=False)

# --- BACKTESTING SECTION ---

# Merge with training data and predictions
merged_results = training_encoded_both.copy()
merged_results['ActualSpread'] = all_data['Spread'].values
merged_results['ActualTotal'] = all_data['Total'].values
merged_results['PredictedSpread'] = spread_model.predict(X_train)
merged_results['PredictedTotal'] = total_model.predict(X_train)
merged_results['HomeTeam'] = training_encoded_both['Home'].values
merged_results['VisitorTeam'] = training_encoded_both['Visitor'].values

# Merge in Pinnacle odds
merged_with_vegas = merged_results.merge(
    pinnacle_odds_df, left_on=['HomeTeam', 'VisitorTeam'], right_on=['Home', 'Visitor'], how='left'
)

# Drop games missing Pinnacle odds
merged_with_vegas = merged_with_vegas.dropna(subset=['VegasSpread', 'VegasTotal'])

# MAE: Model vs Actual
from sklearn.metrics import mean_absolute_error

mae_model_spread = mean_absolute_error(merged_with_vegas['ActualSpread'], merged_with_vegas['PredictedSpread'])
mae_model_total = mean_absolute_error(merged_with_vegas['ActualTotal'], merged_with_vegas['PredictedTotal'])

# MAE: Vegas vs Actual
mae_vegas_spread = mean_absolute_error(merged_with_vegas['ActualSpread'], merged_with_vegas['VegasSpread'])
mae_vegas_total = mean_absolute_error(merged_with_vegas['ActualTotal'], merged_with_vegas['VegasTotal'])

print(f"MAE Spread - Your Model: {mae_model_spread:.2f}, Vegas: {mae_vegas_spread:.2f}")
print(f"MAE Total  - Your Model: {mae_model_total:.2f}, Vegas: {mae_vegas_total:.2f}")

# --- SIMULATED BETTING STRATEGY ---

threshold = 1.5  # points of disagreement to justify betting
merged_with_vegas['BetDiff'] = merged_with_vegas['PredictedSpread'] - merged_with_vegas['VegasSpread']
merged_with_vegas['BetOn'] = merged_with_vegas['BetDiff'].apply(
    lambda x: 'Home' if x < -threshold else ('Visitor' if x > threshold else 'NoBet')
)

# Who actually covered the spread
merged_with_vegas['Covered'] = merged_with_vegas.apply(
    lambda row: 'Home' if row['ActualSpread'] < row['VegasSpread'] else 'Visitor', axis=1
)

# Accuracy and ROI
bets = merged_with_vegas[merged_with_vegas['BetOn'] != 'NoBet']
accuracy = (bets['BetOn'] == bets['Covered']).mean()
print(f"Betting Accuracy (model vs Vegas): {accuracy:.2%} on {len(bets)} bets")

# Optional: ROI simulation
unit = 100
bets['Profit'] = bets['BetOn'] == bets['Covered']
bets['Profit'] = bets['Profit'].apply(lambda x: unit * 0.91 if x else -unit)
roi = bets['Profit'].sum() / (len(bets) * unit) if len(bets) > 0 else 0
print(f"ROI: {roi:.2%}")

# Export for review
merged_with_vegas.to_csv("csv_folder/backtest_model_vs_vegas.csv", index=False)
