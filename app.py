from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data at startup
#team_features = pd.read_csv('team_features_complete.csv')
week1_predictions = pd.read_csv("csv_folder/week1_predictions.csv")
week1_spred_total_predictions = pd.read_csv("csv_folder/week1_spread_total_predictions.csv")
vegas_odds = pd.read_csv("csv_folder/Pinnacle_odds.csv")

merged_df = pd.merge(week1_spred_total_predictions, week1_predictions, on=['Home', 'Visitor'], how='inner')

merged_df = pd.merge(merged_df, vegas_odds, on=['Home', 'Visitor'], how='inner')


def prob_to_moneyline(prob):
    if prob == 0:
        return float('inf') #infinite odds
    elif prob == 1:
        return -float('inf') #certain win
    elif prob > 0.5:
        return round(-100 * (prob / (1-prob)))
    else:
        return round(100 * ((1-prob) / prob))
    


merged_df['PredictedML'] = merged_df['HomeWinProbability'].apply(prob_to_moneyline)


print(merged_df)

@app.route('/')
def index():
    # Show list of week 1 matchups
    games = week1_predictions[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('index.html', games=games)


@app.route('/predict', methods=['POST'])
def predict():
    home = request.form.get('home_team')
    away = request.form.get('away_team')

    # Search for the exact matchup
    match = week1_predictions[
        (week1_predictions['Home'] == home) & (week1_predictions['Visitor'] == away)
    ]

    prob = float(match.iloc[0]['HomeWinProbability'])

    return jsonify({
        'home_team': home,
        'away_team': away,
        'probability_home_win': round(prob, 3)
    })



if __name__ == '__main__':
    app.run(debug=True)
