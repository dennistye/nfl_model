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
    
# def adjust_spread(spread):
#     if spread < 1:
#         spread = spread * -1

#     return spread
    


merged_df['PredictedML'] = merged_df['HomeWinProbability'].apply(prob_to_moneyline)
merged_df['PredictedSpread'] = merged_df['PredictedSpread'].apply(lambda x: x*-1 if x < 1 else x)


#print(merged_df)

@app.route('/')
def index():
    # Show list of week 1 matchups
    games = merged_df[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('index.html', games=games)


@app.route('/predict', methods=['POST'])
def predict():
    home = request.form.get('home_team')
    away = request.form.get('away_team')

    # Search for the exact matchup
    match = merged_df[
        (merged_df['Home'] == home) & (merged_df['Visitor'] == away)
    ]

    prob = float(match.iloc[0]['HomeWinProbability'])
    spread = float(match.iloc[0]['PredictedSpread'])
    total = float(match.iloc[0]['PredictedTotal'])

    return jsonify({
        'home_team': home,
        'away_team': away,
        'probability_home_win': round(prob, 3),
        'predicted_spread': round(spread, 3),
        'predicted_total': round(total, 3)
    })



if __name__ == '__main__':
    app.run(debug=True)
