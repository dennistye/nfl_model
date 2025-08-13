from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load data at startup
#team_features = pd.read_csv('team_features_complete.csv')
# week1_predictions = pd.read_csv("csv_folder/week1_predictions.csv")
# week1_spred_total_predictions = pd.read_csv("csv_folder/week1_spread_total_predictions.csv")
# vegas_odds = pd.read_csv("csv_folder/Pinnacle_odds.csv")

# merged_df = pd.merge(week1_spred_total_predictions, week1_predictions, on=['Home', 'Visitor'], how='inner')

# merged_df = pd.merge(merged_df, vegas_odds, on=['Home', 'Visitor'], how='inner')



prediction_df = pd.read_csv("csv_folder/week1_predictions/week1_predictions_linear_reg.csv")

# def prob_to_moneyline(prob):
#     if prob == 0:
#         return float('inf') #infinite odds
#     elif prob == 1:
#         return -float('inf') #certain win
#     elif prob > 0.5:
#         return round(-100 * (prob / (1-prob)))
#     else:
#         return round(100 * ((1-prob) / prob))


def convert_spread_to_reg(spread):
    if spread > 0:
        spread = spread * -1
    elif spread < 0:
        spread = spread * -1
    return spread



prediction_df['HomeSpread'] = prediction_df['PredictedSpread'].apply(convert_spread_to_reg)
prediction_df['HomeVegasSpread'] = prediction_df['VegasSpread'].apply(convert_spread_to_reg)
prediction_df['DiffSpread'] = prediction_df['HomeSpread'] - prediction_df['HomeVegasSpread']

prediction_df['VisitorSpread'] = prediction_df['HomeSpread'].apply(lambda x: x*-1)
prediction_df['VisitorVegasSpread'] = prediction_df['HomeVegasSpread'].apply(lambda x: x*-1)
prediction_df['DiffVisitorSpread'] = prediction_df['VisitorSpread'] - prediction_df['VisitorVegasSpread']

prediction_df['DiffTotal'] = prediction_df['PredictedTotal'] - prediction_df['VegasTotal']

prediction_df['DiffSpread'] = prediction_df['DiffSpread'].apply(lambda x: x*-1 if x < 0 else x)
prediction_df['DiffTotal'] = prediction_df['DiffTotal'].apply(lambda x: x*-1 if x < 0 else x)

prediction_df = prediction_df.drop(columns=['PredictedSpread', 'VegasSpread']) 

# print(prediction_df)

@app.route('/')
def index():
    # Show list of week 1 matchups
    games = prediction_df[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('index.html', games=games)

@app.route('/best_value')
def predictions():
    return render_template('best_value.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    home = request.form.get('home_team')
    away = request.form.get('away_team')

    # Search for the exact matchup
    match = prediction_df[
        (prediction_df['Home'] == home) & (prediction_df['Visitor'] == away)
    ]

    home_spread = float(match.iloc[0]['HomeSpread'])
    visitor_spread = float(match.iloc[0]['VisitorSpread'])
    total = float(match.iloc[0]['PredictedTotal'])
    vegas_home_spread = float(match.iloc[0]['HomeVegasSpread'])
    vegas_visitor_spread = float(match.iloc[0]['VisitorVegasSpread'])
    vegas_total = float(match.iloc[0]['VegasTotal'])
    diff_spread = float(match.iloc[0]['DiffSpread'])
    diff_visitor_spread = float(match.iloc[0]['DiffVisitorSpread'])
    diff_total = float(match.iloc[0]['DiffTotal'])

    return jsonify({
        'home_team': home,
        'away_team': away,
        'home_spread': round(home_spread, 3),
        'visitor_spread': round(visitor_spread, 3),
        'predicted_total': round(total, 3),
        'vegas_home_spread': round(vegas_home_spread, 3),
        'vegas_visitor_spread': round(vegas_visitor_spread, 3),
        'vegas_total': round(vegas_total, 3),
        'diff_spread': round(diff_spread, 3),
        'diff_visitor_spread': round(diff_visitor_spread, 3),
        'diff_total': round(diff_total, 3)
    })



if __name__ == '__main__':
    app.run(debug=True)
