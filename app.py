from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load data at startup
#team_features = pd.read_csv('team_features_complete.csv')
week1_predictions = pd.read_csv("week1_predictions.csv")


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

    if match.empty:
        return jsonify({'error': 'Matchup not found'}), 404

    prob = float(match.iloc[0]['HomeWinProbability'])

    return jsonify({
        'home_team': home,
        'away_team': away,
        'probability_home_win': round(prob * 100, 2)
    })


# @app.route('/predictions')
# def show_predictions():
#     try:
#         predictions = week1_predictions.to_dict(orient='records')
#         return render_template('predictions.html', predictions=predictions)
#     except Exception as e:
#         return f"Error loading predictions: {e}", 500


if __name__ == '__main__':
    app.run(debug=True)
