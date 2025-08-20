from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sqlite3

app = Flask(__name__)

prediction_df = pd.read_csv("csv_folder/week1_predictions_linear_reg.csv")


@app.route('/')
def index():
    # Show list of week 1 matchups
    games = prediction_df[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('index.html', games=games)

@app.route('/best_value')
def predictions():
    # Read CSV file
    df = pd.read_csv("csv_folder/week1_predictions_linear_reg.csv")

    # Convert DataFrame to HTML table (Bootstrap-friendly)
    table_html = df.to_html(classes="table table-striped", index=False)
    return render_template('best_value.html', table_html=table_html)

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

    def get_starters(team_abbr):
        conn = sqlite3.connect("nfl.db")
        cursor = conn.cursor()

        if(team_abbr == "ARI"):
            team_abbr = "ARZ"
        elif(team_abbr == "LA"):
            team_abbr = "LAR"

        # Get the numeric team ID
        cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", (team_abbr,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return []  # Team not found
        team_id = row[0]

        # cursor.execute(f"SELECT * FROM players ")
        # rows = cursor.fetchall()
        # for row in rows:
        #     print(row)

        # Query defensive starters
        cursor.execute("""
            SELECT name, position, number, side, role, headshot, acquisition, team_id
            FROM players 
            WHERE team_id = ? AND role = '1 string'
        """, (team_id,))
        players = cursor.fetchall()
        print(players)
        print(team_id)
        conn.close()

        # Convert to dict for JSON
        return [{"name": p[0], "position": p[1], "number": p[2], "side": p[3], "role": p[4], "headshot": p[5], "acquisition": p[6], "team_id": p[7]} for p in players]

    home_starters = get_starters(home)
    visitor_starters = get_starters(away)
    

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
        'diff_total': round(diff_total, 3),
        'home_starters': home_starters,
        'visitor_starters': visitor_starters
    })



if __name__ == '__main__':
    app.run(debug=True)
