from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import sqlite3
from datetime import date
import json
from datetime import datetime
from current_nfl_week.current_nfl_week import current_week

app = Flask(__name__)

prediction_df = pd.read_csv("csv_folder/complete_weights.csv")
DB_PATH = "nfl.db"


@app.route('/')
def index():
    games = prediction_df[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('index.html', games=games)

@app.route('/api/')
def api_index():
    odds = prediction_df.to_dict(orient='records')

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM props;"
    props_df = pd.read_sql_query(query, conn)
    conn.close()

    # Convert DataFrame → list of dictionaries
    props = props_df.to_dict(orient='records')

    response = {
        "odds": odds,
        "props": props
    }
    return jsonify(response)
   
@app.route('/matchups')
def matchups():
    games = prediction_df[['Home', 'Visitor']].to_dict(orient='records')
    return render_template('matchups.html', games=games)

@app.route('/api/best_odds')
def best_odds():
   odds = prediction_df.to_dict(orient='records')
   return jsonify(odds)

@app.route('/best_odds')
def best_odds_page():
    return render_template('best_odds.html')

@app.route('/api/best_props')
def best_props():
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM props;"
    props = pd.read_sql_query(query, conn)
    # print(props)
    conn.close()
    return jsonify(json.loads(props.to_json(orient="records")))

@app.route('/best_props')
def props():
    return render_template('best_props.html')

@app.route('/player_stats')
def stats():
    return render_template('player_stats.html')

def get_player_stats(player_name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM players WHERE name = ?", (player_name,))
    row = cursor.fetchone()   # get the first (and likely only) result
    if not row:
        conn.close()
    player_id = row[0]
    query = "SELECT * FROM player_stats WHERE player_id = ?"
    stats = pd.read_sql_query(query, conn, params=(player_id,))
    conn.close()
    return stats

@app.route('/player_stats', methods=["GET", "POST"])
def player_stats():
    stats = None
    if request.method == "POST":
        player_name = request.form.get("player_name")  # Get input from search bar
        if player_name:
            stats = get_player_stats(player_name)
            stats = stats.to_dict(orient="records")  # convert to list of dicts for template
    return render_template("player_stats.html", stats=stats)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/predict', methods=['POST'])
def predict():
    home = request.form.get('home_team')
    away = request.form.get('away_team')

    def get_odds(home, away, x):
        conn = sqlite3.connect("nfl.db")
        cursor = conn.cursor()
        if(x == "HomeVegasSpread"):
                cursor.execute("""
                    SELECT home_spread
                    FROM odds
                    WHERE home_team = ? AND visitor_team = ?
                """, (home, away))
                odds = cursor.fetchall()
                if odds:
                    odds = odds[0][0]
        elif(x == "VisitorVegasSpread"):
                cursor.execute("""
                    SELECT visitor_spread
                    FROM odds
                    WHERE home_team = ? AND visitor_team = ?
                """, (home, away))
                odds = cursor.fetchall()
                if odds:
                    odds = odds[0][0]
        elif(x == "VegasTotal"):
                cursor.execute("""
                    SELECT o_total
                    FROM odds
                    WHERE home_team = ? AND visitor_team = ?
                """, (home, away))
                odds = cursor.fetchall()
                if odds:
                    odds = odds[0][0]
                    odds = odds.lstrip("o")
        return odds
    
    def get_date_time_locaiton(home, away, x):

        closest_week = current_week()
        conn = sqlite3.connect("nfl.db")
        cursor = conn.cursor()
        if(x == "date"):
                cursor.execute("""
                    SELECT Date
                    FROM nfl2025schedule
                    WHERE Week = ? AND Home = ? AND Visitor = ?
                """, (int(closest_week), home, away))
                dtl = cursor.fetchall()
                if dtl:
                    dtl = dtl[0][0]
        elif(x == "time"):
                cursor.execute("""
                    SELECT Time
                    FROM nfl2025schedule
                    WHERE Week = ? AND Home = ? AND Visitor = ?
                """, (int(closest_week), home, away))
                dtl = cursor.fetchall()
                if dtl:
                    dtl = dtl[0][0]
        elif(x == "location"):
                cursor.execute("""
                    SELECT Location
                    FROM nfl2025schedule
                    WHERE Week = ? AND Home = ? AND Visitor = ?
                """, (int(closest_week), home, away))
                dtl = cursor.fetchall()
                if dtl:
                    dtl = dtl[0][0]
        return dtl

    # Search for the exact matchup
    match = prediction_df[
        (prediction_df['Home'] == home) & (prediction_df['Visitor'] == away)
    ]

    home_spread = float(match.iloc[0]['HomeSpread'])
    visitor_spread = float(match.iloc[0]['VisitorSpread'])
    total = float(match.iloc[0]['PredictedTotal'])
    # print(total)
    vegas_home_spread = get_odds(home, away, 'HomeVegasSpread')
    vegas_visitor_spread = get_odds(home, away, 'VisitorVegasSpread')
    vegas_total = get_odds(home, away, 'VegasTotal')

    date = get_date_time_locaiton(home, away, "date")
    print(date)
    time = get_date_time_locaiton(home, away, "time")
    location = get_date_time_locaiton(home, away, "location")
    # diff_spread = float(match.iloc[0]['DiffSpread'])
    # diff_visitor_spread = float(match.iloc[0]['DiffVisitorSpread'])
    # diff_total = float(match.iloc[0]['DiffTotal'])

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
            SELECT id, player_name, position, side, role, player_picture, player_status, team_id, Number
            FROM players4
            WHERE team_id = ? AND role = '1_string'
        """, (team_id,))
        players = cursor.fetchall()
        # print(players)
        # print(team_id)
        players_with_extra = []

        for p in players:
            id = p[0]

            cursor.execute("SELECT Status, Date, Description FROM injuries WHERE ID = ?", (id,))
            injury_row = cursor.fetchone()

            if injury_row:
                injury_status, injury_date, injury_description = injury_row
            else:
                injury_status, injury_date, injury_description = None, None, None

            players_with_extra.append({
                "player_id": p[0],
                "name": p[1],
                "position": p[2],
                "side": p[3],
                "role": p[4],
                "headshot": p[5],
                "status": p[6],
                "team_id": p[7],
                "number": p[8],
                "injury": injury_status,
                "injury_date": injury_date,
                "injury_description": injury_description
            })

        conn.close()

        return players_with_extra

       

    home_starters = get_starters(home)
    visitor_starters = get_starters(away)


    date_obj = datetime.strptime(date, "%m/%d/%Y")

    # Get full day name
    day_name = date_obj.strftime("%A")

    
    

    return jsonify({
        'home_team': home,
        'away_team': away,
        'home_spread': round(home_spread, 3),
        'visitor_spread': round(visitor_spread, 3),
        'predicted_total': round(total, 3),
        'vegas_home_spread': vegas_home_spread,
        'vegas_visitor_spread': vegas_visitor_spread,
        'vegas_total': vegas_total,
        'day': day_name,
        'date': date,
        'time': time,
        'location': location,
        # 'diff_spread': round(diff_spread, 3),
        # 'diff_visitor_spread': round(diff_visitor_spread, 3),
        # 'diff_total': round(diff_total, 3),
        'home_starters': home_starters,
        'visitor_starters': visitor_starters
    })



if __name__ == '__main__':
    app.run(debug=True)
