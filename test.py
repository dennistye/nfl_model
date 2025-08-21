import sqlite3


def get_odds(home = "PHI", away= "DAL", x = "HomeVegasSpread"):
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
        return odds


dam = get_odds()
print(dam)