import requests
import sqlite3
import requests
from bs4 import BeautifulSoup

conn = sqlite3.connect("nfl.db")
cursor = conn.cursor()

nfl_abbr = [
    "ari",  # Arizona Cardinals
    "atl",  # Atlanta Falcons
    "bal",  # Baltimore Ravens
    "buf",  # Buffalo Bills
    "car",  # Carolina Panthers
    "chi",  # Chicago Bears
    "cin",  # Cincinnati Bengals
    "cle",  # Cleveland Browns
    "dal",  # Dallas Cowboys
    "den",  # Denver Broncos
    "det",  # Detroit Lions
    "gb",   # Green Bay Packers
    "hou",  # Houston Texans
    "ind",  # Indianapolis Colts
    "jax",  # Jacksonville Jaguars
    "kc",   # Kansas City Chiefs
    "lv",   # Las Vegas Raiders
    "lac",  # Los Angeles Chargers
    "lar",  # Los Angeles Rams
    "mia",  # Miami Dolphins
    "min",  # Minnesota Vikings
    "ne",   # New England Patriots
    "no",   # New Orleans Saints
    "nyg",  # New York Giants
    "nyj",  # New York Jets
    "phi",  # Philadelphia Eagles
    "pit",  # Pittsburgh Steelers
    "sf",   # San Francisco 49ers
    "sea",  # Seattle Seahawks
    "tb",   # Tampa Bay Buccaneers
    "ten",  # Tennessee Titans
    "wsh"   # Washington Commanders
]


for team in nfl_abbr:
    
    print(team)

    url = f"https://www.espn.com/nfl/team/roster/_/name/{team}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
   
    players = []

    # Loop through all player links
    for a_tag in soup.find_all("a", {"data-resource-id": "AthleteName"}):
        img_tag = a_tag.find("img")
        if img_tag:
            if img_tag.get("title") == "Tre Harris":
                 players.append({
                "name": "Tre' Harris",  # This gets the title attribute
                "headshot": img_tag.get("alt")  # This is the image URL
            })
            elif img_tag.get("title") == "Rakeem Nunez-Roches":    
                players.append({
                    "name": "Rakeem Nuñez-Roches",  # This gets the title attribute
                    "headshot": img_tag.get("alt")  # This is the image URL
                })
            elif img_tag.get("title") == "Jevon Holland":    
                players.append({
                    "name": "Jevón Holland",  # This gets the title attribute
                    "headshot": img_tag.get("alt")  # This is the image URL
                })
            elif img_tag.get("title") == "Cam Ward":    
                players.append({
                "name": "Cameron Ward",  # This gets the title attribute
                "headshot": img_tag.get("alt")  # This is the image URL
            })
            elif img_tag.get("title") == "Broderick Washington":    
                players.append({
                "name": "Broderick Jr.",  # This gets the title attribute
                "headshot": img_tag.get("alt")  # This is the image URL
            })

    for player in players:
         cursor.execute("""
            UPDATE players
            SET headshot = ?
            WHERE name = ?
            """, (player["headshot"], player["name"]))

conn.commit()
conn.close()