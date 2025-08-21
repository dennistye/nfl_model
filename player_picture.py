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


def standardize_name(name):
    name = name.strip()
    if ',' in name:
        last, first = name.split(',', 1)
    else:
        parts = name.split()
        if len(parts) >= 2:
            first, last = parts[0], parts[-1]
        else:
            return name.title()
    first = first.strip().title()
    last = last.strip().title()
    return f"{first} {last}"


for team in nfl_abbr:
    
    print(team)

    url = f"https://www.espn.com/nfl/team/roster/_/name/{team}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
    }

    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")
    #print(soup)

    players = []

    # Loop through all player links
    for a_tag in soup.find_all("a", {"data-resource-id": "AthleteName"}):
        img_tag = a_tag.find("img")
        if img_tag:
            players.append({
                "name": standardize_name(img_tag.get("title")),  # This gets the title attribute
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