
import requests
from bs4 import BeautifulSoup
import sqlite3

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect("nfl.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    position TEXT,
    team_id INTEGER,
    number INTEGER,
    role TEXT, 
    side TEXT, 
    draft_year INTEGER,
    draft_round INTEGER,
    acquisition TEXT,
    FOREIGN KEY (team_id) REFERENCES teams(id)
)
""")

teams = [
    ("Arizona Cardinals", "ARZ"),
    ("Atlanta Falcons", "ATL"),
    ("Baltimore Ravens", "BAL"),
    ("Buffalo Bills", "BUF"),
    ("Carolina Panthers", "CAR"),
    ("Chicago Bears", "CHI"),
    ("Cincinnati Bengals", "CIN"),
    ("Cleveland Browns", "CLE"),
    ("Dallas Cowboys", "DAL"),
    ("Denver Broncos", "DEN"),
    ("Detroit Lions", "DET"),
    ("Green Bay Packers", "GB"),
    ("Houston Texans", "HOU"),
    ("Indianapolis Colts", "IND"),
    ("Jacksonville Jaguars", "JAX"),
    ("Kansas City Chiefs", "KC"),
    ("Las Vegas Raiders", "LV"),
    ("Los Angeles Chargers", "LAC"),
    ("Los Angeles Rams", "LAR"),
    ("Miami Dolphins", "MIA"),
    ("Minnesota Vikings", "MIN"),
    ("New England Patriots", "NE"),
    ("New Orleans Saints", "NO"),
    ("New York Giants", "NYG"),
    ("New York Jets", "NYJ"),
    ("Philadelphia Eagles", "PHI"),
    ("Pittsburgh Steelers", "PIT"),
    ("San Francisco 49ers", "SF"),
    ("Seattle Seahawks", "SEA"),
    ("Tampa Bay Buccaneers", "TB"),
    ("Tennessee Titans", "TEN"),
    ("Washington Commanders", "WAS")
]

cursor.executemany("INSERT INTO teams (name, abbreviation) VALUES (?, ?)", teams)
conn.commit()


for idx, (_, acronym) in enumerate(teams):

    url = f"https://www.ourlads.com/nfldepthcharts/depthchart/{acronym}"

    team_id = idx+1  # Replace with your team identifier
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    # Map color class to acquisition type
    def parse_draft_info(player_name):
        if not player_name:
            return "", None, None
        parts = player_name.rsplit(" ", 1)
        name = parts[0]
        draft_year, draft_round = None, None
        if len(parts) == 2 and "/" in parts[1]:
            year_round = parts[1].split("/")
            try:
                draft_year = int(year_round[0])
            except ValueError:
                draft_year = None
            try:
                draft_round = int(year_round[1])
            except ValueError:
                draft_round = None
        return name, draft_year, draft_round

    color_map = {
        "lc_gold": "Acquired via Trade or FA in 2025",
        "lc_purple": "2025 Rookie Draft Pick",
        "lc_aqua": "2025 UDFA",
        "lc_red": "Injured/Inactive",
        "": "Unknown"
    }
        

    def parse_table(table_id, group_type):
        table_body = soup.find("tbody", id=table_id)
        players = []
        if table_body is None:
            print("No reserve/IR table found — skipping")
        else:
            for row in table_body.find_all("tr"):
                pos = row.find("td").text.strip()  # position
                tds = row.find_all("td")[1:]  # skip Pos column
                for i in range(0, len(tds), 2):
                    num_td = tds[i]
                    player_td = tds[i+1]
                    number = num_td.text.strip() or None
                    player_name = player_td.text.strip() or ""
                    name, draft_year, draft_round = parse_draft_info(player_name)
                    if not name:  # Skip blank names
                        continue
                    if player_td.a:
                        color_class = player_td.a.get("class")[0] if player_td.a.get("class") else ""
                        acquired = color_map.get(color_class, "Unknown")
                    players.append((
                        name,            
                        pos,             
                        team_id,         
                        number,          
                        f"{i//2 + 1} string",  
                        group_type,      
                        draft_year,      
                        draft_round,     
                        acquired         
                    ))
        return players

    print(soup.find("tbody", id="ctl00_phContent_dcTBody"))


    offense_players = parse_table("ctl00_phContent_dcTBody", "offense")
    # Parse defense
    defense_players = parse_table("ctl00_phContent_dcTBody2", "defense")
    # Parse special teams
    special_players = parse_table("ctl00_phContent_dcTBody3", "special")
    # Parse reserves/IR
    reserve_players = parse_table("ctl00_phContent_dcTBody4", "reserve")

    # Combine all
    all_players = offense_players + defense_players + special_players + reserve_players


    cursor.executemany("""
        INSERT INTO players (name, position, team_id, number, role, side, draft_year, draft_round, acquisition)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, all_players)
    conn.commit()

    # List all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", tables)

    # Show contents of each table
    for table_name, in tables:
        print(f"\nTable: {table_name}")
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()
        for row in rows:
            print(row)

conn.close()


