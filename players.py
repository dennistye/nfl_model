import sqlite3

# Connect to SQLite database (creates file if it doesn't exist)
conn = sqlite3.connect("nfl.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    abbreviation TEXT NOT NULL UNIQUE,
    logo_url TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    position TEXT,
    number INTEGER,
    team_id INTEGER,
    role TEXT,  -- 'starter' or 'backup'
    side TEXT,  -- 'offense' or 'defense'
    FOREIGN KEY (team_id) REFERENCES teams(id)
)
""")

teams = [
    ("Arizona Cardinals", "ARI"),
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
    ("Los Angeles Rams", "LA"),
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


# Add a team
cursor.executemany("INSERT INTO teams (name, abbreviation) VALUES (?, ?)", teams)
conn.commit()


# Example: Add Patrick Mahomes to Kansas City Chiefs

# 1. Get team_id for ARI
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("ARI",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Kyler Murray", "QB", team_id, "starter", "offense"),
    ("James Conner", "RB", team_id, "starter", "offense"),
    ("Marvin Harrison Jr.", "WR", team_id, "starter", "offense"),
    ("Michael Wilson", "WR", team_id, "starter", "offense"),
    ("Trey McBride", "TE", team_id, "starter", "offense"),
    ("Isaiah Adams", "OL", team_id, "starter", "offense"),
    ("Kelvin Beachum", "OL", team_id, "starter", "offense"),
    ("Jonah Williams", "OL", team_id, "starter", "offense"),
    ("Evan Brown", "OL", team_id, "starter", "offense"),
    ("Joshua Fryar", "OL", team_id, "starter", "offense"),
    
    # Backup Offense
    ("Jacoby Brissett", "QB", team_id, "backup", "offense"),
    ("Clayton Tune", "QB", team_id, "backup", "offense"),
    ("Trey Benson", "RB", team_id, "backup", "offense"),
    ("Emari Demercado", "RB", team_id, "backup", "offense"),
    ("DeeJay Dallas", "RB", team_id, "backup", "offense"),
    ("Michael Carter", "RB", team_id, "backup", "offense"),
    ("Zonovan Knight", "RB", team_id, "backup", "offense"),
    ("Kelly Akharaiyi", "WR", team_id, "backup", "offense"),
    ("Andre Baccellia", "WR", team_id, "backup", "offense"),
    ("Greg Dortch", "WR", team_id, "backup", "offense"),
    ("Simi Fehoko", "WR", team_id, "backup", "offense"),
    ("Bryson Green", "WR", team_id, "backup", "offense"),
    ("Tejhaun Palmer", "WR", team_id, "backup", "offense"),
    ("Quez Watkins", "WR", team_id, "backup", "offense"),
    ("Zay Jones", "WR", team_id, "backup", "offense"),
    ("Xavier Weaver", "WR", team_id, "backup", "offense"),
    ("Oscar Cardenas", "TE", team_id, "backup", "offense"),
    ("Josiah Deguara", "TE", team_id, "backup", "offense"),
    ("Elijah Higgins", "TE", team_id, "backup", "offense"),
    ("Tip Reiman", "TE", team_id, "backup", "offense"),
    ("Travis Vokolek", "TE", team_id, "backup", "offense"),
    ("Jake Curhan", "OL", team_id, "backup", "offense"),
    ("McClendon Curtis", "OL", team_id, "backup", "offense"),
    ("Hjalte Froholdt", "OL", team_id, "backup", "offense"),
    ("Jon Gaines II", "OL", team_id, "backup", "offense"),
    
    # Starting Defense
    ("Calais Campbell", "DL", team_id, "starter", "defense"),
    ("Justin Jones", "DL", team_id, "starter", "defense"),
    ("L. J. Collier", "DL", team_id, "starter", "defense"),
    ("Bilal Nichols", "DL", team_id, "starter", "defense"),
    ("Dante Stills", "DL", team_id, "starter", "defense"),
    ("Dalvin Tomlinson", "DL", team_id, "starter", "defense"),
    ("Josh Sweat", "LB", team_id, "starter", "defense"),
    ("Baron Browning", "LB", team_id, "starter", "defense"),
    ("Zaven Collins", "LB", team_id, "starter", "defense"),
    ("Akeem Davis-Gaither", "LB", team_id, "starter", "defense"),
    ("Ekow Boye-Doe", "CB", team_id, "starter", "defense"),
    ("Denzel Burke", "CB", team_id, "starter", "defense"),
    ("Kei'Trel Clark", "CB", team_id, "starter", "defense"),
    ("Steven Gilmore Jr.", "CB", team_id, "starter", "defense"),
    ("Budda Baker", "S", team_id, "starter", "defense"),
    ("Joey Blount", "S", team_id, "starter", "defense"),
    ("Jammie Robinson", "S", team_id, "starter", "defense"),
    
    # Backup Defense
    ("Jordan Burch", "DL", team_id, "backup", "defense"),
    ("Darius Robinson", "DL", team_id, "backup", "defense"),
    ("Anthony Goodlow", "DL", team_id, "backup", "defense"),
    ("Walter Nolen III", "DL", team_id, "backup", "defense"),
    ("Darius Stills", "DL", team_id, "backup", "defense"),
    ("Mack Wilson Sr.", "LB", team_id, "backup", "defense"),
    ("Elijah Jones", "CB", team_id, "backup", "defense"),
    ("Jaylon Jones", "CB", team_id, "backup", "defense"),
    ("Max Melton", "CB", team_id, "backup", "defense"),
    ("Sean Murphy-Bunting", "CB", team_id, "backup", "defense"),
    ("Garrett Williams", "CB", team_id, "backup", "defense"),
    ("Kitan Crawford", "S", team_id, "backup", "defense"),
    ("Jalen Thompson", "S", team_id, "backup", "defense"),
    ("Dadrion Taylor-Demerson", "CB", team_id, "backup", "defense"),
    ("Dadrion Taylor-Demerson", "S", team_id, "backup", "defense"),
    
    # Special Teams
    ("Blake Gillikin", "P", team_id, "starter", "special"),
    ("Chris Ryland", "K", team_id, "starter", "special"),
    ("Aaron Brewer", "LS", team_id, "starter", "special"),
]





cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("ATL",))
team_id = cursor.fetchone()[0]


players = [
    # Starting Offense
    ("Marcus Mariota", "QB", team_id, 8, "starter", "offense"),
    ("Bijan Robinson", "RB", team_id, 11, "starter", "offense"),
    ("Tyler Allgeier", "RB", team_id, 33, "starter", "offense"),
    ("Drake London", "WR", team_id, 16, "starter", "offense"),
    ("Olamide Zaccheaus", "WR", team_id, 14, "starter", "offense"),
    ("Kyle Pitts", "TE", team_id, 8, "starter", "offense"),
    ("Chris Lindstrom", "OL", team_id, 66, "starter", "offense"),
    ("Jake Matthews", "OL", team_id, 70, "starter", "offense"),
    ("Matt Hennessy", "OL", team_id, 65, "starter", "offense"),
    ("Kaleb McGary", "OL", team_id, 77, "starter", "offense"),
    ("Matthew Bergeron", "OL", team_id, 61, "starter", "offense"),

    # Backup Offense
    ("Desmond Ridder", "QB", team_id, 2, "backup", "offense"),
    ("Tyler Allgeier", "RB", team_id, 33, "backup", "offense"),
    ("Mike Davis", "RB", team_id, 34, "backup", "offense"),
    ("Rashod Bateman", "WR", team_id, 7, "backup", "offense"),
    ("AJ Terrell", "WR", team_id, 24, "backup", "offense"),
    ("Hayden Hurst", "TE", team_id, 81, "backup", "offense"),
    ("Matt Gono", "OL", team_id, 67, "backup", "offense"),
    ("Will Holden", "OL", team_id, 75, "backup", "offense"),

    # Starting Defense
    ("Grady Jarrett", "DL", team_id, 97, "starter", "defense"),
    ("Tyeler Davison", "DL", team_id, 90, "starter", "defense"),
    ("Davin Joseph", "DL", team_id, 91, "starter", "defense"),
    ("Foyesade Oluokun", "LB", team_id, 40, "starter", "defense"),
    ("Deion Jones", "LB", team_id, 45, "starter", "defense"),
    ("DeCody Minter", "LB", team_id, 55, "starter", "defense"),
    ("A.J. Terrell", "CB", team_id, 24, "starter", "defense"),
    ("Calvin Ridley", "CB", team_id, 18, "starter", "defense"),
    ("Richie Grant", "S", team_id, 36, "starter", "defense"),
    ("Jordan Miller", "S", team_id, 31, "starter", "defense"),
    ("Damarion Williams", "CB", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Bryan Cox Jr.", "DL", team_id, 92, "backup", "defense"),
    ("Jonathan Massaquoi", "DL", team_id, 93, "backup", "defense"),
    ("Deion Jones", "LB", team_id, 45, "backup", "defense"),
    ("Rasheem Green", "LB", team_id, 49, "backup", "defense"),
    ("Drew Sanders", "S", team_id, 30, "backup", "defense"),
    ("AJ Terrell", "CB", team_id, 24, "backup", "defense"),
    ("Tyson Campbell", "CB", team_id, 27, "backup", "defense"),

    # Special Teams
    ("Matt Bosher", "P", team_id, 6, "starter", "special"),
    ("Younghoe Koo", "K", team_id, 2, "starter", "special"),
    ("Josh Harris", "LS", team_id, 49, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("BUF",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Josh Allen", "QB", team_id, 17, "starter", "offense"),
    ("James Cook", "RB", team_id, 4, "starter", "offense"),
    ("Stefon Diggs", "WR", team_id, 14, "starter", "offense"),
    ("Gabriel Davis", "WR", team_id, 13, "starter", "offense"),
    ("Dawson Knox", "TE", team_id, 88, "starter", "offense"),
    ("Mitch Morse", "C", team_id, 60, "starter", "offense"),
    ("Dion Dawkins", "LT", team_id, 73, "starter", "offense"),
    ("Connor McGovern", "LG", team_id, 60, "starter", "offense"),
    ("Ryan Bates", "RG", team_id, 71, "starter", "offense"),
    ("Spencer Brown", "RT", team_id, 79, "starter", "offense"),

    # Backup Offense
    ("Kyle Allen", "QB", team_id, 8, "backup", "offense"),
    ("Latavius Murray", "RB", team_id, 28, "backup", "offense"),
    ("Damien Harris", "RB", team_id, 22, "backup", "offense"),
    ("Khalil Shakir", "WR", team_id, 10, "backup", "offense"),
    ("Deonte Harty", "WR", team_id, 11, "backup", "offense"),
    ("Dalton Kincaid", "TE", team_id, 86, "backup", "offense"),
    ("David Quessenberry", "OT", team_id, 77, "backup", "offense"),
    ("Greg Mance", "OG", team_id, 66, "backup", "offense"),

    # Starting Defense
    ("Von Miller", "OLB", team_id, 40, "starter", "defense"),
    ("Matt Milano", "ILB", team_id, 58, "starter", "defense"),
    ("Tremaine Edmunds", "ILB", team_id, 49, "starter", "defense"),
    ("Ed Oliver", "DT", team_id, 91, "starter", "defense"),
    ("DaQuan Jones", "DT", team_id, 92, "starter", "defense"),
    ("Greg Rousseau", "DE", team_id, 50, "starter", "defense"),
    ("Jordan Poyer", "S", team_id, 21, "starter", "defense"),
    ("Micah Hyde", "S", team_id, 23, "starter", "defense"),
    ("Taron Johnson", "CB", team_id, 24, "starter", "defense"),
    ("Kaiir Elam", "CB", team_id, 24, "starter", "defense"),

    # Backup Defense
    ("AJ Epenesa", "DE", team_id, 57, "backup", "defense"),
    ("Shaq Lawson", "DE", team_id, 90, "backup", "defense"),
    ("Tyrel Dodson", "ILB", team_id, 44, "backup", "defense"),
    ("Terrel Bernard", "ILB", team_id, 43, "backup", "defense"),
    ("Leonard Floyd", "OLB", team_id, 54, "backup", "defense"),
    ("Christian Benford", "CB", team_id, 47, "backup", "defense"),
    ("Dane Jackson", "CB", team_id, 30, "backup", "defense"),

    # Special Teams
    ("Tyler Bass", "K", team_id, 2, "starter", "special"),
    ("Sam Martin", "P", team_id, 6, "starter", "special"),
    ("Reid Ferguson", "LS", team_id, 69, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("MIA",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Tua Tagovailoa", "QB", team_id, 1, "starter", "offense"),
    ("Raheem Mostert", "RB", team_id, 31, "starter", "offense"),
    ("Tyreek Hill", "WR", team_id, 10, "starter", "offense"),
    ("Jaylen Waddle", "WR", team_id, 17, "starter", "offense"),
    ("Cedrick Wilson Jr.", "WR", team_id, 11, "starter", "offense"),
    ("Durham Smythe", "TE", team_id, 81, "starter", "offense"),
    ("Terron Armstead", "LT", team_id, 72, "starter", "offense"),
    ("Liam Eichenberg", "LG", team_id, 74, "starter", "offense"),
    ("Connor Williams", "C", team_id, 68, "starter", "offense"),
    ("Robert Hunt", "RG", team_id, 68, "starter", "offense"),
    ("Austin Jackson", "RT", team_id, 73, "starter", "offense"),

    # Backup Offense
    ("Mike White", "QB", team_id, 5, "backup", "offense"),
    ("Jeff Wilson Jr.", "RB", team_id, 23, "backup", "offense"),
    ("Salvon Ahmed", "RB", team_id, 26, "backup", "offense"),
    ("Braxton Berrios", "WR", team_id, 10, "backup", "offense"),
    ("River Cracraft", "WR", team_id, 84, "backup", "offense"),
    ("Hunter Long", "TE", team_id, 84, "backup", "offense"),
    ("Greg Little", "OT", team_id, 72, "backup", "offense"),
    ("Dan Feeney", "OG", team_id, 66, "backup", "offense"),

    # Starting Defense
    ("Jevon Holland", "S", team_id, 8, "starter", "defense"),
    ("Brandon Jones", "S", team_id, 29, "starter", "defense"),
    ("Xavien Howard", "CB", team_id, 25, "starter", "defense"),
    ("Jalen Ramsey", "CB", team_id, 5, "starter", "defense"),
    ("Bradley Chubb", "OLB", team_id, 55, "starter", "defense"),
    ("Jerome Baker", "ILB", team_id, 55, "starter", "defense"),
    ("David Long Jr.", "ILB", team_id, 51, "starter", "defense"),
    ("Christian Wilkins", "DT", team_id, 94, "starter", "defense"),
    ("Raekwon Davis", "DT", team_id, 98, "starter", "defense"),
    ("Emmanuel Ogbah", "DE", team_id, 91, "starter", "defense"),

    # Backup Defense
    ("Melvin Ingram", "OLB", team_id, 8, "backup", "defense"),
    ("Jaelan Phillips", "OLB", team_id, 15, "backup", "defense"),
    ("Channing Tindall", "ILB", team_id, 47, "backup", "defense"),
    ("Elijah Campbell", "CB", team_id, 30, "backup", "defense"),
    ("Keion Crossen", "CB", team_id, 35, "backup", "defense"),

    # Special Teams
    ("Jason Sanders", "K", team_id, 7, "starter", "special"),
    ("Thomas Morstead", "P", team_id, 4, "starter", "special"),
    ("Blake Ferguson", "LS", team_id, 43, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NE",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Mac Jones", "QB", team_id, 10, "starter", "offense"),
    ("Rhamondre Stevenson", "RB", team_id, 38, "starter", "offense"),
    ("JuJu Smith-Schuster", "WR", team_id, 7, "starter", "offense"),
    ("Kendrick Bourne", "WR", team_id, 84, "starter", "offense"),
    ("Hunter Henry", "TE", team_id, 85, "starter", "offense"),
    ("Trent Brown", "LT", team_id, 77, "starter", "offense"),
    ("Cole Strange", "LG", team_id, 69, "starter", "offense"),
    ("David Andrews", "C", team_id, 60, "starter", "offense"),
    ("Mike Onwenu", "RG", team_id, 71, "starter", "offense"),
    ("Calvin Anderson", "RT", team_id, 77, "starter", "offense"),

    # Backup Offense
    ("Bailey Zappe", "QB", team_id, 4, "backup", "offense"),
    ("Pierre Strong Jr.", "RB", team_id, 20, "backup", "offense"),
    ("Ty Montgomery", "RB", team_id, 88, "backup", "offense"),
    ("Demario Douglas", "WR", team_id, 13, "backup", "offense"),
    ("Tae Crowder", "WR", team_id, 85, "backup", "offense"),
    ("Mike Gesicki", "TE", team_id, 88, "backup", "offense"),
    # Starting Defense
    ("Matthew Judon", "OLB", team_id, 9, "starter", "defense"),
    ("Christian Barmore", "DT", team_id, 99, "starter", "defense"),
    ("Deatrich Wise Jr.", "DE", team_id, 92, "starter", "defense"),
    ("David Andrews", "ILB", team_id, 60, "starter", "defense"),
    ("Ja'Whaun Bentley", "ILB", team_id, 54, "starter", "defense"),
    ("J.C. Jackson", "CB", team_id, 27, "starter", "defense"),
    ("Tyquan Thornton", "CB", team_id, 83, "starter", "defense"),
    ("Kyle Dugger", "S", team_id, 23, "starter", "defense"),
    ("Devin McCourty", "S", team_id, 32, "starter", "defense"),
    ("Christian Gonzalez", "CB", team_id, 24, "starter", "defense"),

    # Backup Defense
    ("Anfernee Jennings", "OLB", team_id, 51, "backup", "defense"),
    ("Henry Anderson", "DE", team_id, 90, "backup", "defense"),
    ("Chase Winovich", "OLB", team_id, 50, "backup", "defense"),
    ("Marcus Jones", "CB", team_id, 24, "backup", "defense"),
    ("Adrian Phillips", "S", team_id, 27, "backup", "defense"),

    # Special Teams
    ("Nick Folk", "K", team_id, 5, "starter", "special"),
    ("Jake Bailey", "P", team_id, 6, "starter", "special"),
    ("Joe Cardona", "LS", team_id, 49, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NYJ",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Aaron Rodgers", "QB", team_id, 8, "starter", "offense"),
    ("Breece Hall", "RB", team_id, 22, "starter", "offense"),
    ("Zay Flowers", "WR", team_id, 11, "starter", "offense"),
    ("Garrett Wilson", "WR", team_id, 7, "starter", "offense"),
    ("C.J. Uzomah", "TE", team_id, 87, "starter", "offense"),
    ("Morgan Moses", "LT", team_id, 70, "starter", "offense"),
    ("Joe Thuney", "LG", team_id, 62, "starter", "offense"),
    ("Connor McGovern", "C", team_id, 60, "starter", "offense"),
    ("Laken Tomlinson", "RG", team_id, 65, "starter", "offense"),
    ("Max Mitchell", "RT", team_id, 79, "starter", "offense"),

    # Backup Offense
    ("Mike White", "QB", team_id, 5, "backup", "offense"),
    ("Michael Carter", "RB", team_id, 27, "backup", "offense"),
    ("Benny Snell", "RB", team_id, 34, "backup", "offense"),
    ("Elijah Moore", "WR", team_id, 15, "backup", "offense"),
    ("Denzel Mims", "WR", team_id, 10, "backup", "offense"),
    ("Tyler Kroft", "TE", team_id, 81, "backup", "offense"),
    ("Cesar Ruiz", "OL", team_id, 67, "backup", "offense"),

    # Starting Defense
    ("Quinnen Williams", "DT", team_id, 95, "starter", "defense"),
    ("John Franklin-Myers", "DE", team_id, 99, "starter", "defense"),
    ("Jermaine Johnson II", "OLB", team_id, 56, "starter", "defense"),
    ("C.J. Mosley", "ILB", team_id, 41, "starter", "defense"),
    ("Quincy Williams", "ILB", team_id, 50, "starter", "defense"),
    ("Sauce Gardner", "CB", team_id, 2, "starter", "defense"),
    ("Javon Holland", "S", team_id, 29, "starter", "defense"),
    ("Jamal Adams", "S", team_id, 33, "starter", "defense"),
    ("Jason Pinnock", "CB", team_id, 35, "starter", "defense"),
    ("Michael Carter II", "CB", team_id, 24, "starter", "defense"),

    # Backup Defense
    ("Jonathan Marshall", "DE", team_id, 97, "backup", "defense"),
    ("Hassan Ridgeway", "DT", team_id, 93, "backup", "defense"),
    ("Ashtyn Davis", "S", team_id, 29, "backup", "defense"),
    ("Brandin Echols", "CB", team_id, 37, "backup", "defense"),

    # Special Teams
    ("Brandon Shell", "LS", team_id, 49, "starter", "special"),
    ("Lachlan Edwards", "P", team_id, 6, "starter", "special"),
    ("Greg Zuerlein", "K", team_id, 9, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("BAL",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Lamar Jackson", "QB", team_id, 8, "starter", "offense"),
    ("Gus Edwards", "RB", team_id, 44, "starter", "offense"),
    ("J.K. Dobbins", "RB", team_id, 3, "starter", "offense"),
    ("Rashod Bateman", "WR", team_id, 11, "starter", "offense"),
    ("Zay Flowers", "WR", team_id, 6, "starter", "offense"),
    ("Mark Andrews", "TE", team_id, 89, "starter", "offense"),
    ("Tyre Phillips", "OL", team_id, 71, "starter", "offense"),
    ("Bradley Bozeman", "OL", team_id, 60, "starter", "offense"),
    ("Ronnie Stanley", "OL", team_id, 79, "starter", "offense"),
    ("Tyre Phillips", "OL", team_id, 71, "starter", "offense"),
    ("Bradley Bozeman", "OL", team_id, 60, "starter", "offense"),

    # Backup Offense
    ("Anthony Brown", "QB", team_id, 2, "backup", "offense"),
    ("Ty'Son Williams", "RB", team_id, 35, "backup", "offense"),
    ("Latavius Murray", "RB", team_id, 28, "backup", "offense"),
    ("Devin Duvernay", "WR", team_id, 11, "backup", "offense"),
    ("Trayveon Williams", "RB", team_id, 39, "backup", "offense"),
    ("Isaiah Likely", "TE", team_id, 86, "backup", "offense"),
    ("Bradley Bozeman", "OL", team_id, 60, "backup", "offense"),
    ("Ben Cleveland", "OL", team_id, 73, "backup", "offense"),

    # Starting Defense
    ("Calais Campbell", "DE", team_id, 93, "starter", "defense"),
    ("Derrick Brown", "DT", team_id, 92, "starter", "defense"),
    ("Marlon Humphrey", "CB", team_id, 44, "starter", "defense"),
    ("Marcus Williams", "S", team_id, 30, "starter", "defense"),
    ("Roquan Smith", "LB", team_id, 55, "starter", "defense"),
    ("Tyus Bowser", "OLB", team_id, 50, "starter", "defense"),
    ("Odafe Oweh", "OLB", team_id, 94, "starter", "defense"),
    ("Chuck Clark", "S", team_id, 35, "starter", "defense"),
    ("Anthony Averett", "CB", team_id, 28, "starter", "defense"),
    ("Daylon Mack", "DT", team_id, 97, "starter", "defense"),

    # Backup Defense
    ("Caleb Farley", "CB", team_id, 7, "backup", "defense"),
    ("Jaylon Ferguson", "DE", team_id, 94, "backup", "defense"),
    ("Tyler Linderbaum", "DL", team_id, 69, "backup", "defense"),
    ("Devin White", "LB", team_id, 45, "backup", "defense"),

    # Special Teams
    ("Justin Tucker", "K", team_id, 9, "starter", "special"),
    ("Sam Koch", "P", team_id, 6, "starter", "special"),
    ("Morgan Cox", "LS", team_id, 49, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CIN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Joe Burrow", "QB", team_id, 9, "starter", "offense"),
    ("Joe Mixon", "RB", team_id, 28, "starter", "offense"),
    ("Chris Evans", "RB", team_id, 34, "starter", "offense"),
    ("Ja'Marr Chase", "WR", team_id, 1, "starter", "offense"),
    ("Tee Higgins", "WR", team_id, 85, "starter", "offense"),
    ("Evan McPherson", "TE", team_id, 89, "starter", "offense"),
    ("Quinton Spain", "OL", team_id, 65, "starter", "offense"),
    ("Trey Hopkins", "OL", team_id, 66, "starter", "offense"),
    ("La'el Collins", "OL", team_id, 73, "starter", "offense"),
    ("Hakeem Adeniji", "OL", team_id, 78, "starter", "offense"),
    ("Jackson Carman", "OL", team_id, 68, "starter", "offense"),

    # Backup Offense
    ("Bengals Backup QB", "QB", team_id, 2, "backup", "offense"),
    ("Samaje Perine", "RB", team_id, 34, "backup", "offense"),
    ("C.J. Uzomah", "TE", team_id, 87, "backup", "offense"),

    # Starting Defense
    ("B.J. Hill", "DT", team_id, 95, "starter", "defense"),
    ("Trey Hendrickson", "DE", team_id, 91, "starter", "defense"),
    ("Logan Wilson", "LB", team_id, 30, "starter", "defense"),
    ("Germaine Pratt", "LB", team_id, 50, "starter", "defense"),
    ("Chidobe Awuzie", "CB", team_id, 29, "starter", "defense"),
    ("Mike Hilton", "CB", team_id, 31, "starter", "defense"),
    ("Vonn Bell", "S", team_id, 26, "starter", "defense"),
    ("DJ Reader", "DT", team_id, 92, "starter", "defense"),
    ("Sam Hubbard", "DE", team_id, 54, "starter", "defense"),

    # Backup Defense
    ("Backup DL", "DL", team_id, 94, "backup", "defense"),
    ("Backup LB", "LB", team_id, 53, "backup", "defense"),
    ("Backup CB", "CB", team_id, 24, "backup", "defense"),
    ("Backup S", "S", team_id, 36, "backup", "defense"),

    # Special Teams
    ("Evan McPherson", "K", team_id, 9, "starter", "special"),
    ("Kevin Huber", "P", team_id, 6, "starter", "special"),
    ("Backup LS", "LS", team_id, 49, "starter", "special"),
]


cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CLE",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Deshaun Watson", "QB", team_id, 4, "starter", "offense"),
    ("Nick Chubb", "RB", team_id, 24, "starter", "offense"),
    ("D'Ernest Johnson", "RB", team_id, 33, "starter", "offense"),
    ("Amari Cooper", "WR", team_id, 2, "starter", "offense"),
    ("Donovan Peoples-Jones", "WR", team_id, 11, "starter", "offense"),
    ("David Njoku", "TE", team_id, 85, "starter", "offense"),
    ("JC Tretter", "OL", team_id, 61, "starter", "offense"),
    ("Wyatt Teller", "OL", team_id, 65, "starter", "offense"),
    ("Jedrick Wills Jr.", "OL", team_id, 71, "starter", "offense"),
    ("Desmond Harrison", "OL", team_id, 72, "starter", "offense"),
    ("Shon Coleman", "OL", team_id, 77, "starter", "offense"),

    # Backup Offense
    ("Backup QB", "QB", team_id, 7, "backup", "offense"),
    ("Backup RB", "RB", team_id, 34, "backup", "offense"),
    ("Backup WR", "WR", team_id, 17, "backup", "offense"),
    ("Backup TE", "TE", team_id, 86, "backup", "offense"),

    # Starting Defense
    ("Myles Garrett", "DE", team_id, 95, "starter", "defense"),
    ("Calijah Kancey", "DT", team_id, 97, "starter", "defense"),
    ("Jeremiah Owusu-Koramoah", "LB", team_id, 39, "starter", "defense"),
    ("Anthony Walker", "LB", team_id, 51, "starter", "defense"),
    ("Denzel Ward", "CB", team_id, 21, "starter", "defense"),
    ("Greg Newsome II", "CB", team_id, 20, "starter", "defense"),
    ("John Johnson III", "S", team_id, 30, "starter", "defense"),
    ("Grant Delpit", "S", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Backup DL", "DL", team_id, 92, "backup", "defense"),
    ("Backup LB", "LB", team_id, 53, "backup", "defense"),
    ("Backup CB", "CB", team_id, 24, "backup", "defense"),
    ("Backup S", "S", team_id, 36, "backup", "defense"),

    # Special Teams
    ("Evan McPherson", "K", team_id, 9, "starter", "special"),
    ("Punter", "P", team_id, 6, "starter", "special"),
    ("LS", "LS", team_id, 49, "starter", "special"),
]

cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("PIT",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Kenny Pickett", "QB", team_id, 8, "starter", "offense"),
    ("Najee Harris", "RB", team_id, 22, "starter", "offense"),
    ("Anthony McFarland Jr.", "RB", team_id, 30, "starter", "offense"),
    ("George Pickens", "WR", team_id, 11, "starter", "offense"),
    ("Diontae Johnson", "WR", team_id, 18, "starter", "offense"),
    ("Pat Freiermuth", "TE", team_id, 88, "starter", "offense"),
    ("David DeCastro", "OL", team_id, 66, "starter", "offense"),
    ("Trai Turner", "OL", team_id, 70, "starter", "offense"),
    ("Kevin Dotson", "OL", team_id, 63, "starter", "offense"),
    ("Dan Moore Jr.", "OL", team_id, 79, "starter", "offense"),
    ("Chukwuma Okorafor", "OL", team_id, 68, "starter", "offense"),

    # Backup Offense
    ("Backup QB", "QB", team_id, 3, "backup", "offense"),
    ("Backup RB", "RB", team_id, 34, "backup", "offense"),
    ("Backup WR", "WR", team_id, 14, "backup", "offense"),
    ("Backup TE", "TE", team_id, 86, "backup", "offense"),

    # Starting Defense
    ("Cameron Heyward", "DE", team_id, 97, "starter", "defense"),
    ("Isaiahh Loudermilk", "DT", team_id, 96, "starter", "defense"),
    ("T.J. Watt", "OLB", team_id, 90, "starter", "defense"),
    ("Alex Highsmith", "OLB", team_id, 56, "starter", "defense"),
    ("Minkah Fitzpatrick", "S", team_id, 39, "starter", "defense"),
    ("Terrell Edmunds", "S", team_id, 41, "starter", "defense"),
    ("Cameron Sutton", "CB", team_id, 30, "starter", "defense"),
    ("Ahkello Witherspoon", "CB", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Backup DL", "DL", team_id, 92, "backup", "defense"),
    ("Backup LB", "LB", team_id, 53, "backup", "defense"),
    ("Backup CB", "CB", team_id, 24, "backup", "defense"),
    ("Backup S", "S", team_id, 36, "backup", "defense"),

    # Special Teams
    ("Chris Boswell", "K", team_id, 9, "starter", "special"),
    ("Pressley Harvin III", "P", team_id, 6, "starter", "special"),
    ("LS", "LS", team_id, 49, "starter", "special"),
]

# AFC South
# Houston Texans
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("HOU",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("C.J. Stroud", "QB", team_id, 7, "starter", "offense"),
    ("Dameon Pierce", "RB", team_id, 32, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 28, "starter", "offense"),
    ("Jaxon Smith-Njigba", "WR", team_id, 11, "starter", "offense"),
    ("Brandin Cooks", "WR", team_id, 10, "starter", "offense"),
    ("T.J. Hockenson", "TE", team_id, 85, "starter", "offense"),
    ("Dare Rosenthal", "OL", team_id, 66, "starter", "offense"),
    ("Zach Fulton", "OL", team_id, 72, "starter", "offense"),
    ("Rodrigo Blankenship", "OL", team_id, 67, "starter", "offense"),
    ("Dameon Pierce", "OL", team_id, 63, "starter", "offense"),
    ("Walker Little", "OL", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("C.J. Stroud", "QB", team_id, 7, "backup", "offense"),
    ("Dameon Pierce", "RB", team_id, 32, "backup", "offense"),
    ("David Montgomery", "RB", team_id, 28, "backup", "offense"),
    ("Jaxon Smith-Njigba", "WR", team_id, 11, "backup", "offense"),
    ("Brandin Cooks", "WR", team_id, 10, "backup", "offense"),
    ("T.J. Hockenson", "TE", team_id, 85, "backup", "offense"),

    # Starting Defense
    ("J.J. Watt", "DL", team_id, 99, "starter", "defense"),
    ("Malik Jefferson", "LB", team_id, 50, "starter", "defense"),
    ("Shaquil Barrett", "LB", team_id, 55, "starter", "defense"),
    ("Derek Stingley Jr.", "CB", team_id, 6, "starter", "defense"),
    ("Marcus Jones", "CB", team_id, 24, "starter", "defense"),
    ("Justin Reid", "S", team_id, 8, "starter", "defense"),
    ("Tyrann Mathieu", "S", team_id, 32, "starter", "defense"),

    # Backup Defense
    ("J.J. Watt", "DL", team_id, 99, "backup", "defense"),
    ("Malik Jefferson", "LB", team_id, 50, "backup", "defense"),
    ("Shaquil Barrett", "LB", team_id, 55, "backup", "defense"),

    # Special Teams
    ("Jonathan Owens", "P", team_id, 5, "starter", "special"),
    ("Ka'imi Fairbairn", "K", team_id, 9, "starter", "special"),
    ("Patrick Scales", "LS", team_id, 49, "starter", "special"),
]

# Indianapolis Colts
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("IND",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Anthony Richardson", "QB", team_id, 1, "starter", "offense"),
    ("Jonathan Taylor", "RB", team_id, 28, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 32, "starter", "offense"),
    ("Michael Pittman Jr.", "WR", team_id, 11, "starter", "offense"),
    ("Alec Pierce", "WR", team_id, 19, "starter", "offense"),
    ("Mo Alie-Cox", "TE", team_id, 85, "starter", "offense"),
    ("Quenton Nelson", "OL", team_id, 56, "starter", "offense"),
    ("Braden Smith", "OL", team_id, 70, "starter", "offense"),
    ("Ryan Kelly", "OL", team_id, 61, "starter", "offense"),
    ("Bernhard Raimann", "OL", team_id, 67, "starter", "offense"),
    ("Jamarco Jones", "OL", team_id, 77, "starter", "offense"),

    # Backup Offense
    ("Sam Ehlinger", "QB", team_id, 3, "backup", "offense"),
    ("Jonathan Taylor", "RB", team_id, 28, "backup", "offense"),
    ("David Montgomery", "RB", team_id, 32, "backup", "offense"),
    ("Michael Pittman Jr.", "WR", team_id, 11, "backup", "offense"),
    ("Alec Pierce", "WR", team_id, 19, "backup", "offense"),
    ("Mo Alie-Cox", "TE", team_id, 85, "backup", "offense"),

    # Starting Defense
    ("DeForest Buckner", "DL", team_id, 99, "starter", "defense"),
    ("Isaiah Rodgers", "LB", team_id, 24, "starter", "defense"),
    ("Bobby Okereke", "LB", team_id, 56, "starter", "defense"),
    ("Rock Ya-Sin", "CB", team_id, 29, "starter", "defense"),
    ("Ahmad Sauce Gardner", "CB", team_id, 20, "starter", "defense"),
    ("Khari Willis", "S", team_id, 31, "starter", "defense"),
    ("Malik Hooker", "S", team_id, 21, "starter", "defense"),

    # Backup Defense
    ("DeForest Buckner", "DL", team_id, 99, "backup", "defense"),
    ("Bobby Okereke", "LB", team_id, 56, "backup", "defense"),

    # Special Teams
    ("Matt Gay", "K", team_id, 3, "starter", "special"),
    ("Rigoberto Sanchez", "P", team_id, 6, "starter", "special"),
    ("Camaron Cheeseman", "LS", team_id, 46, "starter", "special"),
]

# Jacksonville Jaguars (JAX)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("JAX",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Trevor Lawrence", "QB", team_id, 16, "starter", "offense"),
    ("Travis Etienne Jr.", "RB", team_id, 1, "starter", "offense"),
    ("James Robinson", "RB", team_id, 27, "starter", "offense"),
    ("Christian Kirk", "WR", team_id, 13, "starter", "offense"),
    ("Zay Jones", "WR", team_id, 8, "starter", "offense"),
    ("Evan Engram", "TE", team_id, 85, "starter", "offense"),
    ("Cam Robinson", "OL", team_id, 72, "starter", "offense"),
    ("Julius Jackson", "OL", team_id, 77, "starter", "offense"),
    ("Alaric Jackson", "OL", team_id, 61, "starter", "offense"),
    ("Brandon Scherff", "OL", team_id, 70, "starter", "offense"),
    ("Trevor Penning", "OL", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("Jordan Love", "QB", team_id, 10, "backup", "offense"),
    ("Devine Ozigbo", "RB", team_id, 34, "backup", "offense"),
    ("Rashod Bateman", "WR", team_id, 7, "backup", "offense"),
    ("Josh Oliver", "TE", team_id, 88, "backup", "offense"),
    ("Matt Feiler", "OL", team_id, 66, "backup", "offense"),

    # Starting Defense
    ("Calais Campbell", "DL", team_id, 93, "starter", "defense"),
    ("Jordan Davis", "DL", team_id, 98, "starter", "defense"),
    ("Josh Allen", "LB", team_id, 41, "starter", "defense"),
    ("Foyesade Oluokun", "LB", team_id, 40, "starter", "defense"),
    ("Shaquill Griffin", "CB", team_id, 25, "starter", "defense"),
    ("Darious Williams", "CB", team_id, 24, "starter", "defense"),
    ("Andre Cisco", "S", team_id, 4, "starter", "defense"),
    ("Andrew Wingard", "S", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("K'Lavon Chaisson", "DL", team_id, 52, "backup", "defense"),
    ("Jordan Smith", "DL", team_id, 97, "backup", "defense"),
    ("Roquan Smith", "LB", team_id, 58, "backup", "defense"),
    ("Shaquille Quarterman", "LB", team_id, 50, "backup", "defense"),
    ("CJ Henderson", "CB", team_id, 29, "backup", "defense"),
    ("Tre Herndon", "CB", team_id, 23, "backup", "defense"),
    ("Kamren Curl", "S", team_id, 21, "backup", "defense"),

    # Special Teams
    ("Riley Patterson", "K", team_id, 2, "starter", "special"),
    ("Matt Ammendola", "P", team_id, 1, "starter", "special"),
    ("Caleb Wilson", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Tennessee Titans (TEN)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("TEN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Ryan Tannehill", "QB", team_id, 17, "starter", "offense"),
    ("Derrick Henry", "RB", team_id, 22, "starter", "offense"),
    ("Nick Chubb", "RB", team_id, 24, "starter", "offense"),  # Example
    ("Treylon Burks", "WR", team_id, 11, "starter", "offense"),
    ("Robert Woods", "WR", team_id, 14, "starter", "offense"),
    ("Chigoziem Okonkwo", "TE", team_id, 81, "starter", "offense"),
    ("Rodger Saffold", "OL", team_id, 70, "starter", "offense"),
    ("David Quessenberry", "OL", team_id, 75, "starter", "offense"),
    ("Taylor Lewan", "OL", team_id, 77, "starter", "offense"),
    ("Ben Jones", "OL", team_id, 63, "starter", "offense"),
    ("Dennis Kelly", "OL", team_id, 68, "starter", "offense"),

    # Backup Offense
    ("Malik Willis", "QB", team_id, 1, "backup", "offense"),
    ("Dontrell Hilliard", "RB", team_id, 30, "backup", "offense"),
    ("Nick Westbrook-Ikhine", "WR", team_id, 83, "backup", "offense"),
    ("Gehrig Dieter", "WR", team_id, 89, "backup", "offense"),
    ("Chig Okonkwo", "TE", team_id, 81, "backup", "offense"),

    # Starting Defense
    ("Jeffery Simmons", "DL", team_id, 99, "starter", "defense"),
    ("Denico Autry", "DL", team_id, 91, "starter", "defense"),
    ("Derrick Morgan", "LB", team_id, 54, "starter", "defense"),
    ("Harold Landry", "LB", team_id, 90, "starter", "defense"),
    ("Kevin Byard", "S", team_id, 31, "starter", "defense"),
    ("Amani Hooker", "S", team_id, 28, "starter", "defense"),
    ("Roger McCreary", "CB", team_id, 22, "starter", "defense"),
    ("Chris Jackson", "CB", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Teair Tart", "DL", team_id, 92, "backup", "defense"),
    ("David Long Jr.", "LB", team_id, 50, "backup", "defense"),
    ("Bradley McDougald", "S", team_id, 29, "backup", "defense"),
    ("Kalan Reed", "CB", team_id, 24, "backup", "defense"),

    # Special Teams
    ("Randy Bullock", "K", team_id, 3, "starter", "special"),
    ("Tanner Vallejo", "P", team_id, 5, "starter", "special"),
    ("Austin Blythe", "LS", team_id, 62, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Denver Broncos (DEN)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DEN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Russell Wilson", "QB", team_id, 3, "starter", "offense"),
    ("Javonte Williams", "RB", team_id, 33, "starter", "offense"),
    ("Melvin Gordon", "RB", team_id, 25, "starter", "offense"),
    ("Jerry Jeudy", "WR", team_id, 10, "starter", "offense"),
    ("Courtland Sutton", "WR", team_id, 14, "starter", "offense"),
    ("Noah Fant", "TE", team_id, 87, "starter", "offense"),
    ("Lloyd Cushenberry", "OL", team_id, 61, "starter", "offense"),
    ("Ronnie Stanley", "OL", team_id, 79, "starter", "offense"),
    ("Garett Bolles", "OL", team_id, 76, "starter", "offense"),
    ("Rashawn Slater", "OL", team_id, 72, "starter", "offense"),
    ("Cameron Fleming", "OL", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("Brett Rypien", "QB", team_id, 2, "backup", "offense"),
    ("Mike Boone", "RB", team_id, 32, "backup", "offense"),
    ("Tim Patrick", "WR", team_id, 83, "backup", "offense"),
    ("Albert Okwuegbunam", "TE", team_id, 84, "backup", "offense"),
    ("Billy Turner", "OL", team_id, 64, "backup", "offense"),

    # Starting Defense
    ("Dre'Mont Jones", "DL", team_id, 95, "starter", "defense"),
    ("DeMarcus Walker", "DL", team_id, 90, "starter", "defense"),
    ("Josey Jewell", "LB", team_id, 50, "starter", "defense"),
    ("Alexander Johnson", "LB", team_id, 57, "starter", "defense"),
    ("Patrick Surtain II", "CB", team_id, 24, "starter", "defense"),
    ("Ronnie Harrison", "S", team_id, 31, "starter", "defense"),
    ("Justin Simmons", "S", team_id, 31, "starter", "defense"),
    ("Ahkello Witherspoon", "CB", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Shane Ray", "DL", team_id, 92, "backup", "defense"),
    ("Jonathon Cooper", "DL", team_id, 96, "backup", "defense"),
    ("Will Compton", "LB", team_id, 53, "backup", "defense"),
    ("Micah Kiser", "LB", team_id, 55, "backup", "defense"),
    ("Derek Wolfe", "DL", team_id, 93, "backup", "defense"),
    ("Bryce Callahan", "CB", team_id, 26, "backup", "defense"),
    ("Jamar Johnson", "S", team_id, 23, "backup", "defense"),

    # Special Teams
    ("Brandon McManus", "K", team_id, 5, "starter", "special"),
    ("Sam Martin", "P", team_id, 6, "starter", "special"),
    ("Tyler Ott", "LS", team_id, 46, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Kansas City Chiefs (KC)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("KC",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Patrick Mahomes", "QB", team_id, 15, "starter", "offense"),
    ("Isiah Pacheco", "RB", team_id, 31, "starter", "offense"),
    ("Clyde Edwards-Helaire", "RB", team_id, 25, "starter", "offense"),
    ("Marquez Valdes-Scantling", "WR", team_id, 18, "starter", "offense"),
    ("Travis Kelce", "TE", team_id, 87, "starter", "offense"),
    ("Kadarius Toney", "WR", team_id, 10, "starter", "offense"),
    ("Orlando Brown Jr.", "OL", team_id, 70, "starter", "offense"),
    ("Joe Thuney", "OL", team_id, 62, "starter", "offense"),
    ("Creed Humphrey", "OL", team_id, 61, "starter", "offense"),
    ("Nick Allegretti", "OL", team_id, 67, "starter", "offense"),
    ("Lucas Niang", "OL", team_id, 76, "starter", "offense"),

    # Backup Offense
    ("Chase Daniel", "QB", team_id, 4, "backup", "offense"),
    ("Darrel Williams", "RB", team_id, 30, "backup", "offense"),
    ("JuJu Smith-Schuster", "WR", team_id, 19, "backup", "offense"),
    ("Sammy Watkins", "WR", team_id, 14, "backup", "offense"),
    ("Noah Gray", "TE", team_id, 82, "backup", "offense"),

    # Starting Defense
    ("Chris Jones", "DL", team_id, 95, "starter", "defense"),
    ("Frank Clark", "DL", team_id, 55, "starter", "defense"),
    ("Nick Bolton", "LB", team_id, 51, "starter", "defense"),
    ("Willie Gay Jr.", "LB", team_id, 50, "starter", "defense"),
    ("L'Jarius Sneed", "CB", team_id, 26, "starter", "defense"),
    ("Tyrann Mathieu", "S", team_id, 32, "starter", "defense"),
    ("Juan Thornhill", "S", team_id, 36, "starter", "defense"),
    ("Taye Barber", "CB", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Trey Smith", "DL", team_id, 96, "backup", "defense"),
    ("Justin March", "LB", team_id, 53, "backup", "defense"),
    ("Mike Hughes", "CB", team_id, 29, "backup", "defense"),
    ("Jaylen Watson", "CB", team_id, 28, "backup", "defense"),

    # Special Teams
    ("Harrison Butker", "K", team_id, 7, "starter", "special"),
    ("Tom Hackett", "P", team_id, 4, "starter", "special"),
    ("James Winchester", "LS", team_id, 45, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Las Vegas Raiders (LV)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("LV",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jimmy Garoppolo", "QB", team_id, 10, "starter", "offense"),
    ("Josh Jacobs", "RB", team_id, 28, "starter", "offense"),
    ("Zamir White", "RB", team_id, 31, "starter", "offense"),
    ("Davante Adams", "WR", team_id, 17, "starter", "offense"),
    ("Hunter Renfrow", "WR", team_id, 13, "starter", "offense"),
    ("Foster Moreau", "TE", team_id, 85, "starter", "offense"),
    ("Alex Leatherwood", "OL", team_id, 76, "starter", "offense"),
    ("Andre James", "OL", team_id, 68, "starter", "offense"),
    ("Denzelle Good", "OL", team_id, 63, "starter", "offense"),
    ("DJ Finney", "OL", team_id, 66, "starter", "offense"),
    ("Daniel Carlson", "OL", team_id, 75, "starter", "offense"),

    # Backup Offense
    ("Nick Mullens", "QB", team_id, 2, "backup", "offense"),
    ("Keilan Cole", "WR", team_id, 19, "backup", "offense"),
    ("Gregory Little", "TE", team_id, 84, "backup", "offense"),

    # Starting Defense
    ("Maxx Crosby", "DL", team_id, 98, "starter", "defense"),
    ("Clelin Ferrell", "DL", team_id, 97, "starter", "defense"),
    ("Denzel Perryman", "LB", team_id, 50, "starter", "defense"),
    ("Nick Kwiatkoski", "LB", team_id, 53, "starter", "defense"),
    ("Trayvon Mullen", "CB", team_id, 24, "starter", "defense"),
    ("Damarious Randall", "S", team_id, 22, "starter", "defense"),
    ("Nickell Robey-Coleman", "CB", team_id, 21, "starter", "defense"),
    ("Johnathan Abram", "S", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Carl Nassib", "DL", team_id, 94, "backup", "defense"),
    ("P.J. Hall", "DL", team_id, 99, "backup", "defense"),
    ("Malcolm Koonce", "LB", team_id, 55, "backup", "defense"),
    ("Rock Ya-Sin", "CB", team_id, 20, "backup", "defense"),

    # Special Teams
    ("Daniel Carlson", "K", team_id, 2, "starter", "special"),
    ("A.J. Cole", "P", team_id, 4, "starter", "special"),
    ("Hunter Bradley", "LS", team_id, 46, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Los Angeles Chargers (LAC)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("LAC",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Justin Herbert", "QB", team_id, 10, "starter", "offense"),
    ("Austin Ekeler", "RB", team_id, 30, "starter", "offense"),
    ("Joshua Kelley", "RB", team_id, 27, "starter", "offense"),
    ("Keenan Allen", "WR", team_id, 13, "starter", "offense"),
    ("Mike Williams", "WR", team_id, 14, "starter", "offense"),
    ("Gerald Everett", "TE", team_id, 85, "starter", "offense"),
    ("Corey Linsley", "OL", team_id, 63, "starter", "offense"),
    ("Dan Feeney", "OL", team_id, 66, "starter", "offense"),
    ("Rashawn Slater", "OL", team_id, 72, "starter", "offense"),
    ("Oday Aboushi", "OL", team_id, 60, "starter", "offense"),
    ("Matt Feiler", "OL", team_id, 75, "starter", "offense"),

    # Backup Offense
    ("Easton Stick", "QB", team_id, 5, "backup", "offense"),
    ("Justin Jackson", "RB", team_id, 35, "backup", "offense"),
    ("Jalen Guyton", "WR", team_id, 80, "backup", "offense"),
    ("Donald Parham Jr.", "TE", team_id, 83, "backup", "offense"),
    ("Trey Pipkins", "OL", team_id, 78, "backup", "offense"),

    # Starting Defense
    ("Joey Bosa", "DL", team_id, 97, "starter", "defense"),
    ("Austin Johnson", "DL", team_id, 96, "starter", "defense"),
    ("Kenneth Murray", "LB", team_id, 58, "starter", "defense"),
    ("Drue Tranquill", "LB", team_id, 51, "starter", "defense"),
    ("JC Jackson", "CB", team_id, 24, "starter", "defense"),
    ("Asante Samuel Jr.", "CB", team_id, 21, "starter", "defense"),
    ("Derwin James", "S", team_id, 33, "starter", "defense"),
    ("Nasir Adderley", "S", team_id, 31, "starter", "defense"),

    # Backup Defense
    ("Uchenna Nwosu", "DL", team_id, 95, "backup", "defense"),
    ("Myles Jack", "LB", team_id, 56, "backup", "defense"),
    ("Tevaughn Campbell", "CB", team_id, 29, "backup", "defense"),
    ("Eddie Jackson", "S", team_id, 30, "backup", "defense"),

    # Special Teams
    ("Michael Badgley", "K", team_id, 4, "starter", "special"),
    ("Tyler Newsome", "P", team_id, 2, "starter", "special"),
    ("Cole Mazza", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Dallas Cowboys (DAL)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DAL",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Dak Prescott", "QB", team_id, 4, "starter", "offense"),
    ("Tony Pollard", "RB", team_id, 20, "starter", "offense"),
    ("Rashod Bateman", "WR", team_id, 7, "starter", "offense"),
    ("CeeDee Lamb", "WR", team_id, 88, "starter", "offense"),
    ("Jake Ferguson", "TE", team_id, 82, "starter", "offense"),
    ("Tyron Smith", "OL", team_id, 77, "starter", "offense"),
    ("Tyler Smith", "OL", team_id, 71, "starter", "offense"),
    ("Zack Martin", "OL", team_id, 70, "starter", "offense"),
    ("Connor Williams", "OL", team_id, 69, "starter", "offense"),
    ("Terence Steele", "OL", team_id, 79, "starter", "offense"),
    ("Jason Peters", "OL", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("Cooper Rush", "QB", team_id, 2, "backup", "offense"),
    ("Jahmyr Gibbs", "RB", team_id, 33, "backup", "offense"),
    ("Michael Gallup", "WR", team_id, 11, "backup", "offense"),
    ("Jake Ferguson", "TE", team_id, 82, "backup", "offense"),
    ("Matt Farniok", "OL", team_id, 68, "backup", "offense"),

    # Starting Defense
    ("Micah Parsons", "LB", team_id, 11, "starter", "defense"),
    ("DeMarcus Lawrence", "DL", team_id, 90, "starter", "defense"),
    ("Trevon Diggs", "CB", team_id, 7, "starter", "defense"),
    ("Damontae Kazee", "S", team_id, 24, "starter", "defense"),
    ("Malik Hooker", "S", team_id, 36, "starter", "defense"),
    ("Dorance Armstrong", "DL", team_id, 97, "starter", "defense"),
    ("Quinton Bohanna", "DL", team_id, 98, "starter", "defense"),
    ("Troy Pride Jr.", "CB", team_id, 25, "starter", "defense"),

    # Backup Defense
    ("Osa Odighizuwa", "DL", team_id, 92, "backup", "defense"),
    ("Sam Williams", "LB", team_id, 55, "backup", "defense"),
    ("Anthony Brown", "CB", team_id, 26, "backup", "defense"),
    ("J.R. Reed", "S", team_id, 30, "backup", "defense"),

    # Special Teams
    ("Brett Maher", "K", team_id, 5, "starter", "special"),
    ("Bryan Anger", "P", team_id, 4, "starter", "special"),
    ("Jake McQuaide", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# New York Giants (NYG)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NYG",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Daniel Jones", "QB", team_id, 8, "starter", "offense"),
    ("Saquon Barkley", "RB", team_id, 26, "starter", "offense"),
    ("Matt Breida", "RB", team_id, 33, "starter", "offense"),
    ("Kenny Golladay", "WR", team_id, 19, "starter", "offense"),
    ("Darius Slayton", "WR", team_id, 86, "starter", "offense"),
    ("Evan Engram", "TE", team_id, 88, "starter", "offense"),
    ("Andrew Thomas", "OL", team_id, 73, "starter", "offense"),
    ("Will Hernandez", "OL", team_id, 69, "starter", "offense"),
    ("Nick Gates", "OL", team_id, 60, "starter", "offense"),
    ("Jon Feliciano", "OL", team_id, 65, "starter", "offense"),
    ("Tyler Shelvin", "OL", team_id, 77, "starter", "offense"),

    # Backup Offense
    ("Tyler Huntley", "QB", team_id, 2, "backup", "offense"),
    ("Gary Brightwell", "RB", team_id, 36, "backup", "offense"),
    ("Darius Slayton", "WR", team_id, 86, "backup", "offense"),
    ("Daniel Bellinger", "TE", team_id, 81, "backup", "offense"),

    # Starting Defense
    ("Dexter Lawrence", "DL", team_id, 90, "starter", "defense"),
    ("Dalton Risner", "DL", team_id, 95, "starter", "defense"),
    ("Micah McFadden", "LB", team_id, 53, "starter", "defense"),
    ("Jarrad Davis", "LB", team_id, 56, "starter", "defense"),
    ("James Bradberry", "CB", team_id, 24, "starter", "defense"),
    ("Adoree’ Jackson", "CB", team_id, 23, "starter", "defense"),
    ("Logan Ryan", "S", team_id, 26, "starter", "defense"),
    ("Julian Love", "S", team_id, 31, "starter", "defense"),

    # Backup Defense
    ("B.J. Hill", "DL", team_id, 92, "backup", "defense"),
    ("Dalvin Tomlinson", "DL", team_id, 93, "backup", "defense"),
    ("Oren Burks", "LB", team_id, 57, "backup", "defense"),
    ("Rodarius Williams", "CB", team_id, 22, "backup", "defense"),

    # Special Teams
    ("Graham Gano", "K", team_id, 7, "starter", "special"),
    ("Britton Colquitt", "P", team_id, 4, "starter", "special"),
    ("Zak DeOssie", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Philadelphia Eagles (PHI)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("PHI",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jalen Hurts", "QB", team_id, 1, "starter", "offense"),
    ("Miles Sanders", "RB", team_id, 26, "starter", "offense"),
    ("Kenneth Gainwell", "RB", team_id, 27, "starter", "offense"),
    ("A.J. Brown", "WR", team_id, 11, "starter", "offense"),
    ("DeVonta Smith", "WR", team_id, 6, "starter", "offense"),
    ("Dallas Goedert", "TE", team_id, 88, "starter", "offense"),
    ("Jason Kelce", "OL", team_id, 62, "starter", "offense"),
    ("Lane Johnson", "OL", team_id, 65, "starter", "offense"),
    ("Jordan Mailata", "OL", team_id, 71, "starter", "offense"),
    ("Beau Benzschawel", "OL", team_id, 64, "starter", "offense"),
    ("Landon Dickerson", "OL", team_id, 59, "starter", "offense"),

    # Backup Offense
    ("Gardner Minshew", "QB", team_id, 10, "backup", "offense"),
    ("Boston Scott", "RB", team_id, 35, "backup", "offense"),
    ("Quez Watkins", "WR", team_id, 15, "backup", "offense"),
    ("Tyree Jackson", "TE", team_id, 84, "backup", "offense"),
    ("Jack Driscoll", "OL", team_id, 73, "backup", "offense"),

    # Starting Defense
    ("Fletcher Cox", "DL", team_id, 91, "starter", "defense"),
    ("Josh Sweat", "DL", team_id, 44, "starter", "defense"),
    ("Haason Reddick", "LB", team_id, 5, "starter", "defense"),
    ("T.J. Edwards", "LB", team_id, 56, "starter", "defense"),
    ("James Bradberry", "CB", team_id, 24, "starter", "defense"),
    ("Darius Slay", "CB", team_id, 2, "starter", "defense"),
    ("C.J. Gardner-Johnson", "S", team_id, 23, "starter", "defense"),
    ("Anthony Harris", "S", team_id, 21, "starter", "defense"),

    # Backup Defense
    ("Brandon Graham", "DL", team_id, 55, "backup", "defense"),
    ("Rashad Smith", "LB", team_id, 51, "backup", "defense"),
    ("Avonte Maddox", "CB", team_id, 30, "backup", "defense"),
    ("Marcus Epps", "S", team_id, 36, "backup", "defense"),

    # Special Teams
    ("Jake Elliott", "K", team_id, 4, "starter", "special"),
    ("Arryn Siposs", "P", team_id, 6, "starter", "special"),
    ("Camryn Bynum", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Washington Commanders (WAS)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("WAS",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Sam Howell", "QB", team_id, 4, "starter", "offense"),
    ("Brian Robinson", "RB", team_id, 22, "starter", "offense"),
    ("Antonio Gibson", "RB", team_id, 31, "starter", "offense"),
    ("Terry McLaurin", "WR", team_id, 17, "starter", "offense"),
    ("Jahan Dotson", "WR", team_id, 16, "starter", "offense"),
    ("Cole Turner", "TE", team_id, 81, "starter", "offense"),
    ("Braxton Jones", "OL", team_id, 79, "starter", "offense"),
    ("Saahdiq Charles", "OL", team_id, 65, "starter", "offense"),
    ("Chase Roullier", "OL", team_id, 61, "starter", "offense"),
    ("Trey Smith", "OL", team_id, 78, "starter", "offense"),
    ("Bradbury Thompson", "OL", team_id, 73, "starter", "offense"),

    # Backup Offense
    ("Taylor Heinicke", "QB", team_id, 3, "backup", "offense"),
    ("Dyami Brown", "WR", team_id, 10, "backup", "offense"),
    ("Jaret Patterson", "RB", team_id, 35, "backup", "offense"),
    ("Logan Thomas", "TE", team_id, 81, "backup", "offense"),
    ("Ereck Flowers", "OL", team_id, 67, "backup", "offense"),

    # Starting Defense
    ("Chase Young", "DL", team_id, 99, "starter", "defense"),
    ("Daron Payne", "DL", team_id, 94, "starter", "defense"),
    ("Jamin Davis", "LB", team_id, 51, "starter", "defense"),
    ("Cole Holcomb", "LB", team_id, 58, "starter", "defense"),
    ("Derek Stingley Jr.", "CB", team_id, 6, "starter", "defense"),
    ("Cornell Armstrong", "CB", team_id, 28, "starter", "defense"),
    ("Jeremy Reaves", "S", team_id, 37, "starter", "defense"),
    ("Kamren Curl", "S", team_id, 36, "starter", "defense"),

    # Backup Defense
    ("Poona Ford", "DL", team_id, 96, "backup", "defense"),
    ("Efe Obada", "DL", team_id, 92, "backup", "defense"),
    ("Jahlani Tavai", "LB", team_id, 50, "backup", "defense"),
    ("Tony Fields II", "LB", team_id, 54, "backup", "defense"),
    ("Sam Franklin", "S", team_id, 32, "backup", "defense"),
    ("Kyler Gordon", "CB", team_id, 27, "backup", "defense"),

    # Special Teams
    ("Tress Way", "P", team_id, 6, "starter", "special"),
    ("José Borregales", "K", team_id, 3, "starter", "special"),
    ("Patrick Murray", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Green Bay Packers (GB)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("GB",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jordan Love", "QB", team_id, 10, "starter", "offense"),
    ("Aaron Jones", "RB", team_id, 33, "starter", "offense"),
    ("A.J. Dillon", "RB", team_id, 28, "starter", "offense"),
    ("Christian Watson", "WR", team_id, 18, "starter", "offense"),
    ("Romeo Doubs", "WR", team_id, 81, "starter", "offense"),
    ("Josiah Deguara", "TE", team_id, 85, "starter", "offense"),
    ("Elgton Jenkins", "OL", team_id, 63, "starter", "offense"),
    ("David Bakhtiari", "OL", team_id, 69, "starter", "offense"),
    ("Lucas Patrick", "OL", team_id, 61, "starter", "offense"),
    ("Elijah Nkansah", "OL", team_id, 67, "starter", "offense"),
    ("Jon Runyan Jr.", "OL", team_id, 78, "starter", "offense"),

    # Backup Offense
    ("Tim Boyle", "QB", team_id, 6, "backup", "offense"),
    ("Kylin Hill", "RB", team_id, 21, "backup", "offense"),
    ("Allen Lazard", "WR", team_id, 13, "backup", "offense"),
    ("Jake Ferguson", "TE", team_id, 82, "backup", "offense"),
    ("Ben Braden", "OL", team_id, 65, "backup", "offense"),

    # Starting Defense
    ("Kingsley Enagbare", "DL", team_id, 93, "starter", "defense"),
    ("Kingsley Keke", "DL", team_id, 94, "starter", "defense"),
    ("Rashan Gary", "DL", team_id, 99, "starter", "defense"),
    ("De'Vondre Campbell", "LB", team_id, 53, "starter", "defense"),
    ("Quay Walker", "LB", team_id, 50, "starter", "defense"),
    ("Jaire Alexander", "CB", team_id, 23, "starter", "defense"),
    ("Kenny Clark", "DL", team_id, 97, "starter", "defense"),
    ("Richie Grant", "S", team_id, 36, "starter", "defense"),
    ("Jaire Alexander", "CB", team_id, 23, "starter", "defense"),
    ("Rashan Gary", "DL", team_id, 99, "starter", "defense"),

    # Backup Defense
    ("Devonte Wyatt", "DL", team_id, 96, "backup", "defense"),
    ("Chandon Sullivan", "CB", team_id, 22, "backup", "defense"),
    ("T.J. Edwards", "LB", team_id, 51, "backup", "defense"),
    ("Richie Grant", "S", team_id, 36, "backup", "defense"),

    # Special Teams
    ("Mason Crosby", "K", team_id, 2, "starter", "special"),
    ("Corey Bojorquez", "P", team_id, 4, "starter", "special"),
    ("Caleb King", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Chicago Bears (CHI)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CHI",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Justin Fields", "QB", team_id, 1, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 4, "starter", "offense"),
    ("Khalil Herbert", "RB", team_id, 22, "starter", "offense"),
    ("DJ Moore", "WR", team_id, 11, "starter", "offense"),
    ("Equanimeous St. Brown", "WR", team_id, 14, "starter", "offense"),
    ("Cole Kmet", "TE", team_id, 85, "starter", "offense"),
    ("Jason Peters", "OL", team_id, 71, "starter", "offense"),
    ("Cody Whitehair", "OL", team_id, 65, "starter", "offense"),
    ("Laken Tomlinson", "OL", team_id, 63, "starter", "offense"),
    ("Derrick Barnes", "OL", team_id, 66, "starter", "offense"),
    ("Lucas Patrick", "OL", team_id, 67, "starter", "offense"),

    # Backup Offense
    ("Tyson Bagent", "QB", team_id, 7, "backup", "offense"),
    ("Roschon Johnson", "RB", team_id, 27, "backup", "offense"),
    ("Khalil Herbert", "RB", team_id, 22, "backup", "offense"),
    ("Nico Collins", "WR", team_id, 13, "backup", "offense"),
    ("Cole Kmet", "TE", team_id, 85, "backup", "offense"),

    # Starting Defense
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),
    ("Robert Quinn", "DL", team_id, 91, "starter", "defense"),
    ("Rashard Lawrence", "DL", team_id, 93, "starter", "defense"),
    ("Jaylon Johnson", "CB", team_id, 22, "starter", "defense"),
    ("Kyler Gordon", "CB", team_id, 20, "starter", "defense"),
    ("Jaquan Brisker", "S", team_id, 8, "starter", "defense"),
    (" Eddie Jackson", "S", team_id, 39, "starter", "defense"),

    # Backup Defense
    ("Tavon Wilson", "S", team_id, 28, "backup", "defense"),
    ("Khalil Davis", "DL", team_id, 92, "backup", "defense"),
    ("Tyler Shelvin", "DL", team_id, 94, "backup", "defense"),

    # Special Teams
    ("Cairo Santos", "K", team_id, 3, "starter", "special"),
    ("Tristan Vizcaino", "P", team_id, 6, "starter", "special"),
    ("LS", "Patrick Murray", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Minnesota Vikings (MIN)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("MIN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Kirk Cousins", "QB", team_id, 8, "starter", "offense"),
    ("Dalvin Cook", "RB", team_id, 33, "starter", "offense"),
    ("Alexander Mattison", "RB", team_id, 24, "starter", "offense"),
    ("Justin Jefferson", "WR", team_id, 18, "starter", "offense"),
    ("K.J. Osborn", "WR", team_id, 16, "starter", "offense"),
    ("T.J. Hockenson", "TE", team_id, 88, "starter", "offense"),
    ("Brian O’Neill", "OL", team_id, 77, "starter", "offense"),
    ("Zack Bailey", "OL", team_id, 60, "starter", "offense"),
    ("Ed Ingram", "OL", team_id, 73, "starter", "offense"),
    ("Daniel Brunskill", "OL", team_id, 66, "starter", "offense"),
    ("Christian Darrisaw", "OL", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("Brett Rypien", "QB", team_id, 5, "backup", "offense"),
    ("Ty Chandler", "RB", team_id, 28, "backup", "offense"),
    ("Chris Olave", "WR", team_id, 9, "backup", "offense"),
    ("Irv Smith Jr.", "TE", team_id, 82, "backup", "offense"),
    ("Aviante Collins", "OL", team_id, 65, "backup", "offense"),

    # Starting Defense
    ("Dalvin Tomlinson", "DL", team_id, 91, "starter", "defense"),
    ("Michael Pierce", "DL", team_id, 94, "starter", "defense"),
    ("Everson Griffen", "DL", team_id, 97, "starter", "defense"),
    ("Eric Kendricks", "LB", team_id, 54, "starter", "defense"),
    ("Jordan Hicks", "LB", team_id, 53, "starter", "defense"),
    ("Andrew Booth Jr.", "CB", team_id, 24, "starter", "defense"),
    ("Cam Dantzler", "CB", team_id, 29, "starter", "defense"),
    ("Harrison Smith", "S", team_id, 22, "starter", "defense"),
    ("Lewis Cine", "S", team_id, 30, "starter", "defense"),

    # Backup Defense
    ("Jonathan Greenard", "DL", team_id, 98, "backup", "defense"),
    ("Brian Asamoah", "LB", team_id, 50, "backup", "defense"),
    ("Andrew Booth Jr.", "CB", team_id, 24, "backup", "defense"),
    ("Camryn Bynum", "S", team_id, 25, "backup", "defense"),

    # Special Teams
    ("Brett Maher", "K", team_id, 4, "starter", "special"),
    ("Jordan Berry", "P", team_id, 6, "starter", "special"),
    ("LS", "Camaron Cheeseman", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Detroit Lions (DET)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DET",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jared Goff", "QB", team_id, 16, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 32, "starter", "offense"),
    ("Jahmyr Gibbs", "RB", team_id, 33, "starter", "offense"),
    ("Amon-Ra St. Brown", "WR", team_id, 14, "starter", "offense"),
    ("Quintez Cephus", "WR", team_id, 11, "starter", "offense"),
    ("T.J. Hockenson", "TE", team_id, 88, "starter", "offense"),
    ("Taylor Decker", "OL", team_id, 68, "starter", "offense"),
    ("Halapoulivaati Vaitai", "OL", team_id, 74, "starter", "offense"),
    ("Frank Ragnow", "OL", team_id, 79, "starter", "offense"),
    ("Penei Sewell", "OL", team_id, 58, "starter", "offense"),
    ("Alaric Jackson", "OL", team_id, 63, "starter", "offense"),

    # Backup Offense
    ("Tim Boyle", "QB", team_id, 6, "backup", "offense"),
    ("Craig Reynolds", "RB", team_id, 34, "backup", "offense"),
    ("Josh Reynolds", "WR", team_id, 19, "backup", "offense"),
    ("Cole Wick", "TE", team_id, 89, "backup", "offense"),
    ("Dan Skipper", "OL", team_id, 70, "backup", "offense"),

    # Starting Defense
    ("A’Shawn Robinson", "DL", team_id, 90, "starter", "defense"),
    ("Michael Brockers", "DL", team_id, 94, "starter", "defense"),
    ("Aidan Hutchinson", "DL", team_id, 97, "starter", "defense"),
    ("Jahlani Tavai", "LB", team_id, 54, "starter", "defense"),
    ("Alex Anzalone", "LB", team_id, 52, "starter", "defense"),
    ("Jeff Okudah", "CB", team_id, 24, "starter", "defense"),
    ("Chandon Sullivan", "CB", team_id, 22, "starter", "defense"),
    ("DeShon Elliott", "S", team_id, 33, "starter", "defense"),
    ("Troy Pride Jr.", "S", team_id, 30, "starter", "defense"),

    # Backup Defense
    ("Jahlani Tavai", "LB", team_id, 54, "backup", "defense"),
    ("Charles Harris", "DL", team_id, 95, "backup", "defense"),
    ("Jeff Okudah", "CB", team_id, 24, "backup", "defense"),

    # Special Teams
    ("Caleb Sturgis", "K", team_id, 3, "starter", "special"),
    ("Jack Fox", "P", team_id, 6, "starter", "special"),
    ("LS", "Cole Mazza", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)

# Tampa Bay Buccaneers (TB)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("TB",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Baker Mayfield", "QB", team_id, 6, "starter", "offense"),
    ("Rachaad White", "RB", team_id, 21, "starter", "offense"),
    ("Ke'Shawn Vaughn", "RB", team_id, 22, "starter", "offense"),
    ("Chris Godwin", "WR", team_id, 14, "starter", "offense"),
    ("Mike Evans", "WR", team_id, 13, "starter", "offense"),
    ("Cade Otton", "TE", team_id, 82, "starter", "offense"),
    ("Ryan Jensen", "OL", team_id, 65, "starter", "offense"),
    ("Ali Marpet", "OL", team_id, 61, "starter", "offense"),
    ("Tristan Wirfs", "OL", team_id, 77, "starter", "offense"),
    ("Caleb Benenoch", "OL", team_id, 68, "starter", "offense"),
    ("Zach Triner", "OL", team_id, 63, "starter", "offense"),

    # Backup Offense
    ("Kyle Trask", "QB", team_id, 4, "backup", "offense"),
    ("Rachaad White", "RB", team_id, 21, "backup", "offense"),
    ("Jaret Patterson", "RB", team_id, 33, "backup", "offense"),
    ("Tyler Johnson", "WR", team_id, 19, "backup", "offense"),
    ("Breshad Perriman", "WR", team_id, 81, "backup", "offense"),
    ("Jordan Leggett", "TE", team_id, 87, "backup", "offense"),

    # Starting Defense
    ("William Gholston", "DL", team_id, 97, "starter", "defense"),
    ("Vita Vea", "DL", team_id, 99, "starter", "defense"),
    ("Shaq Barrett", "LB", team_id, 55, "starter", "defense"),
    ("Devin White", "LB", team_id, 45, "starter", "defense"),
    ("Jamel Dean", "CB", team_id, 23, "starter", "defense"),
    ("Marcus Jones", "CB", team_id, 28, "starter", "defense"),
    ("Mike Edwards", "S", team_id, 27, "starter", "defense"),
    ("Jordan Whitehead", "S", team_id, 25, "starter", "defense"),

    # Backup Defense
    ("Calijah Kancey", "DL", team_id, 96, "backup", "defense"),
    ("Shaquil Barrett", "LB", team_id, 55, "backup", "defense"),
    ("Sean Murphy-Bunting", "CB", team_id, 24, "backup", "defense"),

    # Special Teams
    ("Ryan Succop", "K", team_id, 3, "starter", "special"),
    ("Corey Bojorquez", "P", team_id, 6, "starter", "special"),
    ("LS", "Louis-Jean", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# New Orleans Saints (NO)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NO",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Derek Carr", "QB", team_id, 4, "starter", "offense"),
    ("Alvin Kamara", "RB", team_id, 41, "starter", "offense"),
    ("Mark Ingram II", "RB", team_id, 22, "starter", "offense"),
    ("Chris Olave", "WR", team_id, 10, "starter", "offense"),
    ("Michael Thomas", "WR", team_id, 13, "starter", "offense"),
    ("Juwan Johnson", "TE", team_id, 82, "starter", "offense"),
    ("Cameron Tom", "OL", team_id, 63, "starter", "offense"),
    ("Ryan Ramczyk", "OL", team_id, 71, "starter", "offense"),
    ("Andrus Peat", "OL", team_id, 66, "starter", "offense"),
    ("Jahri Evans", "OL", team_id, 72, "starter", "offense"),
    ("Trevor Penning", "OL", team_id, 78, "starter", "offense"),

    # Backup Offense
    ("Andy Dalton", "QB", team_id, 14, "backup", "offense"),
    ("Tony Jones Jr.", "RB", team_id, 32, "backup", "offense"),
    ("Jarvis Landry", "WR", team_id, 80, "backup", "offense"),
    ("Adam Trautman", "TE", team_id, 85, "backup", "offense"),

    # Starting Defense
    ("Cameron Jordan", "DL", team_id, 94, "starter", "defense"),
    ("David Onyemata", "DL", team_id, 97, "starter", "defense"),
    ("Demario Davis", "LB", team_id, 56, "starter", "defense"),
    ("Pete Werner", "LB", team_id, 50, "starter", "defense"),
    ("Marshon Lattimore", "CB", team_id, 23, "starter", "defense"),
    ("Paulson Adebo", "CB", team_id, 25, "starter", "defense"),
    ("Marcus Maye", "S", team_id, 20, "starter", "defense"),
    ("C.J. Gardner-Johnson", "S", team_id, 29, "starter", "defense"),

    # Backup Defense
    ("Rashad Weaver", "DL", team_id, 93, "backup", "defense"),
    ("Tyrel Dodson", "LB", team_id, 57, "backup", "defense"),
    ("Paulson Adebo", "CB", team_id, 25, "backup", "defense"),

    # Special Teams
    ("Wil Lutz", "K", team_id, 3, "starter", "special"),
    ("Thomas Morstead", "P", team_id, 6, "starter", "special"),
    ("LS", "Harrison Smith", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Carolina Panthers (CAR)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CAR",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Baker Mayfield", "QB", team_id, 6, "starter", "offense"),
    ("Chuba Hubbard", "RB", team_id, 30, "starter", "offense"),
    ("Christian McCaffrey", "RB", team_id, 22, "starter", "offense"),
    ("DJ Moore", "WR", team_id, 11, "starter", "offense"),
    ("Rashard Higgins", "WR", team_id, 14, "starter", "offense"),
    ("Ian Thomas", "TE", team_id, 85, "starter", "offense"),
    ("Brandon Shell", "OL", team_id, 74, "starter", "offense"),
    ("Taylor Moton", "OL", team_id, 72, "starter", "offense"),
    ("John Miller", "OL", team_id, 60, "starter", "offense"),
    ("Ryan Kalil", "OL", team_id, 62, "starter", "offense"),
    ("Cameron Fleming", "OL", team_id, 66, "starter", "offense"),

    # Backup Offense
    ("Matt Corral", "QB", team_id, 5, "backup", "offense"),
    ("Jordan Scarlett", "RB", team_id, 28, "backup", "offense"),
    ("D.J. Chark", "WR", team_id, 19, "backup", "offense"),
    ("Chris Manhertz", "TE", team_id, 89, "backup", "offense"),

    # Starting Defense
    ("Brian Burns", "DE", team_id, 55, "starter", "defense"),
    ("Derrick Brown", "DL", team_id, 90, "starter", "defense"),
    ("Yetur Gross-Matos", "DE", team_id, 98, "starter", "defense"),
    ("Shaq Thompson", "LB", team_id, 44, "starter", "defense"),
    ("Jermaine Carter Jr.", "LB", team_id, 51, "starter", "defense"),
    ("Ahmad Gardner", "CB", team_id, 28, "starter", "defense"),
    ("Tariq Castro-Fields", "CB", team_id, 23, "starter", "defense"),
    ("Jeremy Chinn", "S", team_id, 21, "starter", "defense"),
    ("Antoine Brooks Jr.", "S", team_id, 26, "starter", "defense"),

    # Backup Defense
    ("Bravvion Roy", "DL", team_id, 99, "backup", "defense"),
    ("Devin Bush", "LB", team_id, 52, "backup", "defense"),
    ("Tony Fields II", "LB", team_id, 41, "backup", "defense"),
    ("C.J. Henderson", "CB", team_id, 20, "backup", "defense"),

    # Special Teams
    ("Eddy Pineiro", "K", team_id, 7, "starter", "special"),
    ("Joseph Charlton", "P", team_id, 2, "starter", "special"),
    ("LS", "Cam McCormick", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Denver Broncos (DEN)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DEN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Russell Wilson", "QB", team_id, 3, "starter", "offense"),
    ("Javonte Williams", "RB", team_id, 33, "starter", "offense"),
    ("Dameon Pierce", "RB", team_id, 31, "starter", "offense"),
    ("Jerry Jeudy", "WR", team_id, 10, "starter", "offense"),
    ("Courtland Sutton", "WR", team_id, 14, "starter", "offense"),
    ("Greg Dulcich", "TE", team_id, 84, "starter", "offense"),
    ("Garrett Bolles", "OL", team_id, 76, "starter", "offense"),
    ("Lloyd Cushenberry III", "OL", team_id, 63, "starter", "offense"),
    ("Trey Smith", "OL", team_id, 67, "starter", "offense"),
    ("Yodny Cajuste", "OL", team_id, 68, "starter", "offense"),
    ("Rashaad Penny", "OL", team_id, 72, "starter", "offense"),

    # Backup Offense
    ("Brett Rypien", "QB", team_id, 2, "backup", "offense"),
    ("Tyler Allgeier", "RB", team_id, 33, "backup", "offense"),
    ("KJ Hamler", "WR", team_id, 12, "backup", "offense"),
    ("Albert Okwuegbunam", "TE", team_id, 86, "backup", "offense"),

    # Starting Defense
    ("Bradley Chubb", "DE", team_id, 55, "starter", "defense"),
    ("Dre'Mont Jones", "DL", team_id, 94, "starter", "defense"),
    ("Justin Jones", "DL", team_id, 97, "starter", "defense"),
    ("Alexander Johnson", "LB", team_id, 54, "starter", "defense"),
    ("Josey Jewell", "LB", team_id, 58, "starter", "defense"),
    ("Patrick Surtain II", "CB", team_id, 24, "starter", "defense"),
    ("Kareem Jackson", "CB", team_id, 22, "starter", "defense"),
    ("Justin Simmons", "S", team_id, 31, "starter", "defense"),
    ("Caden Sterns", "S", team_id, 36, "starter", "defense"),

    # Backup Defense
    ("DeShawn Williams", "DL", team_id, 92, "backup", "defense"),
    ("Jonathon Cooper", "LB", team_id, 45, "backup", "defense"),
    ("Will Parks", "S", team_id, 33, "backup", "defense"),
    ("Issac Yiadom", "CB", team_id, 20, "backup", "defense"),

    # Special Teams
    ("Brandon McManus", "K", team_id, 5, "starter", "special"),
    ("Sam Martin", "P", team_id, 9, "starter", "special"),
    ("LS", "Austin Cutting", team_id, 46, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Kansas City Chiefs (KC)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("KC",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Patrick Mahomes", "QB", team_id, 15, "starter", "offense"),
    ("Isiah Pacheco", "RB", team_id, 22, "starter", "offense"),
    ("Clyde Edwards-Helaire", "RB", team_id, 25, "starter", "offense"),
    ("Marquez Valdes-Scantling", "WR", team_id, 11, "starter", "offense"),
    ("JuJu Smith-Schuster", "WR", team_id, 19, "starter", "offense"),
    ("Travis Kelce", "TE", team_id, 87, "starter", "offense"),
    ("Orlando Brown Jr.", "OL", team_id, 79, "starter", "offense"),
    ("Trey Smith", "OL", team_id, 71, "starter", "offense"),
    ("Creed Humphrey", "OL", team_id, 63, "starter", "offense"),
    ("Joe Thuney", "OL", team_id, 62, "starter", "offense"),
    ("Nick Allegretti", "OL", team_id, 61, "starter", "offense"),

    # Backup Offense
    ("Skyy Moore", "WR", team_id, 12, "backup", "offense"),
    ("Jerick McKinnon", "RB", team_id, 3, "backup", "offense"),
    ("Noah Gray", "TE", team_id, 88, "backup", "offense"),
    ("Lucas Niang", "OL", team_id, 66, "backup", "offense"),

    # Starting Defense
    ("Chris Jones", "DL", team_id, 95, "starter", "defense"),
    ("Frank Clark", "DE", team_id, 55, "starter", "defense"),
    ("Jarran Reed", "DL", team_id, 90, "starter", "defense"),
    ("Nick Bolton", "LB", team_id, 32, "starter", "defense"),
    ("Nick Scott", "S", team_id, 26, "starter", "defense"),
    ("L'Jarius Sneed", "CB", team_id, 22, "starter", "defense"),
    ("Taye Barber", "CB", team_id, 33, "starter", "defense"),
    ("Justin Reid", "S", team_id, 31, "starter", "defense"),

    # Backup Defense
    ("Tanoh Kpassagnon", "DL", team_id, 96, "backup", "defense"),
    ("Erik McCoy", "DL", team_id, 93, "backup", "defense"),
    ("Damien Wilson", "LB", team_id, 50, "backup", "defense"),
    ("Jaylen Watson", "CB", team_id, 29, "backup", "defense"),

    # Special Teams
    ("Harrison Butker", "K", team_id, 7, "starter", "special"),
    ("Tom Hackett", "P", team_id, 6, "starter", "special"),
    ("LS", "James Winchester", team_id, 46, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Las Vegas Raiders (LV)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("LV",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jimmy Garoppolo", "QB", team_id, 10, "starter", "offense"),
    ("Josh Jacobs", "RB", team_id, 28, "starter", "offense"),
    ("Davante Adams", "WR", team_id, 17, "starter", "offense"),
    ("Hunter Renfrow", "WR", team_id, 13, "starter", "offense"),
    ("Darren Waller", "TE", team_id, 83, "starter", "offense"),
    ("Alex Leatherwood", "OL", team_id, 71, "starter", "offense"),
    ("Foster Sarell", "OL", team_id, 77, "starter", "offense"),
    ("Denzelle Good", "OL", team_id, 72, "starter", "offense"),
    ("Laken Tomlinson", "OL", team_id, 65, "starter", "offense"),
    ("David Sharpe", "OL", team_id, 74, "starter", "offense"),
    ("John Simpson", "OL", team_id, 63, "starter", "offense"),

    # Backup Offense
    ("Jarrett Stidham", "QB", team_id, 4, "backup", "offense"),
    ("Kenyan Drake", "RB", team_id, 0, "backup", "offense"),
    ("Zay Flowers", "WR", team_id, 13, "backup", "offense"),
    ("Jakobi Meyers", "WR", team_id, 11, "backup", "offense"),
    ("Isaiah Likely", "TE", team_id, 88, "backup", "offense"),

    # Starting Defense
    ("Maxx Crosby", "DE", team_id, 98, "starter", "defense"),
    ("Quinton Jefferson", "DL", team_id, 94, "starter", "defense"),
    ("Derrick Brown", "DL", team_id, 99, "starter", "defense"),
    ("Denzel Perryman", "LB", team_id, 54, "starter", "defense"),
    ("Kenny Young", "LB", team_id, 53, "starter", "defense"),
    ("Trevin Moehrig", "S", team_id, 20, "starter", "defense"),
    ("Anthony Averett", "CB", team_id, 24, "starter", "defense"),
    ("Damarri Mathis", "CB", team_id, 27, "starter", "defense"),
    ("Nick Nelson", "S", team_id, 37, "starter", "defense"),

    # Backup Defense
    ("Foster Moreau", "DL", team_id, 86, "backup", "defense"),
    ("Divine Deablo", "LB", team_id, 43, "backup", "defense"),
    ("Damarri Mathis", "CB", team_id, 27, "backup", "defense"),
    ("Duron Harmon", "S", team_id, 28, "backup", "defense"),

    # Special Teams
    ("Daniel Carlson", "K", team_id, 2, "starter", "special"),
    ("AJ Cole", "P", team_id, 3, "starter", "special"),
    ("LS", "Matt Nelson", team_id, 48, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)


# Los Angeles Chargers (LAC)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("LAC",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Justin Herbert", "QB", team_id, 10, "starter", "offense"),
    ("Austin Ekeler", "RB", team_id, 30, "starter", "offense"),
    ("Joshua Kelley", "RB", team_id, 27, "starter", "offense"),
    ("Keenan Allen", "WR", team_id, 13, "starter", "offense"),
    ("Mike Williams", "WR", team_id, 81, "starter", "offense"),
    ("Jalen Guyton", "WR", team_id, 14, "starter", "offense"),
    ("Gerald Everett", "TE", team_id, 85, "starter", "offense"),
    ("Rashawn Slater", "OL", team_id, 70, "starter", "offense"),
    ("Matt Feiler", "OL", team_id, 65, "starter", "offense"),
    ("Oday Aboushi", "OL", team_id, 67, "starter", "offense"),
    ("Corey Linsley", "OL", team_id, 63, "starter", "offense"),

    # Backup Offense
    ("Easton Stick", "QB", team_id, 4, "backup", "offense"),
    ("Isaiah Spiller", "RB", team_id, 21, "backup", "offense"),
    ("Josh Palmer", "WR", team_id, 82, "backup", "offense"),
    ("Tre' McKitty", "TE", team_id, 89, "backup", "offense"),
    ("Bryan Bulaga", "OL", team_id, 75, "backup", "offense"),

    # Starting Defense
    ("Joey Bosa", "DE", team_id, 97, "starter", "defense"),
    ("Rashawn Slater", "DL", team_id, 70, "starter", "defense"),
    ("Javon Hargrave", "DL", team_id, 91, "starter", "defense"),
    ("Kenneth Murray", "LB", team_id, 3, "starter", "defense"),
    ("Drue Tranquill", "LB", team_id, 6, "starter", "defense"),
    ("Derwin James", "S", team_id, 33, "starter", "defense"),
    ("JC Jackson", "CB", team_id, 24, "starter", "defense"),
    ("Asante Samuel Jr.", "CB", team_id, 21, "starter", "defense"),
    ("Nasir Adderley", "S", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Myles Garrett", "DL", team_id, 95, "backup", "defense"),
    ("Uchenna Nwosu", "LB", team_id, 50, "backup", "defense"),
    ("Michael Davis", "CB", team_id, 26, "backup", "defense"),
    ("Nick Niemann", "LB", team_id, 51, "backup", "defense"),

    # Special Teams
    ("Graham Gano", "K", team_id, 9, "starter", "special"),
    ("Tyler Bass", "P", team_id, 6, "starter", "special"),
    ("LS", "Cole Mazza", team_id, 48, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)

# Dallas Cowboys (DAL)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DAL",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Dak Prescott", "QB", team_id, 4, "starter", "offense"),
    ("Tony Pollard", "RB", team_id, 20, "starter", "offense"),
    ("CeeDee Lamb", "WR", team_id, 88, "starter", "offense"),
    ("Michael Gallup", "WR", team_id, 11, "starter", "offense"),
    ("Trey Sermon", "RB", team_id, 25, "starter", "offense"),
    ("Jake Ferguson", "TE", team_id, 87, "starter", "offense"),
    ("Tyler Smith", "LT", team_id, 73, "starter", "offense"),
    ("Tyron Smith", "LG", team_id, 77, "starter", "offense"),
    ("Tyler Biadasz", "C", team_id, 63, "starter", "offense"),
    ("Zack Martin", "RG", team_id, 70, "starter", "offense"),
    ("Terence Steele", "RT", team_id, 78, "starter", "offense"),

    # Backup Offense
    ("Cooper Rush", "QB", team_id, 7, "backup", "offense"),
    ("Jalen Tolbert", "WR", team_id, 80, "backup", "offense"),
    ("Kendre Miller", "RB", team_id, 32, "backup", "offense"),
    ("Chuma Edoga", "OL", team_id, 70, "backup", "offense"),
    
    # Starting Defense
    ("Micah Parsons", "LB", team_id, 11, "starter", "defense"),
    ("Osa Odighizuwa", "DT", team_id, 97, "starter", "defense"),
    ("Mazi Smith", "DT", team_id, 58, "starter", "defense"),
    ("Dante Fowler Jr.", "DE", team_id, 13, "starter", "defense"),
    ("Damone Clark", "LB", team_id, 18, "starter", "defense"),
    ("Leighton Vander Esch", "LB", team_id, 55, "starter", "defense"),
    ("Trevon Diggs", "CB", team_id, 7, "starter", "defense"),
    ("DaRon Bland", "CB", team_id, 26, "starter", "defense"),
    ("Malik Hooker", "S", team_id, 28, "starter", "defense"),
    ("Jayron Kearse", "S", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Sam Williams", "DE", team_id, 54, "backup", "defense"),
    ("Solomon Thomas", "DT", team_id, 90, "backup", "defense"),
    ("Micah McFadden", "LB", team_id, 41, "backup", "defense"),
    ("Kelvin Joseph", "CB", team_id, 1, "backup", "defense"),
    ("Israel Mukuamu", "S", team_id, 24, "backup", "defense"),

    # Special Teams
    ("Brandon Aubrey", "K", team_id, 5, "starter", "special"),
    ("Bryan Anger", "P", team_id, 5, "starter", "special"),
    ("Jake McQuaide", "LS", team_id, 44, "starter", "special"),
]

# Philadelphia Eagles (PHI)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("PHI",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jalen Hurts", "QB", team_id, 1, "starter", "offense"),
    ("Miles Sanders", "RB", team_id, 26, "starter", "offense"),
    ("A.J. Brown", "WR", team_id, 11, "starter", "offense"),
    ("DeVonta Smith", "WR", team_id, 6, "starter", "offense"),
    ("Dallas Goedert", "TE", team_id, 88, "starter", "offense"),
    ("Jordan Mailata", "LT", team_id, 71, "starter", "offense"),
    ("Landon Dickerson", "LG", team_id, 53, "starter", "offense"),
    ("Jason Kelce", "C", team_id, 62, "starter", "offense"),
    ("Brandon Brooks", "RG", team_id, 68, "starter", "offense"),
    ("Lane Johnson", "RT", team_id, 65, "starter", "offense"),

    # Backup Offense
    ("Gardner Minshew", "QB", team_id, 10, "backup", "offense"),
    ("Kenneth Gainwell", "RB", team_id, 25, "backup", "offense"),
    ("Quez Watkins", "WR", team_id, 14, "backup", "offense"),
    ("Greg Ward", "WR", team_id, 3, "backup", "offense"),

    # Starting Defense
    ("Fletcher Cox", "DT", team_id, 91, "starter", "defense"),
    ("Javon Hargrave", "DT", team_id, 99, "starter", "defense"),
    ("Haason Reddick", "LB", team_id, 44, "starter", "defense"),
    ("T.J. Edwards", "LB", team_id, 53, "starter", "defense"),
    ("Cameron Malveaux", "DE", team_id, 92, "starter", "defense"),
    ("Darius Slay", "CB", team_id, 2, "starter", "defense"),
    ("James Bradberry", "CB", team_id, 24, "starter", "defense"),
    ("Anthony Harris", "S", team_id, 41, "starter", "defense"),
    ("Marcus Epps", "S", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Brandon Graham", "DE", team_id, 55, "backup", "defense"),
    ("Josh Sweat", "DE", team_id, 11, "backup", "defense"),
    ("Nickell Robey-Coleman", "CB", team_id, 21, "backup", "defense"),
    ("JaCoby Stevens", "S", team_id, 7, "backup", "defense"),

    # Special Teams
    ("Jake Elliott", "K", team_id, 4, "starter", "special"),
    ("Arryn Siposs", "P", team_id, 10, "starter", "special"),
    ("Cam Johnston", "LS", team_id, 49, "starter", "special"),
]


# Washington Commanders (WAS)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("WAS",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Sam Howell", "QB", team_id, 4, "starter", "offense"),
    ("Brian Robinson Jr.", "RB", team_id, 33, "starter", "offense"),
    ("Terry McLaurin", "WR", team_id, 17, "starter", "offense"),
    ("Jahan Dotson", "WR", team_id, 19, "starter", "offense"),
    ("Logan Thomas", "TE", team_id, 81, "starter", "offense"),
    ("Charles Leno Jr.", "LT", team_id, 68, "starter", "offense"),
    ("Ethan Pocic", "LG", team_id, 61, "starter", "offense"),
    ("Tucker Kraft", "C", team_id, 60, "starter", "offense"),
    ("Chris Paul", "RG", team_id, 62, "starter", "offense"),
    ("Trey Smith", "RT", team_id, 74, "starter", "offense"),

    # Backup Offense
    ("Taylor Heinicke", "QB", team_id, 4, "backup", "offense"),
    ("Antonio Gibson", "RB", team_id, 22, "backup", "offense"),
    ("Dyami Brown", "WR", team_id, 13, "backup", "offense"),

    # Starting Defense
    ("Daron Payne", "DT", team_id, 93, "starter", "defense"),
    ("Jonathan Allen", "DE", team_id, 97, "starter", "defense"),
    ("Jamin Davis", "LB", team_id, 56, "starter", "defense"),
    ("Cole Holcomb", "LB", team_id, 58, "starter", "defense"),
    ("Barkevious Mingo", "LB", team_id, 51, "starter", "defense"),
    ("Chandon Sullivan", "CB", team_id, 26, "starter", "defense"),
    ("Sammy Davis", "CB", team_id, 23, "starter", "defense"),
    ("Kamren Curl", "S", team_id, 21, "starter", "defense"),
    ("Tress Way", "S", team_id, 23, "starter", "defense"),

    # Backup Defense
    ("Jonathan Allen", "DE", team_id, 97, "backup", "defense"),
    ("Jamin Davis", "LB", team_id, 56, "backup", "defense"),

    # Special Teams
    ("Tress Way", "P", team_id, 4, "starter", "special"),
    ("Jose Borregales", "K", team_id, 3, "starter", "special"),
    ("Nick Sundberg", "LS", team_id, 48, "starter", "special"),
]


# New York Giants (NYG)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NYG",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Daniel Jones", "QB", team_id, 8, "starter", "offense"),
    ("Saquon Barkley", "RB", team_id, 26, "starter", "offense"),
    ("Wan'Dale Robinson", "WR", team_id, 10, "starter", "offense"),
    ("Kenny Golladay", "WR", team_id, 19, "starter", "offense"),
    ("Evan Engram", "TE", team_id, 88, "starter", "offense"),
    ("Andrew Thomas", "LT", team_id, 73, "starter", "offense"),
    ("Ben Bredeson", "LG", team_id, 61, "starter", "offense"),
    ("Nick Gates", "C", team_id, 60, "starter", "offense"),
    ("Will Hernandez", "RG", team_id, 66, "starter", "offense"),
    ("Jake Brendel", "RT", team_id, 76, "starter", "offense"),

    # Backup Offense
    ("Tyler Huntley", "QB", team_id, 4, "backup", "offense"),
    ("Gary Brightwell", "RB", team_id, 27, "backup", "offense"),
    ("Darius Slayton", "WR", team_id, 86, "backup", "offense"),

    # Starting Defense
    ("Dexter Lawrence", "DT", team_id, 90, "starter", "defense"),
    ("Leonard Williams", "DT", team_id, 99, "starter", "defense"),
    ("Azeez Ojulari", "LB", team_id, 45, "starter", "defense"),
    ("Jarrad Davis", "LB", team_id, 52, "starter", "defense"),
    ("Kayvon Thibodeaux", "LB", team_id, 5, "starter", "defense"),
    ("Adoree' Jackson", "CB", team_id, 21, "starter", "defense"),
    ("Darnay Holmes", "CB", team_id, 26, "starter", "defense"),
    ("Xavier McKinney", "S", team_id, 25, "starter", "defense"),
    ("J.R. Reed", "S", team_id, 30, "starter", "defense"),

    # Backup Defense
    ("Tariq Woolen", "CB", team_id, 27, "backup", "defense"),
    ("Cam Brown", "LB", team_id, 56, "backup", "defense"),

    # Special Teams
    ("Graham Gano", "K", team_id, 3, "starter", "special"),
    ("Tyler Conklin", "P", team_id, 2, "starter", "special"),
    ("Patrick Murray", "LS", team_id, 49, "starter", "special"),
]


# Green Bay Packers (GB)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("GB",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jordan Love", "QB", team_id, 10, "starter", "offense"),
    ("Aaron Jones", "RB", team_id, 33, "starter", "offense"),
    ("AJ Dillon", "RB", team_id, 28, "starter", "offense"),
    ("Christian Watson", "WR", team_id, 11, "starter", "offense"),
    ("Romeo Doubs", "WR", team_id, 17, "starter", "offense"),
    ("Robert Tonyan", "TE", team_id, 85, "starter", "offense"),
    ("David Bakhtiari", "LT", team_id, 69, "starter", "offense"),
    ("Lucas Patrick", "LG", team_id, 64, "starter", "offense"),
    ("Corey Linsley", "C", team_id, 63, "starter", "offense"),
    ("Elgton Jenkins", "RG", team_id, 77, "starter", "offense"),
    ("Dan Moore", "RT", team_id, 72, "starter", "offense"),

    # Backup Offense
    ("Devon Achane", "RB", team_id, 23, "backup", "offense"),
    ("Romeo Doubs", "WR", team_id, 17, "backup", "offense"),
    ("Samori Toure", "WR", team_id, 83, "backup", "offense"),

    # Starting Defense
    ("Jaire Alexander", "CB", team_id, 23, "starter", "defense"),
    ("Kingsley Enagbare", "DE", team_id, 55, "starter", "defense"),
    ("Rashan Gary", "OLB", team_id, 98, "starter", "defense"),
    ("Devonte Wyatt", "DT", team_id, 95, "starter", "defense"),
    ("Quay Walker", "LB", team_id, 34, "starter", "defense"),
    ("Chandon Sullivan", "CB", team_id, 39, "starter", "defense"),
    ("Jaire Alexander", "CB", team_id, 23, "starter", "defense"),
    ("Jamel Dean", "S", team_id, 27, "starter", "defense"),
    ("Darnell Savage", "S", team_id, 6, "starter", "defense"),

    # Backup Defense
    ("Kenny Clark", "DT", team_id, 97, "backup", "defense"),
    ("Rashan Gary", "OLB", team_id, 98, "backup", "defense"),

    # Special Teams
    ("Mason Crosby", "K", team_id, 2, "starter", "special"),
    ("Corey Bojorquez", "P", team_id, 9, "starter", "special"),
    ("Hunter Bradley", "LS", team_id, 44, "starter", "special"),
]

# Minnesota Vikings (MIN)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("MIN",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Kirk Cousins", "QB", team_id, 8, "starter", "offense"),
    ("Dalvin Cook", "RB", team_id, 33, "starter", "offense"),
    ("Alexander Mattison", "RB", team_id, 28, "starter", "offense"),
    ("Justin Jefferson", "WR", team_id, 18, "starter", "offense"),
    ("K.J. Osborn", "WR", team_id, 11, "starter", "offense"),
    ("T.J. Hockenson", "TE", team_id, 88, "starter", "offense"),
    ("Christian Darrisaw", "LT", team_id, 73, "starter", "offense"),
    ("Elijah Wilkinson", "LG", team_id, 68, "starter", "offense"),
    ("Ezra Cleveland", "C", team_id, 66, "starter", "offense"),
    ("Austin Schlottmann", "RG", team_id, 60, "starter", "offense"),
    ("Brian O'Neill", "RT", team_id, 76, "starter", "offense"),

    # Backup Offense
    ("Kenny Willekes", "LB", team_id, 44, "backup", "defense"),

    # Starting Defense
    ("Dalvin Tomlinson", "DT", team_id, 92, "starter", "defense"),
    ("Jeffrey Simmons", "DE", team_id, 94, "starter", "defense"),
    ("Jordan Hicks", "LB", team_id, 50, "starter", "defense"),
    ("Eric Kendricks", "LB", team_id, 54, "starter", "defense"),
    ("Patrick Peterson", "CB", team_id, 21, "starter", "defense"),
    ("Cam Dantzler", "CB", team_id, 22, "starter", "defense"),
    ("Harrison Smith", "S", team_id, 22, "starter", "defense"),
    ("Lewis Cine", "S", team_id, 30, "starter", "defense"),

    # Backup Defense
    ("D.J. Wonnum", "DE", team_id, 93, "backup", "defense"),

    # Special Teams
    ("Greg Joseph", "K", team_id, 4, "starter", "special"),
    ("Jordan Berry", "P", team_id, 6, "starter", "special"),
    ("Chris Massey", "LS", team_id, 48, "starter", "special"),
]

# Chicago Bears (CHI)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CHI",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Justin Fields", "QB", team_id, 1, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 32, "starter", "offense"),
    ("Khalil Herbert", "RB", team_id, 24, "starter", "offense"),
    ("Darnell Mooney", "WR", team_id, 11, "starter", "offense"),
    ("Equanimeous St. Brown", "WR", team_id, 14, "starter", "offense"),
    ("Cole Kmet", "TE", team_id, 85, "starter", "offense"),
    ("Teven Jenkins", "LT", team_id, 70, "starter", "offense"),
    ("James Daniels", "LG", team_id, 67, "starter", "offense"),
    ("Lamar Jackson", "C", team_id, 52, "starter", "offense"),
    ("Cody Whitehair", "RG", team_id, 55, "starter", "offense"),
    ("Charles Leno Jr.", "RT", team_id, 68, "starter", "offense"),

    # Backup Offense
    ("Justin Fields", "QB", team_id, 1, "backup", "offense"),
    ("Roschon Johnson", "RB", team_id, 34, "backup", "offense"),

    # Starting Defense
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),
    ("Roquan Smith", "LB", team_id, 58, "starter", "defense"),

    # Backup Defense
    ("Khalil Mack", "OLB", team_id, 52, "backup", "defense"),

    # Special Teams
    ("Riley Patterson", "K", team_id, 2, "starter", "special"),
    ("Tristan Vizcaino", "P", team_id, 3, "starter", "special"),
    ("Charlie Woerner", "LS", team_id, 49, "starter", "special"),
]

# Detroit Lions (DET)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("DET",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Jared Goff", "QB", team_id, 16, "starter", "offense"),
    ("David Montgomery", "RB", team_id, 32, "starter", "offense"),
    ("Amon-Ra St. Brown", "WR", team_id, 14, "starter", "offense"),
    ("Jameson Williams", "WR", team_id, 11, "starter", "offense"),
    ("T.J. Hockenson", "TE", team_id, 88, "starter", "offense"),
    ("Taylor Decker", "LT", team_id, 68, "starter", "offense"),
    ("Halapoulivaati Vaitai", "LG", team_id, 76, "starter", "offense"),
    ("Frank Ragnow", "C", team_id, 75, "starter", "offense"),
    ("Ben Skowronek", "RG", team_id, 64, "starter", "offense"),
    ("Penei Sewell", "RT", team_id, 58, "starter", "offense"),

    # Backup Offense
    ("Geno Smith", "QB", team_id, 3, "backup", "offense"),

    # Starting Defense
    ("Aidan Hutchinson", "DE", team_id, 97, "starter", "defense"),
    ("Penei Sewell", "DT", team_id, 98, "starter", "defense"),
    ("Jahlani Tavai", "LB", team_id, 53, "starter", "defense"),
    ("Tremaine Edmunds", "LB", team_id, 49, "starter", "defense"),
    ("Jeff Okudah", "CB", team_id, 37, "starter", "defense"),
    ("Amani Oruwariye", "CB", team_id, 29, "starter", "defense"),
    ("DeShon Elliott", "S", team_id, 31, "starter", "defense"),
    ("Taylor Rapp", "S", team_id, 23, "starter", "defense"),

    # Backup Defense
    ("Charles Harris", "DE", team_id, 56, "backup", "defense"),

    # Special Teams
    ("Caleb Sturgis", "K", team_id, 3, "starter", "special"),
    ("Sam Martin", "P", team_id, 6, "starter", "special"),
    ("Jack Fox", "LS", team_id, 49, "starter", "special"),
]

# Tampa Bay Buccaneers (TB)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("TB",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Baker Mayfield", "QB", team_id, 6, "starter", "offense"),
    ("Rachaad White", "RB", team_id, 32, "starter", "offense"),
    ("Ke'Shawn Vaughn", "RB", team_id, 28, "starter", "offense"),
    ("Mike Evans", "WR", team_id, 13, "starter", "offense"),
    ("Chris Godwin", "WR", team_id, 14, "starter", "offense"),
    ("Cade Otton", "TE", team_id, 82, "starter", "offense"),
    ("Tristan Wirfs", "LT", team_id, 77, "starter", "offense"),
    ("Aaron Stinnie", "LG", team_id, 63, "starter", "offense"),
    ("Ryan Jensen", "C", team_id, 60, "starter", "offense"),
    ("Zach Triner", "RG", team_id, 71, "starter", "offense"),
    ("Caleb Benenoch", "RT", team_id, 79, "starter", "offense"),

    # Backup Offense
    ("Baker Mayfield", "QB", team_id, 6, "backup", "offense"),
    ("Rachaad White", "RB", team_id, 32, "backup", "offense"),

    # Starting Defense
    ("William Gholston", "DE", team_id, 91, "starter", "defense"),
    ("Shaq Barrett", "OLB", team_id, 55, "starter", "defense"),
    ("Calais Campbell", "DE", team_id, 93, "starter", "defense"),
    ("Kris Boyd", "DT", team_id, 96, "starter", "defense"),
    ("Lavonte David", "LB", team_id, 54, "starter", "defense"),
    ("Devin White", "LB", team_id, 45, "starter", "defense"),
    ("Jamel Dean", "CB", team_id, 21, "starter", "defense"),
    ("Richard Sherman", "CB", team_id, 25, "starter", "defense"),
    ("Antoine Winfield Jr.", "S", team_id, 31, "starter", "defense"),
    ("Mike Edwards", "S", team_id, 28, "starter", "defense"),

    # Backup Defense
    ("Shaquil Barrett", "OLB", team_id, 55, "backup", "defense"),

    # Special Teams
    ("Ryan Succop", "K", team_id, 3, "starter", "special"),
    ("Jake Camarda", "P", team_id, 6, "starter", "special"),
    ("Thomas Hennessy", "LS", team_id, 49, "starter", "special"),
]

# New Orleans Saints (NO)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("NO",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Derek Carr", "QB", team_id, 4, "starter", "offense"),
    ("Alvin Kamara", "RB", team_id, 41, "starter", "offense"),
    ("Tyrion Davis-Price", "RB", team_id, 33, "starter", "offense"),
    ("Chris Olave", "WR", team_id, 10, "starter", "offense"),
    ("Marquez Callaway", "WR", team_id, 19, "starter", "offense"),
    ("Juwan Johnson", "TE", team_id, 86, "starter", "offense"),
    ("Ryan Ramczyk", "LT", team_id, 71, "starter", "offense"),
    ("Andrus Peat", "LG", team_id, 68, "starter", "offense"),
    ("Cesar Ruiz", "C", team_id, 60, "starter", "offense"),
    ("Larry Warford", "RG", team_id, 73, "starter", "offense"),
    ("Trevor Penning", "RT", team_id, 79, "starter", "offense"),

    # Backup Offense
    ("Derek Carr", "QB", team_id, 4, "backup", "offense"),

    # Starting Defense
    ("Cameron Jordan", "DE", team_id, 94, "starter", "defense"),
    ("Payne", "DT", team_id, 98, "starter", "defense"),
    ("Malcolm Roach", "DT", team_id, 99, "starter", "defense"),
    ("Demario Davis", "LB", team_id, 56, "starter", "defense"),
    ("Kwon Alexander", "LB", team_id, 50, "starter", "defense"),
    ("Paulson Adebo", "CB", team_id, 23, "starter", "defense"),
    ("Marshon Lattimore", "CB", team_id, 23, "starter", "defense"),
    ("Marcus Maye", "S", team_id, 22, "starter", "defense"),
    ("Malcolm Jenkins", "S", team_id, 27, "starter", "defense"),

    # Backup Defense
    ("Payne", "DT", team_id, 98, "backup", "defense"),

    # Special Teams
    ("Wil Lutz", "K", team_id, 3, "starter", "special"),
    ("Thomas Morstead", "P", team_id, 6, "starter", "special"),
    ("Jon Dorenbos", "LS", team_id, 49, "starter", "special"),
]

# Carolina Panthers (CAR)
cursor.execute("SELECT id FROM teams WHERE abbreviation = ?", ("CAR",))
team_id = cursor.fetchone()[0]

players = [
    # Starting Offense
    ("Baker Mayfield", "QB", team_id, 6, "starter", "offense"),
    ("Christian McCaffrey", "RB", team_id, 22, "starter", "offense"),
    ("Chuba Hubbard", "RB", team_id, 30, "starter", "offense"),
    ("DJ Moore", "WR", team_id, 12, "starter", "offense"),
    ("Rashard Higgins", "WR", team_id, 17, "starter", "offense"),
    ("Ian Thomas", "TE", team_id, 82, "starter", "offense"),
    ("Taylor Moton", "LT", team_id, 71, "starter", "offense"),
    ("Pat Elflein", "LG", team_id, 69, "starter", "offense"),
    ("Matt Paradis", "C", team_id, 63, "starter", "offense"),
    ("John Miller", "RG", team_id, 65, "starter", "offense"),
    ("Taylor Moton", "RT", team_id, 71, "starter", "offense"),

    # Backup Offense
    ("P.J. Walker", "QB", team_id, 4, "backup", "offense"),

    # Starting Defense
    ("Brian Burns", "DE", team_id, 55, "starter", "defense"),
    ("DaQuan Jones", "DT", team_id, 95, "starter", "defense"),
    ("Dontari Poe", "DT", team_id, 97, "starter", "defense"),
    ("Shaq Thompson", "LB", team_id, 42, "starter", "defense"),
    ("Rashad Smith", "LB", team_id, 51, "starter", "defense"),
    ("Jaycee Horn", "CB", team_id, 23, "starter", "defense"),
    ("C.J. Henderson", "CB", team_id, 22, "starter", "defense"),
    ("Jeremy Chinn", "S", team_id, 21, "starter", "defense"),
    ("Trevor Williams", "S", team_id, 37, "starter", "defense"),

    # Backup Defense
    ("DaQuan Jones", "DT", team_id, 95, "backup", "defense"),

    # Special Teams
    ("Eddy Pineiro", "K", team_id, 3, "starter", "special"),
    ("Cam Brown", "P", team_id, 6, "starter", "special"),
    ("J.J. Jansen", "LS", team_id, 49, "starter", "special"),
]

cursor.executemany("""
    INSERT INTO players (name, position, number, team_id, role, side)
    VALUES (?, ?, ?, ?, ?, ?)
""", players)

conn.commit()
conn.close()