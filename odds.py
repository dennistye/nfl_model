import requests
import sqlite3
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import re
import numpy as np


url = f"https://www.espn.com/nfl/odds"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36"
}



res = requests.get(url, headers=headers)
soup = BeautifulSoup(res.text, "html.parser")
# Find the <script> tag that contains the JSON
# Find the script tag that contains 'odds'
#print(soup)
# Method 1: Convert soup to string and split by lines
# lines = str(soup).splitlines()
# first_few = lines[:78]  # get first 5 lines
# for line in first_few:
#     print(line)

# print(soup)

df = pd.DataFrame()
values = []



nums = []

# Get game details
for game in soup.select('[data-track-extras]'):
    extras = game['data-track-extras']
    game_info = json.loads(extras)
    print(game_info)
    nums.append(game_info)

df = pd.DataFrame(nums)

#Get each OddsCell
for odds_cell in soup.select('[data-testid="OddsCell"]'):
    moneyline = odds_cell.select_one('.FTMw.FuEs')
    if moneyline:
        print(moneyline.text)
        values.append(moneyline.text)

# df["moneyline"] = values[:len(df)]  # match length just in case
df = df.drop_duplicates()
df = df.loc[np.repeat(df.index, 8)].reset_index(drop=True)
df["moneyline"] = values[:len(df)]  # match length just in case

df.to_csv("games.csv", index=False)

# games = []

# for game_div in soup.select('[data-testid="betSixPack-1"]'):
#     game = {}
    
#     # Get team names safely
#     team_links = game_div.select('a[data-testid="prism-linkbase"]')
#     team_names = []
#     for a in team_links:
#         span = a.find('span')
#         if span and span.text.strip():  # only include non-empty spans
#             team_names.append(span.text.strip())
#     if len(team_names) >= 2:
#         game['away_team'] = team_names[0]
#         game['home_team'] = team_names[1]
    
#     # Get odds
#     odds_cells = game_div.select('[data-testid="OddsCell"]')
#     game['spread'] = odds_cells[0].text.strip() if len(odds_cells) > 0 else None
#     game['total'] = odds_cells[1].text.strip() if len(odds_cells) > 1 else None
#     game['moneyline'] = odds_cells[2].text.strip() if len(odds_cells) > 2 else None
    
#     # Get game id
#     track_extra = game_div.select_one('[data-track-extras]')
#     if track_extra:
#         try:
#             game_info = json.loads(track_extra['data-track-extras'])
#             game['game_id'] = game_info['game_detail'].split()[0]
#         except Exception:
#             game['game_id'] = None
    
#     games.append(game)

# df = pd.DataFrame(games)
# print(df)



# def structure_espn_odds(raw_json):
#     """
#     Takes raw ESPN odds JSON (as dict) and converts it into a structured
#     Python multiline string like the json_snippet format.
#     """
#     structured = {"page": {"content": {"gameOdds": []}}}
    
#     for game in raw_json.get("gameOdd", []):
#         game_entry = {
#             "id": game.get("gameId", ""),
#             "uid": f"s:20~l:28~e:{game.get('gameId','')}~c:{game.get('gameId','')}",
#             "date": game.get("labels", {}).get("line", ""),
#             "gameInfoWithoutId": game.get("gameInfoWithoutId", ""),
#             "odds": []
#         }
        
#         for team in game.get("odds", []):
#             odds_entry = {
#                 "line": {
#                     "primaryText": team["line"]["primaryText"],
#                     "primaryTextFull": team["line"]["primaryTextFull"],
#                     "primaryTextFullWide": team["line"].get("primaryTextFullWide", ""),
#                     "secondaryText": team["line"].get("secondaryText", ""),
#                     "link": team["line"].get("link", "")
#                 },
#                 "pointSpread": {
#                     "primary": team.get("pointSpread", {}).get("primary", ""),
#                     "secondary": team.get("pointSpread", {}).get("secondary", ""),
#                     "mktTxt": team.get("pointSpread", {}).get("mktTxt", "")
#                 },
#                 "moneyline": {
#                     "primary": team.get("moneyline", {}).get("primary", ""),
#                     "mktTxt": team.get("moneyline", {}).get("mktTxt", "")
#                 },
#                 "total": {
#                     "primary": team.get("total", {}).get("primary", ""),
#                     "secondary": team.get("total", {}).get("secondary", ""),
#                     "mktTxt": team.get("total", {}).get("mktTxt", "")
#                 }
#             }
#             game_entry["odds"].append(odds_entry)
        
#         structured["page"]["content"]["gameOdds"].append(game_entry)
    
#     # Convert to pretty-printed JSON string for Python
#     return f"json_snippet = '''\n{json.dumps(structured, indent=2)}\n'''"

# print(structure_espn_odds(soup))


# json_snippet = '''
# {
#   "page": {
#     "content": {
#       "gameOdds": [
#         {
#           "id": "401772851",
#           "uid": "s:20~l:28~e:401772851~c:401772851",
#           "date": "2025-10-05T17:00Z",
#           "gameInfoWithoutId": "Las Vegas Raiders vs Indianapolis Colts",
#           "odds": [
#             {
#               "line": {
#                 "primaryText": "LV",
#                 "primaryTextFull": "Raiders",
#                 "primaryTextFullWide": "Las Vegas Raiders",
#                 "secondaryText": "(0-0)",
#                 "link": "https://www.espn.com/nfl/team/_/name/lv/las-vegas-raiders"
#               },
#               "pointSpread": {
#                 "primary": "+1.5",
#                 "secondary": "-105",
#                 "mktTxt": "pointSpread:LV +1.5"
#               },
#               "moneyline": {
#                 "primary": "+115",
#                 "mktTxt": "moneyline:LV +115"
#               },
#               "total": {
#                 "primary": "o44.5",
#                 "secondary": "-110",
#                 "mktTxt": "total:o44.5"
#               }
#             },
#             {
#               "line": {
#                 "primaryText": "IND",
#                 "primaryTextFull": "Colts",
#                 "primaryTextFullWide": "Indianapolis Colts",
#                 "secondaryText": "(0-0)",
#                 "link": "https://www.espn.com/nfl/team/_/name/ind/indianapolis-colts"
#               },
#               "pointSpread": {
#                 "primary": "-1.5",
#                 "secondary": "-115",
#                 "mktTxt": "pointSpread:IND -1.5"
#               },
#               "moneyline": {
#                 "primary": "-135",
#                 "mktTxt": "moneyline:IND -135"
#               },
#               "total": {
#                 "primary": "u44.5",
#                 "secondary": "-110",
#                 "mktTxt": "total:u44.5"
#               }
#             }
#           ]
#         },
#         {
#           "id": "401772852",
#           "uid": "s:20~l:28~e:401772852~c:401772852",
#           "date": "2025-10-05T17:00Z",
#           "gameInfoWithoutId": "Miami Dolphins vs Carolina Panthers",
#           "odds": [
#             {
#               "line": {
#                 "primaryText": "MIA",
#                 "primaryTextFull": "Dolphins",
#                 "primaryTextFullWide": "Miami Dolphins",
#                 "secondaryText": "(0-0)",
#                 "link": "https://www.espn.com/nfl/team/_/name/mia/miami-dolphins"
#               },
#               "pointSpread": {
#                 "primary": "-1.5",
#                 "secondary": "EVEN",
#                 "mktTxt": "pointSpread:MIA -1.5"
#               },
#               "moneyline": {
#                 "primary": "-115",
#                 "mktTxt": "moneyline:MIA -115"
#               },
#               "total": {
#                 "primary": "o46.5",
#                 "secondary": "-110",
#                 "mktTxt": "total:o46.5"
#               }
#             },
#             {
#               "line": {
#                 "primaryText": "CAR",
#                 "primaryTextFull": "Panthers",
#                 "primaryTextFullWide": "Carolina Panthers",
#                 "secondaryText": "(0-0)",
#                 "link": "https://www.espn.com/nfl/team/_/name/car/carolina-panthers"
#               },
#               "pointSpread": {
#                 "primary": "+1.5",
#                 "secondary": "-120",
#                 "mktTxt": "pointSpread:CAR +1.5"
#               },
#               "moneyline": {
#                 "primary": "-105",
#                 "mktTxt": "moneyline:CAR -105"
#               },
#               "total": {
#                 "primary": "u46.5",
#                 "secondary": "-110",
#                 "mktTxt": "total:u46.5"
#               }
#             }
#           ]
#         }
#       ]
#     }
#   }
# }
# '''

# # Parse the provided JSON snippet
# try:
#     json_data = json.loads(json_snippet)
#     games_data = json_data.get("page", {}).get("content", {}).get("gameOdds", [])
#     print("\nFrom Provided JSON Snippet:")
#     for game in games_data:
#         game_id = game.get("id")
#         game_info = game.get("gameInfoWithoutId")
#         odds = game.get("odds", [])
#         print(f"\nGame: {game_info} (ID: {game_id})")
#         for team_odds in odds:
#             team = team_odds.get("line", {}).get("primaryTextFull")
#             spread = team_odds.get("pointSpread", {})
#             moneyline = team_odds.get("moneyline", {})
#             total = team_odds.get("total", {})
#             print(f"Team: {team}")
#             if spread:
#                 print(f"  Spread: {spread.get('primary')} ({spread.get('secondary')})")
#             if moneyline:
#                 print(f"  Moneyline: {moneyline.get('primary')}")
#             if total:
#                 print(f"  Total: {total.get('primary')} ({total.get('secondary')})")

# except json.JSONDecodeError as e:
#     print(f"Error parsing JSON snippet: {e}")