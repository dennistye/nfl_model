import numpy as np
import pandas as pd
import sqlite3
import re
import time
import random


# df1 = pd.read_csv("2025_pbp_scrape/official_2025_pbp_data1.csv")
# df2 = pd.read_csv("2025_pbp_scrape/official_2025_pbp_data2.csv")

# combined_df = pd.concat([df1, df2], ignore_index=True)

# combined_df.to_csv("2025_pbp_scrape/official_2025_pbp_data_merged.csv")

teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle',
         'dal', 'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan',
         'sdg', 'ram', 'rai', 'mia', 'min', 'nwe', 'nor', 'nyg',
         'nyj', 'phi', 'pit', 'sea', 'sfo', 'tam', 'oti', 'was'
         ]

start_time = time.time()

veg_df = pd.DataFrame()


for team in teams:

    url = 'https://www.pro-football-reference.com/teams/' + team + '/' + str("2025") + '_lines.htm'

    lines_df = pd.read_html(url, header=0, attrs={'id': 'vegas_lines'})[0]

    lines_df.insert(loc=0, column='Season', value=2025)
    lines_df.insert(loc=1, column='Team', value=team.upper())

    veg_df = pd.concat([veg_df, lines_df], ignore_index=True)

    time.sleep(random.randint(4, 5))

end_time = time.time()

elapsed_time = end_time - start_time
print(f'Elapsed time: {elapsed_time} seconds')

print(veg_df.info())
veg_df.to_csv("vegas_lines.csv")