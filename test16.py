import numpy as np
import pandas as pd
import sqlite3
import re


df1 = pd.read_csv("2025_pbp_scrape/official_2025_pbp_data1.csv")
df2 = pd.read_csv("2025_pbp_scrape/official_2025_pbp_data2.csv")

combined_df = pd.concat([df1, df2], ignore_index=True)

combined_df.to_csv("2025_pbp_scrape/official_2025_pbp_data_merged.csv")