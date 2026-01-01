import sqlite3
import pandas as pd
from datetime import date



def current_week(val=None):
    # Connect to the database
    conn = sqlite3.connect("nfl.db")

    # Load schedule
    query = "SELECT * FROM nfl2025schedule"
    df = pd.read_sql_query(query, conn)

    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Find the *last game date* of each week
    df_last = (
        df.sort_values(["Week", "Date", "Time"])
        .groupby("Week", as_index=False)
        .last()
    )

    # Get today's date
    today = pd.to_datetime(date.today())

    # Compute difference between today and each week's *last game*
    df_last["Diff"] = (df_last["Date"] - today).dt.days

    # Determine current week:
    # As soon as the last game of a week has ended, move to the next week
    past_weeks = df_last[df_last["Date"] < today]

    if not past_weeks.empty:
        # If some weeks are already finished, current week is next one
        last_completed_week = past_weeks.iloc[-1]["Week"]
        current_week = last_completed_week + 1
    else:
        # Season hasn’t started yet
        current_week = df_last.iloc[0]["Week"]

    # If we're past the last scheduled week
    if current_week > df_last["Week"].max():
        current_week = df_last["Week"].max()

    # print("Today:", today.date())
    # print(df_last[["Week", "Date", "Diff"]])
    # print("Current NFL Week:", int(current_week))
    val = 18

    if val:
        return val
    else:
        return current_week


def main():
    current_week()


if __name__ == "__main__":
    main()