import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
import time

DB_PATH = Path("data/music.db")

START_YEAR = 2000
END_YEAR = 2025

def fetch_year_end_chart(year):
    url = f"https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_{year}"

    headers = {
        "User-Agent": (
            "MusicEmotionResearch/1.0 "
            "(academic research; contact: dummy@example.com)"
        )
    }

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table", {"class": "wikitable"})

    if table is None:
        raise ValueError(f"No chart table found for {year}")

    df = pd.read_html(str(table))[0]

    # Standardize column names
    df.columns = [c.lower() for c in df.columns]

    # Expected columns: rank, title, artist
    df = df.rename(columns={
        df.columns[0]: "rank",
        df.columns[1]: "title",
        df.columns[2]: "artist"
    })

    df = df[["rank", "title", "artist"]]
    df["year"] = year

    return df


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Fetching Billboard Year-End Top 50 for {year}...")
        try:
            df = fetch_year_end_chart(year)

            for _, row in df.iterrows():
                cur.execute("""
                    INSERT INTO songs (year, rank, title, artist)
                    VALUES (?, ?, ?, ?)
                """, (int(row["year"]), int(row["rank"]), row["title"], row["artist"]))

            conn.commit()
            time.sleep(1)  # polite delay

        except Exception as e:
            print(f"Failed for {year}: {e}")

    conn.close()
    print("Billboard ingestion complete.")


if __name__ == "__main__":
    main()
