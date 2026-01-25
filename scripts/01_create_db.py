import sqlite3
from pathlib import Path

DB_PATH = Path("data/music.db")

DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS songs (
    song_id INTEGER PRIMARY KEY AUTOINCREMENT,
    year INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    UNIQUE(year, rank)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS lyrics (
    song_id INTEGER PRIMARY KEY,
    lyrics_text TEXT,
    source TEXT DEFAULT 'genius',
    fetched_at TEXT,
    fetch_status TEXT,
    FOREIGN KEY(song_id) REFERENCES songs(song_id)
);
""")

conn.commit()
conn.close()

print("Database and tables created.")
