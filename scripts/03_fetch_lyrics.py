import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path

import lyricsgenius

# TODO: remove after running
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path("data/music.db")
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

if not GENIUS_TOKEN:
    raise RuntimeError("GENIUS_ACCESS_TOKEN not set")

genius = lyricsgenius.Genius(
    GENIUS_TOKEN,
    skip_non_songs=True,
    excluded_terms=["(Remix)", "(Live)"],
    remove_section_headers=True,
    timeout=15
)

genius.verbose = False


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get songs without lyrics
    cur.execute("""
        SELECT s.song_id, s.title, s.artist
        FROM songs s
        LEFT JOIN lyrics l ON s.song_id = l.song_id
        WHERE l.song_id IS NULL
    """)

    songs = cur.fetchall()
    print(f"Fetching lyrics for {len(songs)} songs")

    batch_size = 50
    num_batches = (len(songs) + batch_size - 1) // batch_size

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = min(start + batch_size, len(songs))

        print(f"\nStarted processing batch {batch_num + 1}/{num_batches}")

        for song_id, title, artist in songs[start:end]:
            print(f"→ {title} — {artist}")
            try:
                song = genius.search_song(title, artist)

                if song and song.lyrics:
                    cur.execute("""
                    INSERT INTO lyrics (song_id, lyrics_text, fetched_at, fetch_status)
                    VALUES (?, ?, ?, ?)
                """, (
                        song_id,
                        song.lyrics,
                        datetime.utcnow().isoformat(),
                        "success"
                    ))
                else:
                    print(f"Lyrics NOT FOUND for {title}")
                    cur.execute("""
                    INSERT INTO lyrics (song_id, fetched_at, fetch_status)
                    VALUES (?, ?, ?)
                """, (
                        song_id,
                        datetime.utcnow().isoformat(),
                        "not_found"
                    ))

                conn.commit()
                time.sleep(1.2)  # rate limit

            except Exception as e:
                print(f"   FAILED: {e}")
                cur.execute("""
                INSERT OR REPLACE INTO lyrics (song_id, fetched_at, fetch_status)
                VALUES (?, ?, ?)
            """, (
                    song_id,
                    datetime.utcnow().isoformat(),
                    "error"
                ))
                conn.commit()
                time.sleep(2)

        print(f"Completed processing batch: {batch_num + 1}")

    conn.close()
    print("Lyrics ingestion complete.")


if __name__ == "__main__":
    main()
