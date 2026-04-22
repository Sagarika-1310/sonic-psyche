"""
setup_db.py  —  Create all database tables
─────────────────────────────────────────────────────────────────────────────
Run this ONCE before starting any collectors.
Safe to re-run — all tables use CREATE IF NOT EXISTS.

Run:
    python setup_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from db import get_conn, init_db, DB_PATH


def verify_tables():
    """Print all created tables and their column counts."""
    conn = get_conn()
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()

    print("\n" + "═" * 55)
    print("  DATABASE TABLES")
    print("═" * 55)
    print(f"  Location: {DB_PATH}\n")

    for t in tables:
        name = t["name"]
        cols = conn.execute(f"PRAGMA table_info({name})").fetchall()
        print(f"  ✅  {name:25}  ({len(cols)} columns)")

    indexes = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' ORDER BY name"
    ).fetchall()
    print(f"\n  Indexes created: {len(indexes)}")
    for idx in indexes:
        print(f"      {idx['name']}")

    print("═" * 55)
    print("\n  Ready. Run collectors in order:")
    print("    python collectors/01_billboard.py")
    print("    python collectors/02_itunes.py")
    print("    python collectors/03_lastfm.py")
    print("    python collectors/04_musicbrainz.py")
    print("    python collectors/05_audio_librosa.py")
    print("    python collectors/06_audio_essentia.py")
    print("    python collectors/07_lyrics.py")
    print("    python collectors/08_economic.py\n")

    conn.close()


def rename_columns():
    """Rename deezer columns in songs table to itunes"""
    conn = get_conn()
    cursor = conn.cursor()

    try:
        cursor.execute("ALTER TABLE songs RENAME COLUMN status_deezer TO status_itunes")

        conn.commit()
        print("Columns renamed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


def delete_tables(tables):
    conn = get_conn()
    cursor = conn.cursor()
    try:
        for table in tables:
            print(f"Deleting {table} from db...")
            cursor.execute(f"DROP TABLE {table}")

            conn.commit()
            print(f"Deleted {table} successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        conn.close()


def rename_sqlite_index():
    conn = get_conn()
    cursor = conn.cursor()

    try:
        # 1. Drop the old index
        cursor.execute(f"DROP INDEX IF EXISTS idx_songs_deezer")

        # 2. Create the new index with the desired name
        cursor.execute(f"CREATE INDEX idx_songs_itunes ON songs(status_itunes)")

        conn.commit()
        print(f"Index successfully renamed to idx_songs_itunes")
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        conn.close()


if __name__ == "__main__":
    print("Creating database and all tables...")
    init_db()
    verify_tables()
