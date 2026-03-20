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
import sqlite3


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
    print("    python collectors/02_deezer.py")
    print("    python collectors/03_lastfm.py")
    print("    python collectors/04_musicbrainz.py")
    print("    python collectors/05_audio_librosa.py")
    print("    python collectors/06_audio_essentia.py")
    print("    python collectors/07_lyrics.py")
    print("    python collectors/08_economic.py\n")

    conn.close()


if __name__ == "__main__":
    print("Creating database and all tables...")
    init_db()
    verify_tables()
