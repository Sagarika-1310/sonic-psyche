"""
collectors/03_lastfm.py
─────────────────────────────────────────────────────────────────────────────
Fetches from Last.fm API (free, requires key):
  - Listener count (unique listeners)
  - Play count (total scrobbles — proxy for lasting cultural impact)
  - Top 5 community tags on the TRACK (e.g. "pop", "melancholic", "dance")
  - Top 3 community tags on the ARTIST (broader genre context)
  - Last.fm track URL

Why Last.fm tags are valuable:
  Community tags are mood/genre labels applied by listeners, not algorithms.
  They capture dimensions like "melancholic", "uplifting", "chill", "aggressive"
  that audio features alone can't classify. Combining them with audio gives
  you both the "what it sounds like" and "how listeners feel about it."

Run:
    python collectors/03_lastfm.py

Get free key at: https://www.last.fm/api/account/create
"""

import sys, time, json, logging

sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import requests
from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing
from config import LASTFM_API_KEY, RATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("lastfm")

BASE = "http://ws.audioscrobbler.com/2.0/"
SESSION = requests.Session()


# ── API helpers ───────────────────────────────────────────────────────────────

def lastfm_get(method: str, **params) -> dict | None:
    """Generic Last.fm API call. Returns parsed JSON or None on failure."""
    all_params = {
        "method": method,
        "api_key": LASTFM_API_KEY,
        "format": "json",
        **params,
    }
    try:
        r = SESSION.get(BASE, params=all_params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if "error" in data:
            log.debug("Last.fm error %d: %s", data["error"], data.get("message"))
            return None
        return data
    except Exception as e:
        log.debug("Last.fm request failed (%s): %s", method, e)
        return None


def get_track_info(title: str, artist: str) -> dict | None:
    """Fetch track.getInfo — returns tags, listeners, playcount, url."""
    data = lastfm_get("track.getInfo", track=title, artist=artist,
                      autocorrect=1)
    return data.get("track") if data else None


def get_artist_tags(artist: str) -> list[str]:
    """Fetch artist.getTopTags — returns top genre/mood tags for the artist."""
    data = lastfm_get("artist.getTopTags", artist=artist, autocorrect=1)
    if not data:
        return []
    tags = data.get("toptags", {}).get("tag", [])
    # Filter out numeric/garbage tags; return top 3 names
    clean = [t["name"].lower() for t in tags
             if t.get("count", 0) > 10 and not t["name"].isdigit()]
    return clean[:3]


def parse_track_info(song_id: int, track: dict) -> dict:
    """Extract the fields we care about from track.getInfo response."""
    # Tags: list of {name, url, count} dicts or a single dict
    raw_tags = track.get("toptags", {}).get("tag", [])
    if isinstance(raw_tags, dict):
        raw_tags = [raw_tags]  # API sometimes returns dict instead of list

    # Sort by count descending, take top 5
    tags_sorted = sorted(raw_tags,
                         key=lambda t: int(t.get("count", 0) or 0),
                         reverse=True)[:5]
    tag_names = [t["name"].lower() for t in tags_sorted]
    tag_weights = {t["name"].lower(): int(t.get("count", 0) or 0)
                   for t in tags_sorted}

    row = {
        "song_id": song_id,
        "listeners": int(track.get("listeners", 0) or 0),
        "playcount": int(track.get("playcount", 0) or 0),
        "lastfm_url": track.get("url"),
        "tag_weights": json.dumps(tag_weights) if tag_weights else None,
    }

    for i, name in enumerate(tag_names, start=1):
        if i > 5: break
        row[f"tag_{i}"] = name

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()
    songs = get_songs_needing("status_lastfm", conn)
    log.info("%d songs need Last.fm data", len(songs))

    matched = failed = 0

    for song in tqdm(songs, desc="Last.fm"):
        sid = song["id"]
        title = song["title"]
        artist = song["artist"]

        # ── Track info ────────────────────────────────────────────────────────
        track = get_track_info(title, artist)
        time.sleep(RATE["lastfm"])

        if not track:
            set_status(conn, sid, "status_lastfm", 0)
            conn.commit()
            failed += 1
            continue

        row = parse_track_info(sid, track)

        # ── Artist tags ───────────────────────────────────────────────────────
        artist_tags = get_artist_tags(artist)
        time.sleep(RATE["lastfm"])

        for i, tag in enumerate(artist_tags[:3], start=1):
            row[f"artist_tag_{i}"] = tag

        # ── Save ──────────────────────────────────────────────────────────────
        upsert(conn, "lastfm_meta", row)
        set_status(conn, sid, "status_lastfm", 1)
        conn.commit()
        matched += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  LAST.FM COLLECTION COMPLETE")
    print("═" * 52)
    print(f"  Matched: {matched}  |  Failed: {failed}")
    rate = 100 * matched / max(matched + failed, 1)
    print(f"  Match rate: {rate:.1f}%")

    # Top tags overview
    conn2 = get_conn()
    print("\n  Most common track tags in dataset:")
    for col in ["tag_1", "tag_2", "tag_3"]:
        rows = conn2.execute(f"""
            SELECT {col}, COUNT(*) as n FROM lastfm_meta
            WHERE {col} IS NOT NULL
            GROUP BY {col} ORDER BY n DESC LIMIT 8
        """).fetchall()
        if rows:
            tags_str = " | ".join(f"{r[0]} ({r[1]})" for r in rows)
            print(f"  {col}: {tags_str}")
    conn2.close()
    print("═" * 52)


if __name__ == "__main__":
    run()
