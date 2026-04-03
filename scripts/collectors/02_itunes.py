"""
collectors/02_itunes.py
─────────────────────────────────────────────────────────────────────────────
Matches every song in `songs` to the iTunes Search API. No auth required.

Collected per song:
  - 30-second preview URL (M4A)
  - Full track duration
  - Explicit flag (cleaned / explicit / notExplicit)
  - Primary Genre Name
  - Album info, release date, cover art URLs
  - iTunes popularity/collection price context

Fully resumable — skips songs where status_itunes IS NOT NULL.
"""

import sys, time, json, logging
import requests
from tqdm import tqdm

# Maintain your existing local imports
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])
from db import get_conn, upsert, set_status, get_songs_needing
from config import RATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("itunes")

ITUNES_SEARCH = "https://itunes.apple.com/search"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ── iTunes search with smart matching ────────────────────────────────────────

def clean_artist(artist: str) -> str:
    """Strip featuring/ft. noise for better iTunes matching."""
    for sep in [" Featuring ", " ft. ", " Ft. ", " feat. ", " x ", " X "]:
        if sep in artist:
            artist = artist.split(sep)[0]
    return artist.strip()


def score_match(result: dict, title: str, artist: str) -> float:
    """Score an iTunes result for relevance. Higher = better match."""
    t = title.lower()
    a = artist.lower()
    rt = result.get("trackName", "").lower()
    ra = result.get("artistName", "").lower()

    score = 0.0
    # Title matching
    if t in rt or rt in t:           score += 3.0
    if t == rt:                      score += 2.0
    # Artist matching
    if a in ra or ra in a:           score += 3.0
    if a == ra:                      score += 2.0
    # Utility/Quality checks
    if result.get("previewUrl"):     score += 1.0

    return score


def search_itunes(title: str, artist: str) -> dict | None:
    """
    Search iTunes for a song. Returns the best-matching track dict or None.
    """
    clean = clean_artist(artist)

    # iTunes performs best with a combined term query
    for query in [f"{title} {clean}", title]:
        try:
            r = SESSION.get(
                ITUNES_SEARCH,
                params={
                    "term": query,
                    "media": "music",
                    "entity": "song",
                    "limit": 5
                },
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("results", [])

            if results:
                best = max(results, key=lambda x: score_match(x, title, artist))
                if score_match(best, title, artist) >= 3.0:  # Validates at least a partial title/artist match
                    return best
        except Exception as e:
            log.debug("iTunes Search failed ('%s'): %s", query, e)

        # Respecting the rate limit defined in your config
        time.sleep(RATE.get("itunes", 0.5))

    return None


def build_itunes_row(song_id: int, track: dict) -> dict:
    """
    Map iTunes API response to your existing table schema.
    """
    # Map iTunes explicit terms to your integer schema (1: explicit, 0: clean, 2: unknown)
    ex_raw = track.get("trackExplicitness", "notExplicit")
    explicit_int = 1 if ex_raw == "explicit" else 0

    # iTunes returns duration in milliseconds
    duration_ms = track.get("trackTimeMillis", 0)
    duration_s = int(duration_ms / 1000) if duration_ms else None

    # Genre handling
    genre_names = [track.get("primaryGenreName")] if track.get("primaryGenreName") else []

    return {
        "song_id": song_id,
        "itunes_track_id": track.get("trackId"),
        "itunes_title": track.get("trackName"),
        "itunes_artist": track.get("artistName"),
        "album_title": track.get("collectionName"),
        "album_id": track.get("collectionId"),
        "release_date": track.get("releaseDate"),
        "duration_s": duration_s,
        "explicit": explicit_int,
        "bpm": None,  # iTunes API does not provide BPM
        "rank": None,  # iTunes uses trackViewUrl popularity internally; no direct 'rank' int
        "preview_url": track.get("previewUrl"),
        "cover_small": track.get("artworkUrl60"),
        "cover_medium": track.get("artworkUrl100"),
        "itunes_genre_ids": None,
        "genre_names": json.dumps(genre_names) if genre_names else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()
    # Fetching songs that haven't been processed by iTunes yet
    songs = get_songs_needing("status_itunes", conn)
    log.info("%d songs need iTunes data", len(songs))

    matched = failed = 0

    for song in tqdm(songs, desc="iTunes"):
        sid = song["id"]
        title = song["title"]
        artist = song["artist"]

        # ── Search ────────────────────────────────────────────────────────────
        track = search_itunes(title, artist)

        if not track:
            set_status(conn, sid, "status_itunes", 0)
            conn.commit()
            failed += 1
            log.debug("  NO MATCH: %s – %s", title, artist)
            continue

        # ── Build and save row ────────────────────────────────────────────────
        # Note: iTunes provides details in search, so no second 'fetch_track_detail' call is needed
        row = build_itunes_row(sid, track)
        upsert(conn, "itunes_meta", row)

        # ── Update songs table ────────────────────────────────────────────────
        conn.execute("""
            UPDATE songs SET
                itunes_id   = ?,
                preview_url = ?,
                status_itunes = 1
            WHERE id = ?
        """, (track.get("trackId"), track.get("previewUrl"), sid))

        conn.commit()
        matched += 1
        time.sleep(RATE.get("itunes", 0.5))

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  ITUNES COLLECTION COMPLETE")
    print("═" * 52)
    print(f"  Matched:    {matched}")
    print(f"  Failed:     {failed}")
    rate = 100 * matched / max(matched + failed, 1)
    print(f"  Match rate: {rate:.1f}%")
    print()

    # Preview URL availability check
    conn2 = get_conn()
    with_preview = conn2.execute(
        "SELECT COUNT(*) FROM itunes_meta WHERE preview_url IS NOT NULL AND preview_url != ''"
    ).fetchone()[0]
    print(f"  Songs WITH audio preview:    {with_preview}")
    print("═" * 52)
    conn2.close()


if __name__ == "__main__":
    run()
