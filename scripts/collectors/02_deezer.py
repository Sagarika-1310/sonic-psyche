"""
collectors/02_deezer.py
─────────────────────────────────────────────────────────────────────────────
Matches every song in `songs` to Deezer's public API. No auth required.

Collected per song:
  - 30-second preview URL (MP3, direct link)
  - Full track duration
  - Explicit flag (clean / explicit / unknown)
  - BPM field (Deezer provides this for some tracks)
  - Album info, release date, cover art URLs
  - Genre IDs → resolved to genre names via /genre endpoint
  - Deezer popularity rank

Also fetches Deezer genre names from the /genre endpoint (cached locally so
we don't hammer the API).

Fully resumable — skips songs where status_deezer IS NOT NULL.

Run:
    python collectors/02_deezer.py

No pip installs needed beyond `requests`.
"""

import sys, time, json, logging
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import requests
from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing
from config import RATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("deezer")

DEEZER_SEARCH  = "https://api.deezer.com/search"
DEEZER_TRACK   = "https://api.deezer.com/track/{}"
DEEZER_GENRE   = "https://api.deezer.com/genre"
DEEZER_ARTIST  = "https://api.deezer.com/artist/{}"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ── Genre cache ───────────────────────────────────────────────────────────────

def fetch_genre_map() -> dict:
    """
    Returns {genre_id: genre_name} from Deezer's /genre endpoint.
    Cached for the session — only one API call ever.
    """
    try:
        r = SESSION.get(DEEZER_GENRE, timeout=10)
        r.raise_for_status()
        genres = r.json().get("data", [])
        return {g["id"]: g["name"] for g in genres}
    except Exception as e:
        log.warning("Could not fetch genre map: %s", e)
        return {}

GENRE_MAP = {}   # populated on first run


# ── Deezer search with smart matching ────────────────────────────────────────

def clean_artist(artist: str) -> str:
    """Strip featuring/ft. noise for better Deezer matching."""
    for sep in [" Featuring ", " ft. ", " Ft. ", " feat. ", " x ", " X "]:
        if sep in artist:
            artist = artist.split(sep)[0]
    return artist.strip()


def score_match(result: dict, title: str, artist: str) -> float:
    """Score a Deezer result for relevance. Higher = better match."""
    t = title.lower()
    a = artist.lower()
    rt = result.get("title", "").lower()
    ra = result.get("artist", {}).get("name", "").lower()

    score = 0.0
    if t in rt or rt in t:           score += 3.0
    if t == rt:                      score += 2.0
    if a in ra or ra in a:           score += 3.0
    if a == ra:                      score += 2.0
    if result.get("preview"):        score += 1.0   # preview available
    score += result.get("rank", 0) / 1_000_000      # slight popularity bias
    return score


def search_deezer(title: str, artist: str) -> dict | None:
    """
    Search Deezer for a song. Returns the best-matching track dict or None.
    Tries two queries: exact title+artist, then title only as fallback.
    """
    clean = clean_artist(artist)

    for query in [f"{title} {clean}", title]:
        try:
            r = SESSION.get(
                DEEZER_SEARCH,
                params={"q": query, "limit": 10, "order": "RANKING"},
                timeout=10,
            )
            r.raise_for_status()
            results = r.json().get("data", [])
            if results:
                best = max(results, key=lambda x: score_match(x, title, artist))
                if score_match(best, title, artist) >= 3.0:   # at least title match
                    return best
        except Exception as e:
            log.debug("Search failed ('%s'): %s", query, e)
        time.sleep(RATE["deezer"])

    return None


def fetch_track_detail(track_id: int) -> dict | None:
    """
    Fetch full track details from /track/{id}.
    Includes release_date, explicit_lyrics, bpm, gain, contributors.
    """
    try:
        r = SESSION.get(DEEZER_TRACK.format(track_id), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("Track detail fetch failed (id=%d): %s", track_id, e)
        return None


def build_deezer_row(song_id: int, track: dict, detail: dict | None) -> dict:
    """
    Merge search result and detail into a deezer_meta row dict.
    """
    # Genre IDs from track and/or detail
    genre_ids = []
    for src in [track, detail or {}]:
        gid = src.get("genre_id")
        if gid and gid > 0:
            genre_ids.append(gid)

    genre_names = [GENRE_MAP.get(gid, str(gid)) for gid in set(genre_ids)]

    # Explicit flag: Deezer provides explicit_lyrics (bool) and explicit_content_lyrics (int)
    explicit_raw = (detail or track).get("explicit_lyrics")
    explicit_int = {True: 1, False: 0, None: 2}.get(explicit_raw, 2)

    # BPM: often 0 in Deezer — store as-is, will validate later
    bpm = (detail or track).get("bpm", 0) or None

    return {
        "song_id":        song_id,
        "deezer_track_id": track["id"],
        "deezer_title":   track.get("title"),
        "deezer_artist":  track.get("artist", {}).get("name"),
        "album_title":    track.get("album", {}).get("title"),
        "album_id":       track.get("album", {}).get("id"),
        "release_date":   (detail or {}).get("release_date") or track.get("release_date"),
        "duration_s":     track.get("duration"),
        "explicit":       explicit_int,
        "bpm":            bpm,
        "rank":           track.get("rank"),
        "preview_url":    track.get("preview"),
        "cover_small":    track.get("album", {}).get("cover_small"),
        "cover_medium":   track.get("album", {}).get("cover_medium"),
        "deezer_genre_ids": json.dumps(genre_ids) if genre_ids else None,
        "genre_names":    json.dumps(genre_names) if genre_names else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    global GENRE_MAP
    log.info("Fetching Deezer genre map …")
    GENRE_MAP = fetch_genre_map()
    log.info("  %d genres loaded", len(GENRE_MAP))

    conn = get_conn()
    songs = get_songs_needing("status_deezer", conn)
    log.info("%d songs need Deezer data", len(songs))

    matched = failed = 0

    for song in tqdm(songs, desc="Deezer"):
        sid    = song["id"]
        title  = song["title"]
        artist = song["artist"]

        # ── Search ────────────────────────────────────────────────────────────
        track = search_deezer(title, artist)
        if not track:
            set_status(conn, sid, "status_deezer", 0)
            conn.commit()
            failed += 1
            log.debug("  NO MATCH: %s – %s", title, artist)
            time.sleep(RATE["deezer"])
            continue

        # ── Fetch full track detail ───────────────────────────────────────────
        detail = fetch_track_detail(track["id"])
        time.sleep(RATE["deezer"])

        # ── Build and save row ────────────────────────────────────────────────
        row = build_deezer_row(sid, track, detail)
        upsert(conn, "deezer_meta", row)

        # ── Update songs table ────────────────────────────────────────────────
        conn.execute("""
            UPDATE songs SET
                deezer_id   = ?,
                preview_url = ?,
                status_deezer = 1
            WHERE id = ?
        """, (track["id"], track.get("preview"), sid))

        conn.commit()
        matched += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*52)
    print("  DEEZER COLLECTION COMPLETE")
    print("═"*52)
    print(f"  Matched:    {matched}")
    print(f"  Failed:     {failed}")
    rate = 100 * matched / max(matched + failed, 1)
    print(f"  Match rate: {rate:.1f}%")
    print()

    # Preview URL availability
    conn2 = get_conn()
    with_preview = conn2.execute(
        "SELECT COUNT(*) FROM deezer_meta WHERE preview_url IS NOT NULL AND preview_url != ''"
    ).fetchone()[0]
    no_preview = conn2.execute(
        "SELECT COUNT(*) FROM deezer_meta WHERE preview_url IS NULL OR preview_url = ''"
    ).fetchone()[0]
    print(f"  Songs WITH audio preview:    {with_preview}")
    print(f"  Songs WITHOUT audio preview: {no_preview}")
    print("═"*52)
    conn2.close()


if __name__ == "__main__":
    run()