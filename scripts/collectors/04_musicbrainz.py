"""
collectors/04_musicbrainz.py
─────────────────────────────────────────────────────────────────────────────
Fetches from MusicBrainz (free, no auth, strict 1 req/sec rate limit):
  - Recording MBID (canonical identifier)
  - ISRC (International Standard Recording Code)
  - Precise duration (ms)
  - Release date and country
  - Record label
  - Community genre/tags
  - Release type (Single / Album / EP)

Why MusicBrainz matters:
  MusicBrainz is the most authoritative music metadata database (like Wikipedia
  for music). The genre tags come from a different community than Last.fm,
  and ISRCs let you cross-reference other databases later if needed.

IMPORTANT: MusicBrainz enforces exactly 1 request/second.
           Exceeding this will get your IP rate-limited. Do not reduce RATE["musicbrainz"].

Run:
    python collectors/04_musicbrainz.py

No API key needed. Just set MB_USER_AGENT in config.py.
"""

import sys, time, json, logging

sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import requests
from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing
from config import RATE, MB_USER_AGENT

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("musicbrainz")

MB_BASE = "https://musicbrainz.org/ws/2"
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": MB_USER_AGENT,  # MusicBrainz requires a meaningful user agent
    "Accept": "application/json",
})


# ── API helpers ───────────────────────────────────────────────────────────────

def mb_get(endpoint: str, **params) -> dict | None:
    """
    Generic MusicBrainz API call.
    Automatically adds fmt=json. Returns parsed dict or None.
    """
    try:
        r = SESSION.get(
            f"{MB_BASE}/{endpoint}",
            params={"fmt": "json", **params},
            timeout=15,
        )
        if r.status_code == 503:
            log.warning("MusicBrainz rate-limited — waiting 10s")
            time.sleep(10)
            return mb_get(endpoint, **params)  # retry once
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.debug("MB request failed (%s): %s", endpoint, e)
        return None


def search_recording(title: str, artist: str) -> dict | None:
    """
    Search for a recording by title + artist.
    Returns the best match from recording search results.
    """
    # Lucene query syntax: MusicBrainz uses this for precise matching
    query = f'recording:"{title}" AND artist:"{artist}"'
    data = mb_get("recording", query=query, limit=5)
    time.sleep(RATE["musicbrainz"])

    if not data or not data.get("recordings"):
        # Fallback: simpler query without quotes
        query = f'{title} {artist}'
        data = mb_get("recording", query=query, limit=5)
        time.sleep(RATE["musicbrainz"])

    if not data or not data.get("recordings"):
        return None

    recordings = data["recordings"]

    # Score: prefer recordings with high score and matching artist
    def score(rec):
        s = rec.get("score", 0)
        # Bonus if artist name matches
        for ac in rec.get("artist-credit", []):
            name = ac.get("artist", {}).get("name", "").lower()
            if artist.lower()[:10] in name:
                s += 20
        return s

    return max(recordings, key=score)


def fetch_recording_detail(mbid: str) -> dict | None:
    """
    Fetch full recording details using its MBID.
    Includes: ISRCs, tags, release info, label.
    """
    data = mb_get(
        f"recording/{mbid}",
        inc="isrcs+tags+releases+artist-credits+label-rels",
    )
    time.sleep(RATE["musicbrainz"])
    return data


def build_mb_row(song_id: int, recording: dict, detail: dict | None) -> dict:
    """Build a musicbrainz_meta row from API responses."""
    d = detail or recording

    # ISRCs (a track can have multiple; take the first)
    isrcs = d.get("isrcs", [])
    isrc = isrcs[0] if isrcs else None

    # Tags: MusicBrainz tags have name + count
    tags = d.get("tags", [])
    tags_sorted = sorted(tags, key=lambda t: t.get("count", 0), reverse=True)
    tag_names = [t["name"] for t in tags_sorted[:10]]

    # Release info (first release)
    releases = d.get("releases", [])
    release_date = None
    country = None
    primary_type = None
    label = None

    if releases:
        rel = releases[0]
        release_date = rel.get("date")
        country = rel.get("country")
        rg = rel.get("release-group", {})
        primary_type = rg.get("primary-type")
        # Label from label-info if available
        label_infos = rel.get("label-info", [])
        if label_infos:
            label = label_infos[0].get("label", {}).get("name")

    return {
        "song_id": song_id,
        "mbid": d.get("id") or recording.get("id"),
        "isrc": isrc,
        "duration_ms": d.get("length"),
        "release_date": release_date,
        "country": country,
        "mb_tags": json.dumps(tag_names) if tag_names else None,
        "primary_type": primary_type,
        "label": label,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()
    songs = get_songs_needing("status_mb", conn)
    log.info("%d songs need MusicBrainz data", len(songs))

    matched = failed = 0

    for song in tqdm(songs, desc="MusicBrainz"):
        sid = song["id"]
        title = song["title"]
        artist = song["artist"]

        # Search for the recording
        recording = search_recording(title, artist)

        if not recording:
            set_status(conn, sid, "status_mb", 0)
            conn.commit()
            failed += 1
            continue

        mbid = recording.get("id")

        # Fetch full detail
        detail = fetch_recording_detail(mbid) if mbid else None

        # Build and save row
        row = build_mb_row(sid, recording, detail)
        upsert(conn, "musicbrainz_meta", row)

        # Update songs table with MBID
        conn.execute(
            "UPDATE songs SET musicbrainz_id = ?, status_mb = 1 WHERE id = ?",
            (mbid, sid)
        )
        conn.commit()
        matched += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 52)
    print("  MUSICBRAINZ COLLECTION COMPLETE")
    print("═" * 52)
    print(f"  Matched:    {matched}")
    print(f"  Failed:     {failed}")
    print(f"  Match rate: {100 * matched / max(matched + failed, 1):.1f}%")

    # Genre overview
    conn2 = get_conn()
    with_tags = conn2.execute(
        "SELECT COUNT(*) FROM musicbrainz_meta WHERE mb_tags IS NOT NULL AND mb_tags != '[]'"
    ).fetchone()[0]
    print(f"\n  Songs with genre tags: {with_tags}")
    conn2.close()
    print("═" * 52)


if __name__ == "__main__":
    run()
