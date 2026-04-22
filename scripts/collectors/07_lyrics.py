import sys, time, logging, requests
from datetime import datetime

# Path setup for your project
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("lyrics_lrclib")


def fetch_lyrics(title, artist):
    """Fetches clean lyrics from LRCLIB API."""
    url = "https://lrclib.net/api/get"
    params = {
        'artist_name': artist,
        'track_name': title
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            # We want plain lyrics. Sometimes syncedLyrics are available,
            # but plainLyrics is better for NLP analysis.
            return data.get('plainLyrics')
        return None
    except Exception as e:
        log.debug(f"LRCLIB failed for {title}: {e}")
        return None


def run():
    conn = get_conn()
    songs = get_songs_needing("status_lyrics", conn)
    log.info(f"Starting LRCLIB collection for {len(songs)} songs")

    fetched = failed = 0

    for song in tqdm(songs, desc="LRCLIB Fetch"):
        sid, title, artist = song["id"], song["title"], song["artist"]

        lyrics = fetch_lyrics(title, artist)

        if lyrics:
            # Storage logic
            row = {
                "song_id": sid,
                "raw_text": lyrics,
                "clean_text": lyrics.strip(),  # LRCLIB is already clean!
                "source": "lrclib",
                "word_count": len(lyrics.split()),
                "char_count": len(lyrics),
                "fetched_at": datetime.utcnow().isoformat(),
            }
            upsert(conn, "lyrics_raw", row)
            set_status(conn, sid, "status_lyrics", 1)
            fetched += 1
        else:
            # Fallback mark as failed
            set_status(conn, sid, "status_lyrics", 0)
            failed += 1

        # Commit every 10 songs
        if (fetched + failed) % 10 == 0:
            conn.commit()
            time.sleep(1)  # Gentle rate limiting for LRCLIB

    conn.commit()
    conn.close()
    print(f"\nSUCCESS: {fetched} | FAILED: {failed}")


if __name__ == "__main__":
    run()
