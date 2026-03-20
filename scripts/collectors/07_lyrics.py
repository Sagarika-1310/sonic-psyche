"""
collectors/07_lyrics.py
─────────────────────────────────────────────────────────────────────────────
Fetches raw lyrics from Genius API and stores them WITHOUT analysing them.
Analysis is a separate, later step. This file just gets and stores text.

What's stored:
  - raw_text:   exact Genius output (includes [Verse 1], [Chorus] tags)
  - clean_text: tags stripped, whitespace normalised (ready for NLP)
  - word_count: quick sanity check
  - char_count: quick sanity check
  - source:     'genius' or 'failed'

Fully resumable — skips songs where status_lyrics IS NOT NULL.

Run:
    python collectors/07_lyrics.py

Get free Genius token at: https://genius.com/api-clients
Installs: pip install lyricsgenius
"""

import sys, re, time, logging
from datetime import datetime
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing
from config import GENIUS_TOKEN, RATE

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("lyrics")


# ── Genius client (lazy init) ─────────────────────────────────────────────────

_genius = None

def get_genius():
    global _genius
    if _genius is None:
        import lyricsgenius
        _genius = lyricsgenius.Genius(
            GENIUS_TOKEN,
            verbose=False,
            remove_section_headers=False,   # keep raw; we clean separately
            skip_non_songs=True,
            retries=2,
            timeout=15,
        )
    return _genius


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_lyrics(raw: str) -> str:
    """
    Remove Genius metadata lines, section headers [Verse], [Chorus] etc.,
    and normalise whitespace.
    Returns clean lyric text suitable for NLP.
    """
    if not raw:
        return ""

    lines = raw.splitlines()

    # Genius prepends a line like "123 ContributorsSong NameLyrics"
    # Drop lines that look like metadata (start with digit or contain "Lyrics")
    while lines and (not lines[0].strip() or
                     re.match(r"^\d+\s+Contributors", lines[0]) or
                     lines[0].strip().endswith("Lyrics")):
        lines.pop(0)

    text = "\n".join(lines)

    # Remove [Section] tags
    text = re.sub(r"\[.*?\]", "", text)

    # Remove trailing "Embed" or numbers (Genius artifact)
    text = re.sub(r"\d+Embed$", "", text.strip())
    text = re.sub(r"Embed$", "", text.strip())

    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn  = get_conn()
    songs = get_songs_needing("status_lyrics", conn)
    log.info("%d songs need lyrics", len(songs))

    genius  = get_genius()
    fetched = failed = 0

    for song in tqdm(songs, desc="Lyrics"):
        sid    = song["id"]
        title  = song["title"]
        artist = song["artist"]

        # ── Fetch ─────────────────────────────────────────────────────────────
        raw_text = None
        try:
            result = genius.search_song(title, artist)
            if result:
                raw_text = result.lyrics
        except Exception as e:
            log.debug("Genius failed for '%s': %s", title, e)

        time.sleep(RATE["genius"])

        # ── Store ─────────────────────────────────────────────────────────────
        if raw_text:
            clean  = clean_lyrics(raw_text)
            words  = clean.split()
            row = {
                "song_id":    sid,
                "raw_text":   raw_text,
                "clean_text": clean,
                "source":     "genius",
                "word_count": len(words),
                "char_count": len(clean),
                "fetched_at": datetime.utcnow().isoformat(),
            }
            upsert(conn, "lyrics_raw", row)
            set_status(conn, sid, "status_lyrics", 1)
            fetched += 1
        else:
            row = {
                "song_id":    sid,
                "raw_text":   None,
                "clean_text": None,
                "source":     "failed",
                "word_count": 0,
                "char_count": 0,
                "fetched_at": datetime.utcnow().isoformat(),
            }
            upsert(conn, "lyrics_raw", row)
            set_status(conn, sid, "status_lyrics", 0)
            failed += 1

        conn.commit()

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*52)
    print("  LYRICS COLLECTION COMPLETE")
    print("═"*52)
    print(f"  Fetched: {fetched}  |  Failed/not found: {failed}")
    rate = 100 * fetched / max(fetched + failed, 1)
    print(f"  Success rate: {rate:.1f}%")

    # Word count distribution
    conn2 = get_conn()
    stats = conn2.execute("""
        SELECT
            AVG(word_count) as avg_words,
            MIN(word_count) as min_words,
            MAX(word_count) as max_words
        FROM lyrics_raw
        WHERE word_count > 0
    """).fetchone()
    if stats:
        print(f"\n  Word count: avg={stats['avg_words']:.0f}  "
              f"range [{stats['min_words']}–{stats['max_words']}]")

    # Per-period coverage
    print("\n  Lyrics coverage by period:")
    rows = conn2.execute("""
        SELECT s.period, s.year,
               SUM(CASE WHEN s.status_lyrics=1 THEN 1 ELSE 0 END) as ok,
               COUNT(*) as total
        FROM songs s
        GROUP BY s.period, s.year
        ORDER BY s.year
    """).fetchall()
    for r in rows:
        pct = 100 * r["ok"] / max(r["total"], 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {r['year']} [{r['period'][:12]:12}]  "
              f"{r['ok']:3}/{r['total']} ({pct:.0f}%)  {bar}")
    conn2.close()
    print("═"*52)


if __name__ == "__main__":
    run()