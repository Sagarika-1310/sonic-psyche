"""
db.py  —  Database schema and helper functions
─────────────────────────────────────────────────────────────────────────────
Single source of truth for the SQLite database.
Every collector imports from here — no schema is defined elsewhere.

Tables:
  songs             — chart data + identifiers (one row per song × year)
  itunes_meta       — Itunes full metadata (duration, explicit, ID)
  lastfm_meta       — Last.fm tags, listener count, playcount, top tags
  musicbrainz_meta  — Genre, ISRC, recording duration from MusicBrainz
  audio_librosa     — Librosa-extracted audio features
  audio_essentia    — Essentia-extracted audio features (incl. danceability)
  lyrics_raw        — Raw lyrics text (no analysis here)
  economic_annual   — CPI, inflation, unemployment, happiness by year

Design principles:
  - Each table is independently fillable — collectors don't depend on each other
  - All status flags live in `songs` so you can see coverage at a glance
  - NULL = not yet attempted; specific sentinel values for "tried & failed"
"""

import sqlite3
import logging
from pathlib import Path
from config import DB_PATH

log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# TABLE DEFINITIONS
# ═════════════════════════════════════════════════════════════════════════════

SONGS = """
CREATE TABLE IF NOT EXISTS songs (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Chart identity
    title            TEXT    NOT NULL,
    artist           TEXT    NOT NULL,
    year             INTEGER NOT NULL,
    period           TEXT    NOT NULL,   -- pre_covid | during_covid | post_covid
    chart_rank       INTEGER,            -- year-end rank (1 = biggest hit of year)
    peak_rank        INTEGER,            -- highest rank reached during chart run
    weeks_on_chart   INTEGER,            -- total weeks on Hot 100

    -- External IDs (filled by collectors, NULL until matched)
    itunes_id        INTEGER,
    musicbrainz_id   TEXT,               -- MusicBrainz recording MBID (UUID)
    genius_id        INTEGER,

    -- Audio file status
    preview_url      TEXT,               -- Itunes 30s MP3 URL
    audio_path       TEXT,               -- local WAV path once downloaded
    audio_duration_s REAL,              -- actual duration of downloaded audio

    -- Collection status flags
    -- NULL = not tried | 0 = tried, failed | 1 = success
    status_status    INTEGER DEFAULT NULL,
    status_lastfm    INTEGER DEFAULT NULL,
    status_mb        INTEGER DEFAULT NULL,
    status_lyrics    INTEGER DEFAULT NULL,
    status_librosa   INTEGER DEFAULT NULL,
    status_essentia  INTEGER DEFAULT NULL,

    -- Deduplication: same song can appear in multiple years
    -- We track the FIRST year it charted
    first_year       INTEGER,

    UNIQUE(title, artist, year)
);
"""

# Index for fast lookups by status (used by collectors to find unprocessed songs)
SONGS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_songs_year       ON songs(year);",
    "CREATE INDEX IF NOT EXISTS idx_songs_period     ON songs(period);",
    "CREATE INDEX IF NOT EXISTS idx_songs_itunes     ON songs(status_itunes);",
    "CREATE INDEX IF NOT EXISTS idx_songs_librosa    ON songs(status_librosa);",
    "CREATE INDEX IF NOT EXISTS idx_songs_essentia   ON songs(status_essentia);",
    "CREATE INDEX IF NOT EXISTS idx_songs_lyrics     ON songs(status_lyrics);",
]

ITUNES_META = """
CREATE TABLE IF NOT EXISTS itunes_meta (
    song_id          INTEGER PRIMARY KEY REFERENCES songs(id),
    itunes_track_id  INTEGER,
    itunes_title     TEXT,           -- title as it appears on Itunes
    itunes_artist    TEXT,
    album_title      TEXT,
    album_id         INTEGER,
    release_date     TEXT,           -- YYYY-MM-DD or YYYY
    duration_s       INTEGER,        -- full track duration in seconds
    explicit         INTEGER,        -- 0 = clean, 1 = explicit, 2 = unknown
    bpm              REAL,           -- Itunes's own BPM field (often 0 if unlisted)
    rank             INTEGER,        -- Itunes's internal popularity rank
    preview_url      TEXT,           -- 30s MP3 preview
    cover_small      TEXT,           -- album art URL 56x56
    cover_medium     TEXT,           -- album art URL 250x250
    itunes_genre_ids TEXT,           -- JSON array of genre IDs e.g. "[132, 116]"
    -- Genre names from /genre endpoint (fetched separately)
    genre_names      TEXT            -- JSON array e.g. '["Pop", "Dance"]'
);
"""

LASTFM_META = """
CREATE TABLE IF NOT EXISTS lastfm_meta (
    song_id        INTEGER PRIMARY KEY REFERENCES songs(id),
    listeners      INTEGER,     -- unique listeners on Last.fm
    playcount      INTEGER,     -- total plays on Last.fm
    -- Top 5 community tags (most relevant first)
    tag_1          TEXT,
    tag_2          TEXT,
    tag_3          TEXT,
    tag_4          TEXT,
    tag_5          TEXT,
    tag_weights    TEXT,        -- JSON: {"pop": 100, "dance": 87, ...}
    -- Artist-level tags (broader genre context)
    artist_tag_1   TEXT,
    artist_tag_2   TEXT,
    artist_tag_3   TEXT,
    lastfm_url     TEXT
);
"""

MUSICBRAINZ_META = """
CREATE TABLE IF NOT EXISTS musicbrainz_meta (
    song_id             INTEGER PRIMARY KEY REFERENCES songs(id),
    mbid                TEXT,       -- recording MBID
    isrc                TEXT,       -- International Standard Recording Code
    duration_ms         INTEGER,    -- recording duration in ms (from MB)
    release_date        TEXT,
    country             TEXT,       -- release country
    -- Genre/tags from MusicBrainz community (can differ from Last.fm)
    mb_tags             TEXT,       -- JSON array of tags
    primary_type        TEXT,       -- Single | Album | EP | ...
    label               TEXT        -- Record label
);
"""

AUDIO_LIBROSA = """
CREATE TABLE IF NOT EXISTS audio_librosa (
    song_id              INTEGER PRIMARY KEY REFERENCES songs(id),
    -- Source info
    audio_duration_s     REAL,      -- actual clip length analysed

    -- ── Rhythm ──────────────────────────────────────────────────────────────
    tempo_bpm            REAL,      -- estimated BPM (librosa beat tracker)
    tempo_confidence     REAL,      -- confidence of BPM estimate (0–1)
    beat_regularity      REAL,      -- 1/std(beat intervals); high = metronomic
    onset_strength_mean  REAL,      -- mean onset envelope (rhythmic intensity)
    onset_strength_std   REAL,

    -- ── Energy & Loudness ───────────────────────────────────────────────────
    rms_mean             REAL,      -- Root Mean Square energy
    rms_std              REAL,
    loudness_db          REAL,      -- amplitude_to_db of RMS mean

    -- ── Timbre (MFCCs) ──────────────────────────────────────────────────────
    -- First 13 MFCC coefficients (mean and std over time)
    -- mfcc_N_mean: overall spectral shape; mfcc_N_std: how much it varies
    mfcc_1_mean REAL,  mfcc_1_std REAL,
    mfcc_2_mean REAL,  mfcc_2_std REAL,
    mfcc_3_mean REAL,  mfcc_3_std REAL,
    mfcc_4_mean REAL,  mfcc_4_std REAL,
    mfcc_5_mean REAL,  mfcc_5_std REAL,
    mfcc_6_mean REAL,  mfcc_6_std REAL,
    mfcc_7_mean REAL,  mfcc_7_std REAL,
    mfcc_8_mean REAL,  mfcc_8_std REAL,
    mfcc_9_mean REAL,  mfcc_9_std REAL,
    mfcc_10_mean REAL, mfcc_10_std REAL,
    mfcc_11_mean REAL, mfcc_11_std REAL,
    mfcc_12_mean REAL, mfcc_12_std REAL,
    mfcc_13_mean REAL, mfcc_13_std REAL,

    -- ── Spectral ────────────────────────────────────────────────────────────
    spectral_centroid_mean   REAL,  -- brightness: higher = sharper/brighter
    spectral_centroid_std    REAL,
    spectral_bandwidth_mean  REAL,  -- tonal spread
    spectral_bandwidth_std   REAL,
    spectral_rolloff_mean    REAL,  -- freq below which 85% energy falls
    spectral_rolloff_std     REAL,
    spectral_flatness_mean   REAL,  -- 0=tonal, 1=noise-like (acousticness proxy)
    spectral_flatness_std    REAL,
    spectral_contrast_mean   REAL,  -- peak vs valley contrast (clarity)
    zero_crossing_rate_mean  REAL,  -- noise/percussion proxy
    zero_crossing_rate_std   REAL,

    -- ── Tonality ────────────────────────────────────────────────────────────
    key_detected      INTEGER,      -- 0=C, 1=C#, 2=D, ... 11=B
    mode_detected     INTEGER,      -- 1=major, 0=minor
    key_confidence    REAL,         -- Krumhansl-Schmuckler correlation strength
    chroma_mean       REAL,         -- mean energy across all 12 pitch classes
    chroma_std        REAL,         -- tonal variety
    -- Per-pitch-class chroma (C through B)
    chroma_C  REAL, chroma_Cs REAL, chroma_D  REAL, chroma_Ds REAL,
    chroma_E  REAL, chroma_F  REAL, chroma_Fs REAL, chroma_G  REAL,
    chroma_Gs REAL, chroma_A  REAL, chroma_As REAL, chroma_B  REAL,

    -- ── Harmonic / Percussive ────────────────────────────────────────────────
    harmonic_ratio       REAL,      -- harmonic / (harmonic + percussive) energy
    percussive_ratio     REAL,      -- percussive / total energy

    -- ── Derived Approximations ───────────────────────────────────────────────
    -- These approximate Spotify features using open-source methods
    acousticness_approx  REAL,      -- 1 - spectral_flatness (higher = more acoustic)
    speechiness_approx   REAL       -- zcr × (1 - harmonic_ratio)
);
"""

AUDIO_ESSENTIA = """
CREATE TABLE IF NOT EXISTS audio_essentia (
    song_id                    INTEGER PRIMARY KEY REFERENCES songs(id),

    -- ── Rhythm & Danceability ────────────────────────────────────────────────
    danceability               REAL,   -- Essentia Danceability (0–3; higher = more danceable)
    danceability_normalised    REAL,   -- scaled to 0–1 for consistency
    bpm                        REAL,   -- RhythmExtractor2013 BPM (more accurate than librosa)
    bpm_confidence             REAL,
    beats_count                INTEGER,
    beat_loudness_mean         REAL,   -- average loudness at beat positions
    beat_loudness_std          REAL,
    rhythm_strength            REAL,   -- overall beat strength
    rhythm_regularity          REAL,   -- how steady the tempo is

    -- ── Loudness (EBU R128 standard) ────────────────────────────────────────
    loudness_integrated        REAL,   -- integrated loudness (LUFS) — broadcast standard
    loudness_range             REAL,   -- LRA: dynamic range
    loudness_momentary_max     REAL,   -- loudest moment
    dynamic_complexity         REAL,   -- Essentia DynamicComplexity (variation over time)

    -- ── Tonality (Essentia key profiles — more accurate than librosa) ────────
    key_essentia               TEXT,   -- e.g. "C", "F#", "Bb"
    scale_essentia             TEXT,   -- "major" | "minor"
    key_strength               REAL,   -- confidence 0–1

    -- ── Tonal Complexity ─────────────────────────────────────────────────────
    dissonance                 REAL,   -- tonal roughness / harshness (0–1)
    spectral_complexity        REAL,   -- number of peaks in spectrum (complexity)
    hpcp_entropy               REAL,   -- entropy of Harmonic Pitch Class Profile
                                       -- high entropy = tonally ambiguous/complex

    -- ── Timbre & Texture ─────────────────────────────────────────────────────
    mfcc_mean_1  REAL, mfcc_mean_2  REAL, mfcc_mean_3  REAL,
    mfcc_mean_4  REAL, mfcc_mean_5  REAL, mfcc_mean_6  REAL,
    mfcc_mean_7  REAL, mfcc_mean_8  REAL, mfcc_mean_9  REAL,
    mfcc_mean_10 REAL, mfcc_mean_11 REAL, mfcc_mean_12 REAL, mfcc_mean_13 REAL,
    -- GFCC (Gammatone) — perceptually motivated, better for music mood
    gfcc_mean_1  REAL, gfcc_mean_2  REAL, gfcc_mean_3  REAL,
    gfcc_mean_4  REAL, gfcc_mean_5  REAL,

    -- ── Mood Classifiers (Essentia trained ML models) ────────────────────────
    -- These are output probabilities from trained SVM classifiers
    -- trained on AllMusic editorial annotations
    mood_happy       REAL,   -- probability that song is "happy"
    mood_sad         REAL,   -- probability that song is "sad"
    mood_relaxed     REAL,   -- probability that song is "relaxed/chill"
    mood_aggressive  REAL,   -- probability that song is "aggressive/energetic"
    mood_electronic  REAL,   -- probability of electronic production style
    mood_acoustic    REAL,   -- probability of acoustic character

    -- ── Voice / Instrument ───────────────────────────────────────────────────
    voice_instrumental_prob  REAL,  -- probability song is instrumental (0=vocal, 1=instrumental)
    -- Higher values = more likely instrumental

    -- ── Engagement ───────────────────────────────────────────────────────────
    -- Approximated from spectral, rhythm, and loudness features
    energy_essentia  REAL    -- overall perceived energy (Essentia formulation)
);
"""

LYRICS_RAW = """
CREATE TABLE IF NOT EXISTS lyrics_raw (
    song_id     INTEGER PRIMARY KEY REFERENCES songs(id),
    raw_text    TEXT,       -- full raw text from Genius (with section headers)
    clean_text  TEXT,       -- preprocessed: headers removed, whitespace normalized
    source      TEXT,       -- 'genius' | 'failed'
    word_count  INTEGER,    -- quick count for sanity check
    char_count  INTEGER,
    fetched_at  TEXT        -- ISO timestamp
);
"""

ECONOMIC_ANNUAL = """
CREATE TABLE IF NOT EXISTS economic_annual (
    year                  INTEGER PRIMARY KEY,
    period                TEXT,       -- pre_covid | during_covid | post_covid

    -- US CPI (from FRED: CPIAUCSL)
    cpi_us                REAL,       -- annual mean CPI index value
    inflation_rate_us     REAL,       -- YoY % change

    -- US Unemployment (from FRED: UNRATE)
    unemployment_us       REAL,       -- annual mean unemployment rate %

    -- Consumer Sentiment (Univ. Michigan: UMCSENT)
    consumer_sentiment    REAL,       -- annual mean index value

    -- World Happiness Report (Life Ladder score for USA)
    happiness_us          REAL,       -- Cantril ladder score (0–10)
    happiness_global_avg  REAL,       -- global average for reference

    -- World Bank GDP (USA, constant 2015 USD growth %)
    gdp_growth_us         REAL
);
"""


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def get_conn() -> sqlite3.Connection:
    """Return a connection with row_factory set (columns accessible by name)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # safer concurrent writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create all tables and indexes. Safe to call multiple times (IF NOT EXISTS)."""
    conn = get_conn()
    tables = [SONGS, ITUNES_META, LASTFM_META, MUSICBRAINZ_META,
              AUDIO_LIBROSA, AUDIO_ESSENTIA, LYRICS_RAW, ECONOMIC_ANNUAL]
    for ddl in tables:
        conn.execute(ddl)
    for idx in SONGS_INDEXES:
        conn.execute(idx)
    conn.commit()
    conn.close()
    log.info("Database ready at: %s", DB_PATH)


def upsert(conn: sqlite3.Connection, table: str,
           row: dict, pk: str = "song_id"):
    """
    Insert a row; if pk already exists, update all other columns.
    Makes every collector idempotent — safe to re-run without duplicates.
    """
    cols = list(row.keys())
    vals = list(row.values())
    ph = ", ".join("?" * len(cols))
    colstr = ", ".join(cols)
    update = ", ".join(f"{c}=excluded.{c}" for c in cols if c != pk)

    sql = (f"INSERT INTO {table} ({colstr}) VALUES ({ph}) "
           f"ON CONFLICT({pk}) DO UPDATE SET {update}")
    conn.execute(sql, vals)


def get_songs_needing(status_col: str, conn=None) -> list:
    """
    Return all songs where status_col IS NULL (not yet attempted).
    status_col should be one of: status_itunes, status_lastfm, status_mb,
                                  status_lyrics, status_librosa, status_essentia
    """
    close = conn is None
    if conn is None:
        conn = get_conn()

    rows = conn.execute(f"""
        SELECT id, title, artist, year, period
        FROM songs
        WHERE {status_col} IS NULL
        ORDER BY year, chart_rank
    """).fetchall()

    if close:
        conn.close()
    return rows


def set_status(conn: sqlite3.Connection, song_id: int,
               status_col: str, value: int):
    """Update a single status flag for a song (0=failed, 1=success)."""
    conn.execute(f"UPDATE songs SET {status_col}=? WHERE id=?", (value, song_id))
