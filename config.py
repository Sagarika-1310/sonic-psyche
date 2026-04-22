"""
config.py  —  Central configuration
Edit your API keys here. Nothing else needs changing across the project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# ── Project Paths ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
AUDIO_DIR = ROOT / "audio"
OUTPUT_DIR = ROOT / "outputs"
LOG_DIR = ROOT / "logs"
MODELS_DIR = ROOT / "models"

for d in [DATA_DIR, AUDIO_DIR, OUTPUT_DIR, LOG_DIR]:
    d.mkdir(exist_ok=True)

DB_PATH = DATA_DIR / "music_covid.db"

# ── API Keys ──────────────────────────────────────────────────────────────────
# FRED    → free at https://fred.stlouisfed.org/docs/api/api_key.html
# Last.fm → free at https://www.last.fm/api/account/create
# Itunes  → NO KEY NEEDED (public API)
# MusicBrainz → NO KEY NEEDED (just set a user-agent string)

FRED_API_KEY = os.getenv("FRED_API_KEY")
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

# MusicBrainz requires a descriptive User-Agent (replace with your project name)
MB_USER_AGENT = "MusicCovidResearch/1.0 (s*****.notify@gmail.com)"

# ── Time Period Definitions ───────────────────────────────────────────────────
#
#   PRE-COVID   2017–2019  stable streaming baseline, post-social-media era
#   DURING      2020–2021  acute pandemic; lockdowns, social isolation
#   POST-COVID  2022–2024  reopening, inflation surge, recovery dynamics
#
#   Three years pre and post gives balance.
#   Two years "during" = the two most distinctive pandemic years.

PERIODS = {
    "pre_covid": {"years": [2017, 2018, 2019], "label": "Pre-COVID (2017–2019)", "color": "#2196F3"},
    "during_covid": {"years": [2020, 2021], "label": "During-COVID (2020–2021)", "color": "#F44336"},
    "post_covid": {"years": [2022, 2023, 2024], "label": "Post-COVID (2022–2024)", "color": "#4CAF50"},
}

ALL_YEARS = sorted([y for p in PERIODS.values() for y in p["years"]])


# = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

def year_to_period(year: int) -> str:
    for name, cfg in PERIODS.items():
        if year in cfg["years"]:
            return name
    return "unknown"


# ── Rate Limits (seconds between requests) ───────────────────────────────────
RATE = {
    "itunes": 0.2,  # ~50 req/sec
    "lastfm": 0.22,  # ~4.5 req/sec (limit is 5/sec)
    "musicbrainz": 1.1,  # 1 req/sec (MusicBrainz strict)
    "billboard": 0.8,  # polite scraping
    "fred": 0.2,  # FRED is fast & generous
}
