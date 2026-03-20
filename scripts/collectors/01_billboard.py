"""
collectors/01_billboard.py
─────────────────────────────────────────────────────────────────────────────
Collects Billboard year-end Hot 100 charts for every year in ALL_YEARS.
Stores raw chart data in the `songs` table.

Fully resumable — already-inserted songs are skipped (UNIQUE constraint).

Run:
    python collectors/01_billboard.py

Installs:
    pip install billboard.py
"""

import sys, time, logging

sys.path.insert(0, str(__file__.replace("collectors/01_billboard.py", "")))

import billboard
from tqdm import tqdm
from db import get_conn, init_db
from config import ALL_YEARS, RATE, year_to_period

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("billboard")


def collect_charts():
    init_db()
    conn = get_conn()
    total_new = 0

    for year in tqdm(ALL_YEARS, desc="Billboard year-end Hot 100"):
        log.info("Year %d", year)
        try:
            chart = billboard.ChartData('hot-100', date=f'{year}-12-28')
        except Exception as e:
            log.warning("  Billboard fetch FAILED for %d: %s", year, e)
            time.sleep(3)
            continue

        period = year_to_period(year)
        new_cnt = 0

        for rank, entry in enumerate(chart, start=1):
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO songs
                        (title, artist, year, period,
                         chart_rank, peak_rank, weeks_on_chart)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.title.strip(),
                    entry.artist.strip(),
                    year, period, rank,
                    entry.peakPos,
                    entry.weeks,
                ))
                new_cnt += conn.execute("SELECT changes()").fetchone()[0]
            except Exception as e:
                log.debug("  Row insert failed: %s", e)

        conn.commit()
        log.info("  → %d new songs added for %d", new_cnt, year)
        total_new += new_cnt
        time.sleep(RATE["billboard"])

    # ── Mark first_year for each unique title+artist combo ───────────────────
    conn.execute("""
        UPDATE songs
        SET first_year = (
            SELECT MIN(year) FROM songs s2
            WHERE s2.title  = songs.title
              AND s2.artist = songs.artist
        )
        WHERE first_year IS NULL
    """)
    conn.commit()

    # ── Summary ───────────────────────────────────────────────────────────────
    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    print("\n" + "═" * 52)
    print("  BILLBOARD COLLECTION COMPLETE")
    print("═" * 52)
    for year in ALL_YEARS:
        n = conn.execute(
            "SELECT COUNT(*) FROM songs WHERE year=?", (year,)
        ).fetchone()[0]
        period = year_to_period(year)
        print(f"  {year}  [{period:12}]  {n} songs")
    print(f"\n  Total rows in DB: {total}")
    print(f"  New this run:     {total_new}")
    print("═" * 52)

    conn.close()


if __name__ == "__main__":
    collect_charts()
