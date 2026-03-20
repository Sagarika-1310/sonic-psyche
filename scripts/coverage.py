"""
coverage.py  —  Data Collection Coverage Dashboard
─────────────────────────────────────────────────────────────────────────────
Run at any point to see exactly how much data you have and what's missing.
This is your primary tool for knowing when you're ready to move to analysis.

Run:
    python coverage.py

Outputs a clear, colour-coded breakdown per year and per data source.
"""

import sys

sys.path.insert(0, str(__file__).rsplit("/coverage.py", 1)[0])

from db import get_conn


def bar(n, total, width=20):
    if total == 0:
        return "░" * width
    filled = int(width * n / total)
    pct = 100 * n / total
    color = "\033[92m" if pct > 80 else ("\033[93m" if pct > 50 else "\033[91m")
    reset = "\033[0m"
    return f"{color}{'█' * filled}{'░' * (width - filled)}{reset} {n:3}/{total} ({pct:.0f}%)"


def run():
    conn = get_conn()

    # ── Total songs ───────────────────────────────────────────────────────────
    total = conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    if total == 0:
        print("No songs in database yet. Run: python collectors/01_billboard.py")
        return

    print("\n" + "═" * 72)
    print(f"  DATA COLLECTION COVERAGE DASHBOARD   (Total songs: {total})")
    print("═" * 72)

    # ── Per-year breakdown ────────────────────────────────────────────────────
    years = conn.execute(
        "SELECT DISTINCT year FROM songs ORDER BY year"
    ).fetchall()

    print(f"\n  {'Year':>4}  {'Period':14}  {'Songs':>5}  "
          f"{'Deezer':>6}  {'LastFM':>6}  {'MB':>4}  "
          f"{'Librosa':>7}  {'Essentia':>8}  {'Lyrics':>6}")
    print("  " + "─" * 68)

    period_totals = {}
    for row in years:
        year = row["year"]
        r = conn.execute("""
            SELECT
                period,
                COUNT(*) as total,
                SUM(CASE WHEN status_deezer  = 1 THEN 1 ELSE 0 END) as deezer,
                SUM(CASE WHEN status_lastfm  = 1 THEN 1 ELSE 0 END) as lastfm,
                SUM(CASE WHEN status_mb      = 1 THEN 1 ELSE 0 END) as mb,
                SUM(CASE WHEN status_librosa = 1 THEN 1 ELSE 0 END) as librosa,
                SUM(CASE WHEN status_essentia= 1 THEN 1 ELSE 0 END) as essentia,
                SUM(CASE WHEN status_lyrics  = 1 THEN 1 ELSE 0 END) as lyrics
            FROM songs WHERE year = ?
        """, (year,)).fetchone()

        def pct_str(n, t):
            if t == 0: return "  —  "
            p = 100 * n / t
            c = "\033[92m" if p > 80 else ("\033[93m" if p > 50 else "\033[91m")
            return f"{c}{p:4.0f}%\033[0m"

        n = r["total"]
        print(f"  {year:>4}  {r['period']:14}  {n:>5}  "
              f"  {pct_str(r['deezer'], n)}  "
              f"  {pct_str(r['lastfm'], n)}  "
              f"  {pct_str(r['mb'], n)}  "
              f"  {pct_str(r['librosa'], n)}  "
              f"  {pct_str(r['essentia'], n)}  "
              f"  {pct_str(r['lyrics'], n)}")

    # ── Overall totals ────────────────────────────────────────────────────────
    print("\n  " + "─" * 68)
    totals = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status_deezer  = 1 THEN 1 ELSE 0 END) as deezer,
            SUM(CASE WHEN status_lastfm  = 1 THEN 1 ELSE 0 END) as lastfm,
            SUM(CASE WHEN status_mb      = 1 THEN 1 ELSE 0 END) as mb,
            SUM(CASE WHEN status_librosa = 1 THEN 1 ELSE 0 END) as librosa,
            SUM(CASE WHEN status_essentia= 1 THEN 1 ELSE 0 END) as essentia,
            SUM(CASE WHEN status_lyrics  = 1 THEN 1 ELSE 0 END) as lyrics
        FROM songs
    """).fetchone()

    n = totals["total"]
    print(f"  TOTAL              All   {n:>5}  "
          f"  {100 * totals['deezer'] / n:3.0f}%   "
          f"  {100 * totals['lastfm'] / n:3.0f}%   "
          f"  {100 * totals['mb'] / n:3.0f}%   "
          f"  {100 * totals['librosa'] / n:3.0f}%   "
          f"  {100 * totals['essentia'] / n:3.0f}%   "
          f"  {100 * totals['lyrics'] / n:3.0f}%")

    # ── Songs with FULL data (all 6 sources) ──────────────────────────────────
    full = conn.execute("""
        SELECT COUNT(*) FROM songs
        WHERE status_deezer=1 AND status_lastfm=1
          AND status_mb=1 AND status_librosa=1 AND status_lyrics=1
    """).fetchone()[0]

    # Without essentia (optional)
    near_full = conn.execute("""
        SELECT COUNT(*) FROM songs
        WHERE status_deezer=1 AND status_lastfm=1
          AND status_librosa=1 AND status_lyrics=1
    """).fetchone()[0]

    print(f"\n  Songs with ALL sources complete:  {full}/{n}  "
          f"({100 * full / n:.1f}%)")
    print(f"  Songs with core 4 sources:        {near_full}/{n}  "
          f"({100 * near_full / n:.1f}%)")
    print(f"  (Core 4 = Deezer + LastFM + Librosa + Lyrics)")

    # ── Per-period summary ────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  COVERAGE BY PERIOD")
    print("═" * 72)

    for period in ["pre_covid", "during_covid", "post_covid"]:
        r = conn.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN status_librosa=1 THEN 1 ELSE 0 END) as audio,
                   SUM(CASE WHEN status_lyrics=1  THEN 1 ELSE 0 END) as lyrics,
                   SUM(CASE WHEN status_librosa=1 AND status_lyrics=1 THEN 1 ELSE 0 END) as both
            FROM songs WHERE period=?
        """, (period,)).fetchone()

        print(f"\n  {period.upper().replace('_', '-')}")
        print(f"    Total songs:  {r['total']}")
        print(f"    Audio ready:  {bar(r['audio'], r['total'])}")
        print(f"    Lyrics ready: {bar(r['lyrics'], r['total'])}")
        print(f"    Both ready:   {bar(r['both'], r['total'])}  ← analysis-ready")

    # ── Audio download size ───────────────────────────────────────────────────
    import os
    from pathlib import Path
    from config import AUDIO_DIR
    wav_files = list(Path(AUDIO_DIR).glob("*.wav"))
    total_mb = sum(f.stat().st_size for f in wav_files) / (1024 * 1024)
    print(f"\n  Audio files on disk: {len(wav_files)} WAVs  ({total_mb:.0f} MB)")

    # ── Economic data ─────────────────────────────────────────────────────────
    econ_years = conn.execute(
        "SELECT COUNT(*) FROM economic_annual"
    ).fetchone()[0]
    print(f"  Economic data:       {econ_years}/8 years populated")

    # ── What to run next ──────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print("  NEXT STEPS")
    print("═" * 72)

    steps = [
        (totals["deezer"] == 0, "1", "python collectors/01_billboard.py"),
        (totals["deezer"] == 0, "2", "python collectors/02_deezer.py"),
        (totals["lastfm"] == 0, "3", "python collectors/03_lastfm.py"),
        (totals["mb"] == 0, "4", "python collectors/04_musicbrainz.py"),
        (totals["librosa"] == 0, "5", "python collectors/05_audio_librosa.py"),
        (totals["essentia"] == 0, "6", "python collectors/06_audio_essentia.py  (optional)"),
        (totals["lyrics"] == 0, "7", "python collectors/07_lyrics.py"),
        (econ_years == 0, "8", "python collectors/08_economic.py"),
    ]
    any_pending = False
    for condition, step, cmd in steps:
        if condition:
            print(f"  ⬜  Step {step}: {cmd}")
            any_pending = True

    if not any_pending:
        print("  ✅  All collectors complete! You're ready for analysis.")
        print("      Suggested minimum for analysis:")
        print(f"      {near_full} songs have audio + lyrics = enough to proceed")

    print("═" * 72 + "\n")
    conn.close()


if __name__ == "__main__":
    run()
