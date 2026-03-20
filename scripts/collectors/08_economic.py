"""
collectors/08_economic.py
─────────────────────────────────────────────────────────────────────────────
Collects macro-economic and wellbeing data for each year (2017–2024).

Sources:
  FRED API (free, generous):
    - US CPI (CPIAUCSL) — Consumer Price Index, all urban consumers
    - US Unemployment Rate (UNRATE) — monthly, annualised
    - Michigan Consumer Sentiment (UMCSENT) — confidence proxy
    - Real GDP Growth (A191RL1Q225SBEA) — annual

  World Happiness Report (CSV download):
    - Life Ladder score for USA (Cantril 0–10 scale)
    - Global average for reference
    - Download from: https://worldhappiness.report/data/

  Hardcoded fallbacks for both sources so the project works even
  if API keys are not yet set up.

Run:
    python collectors/08_economic.py

Installs: pip install fredapi
"""

import sys, json, logging
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import pandas as pd
import numpy as np
from db import get_conn
from config import FRED_API_KEY, ALL_YEARS, year_to_period, WHR_CSV

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("economic")

# ── WHR_CSV path ──────────────────────────────────────────────────────────────
from pathlib import Path
WHR_CSV = Path(__file__).parent.parent / "data" / "world_happiness_report.csv"


# ── FRED Data ─────────────────────────────────────────────────────────────────

# Hardcoded annual values as fallback
# Source: FRED, Federal Reserve Bank of St. Louis (public domain)
FALLBACK_FRED = {
    # year: (cpi_annual_mean, inflation_rate_pct, unemployment_pct, consumer_sentiment)
    2016: (240.0,  1.3, 4.9, 91.9),
    2017: (245.1,  2.1, 4.4, 95.9),
    2018: (251.1,  2.4, 3.9, 98.4),
    2019: (255.7,  1.8, 3.7, 96.0),
    2020: (258.8,  1.2, 8.1, 76.9),
    2021: (270.9,  4.7, 5.4, 76.8),
    2022: (292.7,  8.0, 3.6, 58.5),
    2023: (304.7,  4.1, 3.6, 65.1),
    2024: (313.5,  2.9, 4.0, 69.0),
}

# US GDP growth (%) — World Bank / BEA
FALLBACK_GDP = {
    2016: 1.7, 2017: 2.3, 2018: 2.9, 2019: 2.3,
    2020: -2.8, 2021: 5.9, 2022: 2.1, 2023: 2.5, 2024: 2.7,
}

# World Happiness Report — US Life Ladder (Cantril scale 0–10)
# Source: World Happiness Report 2024 (free public data)
FALLBACK_WHR_US = {
    2016: 7.10, 2017: 6.99, 2018: 6.89, 2019: 6.89,
    2020: 7.03, 2021: 6.95, 2022: 6.98, 2023: 6.89, 2024: 6.73,
}

# World Happiness Report — Global Average Life Ladder
FALLBACK_WHR_GLOBAL = {
    2016: 5.41, 2017: 5.40, 2018: 5.38, 2019: 5.42,
    2020: 5.47, 2021: 5.53, 2022: 5.55, 2023: 5.54, 2024: 5.53,
}


def fetch_fred_data() -> dict:
    """
    Fetch from FRED API. Returns dict: year → (cpi, inflation, unemployment, sentiment).
    Falls back to hardcoded values if API key not set or request fails.
    """
    if FRED_API_KEY == "YOUR_FRED_KEY_HERE":
        log.info("FRED API key not set — using hardcoded fallback values")
        return FALLBACK_FRED

    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)

        # Fetch monthly series
        cpi     = fred.get_series("CPIAUCSL",         "2016-01-01", "2024-12-31")
        unemp   = fred.get_series("UNRATE",            "2016-01-01", "2024-12-31")
        sent    = fred.get_series("UMCSENT",           "2016-01-01", "2024-12-31")

        # Annualise
        result = {}
        for year in ALL_YEARS + [2016]:
            year_cpi   = cpi[str(year)].mean()   if str(year) in cpi.index.year.astype(str).values   else None
            year_unemp = unemp[str(year)].mean() if str(year) in unemp.index.year.astype(str).values else None
            year_sent  = sent[str(year)].mean()  if str(year) in sent.index.year.astype(str).values  else None

            # Inflation = YoY % change in CPI
            cpi_prev = cpi[str(year - 1)].mean() if str(year - 1) in cpi.index.year.astype(str).values else None
            inflation = 100 * (year_cpi - cpi_prev) / cpi_prev if (year_cpi and cpi_prev) else None

            result[year] = (year_cpi, inflation, year_unemp, year_sent)

        log.info("FRED data fetched for %d years", len(result))
        return result

    except Exception as e:
        log.warning("FRED fetch failed (%s) — using hardcoded fallback", e)
        return FALLBACK_FRED


def load_happiness_data() -> tuple[dict, dict]:
    """
    Load World Happiness Report data.
    Returns (us_scores, global_avg_scores) as {year: score} dicts.

    Download the data file from https://worldhappiness.report/data/
    and save as data/world_happiness_report.csv
    """
    if not WHR_CSV.exists():
        log.info("WHR CSV not found — using hardcoded fallback values")
        return FALLBACK_WHR_US, FALLBACK_WHR_GLOBAL

    try:
        df = pd.read_csv(WHR_CSV)

        # Detect column names (they vary slightly across WHR editions)
        year_col    = next(c for c in df.columns if "year" in c.lower())
        country_col = next(c for c in df.columns if "country" in c.lower() or "name" in c.lower())
        score_col   = next(c for c in df.columns
                           if any(k in c.lower() for k in ["ladder", "happiness score", "life ladder"]))

        # US scores
        us = df[df[country_col].str.contains("United States", na=False)]
        us_scores = dict(zip(us[year_col].astype(int), us[score_col]))

        # Global average by year
        global_avg = df.groupby(year_col)[score_col].mean().to_dict()
        global_avg = {int(k): v for k, v in global_avg.items()}

        log.info("WHR data loaded: %d countries, %d years",
                 df[country_col].nunique(), df[year_col].nunique())
        return us_scores, global_avg

    except Exception as e:
        log.warning("WHR CSV parse failed (%s) — using fallback", e)
        return FALLBACK_WHR_US, FALLBACK_WHR_GLOBAL


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()

    fred_data           = fetch_fred_data()
    whr_us, whr_global  = load_happiness_data()

    inserted = 0
    for year in ALL_YEARS:
        fred_vals = fred_data.get(year, (None, None, None, None))
        cpi, inflation, unemployment, sentiment = fred_vals

        row = {
            "year":                 year,
            "period":               year_to_period(year),
            "cpi_us":               cpi,
            "inflation_rate_us":    inflation,
            "unemployment_us":      unemployment,
            "consumer_sentiment":   sentiment,
            "happiness_us":         whr_us.get(year),
            "happiness_global_avg": whr_global.get(year),
            "gdp_growth_us":        FALLBACK_GDP.get(year),
        }

        conn.execute("""
            INSERT INTO economic_annual
                (year, period, cpi_us, inflation_rate_us, unemployment_us,
                 consumer_sentiment, happiness_us, happiness_global_avg, gdp_growth_us)
            VALUES (:year, :period, :cpi_us, :inflation_rate_us, :unemployment_us,
                    :consumer_sentiment, :happiness_us, :happiness_global_avg, :gdp_growth_us)
            ON CONFLICT(year) DO UPDATE SET
                cpi_us              = excluded.cpi_us,
                inflation_rate_us   = excluded.inflation_rate_us,
                unemployment_us     = excluded.unemployment_us,
                consumer_sentiment  = excluded.consumer_sentiment,
                happiness_us        = excluded.happiness_us,
                happiness_global_avg= excluded.happiness_global_avg,
                gdp_growth_us       = excluded.gdp_growth_us
        """, row)
        inserted += 1

    conn.commit()
    conn.close()

    # ── Print table ───────────────────────────────────────────────────────────
    print("\n" + "═"*80)
    print("  ECONOMIC DATA STORED")
    print("═"*80)
    print(f"  {'Year':>4}  {'Period':15}  {'CPI':>7}  {'Inflation':>9}  "
          f"{'Unemploy':>8}  {'Sentiment':>9}  {'Happiness':>9}")
    print("  " + "─"*76)

    conn2 = get_conn()
    for r in conn2.execute("SELECT * FROM economic_annual ORDER BY year"):
        print(f"  {r['year']:>4}  {r['period']:15}  "
              f"{r['cpi_us'] or 0:>7.1f}  "
              f"{r['inflation_rate_us'] or 0:>8.1f}%  "
              f"{r['unemployment_us'] or 0:>7.1f}%  "
              f"{r['consumer_sentiment'] or 0:>9.1f}  "
              f"{r['happiness_us'] or 0:>9.2f}")
    conn2.close()
    print("═"*80)


if __name__ == "__main__":
    run()