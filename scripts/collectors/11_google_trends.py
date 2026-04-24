import sys, time, logging, random
from datetime import datetime
from pytrends.request import TrendReq
import pandas as pd

# Path setup for project structure
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])
from db import get_conn

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("google_trends_insert")

# Mental Health Lexicon Groups
GROUPS = {
    "depression": ["depression", "antidepressants", "signs of depression", "feeling sad", "loneliness"],
    "anxiety": ["anxiety", "panic attack", "social anxiety", "worrying", "anxiety symptoms"],
    "stress": ["stress", "burnout", "stress relief", "overwhelmed", "work stress"]
}

START_YEAR = 2017
END_YEAR = 2025


def fetch_group_score(keywords, start, end):
    """Fetches trends for a list of keywords and returns the annual average."""
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    timeframe = f"{start}-01-01 {end}-12-31"

    try:
        log.info(f"Fetching data for keywords: {keywords[:2]}...")
        pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo='US')
        df = pytrends.interest_over_time()

        if df is None or df.empty:
            return None

        if 'isPartial' in df.columns:
            df = df.drop(columns=['isPartial'])

        # Calculate the Category Mean (Horizontal average of keywords)
        df['group_score'] = df[keywords].mean(axis=1)

        # Resample to Annual Mean
        try:
            annual = df['group_score'].resample('YE').mean()
        except ValueError:
            annual = df['group_score'].resample('A').mean()

        return annual
    except Exception as e:
        log.error(f"Google API Error: {e}")
        return None


def run():
    conn = get_conn()
    cursor = conn.cursor()
    log.info("Starting Google Trends collection (Insert Mode)...")

    results = {}
    for category, kw_list in GROUPS.items():
        log.info(f"Processing Category: {category.upper()}")
        results[category] = fetch_group_score(kw_list, START_YEAR, END_YEAR)
        # Random sleep to prevent 429/400 errors
        time.sleep(random.uniform(4, 7))

    if all(v is not None for v in results.values()):
        final_df = pd.DataFrame(results)
        final_df.reset_index(inplace=True)
        final_df['year'] = final_df['date'].dt.year
        final_df['total_index'] = final_df[['depression', 'anxiety', 'stress']].mean(axis=1)

        # SQL Insert Query
        insert_query = """
            INSERT INTO annual_google_trends 
            (year, depression_score, anxiety_score, stress_score, total_mental_health_index, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        success_count = 0
        for _, row in final_df.iterrows():
            try:
                cursor.execute(insert_query, (
                    int(row['year']),
                    round(float(row['depression']), 4),
                    round(float(row['anxiety']), 4),
                    round(float(row['stress']), 4),
                    round(float(row['total_index']), 4),
                    datetime.utcnow().isoformat()
                ))
                success_count += 1
            except Exception as e:
                log.warning(f"Could not insert year {int(row['year'])}: {e}")

        conn.commit()
        log.info(f"SUCCESS: Inserted {success_count} years of trend data.")
    else:
        log.error("FAILED: Could not retrieve all categories. Check logs.")

    conn.close()


if __name__ == "__main__":
    run()
