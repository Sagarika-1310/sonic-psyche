"""
collectors/09_lyrics_sentiment.py
─────────────────────────────────────────────────────────────────────────────
Runs sentiment and emotion analysis on clean lyrics from lyrics_raw.

Three complementary models:

  1. VADER  — rule-based, no GPU needed. Scored line-by-line then averaged.
             Captures polarity variance across verses via line-level std dev.

  2. NRCLex — lexicon-based. Maps tokens to 8 primary emotions
             (joy, trust, anticipation, surprise, fear, sadness, disgust,
             anger) plus positive / negative meta-labels. Stored as
             proportions of emotion-tagged tokens.

  3. DistilRoBERTa ('j-hartmann/emotion-english-distilroberta-base')
             — transformer 7-class emotion classifier (joy, sadness, anger,
             fear, disgust, surprise, neutral). Lyrics chunked into ≤512-
             token windows; probabilities averaged across chunks.

What's stored:
  - VADER:         compound, pos, neu, neg, line_std
  - NRCLex:        8 emotion proportions + positive/negative + dominant_emotion
  - DistilRoBERTa: 7 emotion probabilities + argmax label
  - Derived:       proxy_posemo, proxy_negemo, anxiety_score, sentiment_label

Fully resumable — skips songs where status_sentiment IS NOT NULL.
Requires lyrics to already be collected (status_lyrics = 1).

Run:
    python collectors/08_sentiment.py

Installs:
    pip install vaderSentiment nrclex transformers torch
"""

import sys, logging
from datetime import datetime
from statistics import stdev

sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

from tqdm import tqdm
from db import get_conn, upsert, set_status, get_songs_needing

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("sentiment")

# ── Constants ─────────────────────────────────────────────────────────────────

VADER_THRESHOLD = 0.05
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"
CHUNK_TOKENS = 512  # hard model context limit

# ── Lazy model initialisation ─────────────────────────────────────────────────

_vader = None
_emotion_model = None
_emotion_tok = None


def get_vader():
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
    return _vader


def get_emotion_model():
    global _emotion_model, _emotion_tok
    if _emotion_model is None:
        log.info("Loading DistilRoBERTa emotion model "
                 "(first run may download ~300 MB) …")
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        _emotion_tok = AutoTokenizer.from_pretrained(EMOTION_MODEL)
        _emotion_model = AutoModelForSequenceClassification.from_pretrained(
            EMOTION_MODEL)
        _emotion_model.eval()
    return _emotion_model, _emotion_tok


# ── VADER analysis ────────────────────────────────────────────────────────────

def run_vader(clean_text: str) -> dict:
    """
    Score every non-empty line independently; return mean of each VADER
    dimension plus std dev of compound scores across lines.
    Line-level averaging captures mood shifts between verses better than
    scoring the entire blob at once.
    """
    analyzer = get_vader()
    lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

    if not lines:
        return {"compound": 0.0, "pos": 0.0, "neu": 1.0,
                "neg": 0.0, "line_std": 0.0}

    scores = [analyzer.polarity_scores(ln) for ln in lines]
    compounds = [s["compound"] for s in scores]
    n = len(scores)

    return {
        "compound": round(sum(s["compound"] for s in scores) / n, 6),
        "pos": round(sum(s["pos"] for s in scores) / n, 6),
        "neu": round(sum(s["neu"] for s in scores) / n, 6),
        "neg": round(sum(s["neg"] for s in scores) / n, 6),
        "line_std": round(stdev(compounds), 6) if len(compounds) > 1 else 0.0,
    }


# ── NRCLex analysis ───────────────────────────────────────────────────────────

NRC_EMOTIONS = [
    "joy", "trust", "anticipation", "surprise",
    "fear", "sadness", "disgust", "anger",
    "positive", "negative",  # meta-labels
]
NRC_CORE = [e for e in NRC_EMOTIONS if e not in ("positive", "negative")]


def run_nrclex(clean_text: str) -> dict:
    """
    Map tokens to the NRC lexicon; return each emotion as a proportion
    of total emotion-tagged tokens (so scores are comparable across
    songs of different lengths).
    """
    from nrclex import NRCLex
    nrc = NRCLex()
    nrc.load_raw_text(clean_text)
    raw_freq = nrc.raw_emotion_scores
    total = sum(raw_freq.values()) or 1

    out = {f"nrc_{e}": round(raw_freq.get(e, 0) / total, 6)
           for e in NRC_EMOTIONS}

    core_counts = {e: raw_freq.get(e, 0) for e in NRC_CORE}
    out["nrc_dominant_emotion"] = (
        max(core_counts, key=core_counts.get)
        if any(core_counts.values()) else None
    )
    return out


# ── DistilRoBERTa 7-class emotion analysis ────────────────────────────────────

def run_emotion_model(clean_text: str) -> dict:
    """
    Tokenise full lyrics, split into non-overlapping ≤512-token chunks
    (honouring the model's hard context limit), classify each chunk, then
    average softmax probabilities. Label is argmax of averaged scores.

    Label order is read from model.config.id2label so it is always correct
    regardless of checkpoint changes.
    """
    import torch
    import torch.nn.functional as F

    model, tokenizer = get_emotion_model()
    label_order = [model.config.id2label[i].lower()
                   for i in range(len(model.config.id2label))]

    token_ids = tokenizer.encode(clean_text, add_special_tokens=False)
    chunk_size = CHUNK_TOKENS - 2  # reserve 2 slots for [CLS] / [SEP]
    chunks = [token_ids[i: i + chunk_size]
              for i in range(0, max(len(token_ids), 1), chunk_size)]

    accum = [0.0] * len(label_order)
    with torch.no_grad():
        for chunk in chunks:
            ids = ([tokenizer.cls_token_id] + chunk +
                   [tokenizer.sep_token_id])
            tensor = torch.tensor([ids])
            probs = F.softmax(model(tensor).logits, dim=-1).squeeze().tolist()
            for i, p in enumerate(probs):
                accum[i] += p

    n = len(chunks)
    avg = [round(p / n, 6) for p in accum]
    scores = {label_order[i]: avg[i] for i in range(len(label_order))}
    scores["label"] = max(scores, key=scores.get)
    return scores


# ── Derived helpers ───────────────────────────────────────────────────────────

def compound_to_label(compound: float) -> str:
    if compound >= VADER_THRESHOLD: return "positive"
    if compound <= -VADER_THRESHOLD: return "negative"
    return "neutral"


# ── DB query ──────────────────────────────────────────────────────────────────

def get_songs_needing_sentiment(conn):
    """
    Songs that have clean lyrics but have not yet been sentiment-analysed.
    Joins lyrics_raw so clean_text is available without a second query.
    """
    return conn.execute("""
        SELECT s.id, s.title, s.artist, s.year, s.period,
               l.clean_text
        FROM   songs s
        JOIN   lyrics_raw l ON l.song_id = s.id
        WHERE  s.status_sentiment IS NULL
          AND  s.status_lyrics    = 1
          AND  l.clean_text IS NOT NULL
          AND  trim(l.clean_text) != ''
        ORDER  BY s.year, s.chart_rank
    """).fetchall()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()
    songs = get_songs_needing_sentiment(conn)
    log.info("%d songs need sentiment analysis", len(songs))

    if not songs:
        log.info("Nothing to do — exiting.")
        conn.close()
        return

    # Warm up all models before the loop so the first song isn't penalised
    get_vader()
    get_emotion_model()

    ok = failed = 0

    for song in tqdm(songs, desc="Sentiment"):
        sid = song["id"]
        clean_text = song["clean_text"]

        try:
            v = run_vader(clean_text)
            nrc = run_nrclex(clean_text)
            emo = run_emotion_model(clean_text)

            row = {
                "song_id": sid,

                # VADER
                "vader_compound": v["compound"],
                "vader_pos": v["pos"],
                "vader_neu": v["neu"],
                "vader_neg": v["neg"],
                "vader_line_std": v["line_std"],

                # NRCLex
                **{k: nrc[k] for k in nrc},

                # DistilRoBERTa
                "emotion_label": emo["label"],
                "emotion_joy": emo.get("joy"),
                "emotion_sadness": emo.get("sadness"),
                "emotion_anger": emo.get("anger"),
                "emotion_fear": emo.get("fear"),
                "emotion_disgust": emo.get("disgust"),
                "emotion_surprise": emo.get("surprise"),
                "emotion_neutral": emo.get("neutral"),

                # Derived
                "proxy_posemo": round(nrc["nrc_joy"] + nrc["nrc_trust"], 6),
                "proxy_negemo": round(
                    nrc["nrc_sadness"] + nrc["nrc_fear"] +
                    nrc["nrc_anger"] + nrc["nrc_disgust"], 6),
                "anxiety_score": round(
                    0.6 * nrc["nrc_fear"] +
                    0.4 * min(v["line_std"] * v["neg"], 1.0), 6),
                "sentiment_label": compound_to_label(v["compound"]),
                "analyzed_at": datetime.utcnow().isoformat(),
            }

            upsert(conn, "lyrics_sentiment", row)
            set_status(conn, sid, "status_sentiment", 1)
            conn.commit()
            ok += 1

        except Exception as e:
            log.warning("Failed for song_id=%d ('%s'): %s",
                        sid, song["title"], e)
            set_status(conn, sid, "status_sentiment", 0)
            conn.commit()
            failed += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  SENTIMENT ANALYSIS COMPLETE")
    print("═" * 56)
    print(f"  Analysed: {ok}  |  Failed: {failed}")
    rate = 100 * ok / max(ok + failed, 1)
    print(f"  Success rate: {rate:.1f}%")

    conn2 = get_conn()

    # Mean VADER compound per period — bar centred at 0
    print("\n  Mean VADER compound by period:")
    rows = conn2.execute("""
        SELECT s.year, s.period,
               AVG(ls.vader_compound) AS avg_c,
               AVG(ls.vader_pos)      AS avg_p,
               AVG(ls.vader_neg)      AS avg_n,
               COUNT(*)               AS n
        FROM   lyrics_sentiment ls
        JOIN   songs s ON s.id = ls.song_id
        GROUP  BY s.period, s.year
        ORDER  BY s.year
    """).fetchall()
    for r in rows:
        pos = max(0, min(20, int((r["avg_c"] + 1) * 10)))
        bar = "░" * pos + "█" + "░" * (20 - pos)
        print(f"    {r['year']} [{r['period'][:12]:12}]  "
              f"compound={r['avg_c']:+.3f}  "
              f"+{r['avg_p']:.2f}/-{r['avg_n']:.2f}  "
              f"(n={r['n']})  {bar}")

    # NRC dominant emotion distribution
    print("\n  NRC dominant emotion distribution:")
    nrc_rows = conn2.execute("""
        SELECT nrc_dominant_emotion AS emo, COUNT(*) AS n
        FROM   lyrics_sentiment
        WHERE  nrc_dominant_emotion IS NOT NULL
        GROUP  BY emo
        ORDER  BY n DESC
    """).fetchall()
    total_nrc = sum(r["n"] for r in nrc_rows)
    for r in nrc_rows:
        pct = 100 * r["n"] / max(total_nrc, 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {r['emo']:15}  {r['n']:4} ({pct:.0f}%)  {bar}")

    # Sentiment label distribution
    print("\n  Sentiment label distribution (VADER):")
    lbl_rows = conn2.execute("""
        SELECT sentiment_label, COUNT(*) AS n
        FROM   lyrics_sentiment
        GROUP  BY sentiment_label
        ORDER  BY n DESC
    """).fetchall()
    total_lbl = sum(r["n"] for r in lbl_rows)
    for r in lbl_rows:
        pct = 100 * r["n"] / max(total_lbl, 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {r['sentiment_label']:10}  {r['n']:4} ({pct:.0f}%)  {bar}")

    # VADER ↔ DistilRoBERTa agreement rate
    agree = conn2.execute("""
        SELECT COUNT(*) AS n
        FROM   lyrics_sentiment
        WHERE  sentiment_label = emotion_label
    """).fetchone()["n"]
    total_done = conn2.execute(
        "SELECT COUNT(*) AS n FROM lyrics_sentiment"
    ).fetchone()["n"]
    if total_done:
        print(f"\n  VADER ↔ DistilRoBERTa agreement: "
              f"{agree}/{total_done} ({100 * agree / total_done:.1f}%)")

    conn2.close()
    print("═" * 56)


if __name__ == "__main__":
    run()
