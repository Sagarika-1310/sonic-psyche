"""
collectors/10_lyrics_features.py
─────────────────────────────────────────────────────────────────────────────
Extracts linguistic, structural, and semantic features from clean lyrics.

What's computed and stored:

  Lexical richness
    word_count, unique_word_count, type_token_ratio
    mtld — Measure of Textual Lexical Diversity (length-independent;
            more robust than TTR for lyrics of varying length)

  Structure
    line_count, avg_words_per_line, avg_line_length (chars),
    repetition_ratio  — fraction of lines that are exact duplicates,
    rhyme_density     — fraction of consecutive line pairs that rhyme
                        (requires pronouncing / CMU Pronouncing Dict)

  Readability  (textstat)
    flesch_reading_ease, flesch_kincaid_grade,
    syllable_count, avg_syllables_per_word

  POS ratios  (spaCy en_core_web_sm)
    noun_ratio, verb_ratio, adj_ratio, adv_ratio

  LIWC-proxy lexicons  (free word lists; lemma-matched)
    proxy_anx, proxy_sad, proxy_anger, proxy_social,
    proxy_death, proxy_future, proxy_body, proxy_cogmech

  COVID / distress composites  (custom lexicons; lemma-matched)
    isolation_score, hope_score, grief_score,
    agency_score, social_hunger_score

  LDA topic model  (gensim, N_TOPICS = 6)
    Fitted once on the full corpus before the per-song loop.
    lda_dominant_topic, lda_topic_0_prob … lda_topic_5_prob

  SBERT semantic embedding  (sentence-transformers all-MiniLM-L6-v2)
    384-dimensional vector stored as a JSON array (sbert_embedding).

Fully resumable — skips songs where status_features IS NOT NULL.
Requires lyrics to already be collected (status_lyrics = 1).

Run:
    python collectors/09_features.py

Installs:
    pip install nltk textstat pronouncing lexicalrichness
                gensim sentence-transformers torch spacy
    python -m spacy download en_core_web_sm
"""

import sys, json, logging
from datetime import datetime

sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import textstat
from tqdm import tqdm
from db import get_conn, upsert, set_status

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("features")

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ── NLTK bootstrap ────────────────────────────────────────────────────────────
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet",
             "averaged_perceptron_tagger"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as nltk_stopwords

# ── Optional: pronouncing ─────────────────────────────────────────────────────
try:
    import pronouncing

    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False
    log.warning("pronouncing not installed — rhyme_density will be NULL.")

# ── Constants ─────────────────────────────────────────────────────────────────
N_TOPICS = 6
SBERT_MODEL = "all-MiniLM-L6-v2"

# Extra stopwords for LDA preprocessing (lyrics-specific noise)
EXTRA_STOPS = {
    "verse", "chorus", "bridge", "hook", "yeah", "oh", "ooh",
    "ah", "la", "na", "gonna", "gotta", "wanna", "like", "got",
    "know", "just", "let", "get", "go", "say", "mm", "hmm",
}

# ── Lazy model initialisation ─────────────────────────────────────────────────

_spacy_nlp = None
_sbert = None


def get_spacy():
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        # Disable unused pipeline components for speed
        _spacy_nlp = spacy.load("en_core_web_sm",
                                disable=["parser", "ner"])
    return _spacy_nlp


def get_sbert():
    global _sbert
    if _sbert is None:
        log.info("Loading SBERT (%s) …", SBERT_MODEL)
        from sentence_transformers import SentenceTransformer
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


# ── LEXICONS ──────────────────────────────────────────────────────────────────
# Each set is intentionally broad to work across genres and slang registers.
# Matching is done on lemmatised tokens so inflected forms are captured.

LEXICONS = {
    # ── LIWC-proxy categories ─────────────────────────────────────────────────
    "proxy_anx": {
        "afraid", "anxious", "anxiety", "dread", "fear", "fearful", "fright",
        "frighten", "horrify", "horrified", "nervous", "panic", "paranoid",
        "phobia", "scared", "scare", "stress", "terrify", "terror", "threat",
        "tremble", "uneasy", "worry", "worried", "helpless", "overwhelm",
    },
    "proxy_sad": {
        "ache", "alone", "awful", "beg", "bereave", "bleak", "blue", "burden",
        "cry", "dead", "defeat", "depress", "despair", "devastate", "die",
        "disappoint", "doom", "down", "dread", "empty", "fail", "fall",
        "grief", "grieve", "heartbreak", "hopeless", "hurt", "lonely",
        "lose", "loss", "lost", "low", "miss", "mourn", "numb", "pain",
        "regret", "sad", "sorrow", "suffer", "tear", "tragic", "upset",
        "weep", "wish", "yearn",
    },
    "proxy_anger": {
        "anger", "angry", "annoy", "betray", "bitter", "blame", "burn",
        "confront", "crazy", "curse", "damn", "destroy", "enemy", "enrage",
        "explode", "fight", "frustrate", "furious", "fury", "hate", "hatred",
        "hostile", "mad", "manipulate", "offend", "rage", "rebel", "resent",
        "revenge", "savage", "scream", "sick", "spite", "threaten", "violent",
        "war", "wrath",
    },
    "proxy_social": {
        "bond", "belong", "brother", "care", "community", "connect",
        "family", "friend", "gather", "home", "humanity", "join", "kind",
        "love", "loyalty", "meet", "partner", "people", "relation", "share",
        "sister", "social", "team", "together", "trust", "unite", "us",
        "warm", "we", "world", "you",
    },
    "proxy_death": {
        "bury", "coffin", "corpse", "dead", "death", "decay", "die", "dying",
        "end", "eternal", "fade", "fall", "funeral", "ghost", "gone", "grave",
        "kill", "lifeless", "lose", "lost", "mourn", "murder", "perish",
        "rest", "rip", "sacrifice", "spirit", "tomb",
    },
    "proxy_future": {
        "become", "begin", "believe", "build", "change", "come", "dawn",
        "dream", "forward", "future", "gonna", "grow", "hope", "later",
        "next", "plan", "promise", "rise", "shall", "someday",
        "soon", "start", "strive", "tomorrow", "vision", "will", "wish",
        "yet",
    },
    "proxy_body": {
        "arm", "back", "blood", "body", "bone", "brain", "breath", "chest",
        "eye", "face", "feel", "finger", "flesh", "hand", "head", "heart",
        "hold", "hurt", "kiss", "leg", "lip", "mind", "muscle", "neck",
        "pulse", "run", "sense", "shake", "skin", "smile", "soul", "tear",
        "touch", "tremble", "vein", "voice",
    },
    "proxy_cogmech": {
        "analyze", "because", "cause", "consider", "decide", "doubt",
        "explain", "figure", "feel", "guess", "if", "imagine", "know",
        "learn", "maybe", "mean", "notice", "perhaps", "realize", "reason",
        "reflect", "remember", "seem", "should", "suppose", "think",
        "thought", "understand", "wonder", "would",
    },

    # ── COVID / distress composites ───────────────────────────────────────────
    "isolation": {
        "alone", "apart", "away", "cage", "closed", "confine", "dark",
        "distance", "empty", "exclude", "exile", "ghost", "hollow", "home",
        "isolate", "lock", "lonely", "loneliness", "miss", "nothing",
        "prison", "quarantine", "quiet", "separate", "silence", "solitary",
        "stuck", "trap", "void", "wall",
    },
    "hope": {
        "arise", "begin", "better", "believe", "bless", "bright", "chance",
        "dawn", "dream", "faith", "free", "freedom", "grateful", "grow",
        "heal", "heaven", "light", "love", "lucky", "miracle",
        "new", "ok", "okay", "peace", "pray", "promise", "recover", "rise",
        "save", "shine", "smile", "stand", "strength", "stronger", "survive",
        "through", "together", "trust", "victory", "win",
    },
    "grief": {
        "ache", "broken", "bury", "cry", "dead", "death", "depression",
        "devastate", "die", "disappear", "dying", "empty", "fade", "fall",
        "farewell", "ghost", "gone", "goodbye", "grieve", "grief",
        "heartbreak", "hollow", "hopeless", "lose", "loss", "lost", "miss",
        "mourn", "numb", "pain", "sorrow", "suffer", "tears", "tragic",
        "without", "wound",
    },
    "agency": {
        "achieve", "brave", "build", "challenge", "change", "choose", "claim",
        "command", "conquer", "control", "create", "decide", "defeat", "defy",
        "fight", "force", "free", "lead", "master", "me", "mine", "myself",
        "overcome", "own", "power", "push", "reclaim", "refuse", "resist",
        "rise", "rule", "stand", "strive", "take", "triumph", "unbreakable",
        "will", "win",
    },
    "social_hunger": {
        "beside", "close", "embrace", "feel", "hand", "hold", "hug", "kiss",
        "miss", "near", "need", "reach", "return", "share", "skin", "stay",
        "touch", "together", "us", "wait", "want", "warm", "we", "yearn",
    },
}

# ── Text helpers ──────────────────────────────────────────────────────────────

_lemmatizer = WordNetLemmatizer()
_stop_words = set(nltk_stopwords.words("english"))


def tokenize(text: str) -> list:
    """Lowercase alphabetic word tokens — no punctuation."""
    return [w.lower() for w in word_tokenize(text) if w.isalpha()]


def lemmatize(tokens: list) -> list:
    return [_lemmatizer.lemmatize(t) for t in tokens]


def get_lines(text: str) -> list:
    """Non-empty lines from the cleaned text."""
    return [l.strip() for l in text.splitlines() if l.strip()]


def lexicon_prop(lemmas: list, lexicon: set) -> float:
    """Fraction of total lemmas that appear in a given lexicon."""
    if not lemmas:
        return 0.0
    return round(sum(1 for l in lemmas if l in lexicon) / len(lemmas), 6)


# ── Structural features ───────────────────────────────────────────────────────

def repetition_ratio(lines: list) -> float:
    """
    Proportion of lines that are exact duplicates of any earlier line.
    High values indicate hook-heavy / chorus-dominated songs.
    """
    if not lines:
        return 0.0
    seen = set()
    repeated = 0
    for line in lines:
        norm = line.lower().strip()
        if norm in seen:
            repeated += 1
        else:
            seen.add(norm)
    return round(repeated / len(lines), 4)


def rhyme_density(lines: list):
    """
    Fraction of consecutive line pairs whose final words rhyme.
    Uses the CMU Pronouncing Dictionary via the pronouncing library.
    Returns None if pronouncing is unavailable or < 2 lines exist.
    """
    if not HAS_PRONOUNCING or len(lines) < 2:
        return None
    pairs = rhyming = 0
    for a, b in zip(lines, lines[1:]):
        wa = a.split()[-1].lower() if a.split() else ""
        wb = b.split()[-1].lower() if b.split() else ""
        if not wa or not wb:
            continue
        pa = pronouncing.phones_for_word(wa)
        pb = pronouncing.phones_for_word(wb)
        if pa and pb:
            ra = pronouncing.rhyming_part(pa[0])
            rb = pronouncing.rhyming_part(pb[0])
            if ra and rb and ra == rb:
                rhyming += 1
        pairs += 1
    return round(rhyming / pairs, 4) if pairs else None


# ── Lexical diversity ─────────────────────────────────────────────────────────

def compute_mtld(tokens: list):
    """
    Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Length-independent — more reliable than TTR for songs of varying length.
    Requires ≥50 tokens; returns None for shorter texts.
    Falls back gracefully if lexicalrichness is not installed.
    """
    if len(tokens) < 50:
        return None
    try:
        from lexicalrichness import LexicalRichness
        lex = LexicalRichness(" ".join(tokens))
        return round(lex.mtld(threshold=0.72), 4)
    except Exception as e:
        log.debug("MTLD failed: %s", e)
        return None


# ── POS ratios (spaCy) ────────────────────────────────────────────────────────

def pos_ratios(text: str) -> dict:
    """
    Compute noun / verb / adjective / adverb ratios using spaCy.
    PROPN (proper nouns) are merged into the noun count.
    Returns None for all fields if spaCy processing fails.
    """
    nlp = get_spacy()
    doc = nlp(text[:100_000])  # cap to avoid memory spikes on very long lyrics
    non_space = [t for t in doc if not t.is_space]
    total = len(non_space)

    if not total:
        return {"noun_ratio": None, "verb_ratio": None,
                "adj_ratio": None, "adv_ratio": None}

    counts = {"NOUN": 0, "PROPN": 0, "VERB": 0, "ADJ": 0, "ADV": 0}
    for t in non_space:
        if t.pos_ in counts:
            counts[t.pos_] += 1

    return {
        "noun_ratio": round((counts["NOUN"] + counts["PROPN"]) / total, 6),
        "verb_ratio": round(counts["VERB"] / total, 6),
        "adj_ratio": round(counts["ADJ"] / total, 6),
        "adv_ratio": round(counts["ADV"] / total, 6),
    }


# ── LDA topic model ───────────────────────────────────────────────────────────

def build_lda(all_texts: list):
    """
    Fit an LDA model on the full lyrics corpus.
    Must be called once before the per-song loop.
    Returns (lda_model, dictionary).
    """
    from gensim import corpora, models

    log.info("Building LDA model on %d documents …", len(all_texts))
    stop = _stop_words | EXTRA_STOPS

    def preprocess(text):
        return [t for t in lemmatize(tokenize(text))
                if t not in stop and len(t) > 2]

    processed = [preprocess(t) for t in all_texts]
    dictionary = corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=3, no_above=0.85)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed]

    lda = models.LdaModel(
        bow_corpus,
        num_topics=N_TOPICS,
        id2word=dictionary,
        passes=15,
        random_state=42,
        alpha="auto",
        eta="auto",
    )
    log.info("LDA training complete.")
    return lda, dictionary


def get_lda_probs(text: str, lda, dictionary) -> dict:
    """Return per-topic probabilities for a single document."""
    toks = lemmatize(tokenize(text))
    bow = dictionary.doc2bow(toks)
    probs = dict(lda.get_document_topics(bow, minimum_probability=0.0))
    out = {"lda_dominant_topic": max(probs, key=probs.get) if probs else 0}
    for i in range(N_TOPICS):
        out[f"lda_topic_{i}_prob"] = round(probs.get(i, 0.0), 6)
    return out


# ── DB query ──────────────────────────────────────────────────────────────────

def get_songs_needing_features(conn):
    """
    Songs that have clean lyrics but have not yet had features extracted.
    Joins lyrics_raw so clean_text is available without a second query.
    """
    return conn.execute("""
        SELECT s.id, s.title, s.artist, s.year, s.period,
               l.clean_text
        FROM   songs s
        JOIN   lyrics_raw l ON l.song_id = s.id
        WHERE  s.status_features IS NULL
          AND  s.status_lyrics   = 1
          AND  l.clean_text IS NOT NULL
          AND  trim(l.clean_text) != ''
        ORDER  BY s.year, s.chart_rank
    """).fetchall()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn = get_conn()
    songs = get_songs_needing_features(conn)
    log.info("%d songs need feature extraction", len(songs))

    if not songs:
        log.info("Nothing to do — exiting.")
        conn.close()
        return

    # ── Fit LDA on full corpus before per-song loop ───────────────────────────
    # LDA requires the complete vocabulary upfront; this can't be deferred.
    all_texts = [s["clean_text"] for s in songs]
    lda_model, lda_dict = build_lda(all_texts)

    # ── Warm up remaining models ──────────────────────────────────────────────
    get_spacy()
    get_sbert()

    ok = failed = 0

    for song in tqdm(songs, desc="Features"):
        sid = song["id"]
        clean_text = song["clean_text"]

        try:
            tokens = tokenize(clean_text)
            lemmas = lemmatize(tokens)
            lines = get_lines(clean_text)
            syl_cnt = textstat.syllable_count(clean_text)
            pos = pos_ratios(clean_text)

            row = {
                "song_id": sid,

                # ── Lexical richness ──────────────────────────────────────────
                "word_count": len(tokens),
                "unique_word_count": len(set(tokens)),
                "type_token_ratio": round(len(set(tokens)) / len(tokens), 6)
                if tokens else 0.0,
                "mtld": compute_mtld(tokens),

                # ── Structure ─────────────────────────────────────────────────
                "line_count": len(lines),
                "avg_words_per_line": round(len(tokens) / len(lines), 4)
                if lines else 0.0,
                "avg_line_length": round(
                    sum(len(l) for l in lines) / len(lines), 4)
                if lines else 0.0,
                "repetition_ratio": repetition_ratio(lines),
                "rhyme_density": rhyme_density(lines),

                # ── Readability ───────────────────────────────────────────────
                "flesch_reading_ease": round(
                    textstat.flesch_reading_ease(clean_text), 4),
                "flesch_kincaid_grade": round(
                    textstat.flesch_kincaid_grade(clean_text), 4),
                "syllable_count": syl_cnt,
                "avg_syllables_per_word": round(syl_cnt / len(tokens), 4)
                if tokens else 0.0,

                # ── POS ratios ────────────────────────────────────────────────
                **pos,

                # ── LIWC-proxy lexicons ───────────────────────────────────────
                "proxy_anx": lexicon_prop(lemmas, LEXICONS["proxy_anx"]),
                "proxy_sad": lexicon_prop(lemmas, LEXICONS["proxy_sad"]),
                "proxy_anger": lexicon_prop(lemmas, LEXICONS["proxy_anger"]),
                "proxy_social": lexicon_prop(lemmas, LEXICONS["proxy_social"]),
                "proxy_death": lexicon_prop(lemmas, LEXICONS["proxy_death"]),
                "proxy_future": lexicon_prop(lemmas, LEXICONS["proxy_future"]),
                "proxy_body": lexicon_prop(lemmas, LEXICONS["proxy_body"]),
                "proxy_cogmech": lexicon_prop(lemmas, LEXICONS["proxy_cogmech"]),

                # ── COVID / distress composites ───────────────────────────────
                "isolation_score": lexicon_prop(lemmas, LEXICONS["isolation"]),
                "hope_score": lexicon_prop(lemmas, LEXICONS["hope"]),
                "grief_score": lexicon_prop(lemmas, LEXICONS["grief"]),
                "agency_score": lexicon_prop(lemmas, LEXICONS["agency"]),
                "social_hunger_score": lexicon_prop(lemmas, LEXICONS["social_hunger"]),

                # ── LDA ───────────────────────────────────────────────────────
                **get_lda_probs(clean_text, lda_model, lda_dict),

                # ── SBERT embedding ───────────────────────────────────────────
                "sbert_embedding": json.dumps([
                    round(v, 6) for v in
                    get_sbert().encode(
                        clean_text, show_progress_bar=False).tolist()
                ]),

                "analyzed_at": datetime.utcnow().isoformat(),
            }

            upsert(conn, "lyrics_features", row)
            set_status(conn, sid, "status_features", 1)
            conn.commit()
            ok += 1

        except Exception as e:
            log.warning("Failed for song_id=%d ('%s'): %s",
                        sid, song["title"], e)
            set_status(conn, sid, "status_features", 0)
            conn.commit()
            failed += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  FEATURE EXTRACTION COMPLETE")
    print("═" * 56)
    print(f"  Extracted: {ok}  |  Failed: {failed}")
    rate = 100 * ok / max(ok + failed, 1)
    print(f"  Success rate: {rate:.1f}%")

    conn2 = get_conn()

    # Lexical diversity by period
    print("\n  Mean lexical diversity by period:")
    rows = conn2.execute("""
        SELECT s.year, s.period,
               AVG(f.type_token_ratio) AS avg_ttr,
               AVG(f.mtld)             AS avg_mtld,
               AVG(f.repetition_ratio) AS avg_rep,
               COUNT(*)                AS n
        FROM   lyrics_features f
        JOIN   songs s ON s.id = f.song_id
        GROUP  BY s.period, s.year
        ORDER  BY s.year
    """).fetchall()
    for r in rows:
        mtld_str = f"{r['avg_mtld']:.1f}" if r["avg_mtld"] else "  N/A"
        print(f"    {r['year']} [{r['period'][:12]:12}]  "
              f"TTR={r['avg_ttr']:.3f}  MTLD={mtld_str:>6}  "
              f"rep={r['avg_rep']:.2f}  (n={r['n']})")

    # Coverage by period
    print("\n  Feature extraction coverage by period:")
    cov = conn2.execute("""
        SELECT s.year, s.period,
               SUM(CASE WHEN s.status_features=1 THEN 1 ELSE 0 END) AS ok,
               COUNT(*) AS total
        FROM   songs s
        GROUP  BY s.period, s.year
        ORDER  BY s.year
    """).fetchall()
    for r in cov:
        pct = 100 * r["ok"] / max(r["total"], 1)
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {r['year']} [{r['period'][:12]:12}]  "
              f"{r['ok']:3}/{r['total']} ({pct:.0f}%)  {bar}")

    # LDA topic inspection
    print(f"\n  Top words per LDA topic (N={N_TOPICS}):")
    for tid in range(N_TOPICS):
        top = lda_model.show_topic(tid, topn=8)
        words = ", ".join(w for w, _ in top)
        print(f"    Topic {tid}: {words}")

    conn2.close()
    print("═" * 56)


if __name__ == "__main__":
    run()
