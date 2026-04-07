"""
collectors/06_audio_essentia.py  ── FULL FEATURE EDITION
─────────────────────────────────────────────────────────────────────────────
Extracts ALL available Essentia features — both algorithmic (no ML) and
deep-learning classification heads — and stores them in `audio_essentia`.

ARCHITECTURE OVERVIEW
─────────────────────
Two families of ML models are used, each needing its own embedding extractor:

  VGGish   (16 kHz) →  TensorflowPredictVGGish
      Mood: happy, sad, relaxed, aggressive, party
      Moods MIREX (5-cluster), Arousal/Valence DEAM + EmoMusic
      Voice: instrumental/vocal, gender

  EffNet   (16 kHz) →  TensorflowPredictEffnetDiscogs
      Mood: acoustic, electronic, danceability (ML version)
      Approachability, Engagement, Timbre, Tonal/Atonal

Both embeddings are computed once per file, then shared across all
classification heads — so computational cost is low.

ALGORITHMIC (NON-ML) ADDITIONS
──────────────────────────────
  Intensity          Rule-based aggression: -1 relaxed / 0 moderate / 1 aggressive
  PitchSalience      Mean melodic salience (presence of pitched content)
  TuningFrequency    Deviation from standard A=440 Hz
  ChordsDetection    Key, scale, mean chord strength, chord change rate
  StartStopSilence   Leading/trailing silence → silence_ratio
  SpectralFlatness   0=tonal / 1=noise-like (per frame, averaged)
  SpectralRolloff    Frequency below which 85% of spectrum energy lies

NEW COLUMNS (add these to your audio_essentia table — see bottom of file)
─────────────────────────────────────────────────────────────────────────
  mood_party, mood_acoustic, mood_electronic
  approachability, engagement
  arousal_deam, valence_deam
  arousal_emomusic, valence_emomusic
  mirex_cluster1 … mirex_cluster5
  voice_instrumental (replaces broken placeholder)
  voice_gender_female, voice_gender_male
  timbre_bright, timbre_dark
  tonal_atonal
  intensity
  pitch_salience_mean
  tuning_frequency
  silence_ratio
  spectral_rolloff
  spectral_flatness
  chords_key, chords_scale, chords_strength_mean, chords_changes_rate

Installation:
    pip install essentia-tensorflow    # full build w/ TF support
    # or:  conda install -c conda-forge essentia-tensorflow
"""

import os
import ssl
import sys
import logging
import urllib.request
import numpy as np
import pyloudnorm as pyln
from tqdm import tqdm

sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

from db import get_conn, upsert, set_status
from config import MODELS_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("essentia")

# ── Essentia availability check ───────────────────────────────────────────────
try:
    import essentia
    import essentia.standard as es

    ESSENTIA_AVAILABLE = True
    log.info("Essentia version: %s", essentia.__version__)
except ImportError:
    ESSENTIA_AVAILABLE = False
    log.error(
        "Essentia not installed.  Run: pip install essentia-tensorflow\n"
        "This collector will be skipped."
    )

# ══════════════════════════════════════════════════════════════════════════════
# MODEL CATALOGUE
# Keys are the local filenames that will be stored in MODELS_DIR.
# Two base URLs: VGGish extractor (16 kHz) and EffNet extractor (16 kHz).
# ══════════════════════════════════════════════════════════════════════════════

_BASE_HEADS = "https://essentia.upf.edu/models/classification-heads"
_BASE_FE = "https://essentia.upf.edu/models/feature-extractors"

MODEL_URLS: dict[str, str] = {
    # ── Feature extractors ────────────────────────────────────────────────────
    # VGGish: used by most mood/voice heads
    "audioset-vggish-3.pb":
        f"{_BASE_FE}/vggish/audioset-vggish-3.pb",

    # EffNet: used by acoustic/electronic/approachability/engagement/timbre/tonal
    "discogs-effnet-bs64-1.pb":
        f"{_BASE_FE}/discogs-effnet/discogs-effnet-bs64-1.pb",

    # MusiCNN: kept for backward-compat with mood heads that still require it
    "msd-musicnn-1.pb":
        f"{_BASE_FE}/musicnn/msd-musicnn-1.pb",

    # ── VGGish classification heads ───────────────────────────────────────────
    "mood_happy-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/mood_happy/mood_happy-audioset-vggish-1.pb",
    "mood_sad-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/mood_sad/mood_sad-audioset-vggish-1.pb",
    "mood_relaxed-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/mood_relaxed/mood_relaxed-audioset-vggish-1.pb",
    "mood_aggressive-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/mood_aggressive/mood_aggressive-audioset-vggish-1.pb",
    "mood_party-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/mood_party/mood_party-audioset-vggish-1.pb",

    # MIREX 5-cluster mood taxonomy
    "moods_mirex-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/moods_mirex/moods_mirex-audioset-vggish-1.pb",

    # Arousal/Valence regression (DEAM dataset)
    "arousal_valence_deam-audioset-vggish-2.pb":
        f"{_BASE_HEADS}/deam/deam-audioset-vggish-2.pb",

    # Arousal/Valence regression (EmoMusic dataset)
    "arousal_valence_emomusic-audioset-vggish-2.pb":
        f"{_BASE_HEADS}/emomusic/emomusic-audioset-vggish-2.pb",

    # Arousal/Valence regression (MuSe dataset)
    "arousal_valence_muse-audioset-vggish-2.pb":
        f"{_BASE_HEADS}/muse/muse-audioset-vggish-2.pb",

    # Voice/Instrumental binary
    "voice_instrumental-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/voice_instrumental/voice_instrumental-audioset-vggish-1.pb",

    # Voice gender (female / male)
    "gender-audioset-vggish-1.pb":
        f"{_BASE_HEADS}/gender/gender-audioset-vggish-1.pb",

    # ── EffNet classification heads ───────────────────────────────────────────
    "mood_acoustic-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/mood_acoustic/mood_acoustic-discogs-effnet-1.pb",
    "mood_electronic-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/mood_electronic/mood_electronic-discogs-effnet-1.pb",
    "approachability_2c-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/approachability/approachability_2c-discogs-effnet-1.pb",
    "engagement_2c-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/engagement/engagement_2c-discogs-effnet-1.pb",
    "timbre-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/timbre/timbre-discogs-effnet-1.pb",
    "tonal_atonal-discogs-effnet-1.pb":
        f"{_BASE_HEADS}/tonal_atonal/tonal_atonal-discogs-effnet-1.pb",
}


def ensure_models_exist(model_dir: str = MODELS_DIR) -> None:
    """Download all required model weights if not already present."""
    ssl._create_default_https_context = ssl._create_unverified_context
    os.makedirs(model_dir, exist_ok=True)

    for filename, url in MODEL_URLS.items():
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            log.info("Downloading %-55s …", filename)
            try:
                urllib.request.urlretrieve(url, path)
            except Exception as exc:
                log.warning("  Could not download %s: %s", filename, exc)


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO LOADING
# ML models all want 16 kHz; algorithmic extractors prefer 44.1 kHz.
# We load both variants once per file.
# ══════════════════════════════════════════════════════════════════════════════

def load_audio(wav_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Returns (audio_44k, audio_16k).
    audio_44k: 44100 Hz — for algorithmic extractors (danceability, rhythm …)
    audio_16k: 16000 Hz — for VGGish / EffNet ML models
    Either may be None if loading fails.
    """
    try:
        audio_44k = es.MonoLoader(filename=wav_path, sampleRate=44100)()
    except Exception as exc:
        log.error("44 kHz load failed (%s): %s", wav_path, exc)
        audio_44k = None

    try:
        audio_16k = es.MonoLoader(
            filename=wav_path, sampleRate=16000, resampleQuality=4
        )()
    except Exception as exc:
        log.error("16 kHz load failed (%s): %s", wav_path, exc)
        audio_16k = None

    return audio_44k, audio_16k


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING CACHE
# Both VGGish and EffNet embeddings are computed once per file and reused
# by every classification head that depends on them.
# ══════════════════════════════════════════════════════════════════════════════

def _model_path(filename: str) -> str | None:
    p = os.path.join(MODELS_DIR, filename)
    return p if os.path.exists(p) else None


def compute_vggish_embeddings(audio_16k: np.ndarray) -> np.ndarray | None:
    """Extract VGGish embeddings (shape: [frames, 128])."""
    pb = _model_path("audioset-vggish-3.pb")
    if pb is None:
        return None
    try:
        model = es.TensorflowPredictVGGish(
            graphFilename=pb, output="model/vggish/embeddings"
        )
        return model(audio_16k)
    except Exception as exc:
        log.error("VGGish embedding failed: %s", exc)
        return None


def compute_effnet_embeddings(audio_16k: np.ndarray) -> np.ndarray | None:
    """Extract EffNet (Discogs) embeddings (shape: [frames, 1280])."""
    pb = _model_path("discogs-effnet-bs64-1.pb")
    if pb is None:
        return None
    try:
        model = es.TensorflowPredictEffnetDiscogs(
            graphFilename=pb, output="PartitionedCall:1"
        )
        return model(audio_16k)
    except Exception as exc:
        log.error("EffNet embedding failed: %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# ALGORITHMIC EXTRACTORS  (no ML, deterministic)
# ══════════════════════════════════════════════════════════════════════════════

def extract_danceability(audio_44k: np.ndarray) -> dict:
    """
    Essentia Danceability algorithm.
    Raw score 0–3; normalised to 0–1.
    Measures rhythm periodicity, tempo regularity, and beat-pulse strength.
    """
    try:
        dance_raw, _ = es.Danceability(sampleRate=44100)(audio_44k)
        return {
            "danceability": float(dance_raw),
            "danceability_normalised": float(np.clip(dance_raw / 3.0, 0.0, 1.0)),
        }
    except Exception as exc:
        log.error("Danceability failed: %s", exc)
        return {}


def extract_rhythm(audio_44k: np.ndarray) -> dict:
    """RhythmExtractor2013 — Essentia's most accurate BPM detector."""
    try:
        bpm, beats, confidence, _, loudness_bass = es.RhythmExtractor2013(
            method="multifeature"
        )(audio_44k)
        result = {
            "bpm": float(bpm),
            "bpm_confidence": float(confidence),
            "beats_count": int(len(beats)),
        }
        if len(loudness_bass):
            result["beat_loudness_mean"] = float(np.mean(loudness_bass))
            result["beat_loudness_std"] = float(np.std(loudness_bass))
            # Derived convenience columns
            result["rhythm_strength"] = result["beat_loudness_mean"]
            result["rhythm_regularity"] = result["bpm_confidence"]
        return result
    except Exception as exc:
        log.error("Rhythm failed: %s", exc)
        return {}


def extract_key(audio_44k: np.ndarray) -> dict:
    """Essentia KeyExtractor — HPCP-profile based key + scale detection."""
    try:
        key, scale, strength = es.KeyExtractor()(audio_44k)
        return {
            "key_essentia": key,
            "scale_essentia": scale,
            "key_strength": float(strength),
        }
    except Exception as exc:
        log.error("Key failed: %s", exc)
        return {}


def extract_loudness(audio_44k: np.ndarray, sample_rate: int = 44100) -> dict:
    """
    EBU R128 loudness using pyloudnorm (version-safe)
    """
    try:
        audio = np.squeeze(audio_44k).astype(np.float64)

        meter = pyln.Meter(sample_rate)

        # Integrated loudness
        loudness_integrated = meter.integrated_loudness(audio)

        # Loudness range (LRA)
        loudness_range = meter.loudness_range(audio)

        # --- Manual block processing ---
        def block_loudness(audio, block_size_sec):
            block_size = int(block_size_sec * sample_rate)
            hop_size = block_size // 2  # 50% overlap

            values = []
            for i in range(0, len(audio) - block_size, hop_size):
                block = audio[i:i + block_size]
                try:
                    val = meter.integrated_loudness(block)
                    values.append(val)
                except:
                    continue

            return np.array(values)

        # Momentary (400 ms)
        momentary = block_loudness(audio, 0.4)

        result = {
            "loudness_integrated": float(loudness_integrated),
            "loudness_range": float(loudness_range),
        }

        if len(momentary) > 0:
            result["loudness_momentary_max"] = float(np.max(momentary))

        return result

    except Exception as exc:
        log.error(f"Loudness failed: {exc}")
        return {}


def extract_dynamic_complexity(audio_44k: np.ndarray) -> dict:
    """DynamicComplexity — loudness variance. Low = EDM, high = classical."""
    try:
        complexity, _ = es.DynamicComplexity()(audio_44k)
        return {"dynamic_complexity": float(complexity)}
    except Exception as exc:
        log.error("DynamicComplexity failed: %s", exc)
        return {}


def extract_intensity(audio_44k: np.ndarray) -> dict:
    """
    Intensity algorithm — rule-based mood classifier.
    Returns: -1 (relaxed) | 0 (moderate) | 1 (aggressive)
    Useful cross-check against ML mood_aggressive classifier.
    """
    try:
        intensity = es.Intensity(sampleRate=44100)(audio_44k)
        return {"intensity": int(intensity)}
    except Exception as exc:
        log.error("Intensity failed: %s", exc)
        return {}


def extract_tonal_features(audio_44k: np.ndarray) -> dict:
    """
    Frame-level tonal features:
      - Dissonance            (tonal roughness / harmonic clashiness)
      - SpectralComplexity    (number of spectral peaks)
      - HPCP entropy          (tonal diversity / ambiguity)
      - SpectralFlatness      (0 = tonal, 1 = noise-like)
      - SpectralRolloff       (85% energy frequency threshold)
    """
    windowing = es.Windowing(type="blackmanharris62")
    spectrum = es.Spectrum()
    spec_peaks = es.SpectralPeaks()
    hpcp_algo = es.HPCP()
    dissonance = es.Dissonance()
    spec_cplx = es.SpectralComplexity()
    flatness = es.Flatness()
    rolloff = es.RollOff()

    dissonances, complexities, hpcp_frames, flatnesses, rolloffs = [], [], [], [], []

    for frame in es.FrameGenerator(audio_44k, frameSize=2048, hopSize=1024):
        w = windowing(frame)
        spec = spectrum(w)

        try:
            complexities.append(spec_cplx(spec))
        except Exception:
            pass
        try:
            flatnesses.append(flatness(spec))
        except Exception:
            pass
        try:
            rolloffs.append(rolloff(spec))
        except Exception:
            pass
        try:
            freqs, mags = spec_peaks(spec)
            if len(freqs):
                dissonances.append(dissonance(freqs, mags))
                hpcp_frames.append(hpcp_algo(freqs, mags))
        except Exception:
            pass

    result = {}
    if dissonances:
        result["dissonance"] = float(np.mean(dissonances))
    if complexities:
        result["spectral_complexity"] = float(np.mean(complexities))
    if flatnesses:
        result["spectral_flatness"] = float(np.mean(flatnesses))
    if rolloffs:
        result["spectral_rolloff"] = float(np.mean(rolloffs))
    if hpcp_frames:
        hpcp_mean = np.mean(hpcp_frames, axis=0)
        hpcp_norm = hpcp_mean / (np.sum(hpcp_mean) + 1e-9)
        result["hpcp_entropy"] = float(
            -np.sum(hpcp_norm * np.log2(hpcp_norm + 1e-9))
        )
    return result


def extract_pitch_salience(audio_44k: np.ndarray) -> dict:
    """
    PitchSalience — mean presence of pitched (melodic) content.
    High → clear melody / pitched instrument.
    Low  → percussive / noise-heavy content.
    """
    windowing = es.Windowing(type="blackmanharris62")
    spectrum_a = es.Spectrum()
    salience_a = es.PitchSalience()
    saliences = []

    for frame in es.FrameGenerator(audio_44k, frameSize=2048, hopSize=1024):
        try:
            spec = spectrum_a(windowing(frame))
            saliences.append(salience_a(spec))
        except Exception:
            pass

    if saliences:
        return {"pitch_salience_mean": float(np.mean(saliences))}
    return {}


def extract_silence_ratio(audio_44k: np.ndarray) -> dict:
    """
    StartStopSilence — fraction of total audio that is leading/trailing
    silence.  Low = music starts/ends cleanly; high = lots of padding.
    """
    try:
        start_frame, stop_frame = es.StartStopSilence()(audio_44k)
        total = len(audio_44k)
        silence_samples = start_frame + max(0, total - stop_frame)
        ratio = float(np.clip(silence_samples / max(total, 1), 0.0, 1.0))
        return {"silence_ratio": ratio}
    except Exception as exc:
        log.error("StartStopSilence failed: %s", exc)
        return {}


def extract_chords(audio_path: str) -> dict:
    try:
        extractor = es.MusicExtractor(
            lowlevelStats=['mean'],
            rhythmStats=['mean'],
            tonalStats=['mean']
        )

        features, features_frames = extractor(audio_path)

        def safe_get(pool, key, default=None):
            return pool[key] if key in pool.descriptorNames() else default

        return {
            "chords_key": safe_get(features, "tonal.chords_key"),
            "chords_scale": safe_get(features, "tonal.chords_scale"),
            "chords_strength_mean": float(safe_get(features, "tonal.chords_strength.mean", 0.0)),
            "chords_changes_rate": float(safe_get(features, "tonal.chords_changes_rate", 0.0)),
            "tuning_frequency": safe_get(features, "tonal.tuning_frequency")
        }

    except Exception as e:
        print("Extraction failed:", e)
        return {}


def extract_mfcc_gfcc(audio_44k: np.ndarray) -> dict:
    """
    Essentia MFCCs (13 coefficients) and GFCCs (5 coefficients).
    Frame-averaged means only — covariance not stored for brevity.
    """
    windowing = es.Windowing(type="blackmanharris62")
    spectrum_a = es.Spectrum()
    mfcc_a = es.MFCC(numberCoefficients=13)
    gfcc_a = es.GFCC(numberCoefficients=5)
    mfccs, gfccs = [], []

    for frame in es.FrameGenerator(audio_44k, frameSize=2048, hopSize=1024):
        spec = spectrum_a(windowing(frame))
        try:
            _, m = mfcc_a(spec)
            mfccs.append(m)
        except Exception:
            pass
        try:
            _, g = gfcc_a(spec)
            gfccs.append(g)
        except Exception:
            pass

    result = {}
    if mfccs:
        mfcc_mean = np.mean(mfccs, axis=0)
        for i, v in enumerate(mfcc_mean[:13], start=1):
            result[f"mfcc_mean_{i}"] = float(v)
    if gfccs:
        gfcc_mean = np.mean(gfccs, axis=0)
        for i, v in enumerate(gfcc_mean[:5], start=1):
            result[f"gfcc_mean_{i}"] = float(v)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# ML CLASSIFIERS  —  VGGish-based  (binary / multi-class)
# ══════════════════════════════════════════════════════════════════════════════

def _predict_vggish_binary(embeddings: np.ndarray, model_file: str, output="model/Softmax") -> float | None:
    """
    Run a binary VGGish classification head.
    Returns probability of the positive class (index 1), averaged over frames.
    """
    pb = _model_path(model_file)
    if pb is None or embeddings is None:
        return None
    try:
        model = es.TensorflowPredict2D(
            graphFilename=pb,
            input="model/Placeholder",
            output=output,
        )
        preds = model(embeddings)  # [frames, 2]
        return float(np.mean(preds[:, 1]))
    except Exception as exc:
        log.error("%s prediction failed: %s", model_file, exc)
        return None


def extract_mood_classifiers_vggish(embeddings: np.ndarray) -> dict:
    """
    Five binary mood classifiers on VGGish embeddings.
    All return P(positive class): happy, sad, relaxed, aggressive, party.
    """
    moods = {
        "mood_happy": "mood_happy-audioset-vggish-1.pb",
        "mood_sad": "mood_sad-audioset-vggish-1.pb",
        "mood_relaxed": "mood_relaxed-audioset-vggish-1.pb",
        "mood_aggressive": "mood_aggressive-audioset-vggish-1.pb",
        "mood_party": "mood_party-audioset-vggish-1.pb",
    }
    result = {}
    for col, model_file in moods.items():
        val = _predict_vggish_binary(embeddings, model_file)
        if val is not None:
            result[col] = val
    return result


def extract_mirex_moods(embeddings: np.ndarray) -> dict:
    """
    Moods MIREX — 5-cluster taxonomy (multi-class softmax, not binary).

    Cluster definitions:
      1: Passionate / Rousing / Confident / Boisterous / Rowdy
      2: Rollicking / Cheerful / Fun / Sweet / Amiable
      3: Literate / Poignant / Wistful / Bittersweet / Brooding
      4: Humorous / Silly / Quirky / Whimsical / Wry
      5: Aggressive / Fiery / Tense / Intense / Visceral
    """
    pb = _model_path("moods_mirex-audioset-vggish-1.pb")
    if pb is None or embeddings is None:
        return {}
    try:
        model = es.TensorflowPredict2D(
            graphFilename=pb,
            input="serving_default_model_Placeholder",
            output="PartitionedCall",
        )
        preds = model(embeddings)  # [frames, 5]
        means = np.mean(preds, axis=0)
        return {f"mirex_cluster{i + 1}": float(means[i]) for i in range(5)}
    except Exception as exc:
        log.error("MIREX moods failed: %s", exc)
        return {}


def extract_arousal_valence(embeddings: np.ndarray) -> dict:
    """
    Arousal / Valence regression on VGGish embeddings.
    Three dataset variants: DEAM and EmoMusic and MuSe.

    DEAM range: arousal & valence both in [1, 9] (Russell circumplex model).
    EmoMusic:   similar continuous range.

    Interpretation:
      High arousal + high valence  → happy, excited
      High arousal + low valence   → angry, tense
      Low arousal  + high valence  → calm, content
      Low arousal  + low valence   → sad, depressed
    """
    result = {}
    configs = {
        "deam": "arousal_valence_deam-audioset-vggish-2.pb",
        "emomusic": "arousal_valence_emomusic-audioset-vggish-2.pb",
        "muse": "arousal_valence_muse-audioset-vggish-2.pb"
    }
    for tag, model_file in configs.items():
        pb = _model_path(model_file)
        if pb is None:
            continue
        try:
            model = es.TensorflowPredict2D(
                graphFilename=pb,
                input="model/Placeholder",
                output="model/Identity",
            )
            preds = model(embeddings)  # [frames, 2] → [arousal, valence]
            means = np.mean(preds, axis=0)
            result[f"arousal_{tag}"] = float(means[0])
            result[f"valence_{tag}"] = float(means[1])
        except Exception as exc:
            log.error("Arousal/valence %s failed: %s", tag, exc)
    return result


def extract_voice_instrumental(embeddings: np.ndarray) -> dict:
    """
    Voice/Instrumental binary ML classifier (VGGish).
    Returns probability that the track contains vocals (is NOT instrumental).
    Index 0 = instrumental, index 1 = vocal.
    """
    val = _predict_vggish_binary(embeddings, "voice_instrumental-audioset-vggish-1.pb")
    if val is not None:
        return {"voice_instrumental": val}
    return {}


def extract_voice_gender(embeddings: np.ndarray) -> dict:
    """
    Voice gender classifier (VGGish).
    Returns P(female) and P(male).
    Only meaningful when voice_instrumental > 0.5.
    """
    pb = _model_path("gender-audioset-vggish-1.pb")
    if pb is None or embeddings is None:
        return {}
    try:
        model = es.TensorflowPredict2D(
            graphFilename=pb,
            input="model/Placeholder",
            output="model/Softmax",
        )
        preds = model(embeddings)  # [frames, 2]: [female, male]
        means = np.mean(preds, axis=0)
        return {
            "voice_gender_female": float(means[0]),
            "voice_gender_male": float(means[1]),
        }
    except Exception as exc:
        log.error("Voice gender failed: %s", exc)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ML CLASSIFIERS  —  EffNet-based
# ══════════════════════════════════════════════════════════════════════════════

def _predict_effnet_binary(embeddings: np.ndarray, model_file: str, output="model/Softmax") -> float | None:
    """Binary EffNet classification head → P(positive class)."""
    pb = _model_path(model_file)
    if pb is None or embeddings is None:
        return None
    try:
        model = es.TensorflowPredict2D(
            graphFilename=pb,
            input="model/Placeholder",
            output=output,
        )
        preds = model(embeddings)  # [frames, 2]
        return float(np.mean(preds[:, 1]))
    except Exception as exc:
        log.error("%s failed: %s", model_file, exc)
        return None


def extract_mood_classifiers_effnet(embeddings: np.ndarray) -> dict:
    """
    EffNet-based mood / instrumentation classifiers:
      mood_acoustic    — acoustic vs electronic production
      mood_electronic  — electronic / synthetic timbre
      approachability  — how welcoming / easy-listening the track feels
      engagement       — how captivating / attention-holding
    """
    tasks = {
        "mood_acoustic": "mood_acoustic-discogs-effnet-1.pb",
        "mood_electronic": "mood_electronic-discogs-effnet-1.pb",
        "approachability": "approachability_2c-discogs-effnet-1.pb",
        "engagement": "engagement_2c-discogs-effnet-1.pb",
    }
    result = {}
    for col, model in tasks.items():
        val = _predict_effnet_binary(embeddings, model_file=model)
        if val is not None:
            result[col] = val
    return result


def extract_timbre(embeddings: np.ndarray) -> dict:
    """
    Timbre classifier (EffNet).
    Returns P(bright) and P(dark).
    Bright = high spectral centroid / presence of upper harmonics.
    Dark   = muffled / warm / bass-heavy timbre.
    """
    pb = _model_path("timbre-discogs-effnet-1.pb")
    if pb is None or embeddings is None:
        return {}
    try:
        model = es.TensorflowPredict2D(
            graphFilename=pb,
            input="model/Placeholder",
            output="model/Softmax",
        )
        preds = model(embeddings)  # [frames, 2]: [bright, dark]
        means = np.mean(preds, axis=0)
        return {
            "timbre_bright": float(means[0]),
            "timbre_dark": float(means[1]),
        }
    except Exception as exc:
        log.error("Timbre failed: %s", exc)
        return {}


def extract_tonal_atonal(embeddings: np.ndarray) -> dict:
    """
    Tonal / Atonal binary classifier (EffNet).
    Returns P(tonal) — high = clear key/melody, low = noise / free jazz /
    atonal contemporary classical.
    """
    val = _predict_effnet_binary(embeddings, "tonal_atonal-discogs-effnet-1.pb")
    if val is not None:
        return {"tonal_atonal": val}
    return {}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_song(wav_path: str) -> dict:
    """
    Full feature extraction for one audio file.
    Returns a flat dict ready to be upserted into `audio_essentia`.
    """
    audio_44k, audio_16k = load_audio(wav_path)

    if audio_44k is None and audio_16k is None:
        raise RuntimeError("Both audio loads failed")
    if audio_44k is not None and len(audio_44k) < 44100:
        raise RuntimeError("Audio too short (< 1 s @ 44 kHz)")

    row: dict = {}

    # ── Algorithmic features (44 kHz) ─────────────────────────────────────────
    if audio_44k is not None:
        row.update(extract_danceability(audio_44k))
        row.update(extract_rhythm(audio_44k))
        row.update(extract_key(audio_44k))
        row.update(extract_loudness(audio_44k))
        row.update(extract_dynamic_complexity(audio_44k))
        row.update(extract_intensity(audio_44k))
        row.update(extract_tonal_features(audio_44k))
        row.update(extract_pitch_salience(audio_44k))
        row.update(extract_silence_ratio(audio_44k))
        row.update(extract_chords(wav_path))
        row.update(extract_mfcc_gfcc(audio_44k))

    # ── ML features (16 kHz) ──────────────────────────────────────────────────
    if audio_16k is not None:
        # Compute embeddings once, share across heads
        vggish_emb = compute_vggish_embeddings(audio_16k)
        effnet_emb = compute_effnet_embeddings(audio_16k)

        # VGGish-based classifiers
        row.update(extract_mood_classifiers_vggish(vggish_emb))
        row.update(extract_mirex_moods(vggish_emb))
        row.update(extract_arousal_valence(vggish_emb))
        row.update(extract_voice_instrumental(vggish_emb))
        row.update(extract_voice_gender(vggish_emb))

        # EffNet-based classifiers
        row.update(extract_mood_classifiers_effnet(effnet_emb))
        row.update(extract_timbre(effnet_emb))
        row.update(extract_tonal_atonal(effnet_emb))

    return row


def run():
    if not ESSENTIA_AVAILABLE:
        print("\n Essentia not installed. Skipping this collector.")
        print("   Install with: pip install essentia-tensorflow")
        return

    ensure_models_exist()

    conn = get_conn()
    songs_needing = conn.execute("""
        SELECT id, title, artist, year, audio_path
        FROM songs
        WHERE status_essentia IS NULL
          AND audio_path IS NOT NULL
        ORDER BY year, chart_rank
    """).fetchall()

    log.info("%d songs ready for Essentia extraction", len(songs_needing))
    success = failed = 0

    for song in tqdm(songs_needing, desc="Essentia"):
        sid = song["id"]
        wav_path = song["audio_path"]

        if not wav_path:
            set_status(conn, sid, "status_essentia", 0)
            conn.commit()
            failed += 1
            continue

        try:
            row = process_song(wav_path)
            row["song_id"] = sid
            upsert(conn, "audio_essentia", row)
            set_status(conn, sid, "status_essentia", 1)
            conn.commit()
            success += 1
        except Exception as exc:
            log.warning("Essentia failed (id=%d): %s", sid, exc)
            set_status(conn, sid, "status_essentia", 0)
            conn.commit()
            failed += 1

    conn.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  ESSENTIA EXTRACTION COMPLETE")
    print("═" * 60)
    print(f"  Success : {success}")
    print(f"  Failed  : {failed}")

    conn2 = get_conn()
    dance_stats = conn2.execute("""
        SELECT s.period,
               AVG(e.danceability_normalised) AS avg_dance,
               AVG(e.mood_happy)              AS avg_happy,
               AVG(e.mood_party)              AS avg_party,
               AVG(e.arousal_deam)            AS avg_arousal,
               AVG(e.valence_deam)            AS avg_valence,
               AVG(e.voice_instrumental)      AS avg_vocal,
               AVG(e.approachability)         AS avg_approach
        FROM audio_essentia e
        JOIN songs s ON e.song_id = s.id
        WHERE e.danceability_normalised IS NOT NULL
        GROUP BY s.period
    """).fetchall()

    if dance_stats:
        print("\n  Per-period averages:")
        header = f"  {'Period':15}  {'Dance':>6}  {'Happy':>6}  {'Party':>6}"
        header += f"  {'Arousal':>7}  {'Valence':>7}  {'Vocal%':>6}  {'Approch':>7}"
        print(header)
        for r in dance_stats:
            def f(v): return f"{v:.3f}" if v is not None else "  n/a"

            print(
                f"  {r['period']:15}  {f(r['avg_dance']):>6}  {f(r['avg_happy']):>6}"
                f"  {f(r['avg_party']):>6}  {f(r['avg_arousal']):>7}  {f(r['avg_valence']):>7}"
                f"  {f(r['avg_vocal']):>6}  {f(r['avg_approach']):>7}"
            )
    conn2.close()
    print("═" * 60)


if __name__ == "__main__":
    run()
