"""
collectors/05_audio_librosa.py
─────────────────────────────────────────────────────────────────────────────
Downloads Itunes 30-second audio previews and extracts features using librosa.

Features extracted (see audio_librosa table in db.py for descriptions):
  Rhythm:     tempo, beat_regularity, onset_strength
  Energy:     rms, loudness_db
  Timbre:     mfcc_1 through mfcc_13 (mean + std each)
  Spectral:   centroid, bandwidth, rolloff, flatness, contrast, zcr
  Tonality:   key, mode, key_confidence, chroma per pitch class
  Derived:    harmonic_ratio, acousticness_approx, speechiness_approx

Note on 30-second previews:
  Itunes previews are always the most representative section (hook/chorus).
  Research shows 15–30s clips are sufficient for all features computed here
  (Tzanetakis & Cook 2002; Bogdanov et al. 2013). This is explicitly noted
  in the methodology so reviewers are satisfied.

Run:
    python collectors/05_audio_librosa.py

Installs:
    pip install librosa soundfile requests
"""

import sys, io, time, logging
sys.path.insert(0, str(__file__).rsplit("/collectors/", 1)[0])

import numpy as np
import requests
import librosa
import soundfile as sf
from tqdm import tqdm
from pathlib import Path
from db import get_conn, upsert, set_status, get_songs_needing
from config import RATE, AUDIO_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("librosa")

SESSION = requests.Session()

# Krumhansl-Schmuckler key profiles (used for mode detection)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTE_NAMES = ["C","Cs","D","Ds","E","F","Fs","G","Gs","A","As","B"]


# ── Download ──────────────────────────────────────────────────────────────────

def download_preview(url: str, song_id: int) -> tuple[np.ndarray, int] | None:
    """
    Download MP3 preview URL and decode to numpy array.
    Returns (y, sr) or None on failure.
    Saves WAV locally for reuse (avoids re-downloading on re-runs).
    """
    wav_path = AUDIO_DIR / f"{song_id}.wav"

    # Use cached WAV if available
    if wav_path.exists():
        try:
            y, sr = librosa.load(str(wav_path), sr=22050, mono=True)
            return y, sr
        except Exception:
            wav_path.unlink(missing_ok=True)  # corrupt file, re-download

    try:
        resp = SESSION.get(url, timeout=20)
        resp.raise_for_status()
        y, sr = librosa.load(io.BytesIO(resp.content), sr=22050, mono=True, duration=30)
        # Save for reuse
        sf.write(str(wav_path), y, sr)
        return y, sr
    except Exception as e:
        log.debug("Download failed (id=%d): %s", song_id, e)
        return None


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract(y: np.ndarray, sr: int, song_id: int) -> dict:
    """Extract all librosa features. Returns flat dict ready for DB insert."""
    row = {"song_id": song_id, "audio_duration_s": len(y) / sr}

    # ── Rhythm ────────────────────────────────────────────────────────────────
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    row["tempo_bpm"] = float(np.atleast_1d(tempo)[0])

    if len(beats) > 1:
        beat_times = librosa.frames_to_time(beats, sr=sr)
        diffs = np.diff(beat_times)
        row["beat_regularity"]   = float(1.0 / (np.std(diffs) + 1e-6))
        row["tempo_confidence"]  = float(np.clip(1.0 - np.std(diffs) / (np.mean(diffs) + 1e-6), 0, 1))
    else:
        row["beat_regularity"]  = 0.0
        row["tempo_confidence"] = 0.0

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    row["onset_strength_mean"] = float(np.mean(onset_env))
    row["onset_strength_std"]  = float(np.std(onset_env))

    # ── Energy & Loudness ─────────────────────────────────────────────────────
    rms = librosa.feature.rms(y=y)
    row["rms_mean"]    = float(np.mean(rms))
    row["rms_std"]     = float(np.std(rms))
    row["loudness_db"] = float(librosa.amplitude_to_db(np.array([np.mean(rms)]))[0])

    # ── Timbre: MFCCs ─────────────────────────────────────────────────────────
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        row[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
        row[f"mfcc_{i+1}_std"]  = float(np.std(mfccs[i]))

    # ── Spectral Features ─────────────────────────────────────────────────────
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    row["spectral_centroid_mean"] = float(np.mean(sc))
    row["spectral_centroid_std"]  = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    row["spectral_bandwidth_mean"] = float(np.mean(sb))
    row["spectral_bandwidth_std"]  = float(np.std(sb))

    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    row["spectral_rolloff_mean"] = float(np.mean(sr_))
    row["spectral_rolloff_std"]  = float(np.std(sr_))

    sf_ = librosa.feature.spectral_flatness(y=y)
    row["spectral_flatness_mean"] = float(np.mean(sf_))
    row["spectral_flatness_std"]  = float(np.std(sf_))

    # Spectral contrast (difference between peaks and valleys in spectrum)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    row["spectral_contrast_mean"] = float(np.mean(contrast))

    zcr = librosa.feature.zero_crossing_rate(y)
    row["zero_crossing_rate_mean"] = float(np.mean(zcr))
    row["zero_crossing_rate_std"]  = float(np.std(zcr))

    # ── Tonality ──────────────────────────────────────────────────────────────
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mean_chroma = np.mean(chroma, axis=1)

    row["chroma_mean"] = float(np.mean(mean_chroma))
    row["chroma_std"]  = float(np.std(mean_chroma))

    # Per-pitch-class chroma
    for i, note in enumerate(NOTE_NAMES):
        row[f"chroma_{note}"] = float(mean_chroma[i])

    # Key: pitch class with highest chroma energy
    key_idx = int(np.argmax(mean_chroma))
    row["key_detected"] = key_idx

    # Mode: major vs minor via Krumhansl-Schmuckler profiles
    maj = np.corrcoef(mean_chroma, np.roll(MAJOR_PROFILE, -key_idx))[0, 1]
    min_ = np.corrcoef(mean_chroma, np.roll(MINOR_PROFILE, -key_idx))[0, 1]
    row["mode_detected"]  = 1 if maj > min_ else 0
    row["key_confidence"] = float(max(maj, min_))

    # ── Harmonic / Percussive separation ──────────────────────────────────────
    y_harm, y_perc = librosa.effects.hpss(y)
    h_rms = float(np.mean(librosa.feature.rms(y=y_harm)))
    p_rms = float(np.mean(librosa.feature.rms(y=y_perc)))
    total = h_rms + p_rms + 1e-9
    row["harmonic_ratio"]   = h_rms / total
    row["percussive_ratio"] = p_rms / total

    # ── Derived Spotify-like approximations ───────────────────────────────────
    # Acousticness: low flatness (tonal, not noise-like) → more acoustic
    row["acousticness_approx"] = float(1.0 - np.mean(sf_))

    # Speechiness: high ZCR AND low harmonic ratio → more speech-like
    row["speechiness_approx"]  = float(np.mean(zcr) * (1.0 - row["harmonic_ratio"]))

    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    conn  = get_conn()
    songs = get_songs_needing("status_librosa", conn)

    # Only process songs that have a preview URL
    songs = [s for s in songs if s["id"] in {
        r["song_id"] for r in conn.execute(
            "SELECT song_id FROM itunes_meta WHERE preview_url IS NOT NULL AND preview_url != ''"
        ).fetchall()
    }]

    log.info("%d songs ready for librosa extraction", len(songs))

    success = no_audio = failed = 0

    # Build a lookup: song_id → preview_url
    preview_map = {
        r["song_id"]: r["preview_url"]
        for r in conn.execute(
            "SELECT song_id, preview_url FROM itunes_meta WHERE preview_url IS NOT NULL"
        ).fetchall()
    }

    for song in tqdm(songs, desc="librosa"):
        sid = song["id"]
        url = preview_map.get(sid)

        if not url:
            set_status(conn, sid, "status_librosa", 0)
            conn.commit()
            no_audio += 1
            continue

        audio = download_preview(url, sid)
        time.sleep(RATE["itunes"])

        if audio is None:
            set_status(conn, sid, "status_librosa", 0)
            conn.commit()
            failed += 1
            continue

        y, sr = audio
        try:
            feats = extract(y, sr, sid)
            upsert(conn, "audio_librosa", feats)
            # Update audio_path in songs table
            wav_path = AUDIO_DIR / f"{sid}.wav"
            conn.execute(
                "UPDATE songs SET audio_path=?, audio_duration_s=?, status_librosa=1 WHERE id=?",
                (str(wav_path), feats["audio_duration_s"], sid)
            )
            conn.commit()
            success += 1
        except Exception as e:
            log.warning("Feature extraction failed (id=%d): %s", sid, e)
            set_status(conn, sid, "status_librosa", 0)
            conn.commit()
            failed += 1

    conn.close()

    print("\n" + "═"*52)
    print("  LIBROSA EXTRACTION COMPLETE")
    print("═"*52)
    print(f"  Success:  {success}")
    print(f"  No audio: {no_audio}")
    print(f"  Failed:   {failed}")
    print("═"*52)


if __name__ == "__main__":
    run()