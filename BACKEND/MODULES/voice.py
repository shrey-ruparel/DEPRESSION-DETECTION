import librosa
import numpy as np
import pyaudio
import struct
import threading

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
SAMPLE_RATE       = 16000   # 16000 samples per second (speech standard)
CHUNK_DURATION    = 2       # seconds per analysis chunk
CHUNK_SIZE        = 512     # PyAudio internal read buffer (frames per read)
BASELINE_SECS     = 3       # seconds of baseline calibration recording
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION   # = 32000 samples
BYTES_PER_CHUNK   = SAMPLES_PER_CHUNK * 2          # 2 bytes per int16 sample


# ---------------------------------------------------------------
# STEP 1 — AUDIO CONVERSION UTILITY
# ---------------------------------------------------------------
def raw_bytes_to_float32(raw_bytes):
    """
    PyAudio se aane wale raw bytes (int16) ko
    float32 numpy array mein convert karta hai [-1.0 to 1.0].
    Returns None if byte count is invalid.
    """
    if len(raw_bytes) % 2 != 0:
        return None
    num_samples = len(raw_bytes) // 2
    samples = struct.unpack(f'{num_samples}h', raw_bytes)
    arr = np.array(samples, dtype=np.float32)
    arr /= 32768.0
    return arr


# ---------------------------------------------------------------
# STEP 2 — FEATURE EXTRACTION
# ---------------------------------------------------------------
def extract_features(audio_chunk, sr=SAMPLE_RATE):
    """
    Ek 2-second audio chunk se features nikalata hai.

    Features:
      - rms            : loudness (Root Mean Square energy)
      - pitch_mean     : average fundamental frequency in Hz
                         NOTE: Yeh Hz mein hai, MEL NAHI.
                         Mel ek alag perceptual scale hai jo human hearing
                         ki non-linearity capture karta hai.
                         Pitch mean = actual vocal cord vibration frequency.
      - pitch_variance : variation in pitch — LOW = monotone = depression indicator
      - mfcc_means     : 13 Mel-Frequency Cepstral Coefficients
                         (voice ki texture/timbre capture karta hai, pitch nahi)

    Returns None if chunk is too short to analyze.
    """
    if len(audio_chunk) < 2048:
        return None

    # --- RMS Energy ---
    rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

    # --- Pitch (F0) via YIN algorithm ---
    # fmin=50 Hz, fmax=300 Hz — normal human speech range covers this
    f0 = librosa.yin(audio_chunk, fmin=50, fmax=300, sr=sr)
    f0_voiced = f0[f0 > 0]   # unvoiced/silent frames hata do

    pitch_mean     = float(np.mean(f0_voiced))  if len(f0_voiced) > 0 else 0.0
    pitch_variance = float(np.var(f0_voiced))   if len(f0_voiced) > 0 else 0.0

    # --- MFCCs (13 coefficients) ---
    mfccs      = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)   # shape: (13,)

    return {
        'rms':            rms,
        'pitch_mean':     pitch_mean,
        'pitch_variance': pitch_variance,
        'mfccs':          mfcc_means,
    }


# ---------------------------------------------------------------
# STEP 3 — CHUNK SCORING
# ---------------------------------------------------------------
def score_chunk(features, baseline_rms=None):
    """
    Ek chunk ko 0-3 score deta hai.

    Scoring logic:
      RMS (energy):
        - baseline ke 75%+ → 0  (normal)
        - baseline ke 40-75% → 1  (moderate)
        - baseline ke 40% se kam → 2  (low energy)

      Pitch variance (monotone check):
        - variance < 5  → +1  (very flat, monotone)
        - variance < 15 → +0  (normal variation)

    Agar baseline nahi hai toh absolute thresholds use hote hain
    (less reliable — mic distance se RMS change ho jaata hai).
    """
    if features is None:
        return 0

    score = 0

    # --- RMS scoring ---
    rms = features['rms']
    if baseline_rms and baseline_rms > 0:
        ratio = rms / baseline_rms
        if   ratio > 0.75:  score += 0
        elif ratio > 0.40:  score += 1
        else:               score += 2
    else:
        if   rms > 0.05:  score += 0
        elif rms > 0.02:  score += 1
        else:             score += 2

    # --- Pitch variance scoring ---
    pv = features['pitch_variance']
    if pv < 5:    score += 1
    elif pv < 15: score += 0

    return score


# ---------------------------------------------------------------
# STEP 4 — FINAL ASSESSMENT
# ---------------------------------------------------------------
def final_assessment(chunk_scores):
    """
    Saare chunks ke scores ka average lekar final label deta hai.

      avg < 0.7  → 0 = Normal
      avg < 1.4  → 1 = Moderate depressive tone
      avg >= 1.4 → 2 = High depressive tone
    """
    if not chunk_scores:
        return None, "No data collected"

    avg = float(np.mean(chunk_scores))

    if   avg < 0.7:  return 0, "Normal"
    elif avg < 1.4:  return 1, "Moderate depressive tone"
    else:            return 2, "High depressive tone"


# ---------------------------------------------------------------
# STEP 5 — BASELINE RECORDING (fixed 3 seconds)
# ---------------------------------------------------------------
def record_baseline(stream):
    """
    3 seconds ki normal speech record karta hai.
    Iska RMS baad mein relative thresholds ke liye use hoga.
    Returns float32 numpy array.
    """
    total_needed   = SAMPLE_RATE * BASELINE_SECS
    collected      = []
    samples_so_far = 0

    print("Recording", end="", flush=True)
    while samples_so_far < total_needed:
        raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        arr = raw_bytes_to_float32(raw)
        if arr is not None:
            collected.append(arr)
            samples_so_far += len(arr)
        print(".", end="", flush=True)
    print()

    full = np.concatenate(collected)
    return full[:total_needed]


# ---------------------------------------------------------------
# STEP 6 — STOP LISTENER (background thread)
# ---------------------------------------------------------------
def listen_for_stop(stop_flag):
    """
    Yeh function ek alag thread mein chalta hai.
    Sirf Enter press hone ka wait karta hai.
    Press hone pe stop_flag.set() call karta hai — yeh signal hai
    main recording loop ko band hone ka.

    Threading kyun zaruri hai:
      stream.read() ek blocking call hai — jab tak wo frame
      nahi padhta, program wahan ruk jaata hai.
      Toh main thread keyboard nahi sun sakta.
      Isliye ek alag thread background mein keyboard sunta hai.
    """
    input()
    stop_flag.set()
    print("\n[Stop signal received — finishing current chunk...]\n")


# ---------------------------------------------------------------
# STEP 7 — MAIN RECORDING LOOP (Enter se stop)
# ---------------------------------------------------------------
def record_until_stopped(stream, baseline_rms=None):
    """
    Continuously 2-second chunks record aur analyze karta hai.
    Jab user Enter dabaye, recording band ho jaati hai.

    Buffer logic:
      - Har PyAudio read sirf 512 frames deta hai (~0.032 sec)
      - Hum unhe buffer_raw mein jama karte rehte hain
      - Jab buffer mein >= 32000 samples (2 sec) aa jaayein
        tabhi feature extraction karte hain
      - Bacha hua data (leftover) next chunk ke liye rakhte hain
        (isse koi sample waste nahi hota)
    """
    stop_flag = threading.Event()

    stop_thread = threading.Thread(
        target=listen_for_stop,
        args=(stop_flag,),
        daemon=True   # main program band hone pe yeh bhi band ho jaata hai
    )
    stop_thread.start()

    print("Start speaking when ready ...")
    print(">>> Press Enter to Stop <<<\n")

    all_scores  = []
    chunk_index = 0
    buffer_raw  = b''

    while not stop_flag.is_set():
        try:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except OSError as e:
            print(f"  [Mic read error: {e}]")
            continue

        buffer_raw += raw

        if len(buffer_raw) >= BYTES_PER_CHUNK:
            chunk_bytes = buffer_raw[:BYTES_PER_CHUNK]
            buffer_raw  = buffer_raw[BYTES_PER_CHUNK:]

            chunk_index += 1
            audio_array = raw_bytes_to_float32(chunk_bytes)

            if audio_array is None:
                print(f"  Chunk {chunk_index:02d}: conversion failed, skipping.")
                continue

            features = extract_features(audio_array)
            if features is None:
                print(f"  Chunk {chunk_index:02d}: too short, skipping.")
                continue

            score = score_chunk(features, baseline_rms=baseline_rms)
            all_scores.append(score)

            print(
                f"  Chunk {chunk_index:02d} | "
                f"RMS: {features['rms']:.5f} | "
                f"Pitch mean: {features['pitch_mean']:6.1f} Hz | "
                f"Pitch var: {features['pitch_variance']:7.2f} | "
                f"Score: {score}"
            )

    return all_scores


# ---------------------------------------------------------------
# STEP 8 — FULL PIPELINE
# ---------------------------------------------------------------
def run_voice_analysis():
    """
    Complete pipeline ek jagah:
      Phase 1 → Baseline calibration (3s fixed)
      Phase 2 → Main recording (user Enter se stop kare)
      Phase 3 → Final assessment + structured JSON-style output
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    # ---- PHASE 1: Baseline ----
    print("=" * 60)
    print("PHASE 1 — Baseline Calibration")
    print(f"Speak normally for  ({BASELINE_SECS} seconds).")
    print("This Captures your natural voice reference.")
    print("=" * 60)

    baseline_audio    = record_baseline(stream)
    baseline_features = extract_features(baseline_audio)

    if baseline_features:
        baseline_rms = baseline_features['rms']
        print(f"Baseline RMS captured : {baseline_rms:.5f}\n")
    else:
        baseline_rms = None
        print("Baseline capture failed- absolute thresholds.\n")

    # ---- PHASE 2: Main recording ----
    print("=" * 60)
    print("PHASE 2 — Main Recording")
    print("Speak freely for as long as you want.")
    print("=" * 60)

    all_scores = record_until_stopped(stream, baseline_rms=baseline_rms)

    # ---- Cleanup ----
    stream.stop_stream()
    stream.close()
    p.terminate()

    # ---- PHASE 3: Result ----
    print("=" * 60)
    print("RESULT")
    print("=" * 60)

    if not all_scores:
        print("No chunks were analyzed. Check your microphone.")
        return None

    final_score, label = final_assessment(all_scores)
    avg_score          = float(np.mean(all_scores))

    print(f"Total chunks analyzed : {len(all_scores)}")
    print(f"Chunk scores          : {all_scores}")
    print(f"Average score         : {avg_score:.3f}")
    print(f"Final assessment      : {final_score} → {label}")
    print("=" * 60)

    # Structured output — baaki team members ke saath integrate karne ke liye
    result = {
        "voice_score":  final_score,
        "label":        label,
        "score_avg":    avg_score,
        "chunk_scores": all_scores,
        "chunks_count": len(all_scores),
        "baseline_rms": baseline_rms,
    }

    return result


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    result = run_voice_analysis()