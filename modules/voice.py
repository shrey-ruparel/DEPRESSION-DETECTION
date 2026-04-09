import librosa
import numpy as np

def extract_features(audio_chunk, sr=16000):
    # RMS Energy
    rms = np.sqrt(np.mean(audio_chunk**2))
    
    # Pitch (F0) using librosa's YIN algorithm
    f0 = librosa.yin(audio_chunk, fmin=50, fmax=300)
    f0_voiced = f0[f0 > 0]  # remove unvoiced frames
    
    pitch_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
    pitch_variance = np.var(f0_voiced) if len(f0_voiced) > 0 else 0
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)  # 13 values
    
    return {
        'rms': rms,
        'pitch_mean': pitch_mean,
        'pitch_variance': pitch_variance,
        'mfccs': mfcc_means
    }


def score_chunk(features):
    score = 0
    
    # RMS thresholds (you need to tune these)
    if features['rms'] > 0.05:
        score += 0   # Normal energy
    elif features['rms'] > 0.02:
        score += 1   # Moderate
    else:
        score += 2   # Low energy
    
    # Pitch variance (flat pitch = depressive indicator)
    if features['pitch_variance'] < 10:
        score += 1   # Monotone
    
    return score


def final_assessment(chunk_scores):
    avg = np.mean(chunk_scores)
    if avg < 0.7:
        return 0, "Normal"
    elif avg < 1.4:
        return 1, "Moderate"
    else:
        return 2, "High Depressive Tone"
    
import pyaudio
import struct

# ---- Settings ----
SAMPLE_RATE = 16000       # 16000 samples per second (standard for speech)
CHUNK_DURATION = 2        # seconds per chunk
CHUNK_SIZE = 512          # how many frames PyAudio reads at once (internal buffer)
RECORD_SECONDS = 10       # total recording time
SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_DURATION  # = 32000 samples per chunk

def record_and_analyze():
    # Step 1: Open the microphone
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,    # 16-bit audio (standard)
        channels=1,                 # mono (1 microphone)
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("Recording... speak now!")

    all_scores = []
    buffer = []         # accumulates raw frames until we have 2 seconds worth
    samples_collected = 0
    total_samples_needed = SAMPLE_RATE * RECORD_SECONDS

    # Step 2: Keep reading from mic until total time is up
    while samples_collected < total_samples_needed:
        raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        buffer.append(raw_data)
        samples_collected += CHUNK_SIZE

        # Step 3: Once buffer has 2 seconds of audio, process it
        if samples_collected % SAMPLES_PER_CHUNK < CHUNK_SIZE:
            # Combine all raw bytes in buffer into one block
            combined = b''.join(buffer)
            buffer = []  # reset buffer for next chunk

            # Step 4: Convert raw bytes → numpy float array
            # paInt16 means each sample is a 16-bit integer
            num_samples = len(combined) // 2  # 2 bytes per sample
            audio_array = np.array(struct.unpack(f'{num_samples}h', combined), dtype=np.float32)
            audio_array = audio_array / 32768.0  # normalize to range [-1.0, 1.0]

            # Step 5: Feed into YOUR functions
            features = extract_features(audio_array, sr=SAMPLE_RATE)
            score = score_chunk(features)
            all_scores.append(score)

            print(f"Chunk processed — RMS: {features['rms']:.4f} | Pitch Var: {features['pitch_variance']:.2f} | Score: {score}")

    # Step 6: Cleanup
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Step 7: Final result
    final_score, label = final_assessment(all_scores)
    print(f"\n--- RESULT ---")
    print(f"All chunk scores: {all_scores}")
    print(f"Final Score: {final_score} → {label}")

# Run it
record_and_analyze()