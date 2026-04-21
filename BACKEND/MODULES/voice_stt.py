import whisper
import pyaudio
import numpy as np
import struct

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
SAMPLE_RATE   = 16000   # Whisper expects 16kHz audio
CHUNK_SIZE    = 512     # PyAudio internal buffer
RECORD_SECS   = 180      # Default recording time per question

# Whisper model load — pehli baar internet se download hoga (~74MB)
# Baad mein cache se load hoga, fast hoga
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model ready.\n")


# ---------------------------------------------------------------
# STEP 1 — RECORD AUDIO (fixed seconds ya Enter se stop)
# ---------------------------------------------------------------
import threading

def record_audio(max_seconds=RECORD_SECS):
    """
    Audio record karta hai.
    Ya toh max_seconds complete ho jaayein,
    ya user Enter dabaye — jo pehle ho.

    Returns: float32 numpy array (16kHz mono)
    """
    stop_flag = threading.Event()

    def listen_for_enter():
        input()
        stop_flag.set()

    stop_thread = threading.Thread(target=listen_for_enter, daemon=True)
    stop_thread.start()

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )

    print("  [Recording... Press Enter to stop]", flush=True)

    collected      = []
    samples_so_far = 0
    max_samples    = SAMPLE_RATE * max_seconds

    while not stop_flag.is_set() and samples_so_far < max_samples:
        try:
            raw = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except OSError:
            continue

        # int16 bytes → float32
        num = len(raw) // 2
        samples = struct.unpack(f'{num}h', raw)
        arr = np.array(samples, dtype=np.float32) / 32768.0
        collected.append(arr)
        samples_so_far += num

    stream.stop_stream()
    stream.close()
    p.terminate()

    if not collected:
        return None

    return np.concatenate(collected)


# ---------------------------------------------------------------
# STEP 2 — TRANSCRIBE AUDIO WITH WHISPER
# ---------------------------------------------------------------
def transcribe(audio_array, language='En'):
    """
    Whisper se audio array ko text mein convert karta hai.

    language param:
      "hi"   → Hindi
      "en"   → English
      None   → Whisper auto-detect karega (Hinglish ke liye best)

    Returns: transcript string (lowercase, stripped)
    """
    if audio_array is None or len(audio_array) == 0:
        return ""

    audio_float32 = audio_array.astype(np.float32)
    result = model.transcribe(audio_float32, language=language, fp16=False)

    transcript = result["text"].strip().lower()
    return transcript


# ---------------------------------------------------------------
# STEP 3 — RECORD + TRANSCRIBE (combined function)
# ---------------------------------------------------------------
def record_and_transcribe(question_text, max_seconds=RECORD_SECS, language='hi'):
    """
    Ek question display karta hai, user ka jawab record karta hai,
    aur transcript return karta hai.

    Args:
      question_text : screen pe dikhane wala question
      max_seconds   : maximum recording time
      language      : "hi", "en", ya None (auto-detect)

    Returns:
      {
        "question"   : question string,
        "transcript" : what the person said,
        "audio"      : raw numpy array (for voice features if needed)
      }
    """
    print("\n" + "=" * 60)
    print(f"QUESTION: {question_text}")
    print("=" * 60)

    audio = record_audio(max_seconds=max_seconds)

    if audio is None:
        print("  [No audio captured]")
        return {"question": question_text, "transcript": "", "audio": None}

    print("  [Transcribing...]", end="", flush=True)
    transcript = transcribe(audio, language=language)
    print(f" Done.\n")
    print(f"  You said: \"{transcript}\"")

    return {
        "question":   question_text,
        "transcript": transcript,
        "audio":      audio          # voice features ke liye bhi rakh sakte ho
    }


# ---------------------------------------------------------------
# STEP 4 — RUN FULL QUESTIONNAIRE
# ---------------------------------------------------------------

# PHQ-9 inspired questions — clinically relevant
# Har question ek specific depression indicator se map hota hai
QUESTIONS = [
    {
        "id": "q1",
        "text": "Please describe how you have been feeling over the past two weeks.",
        "indicator": "general_mood"
    },
    {
        "id": "q2",
        "text": "Are there activities or hobbies that you used to enjoy? Do you still find interest in them?",
        "indicator": "anhedonia"
    },
    {
        "id": "q3",
        "text": "How has your sleep been lately — do you sleep too much, too little, or wake up often?",
        "indicator": "sleep_disturbance"
    },
    {
        "id": "q4",
        "text": "How do you feel about yourself and your future right now?",
        "indicator": "hopelessness_self_worth"
    },
    {
        "id": "q5",
        "text": "Have you been feeling tired or low on energy even without much physical activity?",
        "indicator": "fatigue"
    },
]


def run_questionnaire():
    """
    Saare questions ke liye record + transcribe karta hai.
    Ek list return karta hai — har entry mein question, transcript, audio.
    Yeh output directly NLP layer ko jaayega (next module).
    """
    print("=" * 60)
    print("DEPRESSION SCREENING — Voice Questionnaire")
    print("Answer each question by speaking.")
    print("Press Enter after each answer to move to the next.")
    print("=" * 60)

    responses = []

    for i, q in enumerate(QUESTIONS):
        print(f"\n[Question {i+1} of {len(QUESTIONS)}]")
        result = record_and_transcribe(
            question_text=q["text"],
            max_seconds=30,      # 30 seconds max per question
            language=None        # auto-detect Hindi/English
        )
        result["indicator"] = q["indicator"]
        result["q_id"]      = q["id"]
        responses.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("TRANSCRIPTION COMPLETE — All responses captured")
    print("=" * 60)
    for r in responses:
        print(f"\n[{r['q_id']}] {r['question']}")
        print(f"  → \"{r['transcript']}\"")

    return responses


# ---------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------
if __name__ == "__main__":
    responses = run_questionnaire()

    # responses ki structure kuch aisi hogi:
    # [
    #   {
    #     "q_id"      : "q1",
    #     "indicator" : "general_mood",
    #     "question"  : "Please describe how...",
    #     "transcript": "main theek nahi hoon sab kuch bahut mushkil lag raha hai",
    #     "audio"     : np.array([...])   # voice features ke liye
    #   },
    #   ...
    # ]

    # voice_NLP.py ke bottom mein
