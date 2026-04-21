from transformers import pipeline
import numpy as np

# ---------------------------------------------------------------
# STEP 1 — MODEL LOAD
# ---------------------------------------------------------------
# j-hartmann/emotion-english-distilroberta-base
# 7 emotions detect karta hai:
#   joy, sadness, anger, fear, disgust, surprise, neutral
# Pehli baar ~250MB download hoga, baad mein cache

print("Loading emotion model...")
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None   # saari emotions ka score chahiye, sirf top 1 nahi
)
print("Emotion model ready.\n")


# ---------------------------------------------------------------
# STEP 2 — CLINICAL WEIGHTS
# ---------------------------------------------------------------
# Har question ka indicator aur uska weight
# Anhedonia aur hopelessness ko 2x weight —
# yeh PHQ-9 mein bhi sabse strong depression predictors hain

INDICATOR_WEIGHTS = {
    "general_mood":        1.5,
    "anhedonia":           2.0,   # hobbies/interest khatam hona
    "sleep_disturbance":   1.5,
    "hopelessness_self_worth": 2.0,   # future + self image
    "fatigue":             1.0,
}

# MAX possible score (agar har cheez 1.0 depression score de)
MAX_SCORE = sum(INDICATOR_WEIGHTS.values())   # = 8.0

# Yeh emotions depression indicate karti hain
DEPRESSIVE_EMOTIONS = ["sadness", "fear", "disgust"]

# Yeh specific phrases strong clinical signals hain —
# emotion model inhe miss kar sakta hai (short/neutral lage)
# toh separately check karte hain
HIGH_RISK_PHRASES = [
    "no point", "what's the point", "give up", "can't go on",
    "don't want to", "no interest", "nothing matters",
    "hate myself", "burden", "worthless", "hopeless",
    "want to die", "end it", "not worth", "nobody cares",
    "always tired", "exhausted all the time", "can't sleep",
    "stopped enjoying", "don't enjoy", "not doing anything anymore",
    "just sitting", "wasting time", "no motivation",
]


# ---------------------------------------------------------------
# STEP 3 — SINGLE TRANSCRIPT ANALYZE
# ---------------------------------------------------------------
def analyze_transcript(transcript, indicator):
    """
    Ek transcript ko analyze karta hai aur weighted depression score deta hai.

    Process:
      1. Emotion model se 7 emotions ka score nikalo
      2. Depressive emotions (sadness + fear + disgust) sum karo = base_score
      3. High-risk phrases check karo — milne pe +0.2 bonus
      4. Indicator weight se multiply karo = weighted_score

    Returns dict with all intermediate values (debugging ke liye useful)
    """
    if not transcript or transcript.strip() == "":
        return {
            "indicator":      indicator,
            "transcript":     transcript,
            "emotions":       {},
            "base_score":     0.0,
            "phrase_bonus":   0.0,
            "weighted_score": 0.0,
            "weight_used":    INDICATOR_WEIGHTS.get(indicator, 1.0),
        }

    # --- Emotion Analysis ---
    raw_emotions = emotion_classifier(transcript)[0]
    # raw_emotions format:
    # [{'label': 'sadness', 'score': 0.82}, {'label': 'joy', 'score': 0.05}, ...]

    emotions_dict = {item['label']: round(item['score'], 4) for item in raw_emotions}

    # Sum of depressive emotion scores
    base_score = sum(
        emotions_dict.get(emotion, 0.0)
        for emotion in DEPRESSIVE_EMOTIONS
    )
    base_score = min(base_score, 1.0)   # cap at 1.0

    # --- High Risk Phrase Check ---
    transcript_lower = transcript.lower()
    phrase_hits = [p for p in HIGH_RISK_PHRASES if p in transcript_lower]
    phrase_bonus = min(len(phrase_hits) * 0.2, 0.4)   # max +0.4 bonus

    # Final score for this response
    combined_score = min(base_score + phrase_bonus, 1.0)
    weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
    weighted_score = combined_score * weight

    return {
        "indicator":      indicator,
        "transcript":     transcript,
        "emotions":       emotions_dict,
        "base_score":     round(base_score, 4),
        "phrase_bonus":   round(phrase_bonus, 4),
        "phrase_hits":    phrase_hits,
        "weighted_score": round(weighted_score, 4),
        "weight_used":    weight,
    }


# ---------------------------------------------------------------
# STEP 4 — FINAL DEPRESSION ASSESSMENT
# ---------------------------------------------------------------
def assess_depression(responses):
    """
    Saare question responses ko analyze karke final depression level deta hai.

    Args:
      responses: list of dicts from whisper_transcription.py
                 Each dict has: q_id, indicator, question, transcript

    Returns:
      final_result dict with:
        - per_question analysis
        - total_weighted_score
        - normalized_score (0.0 to 1.0)
        - depression_level: "Normal" / "Moderate" / "Severe"
        - recommended_action
    """
    print("=" * 60)
    print("NLP ANALYSIS — Processing responses...")
    print("=" * 60)

    per_question = []
    total_weighted = 0.0

    for r in responses:
        transcript = r.get("transcript", "")
        indicator  = r.get("indicator", "general_mood")
        q_id       = r.get("q_id", "")
        question   = r.get("question", "")

        analysis = analyze_transcript(transcript, indicator)
        analysis["q_id"]     = q_id
        analysis["question"] = question
        per_question.append(analysis)

        total_weighted += analysis["weighted_score"]

        # Per-question print
        print(f"\n[{q_id}] {question[:55]}...")
        print(f"  Transcript : \"{transcript[:80]}...\"" if len(transcript) > 80 else f"  Transcript : \"{transcript}\"")
        print(f"  Top emotion: {max(analysis['emotions'], key=analysis['emotions'].get)} "
              f"({max(analysis['emotions'].values()):.2f})")
        print(f"  Base score : {analysis['base_score']:.3f}  |  "
              f"Phrase bonus: +{analysis['phrase_bonus']:.3f}  |  "
              f"Weight: x{analysis['weight_used']}  |  "
              f"Weighted: {analysis['weighted_score']:.3f}")
        if analysis['phrase_hits']:
            print(f"  Risk phrases found: {analysis['phrase_hits']}")

    # --- Normalize ---
    normalized = total_weighted / MAX_SCORE   # 0.0 to 1.0

    # --- Final Level ---
    if normalized < 0.30:
        level  = "Normal"
        action = "No immediate concern. Continue monitoring."
    elif normalized < 0.55:
        level  = "Moderate"
        action = "Consider speaking to a counselor or trusted person."
    else:
        level  = "Severe"
        action = "Professional consultation strongly recommended. Please reach out for help."

    # --- Print Result ---
    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Total weighted score : {total_weighted:.3f} / {MAX_SCORE:.1f}")
    print(f"Normalized score     : {normalized:.3f}  ({normalized*100:.1f}%)")
    print(f"Depression level     : {level}")
    print(f"Recommended action   : {action}")
    print("=" * 60)

    return {
        "per_question":         per_question,
        "total_weighted_score": round(total_weighted, 4),
        "max_possible_score":   MAX_SCORE,
        "normalized_score":     round(normalized, 4),
        "depression_level":     level,
        "recommended_action":   action,
    }


# ---------------------------------------------------------------
# STEP 5 — STANDALONE TEST (without microphone)
# Test karne ke liye tere actual transcripts use kar rahe hain
# ---------------------------------------------------------------
if __name__ == "__main__":
    from voice_stt import run_questionnaire

    # Step 1: Mic se responses lo (voice_stt.py chalega)
    responses = run_questionnaire()

    # Step 2: NLP analysis karo
    if responses:
        result = assess_depression(responses)
    else:
        print("No responses captured.")

