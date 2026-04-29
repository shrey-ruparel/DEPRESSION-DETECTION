"""
Fusion module to combine face and voice depression scores.
Weights match BACKEND/MODULES/fusion.py
"""

FACE_WEIGHT  = 0.35
VOICE_WEIGHT = 0.65

def get_depression_level(fused_score: float) -> tuple:
    """Return (level_str, action_str) from a 0-1 fused depression score."""
    if fused_score < 0.30:
        return ("Normal",   "No immediate concern. Continue monitoring.")
    elif fused_score < 0.55:
        return ("Moderate", "Consider speaking to a counselor or trusted person.")
    else:
        return ("Severe",   "Professional consultation strongly recommended. Please reach out for help.")

def get_fused_score(face_depression_score: float, voice_normalized_score: float, has_face: bool = True, has_voice: bool = True):
    """
    Combines the face depression score and voice normalized score using predefined weights.
    Returns the fused score (0 to 1), level, and recommended action.
    """
    face_w  = FACE_WEIGHT
    voice_w = VOICE_WEIGHT

    if has_face and not has_voice:
        face_w, voice_w = 1.0, 0.0
    elif has_voice and not has_face:
        face_w, voice_w = 0.0, 1.0

    fused = (face_depression_score * face_w) + (voice_normalized_score * voice_w)
    fused = round(min(max(fused, 0.0), 1.0), 4)

    level, action = get_depression_level(fused)

    return {
        "fused_score": fused,
        "fused_level": level,
        "recommended_action": action,
        "face_weight_used": face_w,
        "voice_weight_used": voice_w
    }
