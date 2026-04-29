"""
Face Analysis Module wrapper
Decodes base64 frames from the frontend and uses DeepFace to analyze emotion.
"""

import cv2
import base64
import numpy as np
from deepface import DeepFace

EMOTION_SCORES = {
    "happy":    1.0,
    "surprise": 0.6,
    "neutral":  0.5,
    "fear":     0.2,
    "sad":      0.1,
    "angry":    0.0,
    "disgust":  0.0,
}

def decode_base64_image(b64_string):
    """Decodes a base64 string to a cv2 image."""
    try:
        # Remove the 'data:image/jpeg;base64,' prefix if present
        if ',' in b64_string:
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def analyze_frames(frames_data):
    """
    Analyzes a list of frame dictionaries containing base64 data.
    Expected format: [{'timestamp': 1234, 'data': 'base64str'}, ...]
    Returns a dictionary with face analysis results.
    """
    if not frames_data:
        return {
            "avg_face_score": 0.5,
            "face_depression_score": 0.5,
            "dominant_emotion_overall": "neutral",
            "emotion_distribution": {},
            "face_detected_ratio": 0.0,
            "snapshots_captured": 0
        }

    valid_frames = 0
    face_scores = []
    emotion_totals = {}

    for frame_obj in frames_data:
        b64_str = frame_obj.get("data")
        if not b64_str:
            continue
            
        img = decode_base64_image(b64_str)
        if img is None:
            continue

        try:
            results = DeepFace.analyze(
                img_path=img,
                actions=["emotion"],
                detector_backend="opencv",
                enforce_detection=True,
                silent=True,
            )
            analysis = results[0] if isinstance(results, list) else results
            raw_emotions = analysis["emotion"]
            dominant = analysis["dominant_emotion"]
            
            # Normalize emotions
            total = sum(raw_emotions.values()) or 1.0
            norm_emotions = {k: v / total for k, v in raw_emotions.items()}
            
            # Calculate positivity score for this frame
            frame_score = min(max(
                sum(EMOTION_SCORES.get(e, 0.5) * p for e, p in norm_emotions.items()),
                0.0), 1.0)
                
            face_scores.append(frame_score)
            valid_frames += 1
            
            # Accumulate emotion totals
            for emo, prob in norm_emotions.items():
                emotion_totals[emo] = emotion_totals.get(emo, 0.0) + prob
                
        except Exception as e:
            # Face not detected in this frame
            pass

    if valid_frames == 0:
        return {
            "avg_face_score": 0.5,
            "face_depression_score": 0.5, # 1.0 - 0.5
            "dominant_emotion_overall": "neutral",
            "emotion_distribution": {},
            "face_detected_ratio": 0.0,
            "snapshots_captured": 0
        }

    avg_score = round(float(sum(face_scores) / valid_frames), 4)
    face_depression_score = round(1.0 - avg_score, 4) # Flip positivity to depression
    
    emotion_dist = {str(k): round(float(v) / valid_frames, 4) for k, v in emotion_totals.items()}
    dominant_overall = str(max(emotion_dist, key=emotion_dist.get)) if emotion_dist else "neutral"
    face_detected_ratio = round(float(valid_frames) / len(frames_data), 4)

    return {
        "avg_face_score": float(avg_score),
        "face_depression_score": float(face_depression_score),
        "dominant_emotion_overall": dominant_overall,
        "emotion_distribution": emotion_dist,
        "face_detected_ratio": float(face_detected_ratio),
        "snapshots_captured": int(valid_frames)
    }
