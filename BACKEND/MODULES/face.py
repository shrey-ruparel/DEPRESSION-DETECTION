import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# =========================
# LOAD MODEL
# =========================
model = load_model(r"C:\Users\rutuja\Desktop\dep_detection\DEPRESSION-DETECTION\BACKEND\MODELS\emotion_model_best.h5")

# =========================
# FACE DETECTOR
# =========================
detector = MTCNN()

# =========================
# CONFIG
# =========================
IMG_SIZE = 160

emotions = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

# smoothing
prediction_history = []

# =========================
# EMOTION SCORE (optional)
# =========================
def emotion_to_score(emotion):
    mapping = {
        "Happy": 0,
        "Neutral": 1,
        "Surprise": 1,
        "Sad": 2,
        "Disgust": 2,
        "Angry": 3,
        "Fear": 3
    }
    return mapping.get(emotion, 1)

# =========================
# DETECT FUNCTION
# =========================
def detect_emotion(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    results = []

    for face in faces:
        x, y, w, h = face['box']

        # fix negative coords
        x, y = max(0, x), max(0, y)

        face_img = rgb[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        # preprocess
        face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
        face_img = face_img / 255.0
        face_img = np.reshape(face_img, (1, IMG_SIZE, IMG_SIZE, 3))

        # prediction
        pred = model.predict(face_img, verbose=0)

        # 🔥 smoothing
        prediction_history.append(pred)
        if len(prediction_history) > 10:
            prediction_history.pop(0)

        avg_pred = np.mean(prediction_history, axis=0)

        emotion = emotions[np.argmax(avg_pred)]
        confidence = float(np.max(avg_pred))
        score = emotion_to_score(emotion)

        results.append({
            "box": [x, y, w, h],
            "emotion": emotion,
            "confidence": confidence,
            "score": score
        })

    return results

# =========================
# WEBCAM RUN
# =========================
def run_emotion_detection():
    cap = cv2.VideoCapture(0)

    print("Press ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_emotion(frame)

        for res in results:
            x, y, w, h = res["box"]
            emotion = res["emotion"]
            conf = res["confidence"]

            text = f"{emotion} ({conf:.2f})"

            cv2.putText(frame, text, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# RUN
# =========================
if __name__ == "__main__":
    run_emotion_detection()