# quiz.py

import numpy as np
import joblib
import os


# Load trained model
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "BACKEND", "models", "quiz_model.pkl")

model = joblib.load(MODEL_PATH)

labels = ["Low", "Medium", "High"]

# -------------------------------
# QUESTIONS
# -------------------------------

questions = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling/staying asleep or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself or that you are a failure?",
    "Trouble concentrating on things?",
    "Moving/speaking slowly OR being restless?",
    "Thoughts that you would be better off dead?"
]

# -------------------------------
# INPUT FUNCTION
# -------------------------------

def ask_questions():
    answers = []

    print("\n🧠 Depression Assessment Quiz\n")
    print("0 = Not at all | 1 = Several days | 2 = More than half | 3 = Nearly every day\n")

    for i, q in enumerate(questions):
        while True:
            print(f"{i+1}. {q}")
            ans = input("Your answer (0-3): ")

            if ans in ["0", "1", "2", "3"]:
                answers.append(int(ans))
                break
            else:
                print("❌ Invalid input. Enter 0–3\n")

    return answers


# -------------------------------
# PREDICTION
# -------------------------------

def predict_result(answers):
    X = np.array(answers).reshape(1, -1)

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]

    return {
        "prediction": labels[pred],
        "confidence": round(max(probs) * 100, 2)
    }


# -------------------------------
# RUN
# -------------------------------

if __name__ == "__main__":
    answers = ask_questions()
    result = predict_result(answers)

    print("\n--- RESULT ---")
    print("Risk Level:", result["prediction"])
    print("Confidence:", result["confidence"], "%")