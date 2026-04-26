from flask import Flask, request, jsonify, session, render_template, url_for, redirect, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import os

# Import modules
from modules.quiz import predict_result, questions as quiz_questions
from modules.voice import assess_depression
from db import init_db, create_user, get_user_by_email, save_result, get_results_by_email

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.secret_key = "super_secret_key_123"

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_PATH = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_PATH, exist_ok=True)

# Initialize the database on startup
init_db()


# ==============================
# REGISTER
# ==============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        if not email or not password:
            flash("Email and password are required.", "error")
            return redirect(url_for('register'))

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            create_user(email, hashed_password)
            flash("Account created! Please sign in.", "success")
            return redirect(url_for('home'))
        except:
            flash("An account with that email already exists.", "error")
            return redirect(url_for('register'))

    return render_template("register.html")

# ==============================
# HOME
# ==============================
@app.route('/')
def home():
    return render_template("index.html")

# ==============================
# LOGIN
# ==============================
@app.route('/login', methods=['POST'])
def login():
    email = request.form.get('email')
    password = request.form.get('password')

    user = get_user_by_email(email)

    if not user:
        flash("User does not exist", "error")
        return redirect("/")

    if not check_password_hash(user["password"], password):
        flash("Incorrect password", "error")
        return redirect("/")

    session['user'] = email
    return redirect("/main")

# ==============================
# MAIN
# ==============================
@app.route('/main')
def main():
    return render_template("main.html")

# ==============================
# ASSESSMENT
# ==============================
@app.route('/assessment', methods=['GET'])
def assessment():
    return render_template('assessment.html')

# ==============================
# RESULT
# ==============================
@app.route('/results_page', methods=['GET'])
def results_page():
    if 'user' not in session:
        return redirect(url_for('home'))

    results = get_results_by_email(session['user'])
    latest = results[0] if results else None

    return render_template("results.html", results=results, latest=latest)

# ==============================
# TEXT-QUIZ
# ==============================
@app.route('/text_quiz', methods=['GET', 'POST'])
def predict_quiz():
    if 'user' not in session:
        return redirect(url_for('home'))

    if request.method == 'GET':
        return render_template('text-quiz.html', questions=list(enumerate(quiz_questions, 1)))

    # POST — collect answers
    answers = []
    for i in range(1, 10):
        val = request.form.get(f'q{i}')
        if val is None:
            flash("Please answer all questions before submitting.", "error")
            return render_template('text-quiz.html', questions=list(enumerate(quiz_questions, 1)))
        answers.append(int(val))

    result = predict_result(answers)

    save_result(
        session['user'],
        result.get("prediction"),
        result.get("confidence", 0),
        "quiz"
    )

    return redirect(url_for('results_page'))

# ==============================
# VOICE ANALYSIS
# ==============================

# The 5 open-ended questions from voice_stt.py
VOICE_QUESTIONS = [
    {"id": "q1", "text": "Please describe how you have been feeling over the past two weeks.", "indicator": "general_mood"},
    {"id": "q2", "text": "Are there activities or hobbies that you used to enjoy? Do you still find interest in them?", "indicator": "anhedonia"},
    {"id": "q3", "text": "How has your sleep been lately — do you sleep too much, too little, or wake up often?", "indicator": "sleep_disturbance"},
    {"id": "q4", "text": "How do you feel about yourself and your future right now?", "indicator": "hopelessness_self_worth"},
    {"id": "q5", "text": "Have you been feeling tired or low on energy even without much physical activity?", "indicator": "fatigue"},
]

@app.route('/voice_analysis', methods=['GET'])
def voice_analysis_page():
    if 'user' not in session:
        return redirect(url_for('home'))
    return render_template('voice-analysis.html', questions=VOICE_QUESTIONS)


@app.route('/analyze_voice', methods=['POST'])
def analyze_voice():
    if 'user' not in session:
        return redirect(url_for('home'))

    # Build responses list from submitted transcripts
    responses = []
    for q in VOICE_QUESTIONS:
        transcript = request.form.get(q["id"], "").strip()
        responses.append({
            "q_id":       q["id"],
            "question":   q["text"],
            "indicator":  q["indicator"],
            "transcript": transcript,
        })

    result = assess_depression(responses)

    depression_level = result.get("depression_level", "Unknown")
    normalized_score = result.get("normalized_score", 0.0)
    score_pct = round(normalized_score * 100)

    save_result(
        session['user'],
        depression_level,
        score_pct,
        "voice"
    )

    return redirect(url_for('results_page'))

# ==============================
# VIDEO ANALYSIS
# ==============================
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'user' not in session:
        return redirect(url_for('home'))

    if 'video' not in request.files or request.files['video'].filename == '':
        flash("Please select a video file before submitting.", "error")
        return redirect(url_for('video_analysis_page'))

    file = request.files['video']

    # Save file
    file_path = os.path.join(TEMP_PATH, "temp.webm")
    file.save(file_path)

    # PLACEHOLDER ANALYSIS
    voice_score = 1
    face_score = 1
    final_score = voice_score + face_score

    if final_score <= 1:
        label = "Low"
    elif final_score <= 3:
        label = "Medium"
    else:
        label = "High"

    save_result(session['user'], label, final_score, "video")
    return redirect(url_for('results_page'))

# ==============================
# LOGOUT
# ==============================
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# ==============================
# GET USER HISTORY
# ==============================
@app.route('/get_results', methods=['GET'])
def get_results():
    if 'user' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    results = get_results_by_email(session['user'])
    return jsonify(results)

# ==============================
# RUN
# ==============================
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")