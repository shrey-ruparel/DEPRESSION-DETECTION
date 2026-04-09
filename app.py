from flask import Flask, request, jsonify, session, render_template, url_for, redirect, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
import sqlite3
import os

# Import your modules
from modules.quiz import predict_result

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.secret_key = "super_secret_key_123"

# ==============================
# PATH SETUP
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "users.db")
TEMP_PATH = os.path.join(BASE_DIR, "temp")

os.makedirs(os.path.join(BASE_DIR, "database"), exist_ok=True)
os.makedirs(TEMP_PATH, exist_ok=True)

# ==============================
# INIT DATABASE
# ==============================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # USERS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    # RESULTS TABLE
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT,
        prediction TEXT,
        score INTEGER,
        type TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

init_db()


# ==============================
# REGISTER
# ==============================
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        hashed_password = generate_password_hash(password)

        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute(
                "INSERT INTO users (email, password) VALUES (?, ?)",
                (email, hashed_password)
            )

            conn.commit()
            conn.close()

            return redirect(url_for('home'))

        except:
            return "User already exists"

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

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM users WHERE email=?", (email,))
    user = cursor.fetchone()

    conn.close()

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

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT prediction, score, type, created_at
        FROM results
        WHERE user_email = ?
        ORDER BY created_at DESC
    """, (session['user'],))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "prediction": row[0],
            "score": row[1],
            "type": row[2],
            "date": row[3]
        })

    latest = results[0] if results else None

    return render_template("results.html", results=results, latest=latest)

# ==============================
# TEXT-QUIZ
# ==============================
@app.route('/text_quiz', methods=['POST'])
def predict_quiz():
    if 'user' not in session:
        return redirect(url_for('home'))

    answers = []
    for i in range(1, 10):
        val = request.form.get(f'q{i}')
        if val is None:
            return "Invalid input"
        answers.append(int(val))

    result = predict_result(answers)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO results (user_email, prediction, score, type)
        VALUES (?, ?, ?, ?)
    """, (
        session['user'],
        result.get("prediction"),
        result.get("score", 0),
        "quiz"
    ))

    conn.commit()
    conn.close()

    return redirect(url_for('results_page'))

# ==============================
# VIDEO ANALYSIS
# ==============================
@app.route('/analyze_video', methods=['POST'])
def analyze_video():
    if 'user' not in session:
        return redirect(url_for('home'))

    if 'video' not in request.files:
        return "No video uploaded"

    file = request.files['video']

    if file.filename == "":
        return "No file selected"

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

    # SAVE RESULT
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO results (user_email, prediction, score, type)
        VALUES (?, ?, ?, ?)
    """, (
        session['user'],
        label,
        final_score,
        "video"
    ))

    conn.commit()
    conn.close()

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

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT prediction, score, type, created_at
        FROM results
        WHERE user_email = ?
        ORDER BY created_at DESC
    """, (session['user'],))

    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "prediction": row[0],
            "score": row[1],
            "type": row[2],
            "date": row[3]
        })

    return jsonify(results)

# ==============================
# RUN
# ==============================
if __name__ == '__main__':
    app.run(debug=True, host = "0.0.0.0")